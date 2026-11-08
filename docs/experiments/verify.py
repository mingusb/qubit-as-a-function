"""Reproduce the mathematical and execution checks without contacting remote providers."""
import argparse
import importlib.metadata as meta
import json
from pathlib import Path
import numpy as np
from qaf.interpreter import atlas
from qaf.interpreter import hilbert
from qaf.interpreter import run
from qaf.qaf_core import I, X, Y, Z, H, Wave, BUILTINS, density
from qaf.qaf_language import prepare
from qaf.qaf_cert import lower, quil, certify_quil, ccx


ROOT=Path(__file__).resolve().parents[2]
checks={}


def src(name):
    return (ROOT/"examples"/(name+".qaf")).read_text()


def near(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)


def rejects(fn):
    try:
        fn()
    except (ArithmeticError,ValueError):
        return
    raise AssertionError("expected rejection")


def compare(a,b):
    if isinstance(a,dict):
        for key in set(a)|set(b): near(a.get(key,0),b.get(key,0))
    elif isinstance(a,list):
        if len(a)!=len(b): raise AssertionError("incorrect number of results")
        for x,y in zip(a,b): compare(x,y)
    else: near(a,b)


def nonlocal_gate(psi, matrix, wires, width):
    """Independent tensor-axis action used for comparison with bit-index kernels."""
    rest=[i for i in range(width) if i not in wires]
    order=list(wires)+rest
    blocks=psi.reshape((2,)*width).transpose(order).reshape(len(matrix),-1)
    return (matrix@blocks).reshape((2,)*width).transpose(np.argsort(order)).flatten()


def check_ccx():
    import sympy as s
    t=(1+s.I)/s.sqrt(2)
    expected=s.eye(8)
    expected.row_swap(6,7)
    for i in range(8):
        v=s.eye(8)[:,i]
        for name,*w in ccx(0,1,2):
            dest=s.zeros(8,1)
            for k in range(8):
                mask=1<<(2-w[-1])
                b=bool(k&mask)
                if name=="cx":
                    dest[k^mask if k&(1<<(2-w[0])) else k]+=v[k]
                elif name=="h":
                    dest[k&~mask]+=v[k]/s.sqrt(2)
                    dest[k|mask]+=(-1 if b else 1)*v[k]/s.sqrt(2)
                else: dest[k]+=v[k]*((t if name=="t" else s.conjugate(t)) if b else 1)
            v=dest.applyfunc(s.simplify)
        assert v==expected[:,i]
    checks["exact_toffoli"]="all eight columns equal, including phase"


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--gpu",action="store_true")
    cli.add_argument("--providers",action="store_true")
    cli.add_argument("--record",type=Path)
    opt=cli.parse_args()
    if opt.gpu:
        import cupy as cp
        from qaf.qaf_gpu import evaluate
        checks["device"]=cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    def both(source):
        branches,answers=run(source)
        if opt.gpu:
            _,gpu=evaluate(source)
            compare(answers,gpu)
        return branches,answers
    rng=np.random.default_rng(20261108)
    for name in ("callable","teleport","recursion","grover","functions","algebra","bell","bell_state"):
        branches,answers=both(src(name))
        if name=="callable":
            near(answers[0],[.5,0,0,.5]); near(answers[1],.5)
        elif name=="teleport":
            target=np.array([np.cos(np.pi/6),np.exp(1j*np.pi/5)*.5])
            for branch in branches:
                rho=branch.wave.values().reshape(4,2)
                near(rho.T@rho.conj(),np.outer(target,target.conj()))
                near(branch.weight,.25)
        elif name=="grover": near(answers[0],[0,0,0,1])
        elif name=="recursion": near(answers[0],[.5,0,0,0,0,0,0,.5])
        elif name=="algebra": near(answers,[[0,2**-.5],[2**-.5,0]])
        checks[name]=answers
    nested=("qubits 2; bits 1; apply H on 0; measure 0 -> c[0]; "
            "if c[0] == 1 { measure 1 -> c[0]; apply X on 1; } show probs; show bits;")
    _,ans=both(nested); near(ans[0],[.5,0,0,.5]); near(ans[1]["0"],1)
    checks["conditional_record_overwrite"]="passed"
    code=atlas.order_finding()
    _,ans=both(code)
    peaks=np.array(ans[0]).reshape(8,16).sum(axis=1)
    near(peaks,[.25,0,.25,0,.25,0,.25,0])
    checks["order_mod_15"]=peaks.tolist()
    assert code==(ROOT/"examples/order15.qaf").read_text()
    for tries in range(10):
        n=4
        v=rng.normal(size=16)+1j*rng.normal(size=16)
        v/=np.linalg.norm(v)
        wave=Wave.from_values(v)
        a,b=wave.split()
        near(Wave.join(a,b).values(),v)
        w=[int(x) for x in rng.choice(n,size=2,replace=False)]
        # Generate independently chosen nonadjacent two-wire actions.
        source="qubits 4; state(b) = "+"+".join(
            f"eq(b,{k})*({x.real:.17g}+i*({x.imag:.17g}))" for k,x in enumerate(v))+";"
        source+=f"apply CX on {w[0]},{w[1]}; show density;"
        branch,_=both(source)
        ref=nonlocal_gate(v,BUILTINS["CX"].unitary(),w,n)
        near(density(branch),np.outer(ref,ref.conj()))
    checks["random_registers"]=10
    invalid=["qubits 1; state(b)=0;","qubits 1; gate Bad[1](y,x)=0; apply Bad on 0;",
             "qubits 1; state(b)=eq(b,0)/(1-1);","qubits 1; show response 0 [1,1,0,0];",
             "qubits 2; apply CX on 0,0;","qubits 1; apply Reset on 0; show amp \"0\";"]
    for source in invalid:
        rejects(lambda:run(source))
        if opt.gpu: rejects(lambda:evaluate(source))
    checks["rejected_programs"]=len(invalid)
    for _ in range(10):
        mat=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
        rho=mat@mat.conj().T; rho/=np.trace(rho)
        near(atlas.from_bloch(atlas.bloch(rho)),rho)
        near(atlas.from_sic(atlas.sic(rho)),rho)
        _,recovered=atlas.discrete_wigner(rho); near(recovered,rho)
        kd=atlas.kirkwood_dirac(rho,I,H)
        near(kd.sum(axis=1),np.diag(rho)); near(kd.sum(axis=0),np.diag(H@rho@H))
    p=np.array([1,1j])/np.sqrt(2)
    near(p@atlas.majorana_root(p),0); near(atlas.bargmann([1,2],.25),1.5)
    rho=np.outer(p,p.conj())
    K=BUILTINS["Reset"].matrices
    near(atlas.choi_apply(atlas.choi(K),rho),sum(k@rho@k.conj().T for k in K))
    a=np.array([[.2,.3j],[.1,.45]])
    near(atlas.cubic_svt(a),4*a@a.conj().T@a-3*a)
    near(atlas.two_step([I]),np.diag([1,0]))
    near(atlas.two_step(BUILTINS["Reset"].matrices),I/2)
    energies=[atlas.field_grid(k) for k in (2,3,4,5,6)]
    assert all(x<y for x,y in zip(energies,energies[1:])) and energies[-1]<np.pi**2/2
    checks["box_energies"]=energies
    checks["tomography_and_functional_calculus"]="passed"
    for name in ("hilbert","wave_packet"):
        ast=prepare(src(name)); state=hilbert.parse_state(ast)
        answers=hilbert.evaluate(ast); checks[name]=answers
        rank=8
        emitted=hilbert.register_source(ast,rank)
        _,actual=both(emitted)
        raw=np.array([[sum(complex(re,im)*float(atom.at(i,j)) for re,im,atom in state)
                       for j in range(rank)] for i in range(rank)])
        near(np.sum(abs(raw)**2),float(hilbert.norm2(state,rank)))
        near(actual[0],(abs(raw)**2/np.sum(abs(raw)**2)).flatten())
    assert checks["hilbert"]==[{"tail":"1/43046721"},{"fidelity":"24/25"},{"svm":"-64/225"}]
    from fractions import Fraction
    from numpy.polynomial.hermite import hermgauss
    nodes,weights=hermgauss(20)
    def quadrature(c):
        x=np.sqrt(2)*nodes[:,None]
        y=c*x+np.sqrt(2*(1-c*c))*nodes[None,:]
        return np.sum(weights[:,None]*weights[None,:]*(
            np.sin(x)*np.sin(y)+c*np.cos(x)*np.cos(y)))/np.pi
    for c in (Fraction(0),Fraction(3,4),Fraction(8,9),Fraction(24,25),Fraction(1)):
        near(float(hilbert.sine_kernel(c)),quadrature(float(c)))
    result=hilbert.evaluate(prepare(src("nonlinear")))[0]["network"]
    alpha=min(2,1/(1-quadrature(.75)))
    near(float(result["decimal"]),alpha*(quadrature(8/9)-quadrature(24/25)))
    checks["nonlinear_infinite_width"]=result
    rejects(lambda:hilbert.evaluate(prepare("hilbert {1 * diagonal(1);} show tail 2;")))
    rejects(lambda:hilbert.evaluate(prepare("hilbert {1 * basis(0,1); -1 * basis(0,1);} show tail 2;")))
    check_ccx()
    portable=lower(src("bell_state")); target=quil(portable)
    checks["bell_certificate"]=certify_quil(portable,target)
    for tamper in [target.replace("CNOT 0 1","CNOT 1 0"),target.replace("ro[0]","ro[1]"),
                   target.replace("H 0","X 0")]:
        rejects(lambda:certify_quil(portable,tamper))
    if opt.providers:
        from qaf.qaf_cloud import sample, ionq_payload, google_circuit
        from qaf.qaf_energy import exhaustive,ocean
        from dimod import ExactSolver
        for name in ("braket-local","cirq-local"):
            ans=sample(portable,name,"",100)["queries"][0]
            near(ans[1:3],[0,0]); near(ans[0]+ans[3],1)
            assert min(ans[0],ans[3])>0
            checks[name]=ans
        # A reset of the branch bit must not change which body was selected.
        nested_sample=sample(lower(nested),"cirq-local","",128)["queries"]
        near(nested_sample[0][1:3],[0,0]); near(nested_sample[1]["0"],1)
        checks["ionq_payload"]=ionq_payload(portable)
        import cirq_google, cirq
        device=cirq_google.Sycamore
        native=google_circuit(portable,device)
        sampled=cirq.Simulator(seed=22).run(native,repetitions=100)
        near(sampled.measurements["m0"],sampled.measurements["m1"])
        assert sum(sampled.measurements["m0"])[0] in range(1,100)
        plain=lower("qubits 2; apply CX on 0,1; show probs;")
        grid=google_circuit(plain,device)
        order=[next(o.qubits[0] for o in grid.all_operations() if cirq.is_measurement(o)
                    and cirq.measurement_key_name(o)==f"m{i}") for i in range(2)]
        unitary=cirq.drop_terminal_measurements(grid).unitary(qubit_order=order)
        cirq.testing.assert_allclose_up_to_global_phase(unitary,cirq.unitary(cirq.CNOT),atol=1e-12)
        checks["google_grid_device"]="Sycamore metadata validation and local sampling"
        for name in ("onehot","antiferromagnet"):
            model=prepare(src(name)); expected=exhaustive(model,np)
            if opt.gpu: assert exhaustive(model,cp)==expected
            actual=ocean(model,ExactSolver())
            assert actual["word"] in expected["words"]
            near(actual["best_observed_energy"],expected["minimum"])
            checks[name]=expected
    names=["TatSu","numpy","cupy-cuda13x","cirq-google","amazon-braket-sdk","azure-quantum",
           "dimod","dwave-system","matplotlib","sympy","latexminted","Pygments"]
    versions={}
    for name in names:
        try: versions[name]=meta.version(name)
        except meta.PackageNotFoundError: pass
    checks["versions"]=versions
    out=json.dumps(checks,indent=2,allow_nan=False)
    if opt.record: opt.record.write_text(out+"\n",encoding="utf8")
    print(out)


if __name__=="__main__": main()
