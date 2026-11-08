"""Generate/replay a factoring experiment; no factor or order enters circuit construction."""
import argparse
import hashlib
import json
from math import gcd, isqrt
from pathlib import Path
import numpy as np

from importlib.resources import files


def parameters(n, base, method):
    if n<9 or n%2==0 or not 1<base<n or gcd(base,n)!=1:
        raise ValueError("use an odd composite and a nontrivial coprime base")
    bits=n.bit_length()
    if method=="shor": return bits, bits, 0
    if method!="shortlog": raise ValueError("unknown factoring method")
    m=(bits+1)//2-1
    return bits,m,m


def circuit(n,base,method):
    bits,m,l=parameters(n,base,method)
    head=files("qaf_examples").joinpath("prefix", "factoring.qaf").read_text(encoding="utf8").split("\nqubits ")[0]
    if method=="shor": width=bits+2*m; call=f"order_finding {n} {base} {bits} {2*m}"
    else: width=bits+m+2*l; call=f"short_log {n} {base} {bits} {m} {l}"
    return f"{head}\nqubits {width}\napply X {width-1}\ndo {call}\nshow probs\n"


def oracle_spectrum(n,g,method):
    """Independent Fourier reference: enumerate modular values, transform each register."""
    bits,m,l=parameters(n,g,method)
    a_size=2**(m+l if l else 2*m)
    b_size=2**l
    x=pow(g,(n-1)//2-2**m,n) if l else 1
    labels=np.empty((a_size,b_size),dtype=int)
    for a in range(a_size):
        for b in range(b_size):
            labels[a,b]=pow(g,a,n)*pow(x,-b,n)%n
    probs=np.zeros((a_size,b_size))
    for z in np.unique(labels):
        wave=(labels==z)/np.sqrt(a_size*b_size)
        probs+=abs(np.fft.fftn(wave,axes=(0,1),norm="ortho"))**2
    return probs


def decode(n,g,method,j,k):
    from quaspy.math.groups import IntegerModRingMulSubgroupElement as Element
    from quaspy.orderfinding.general.postprocessing.ekera import solve_j_for_r_mod_N
    from quaspy.logarithmfinding.short.postprocessing import solve_j_k_for_d
    _,m,l=parameters(n,g,method)
    if method=="shor":
        r=solve_j_for_r_mod_N(j,m,m,g,n,B=2)
        if r is None or int(r)%2: return None
        r=int(r)
        if pow(g,r,n)!=1: return None
        h=pow(g,r//2,n)
        p=next((v for v in [gcd(h-1,n),gcd(h+1,n)] if 1<v<n),None)
        if p is None: return None
        return {"order":r,"factors":sorted([p,n//p])}
    x=pow(g,(n-1)//2-2**m,n)
    # Both Fourier transforms use the same negative sign. Simultaneous negation
    # of j and k leaves the real input state's joint outcome law unchanged.
    d=solve_j_k_for_d(j,k,m,l,Element(g,n),Element(x,n),tau=min(2,l),t=min(1,m-1))
    if d is None: return None
    d=int(d)
    if not (0<=d<2**m and pow(g,d,n)==x): return None
    total=2*(d+2**m+1); disc=total**2-4*n
    if disc<0: return None
    root=isqrt(disc)
    if root*root!=disc or (total-root)%2: return None
    p=(total-root)//2; q=(total+root)//2
    if p<=1 or p*q!=n: return None
    return {"short_log":d,"factors":[p,q]}


def experiment(n,g,method,shots,seed,backend,verify=False):
    from .interpreter import run
    if shots<1: raise ValueError("shots must be positive")
    bits,m,l=parameters(n,g,method)
    source=circuit(n,g,method)
    if backend=="cuda":
        from .qaf_gpu import evaluate
        wave,_=evaluate(source)
        import cupy as cp
        marginal=(cp.abs(wave.reshape(-1,2**l,2**bits))**2).sum(axis=2).get()
    else:
        _,answers=run(source)
        marginal=np.array(answers[0]).reshape(-1,2**l,2**bits).sum(axis=2)
    probability=marginal.flatten()
    reference_error=None
    if verify:
        reference_error=float(np.max(np.abs(marginal-oracle_spectrum(n,g,method))))
        if reference_error>1e-10: raise ArithmeticError("modular spectrum mismatch")
    rng=np.random.default_rng(seed)
    outcomes=rng.choice(len(probability),size=shots,p=probability/probability.sum())
    attempts=[]
    for word in outcomes:
        j,k=divmod(int(word),2**l)
        recovered=decode(n,g,method,j,k)
        attempts.append({"frequency":[j,k] if l else [j],"recovered":recovered})
    return {"modulus":n,"base":g,"method":method,"qubits":bits+(m+2*l if l else 2*m),
            "backend":backend,"seed":seed,"shots":shots,"verified_shots":sum(
                v["recovered"] is not None for v in attempts),
            "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
            "max_frequency_error":reference_error,"samples":attempts}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("modulus",type=int); cli.add_argument("--base",type=int,default=2)
    cli.add_argument("--method",choices=["shor","shortlog"],default="shortlog")
    cli.add_argument("--shots",type=int,default=16); cli.add_argument("--seed",type=int,default=20261108)
    cli.add_argument("--backend",choices=["cuda","cpu"],default="cuda")
    cli.add_argument("--emit",type=Path); cli.add_argument("--verify",action="store_true")
    cli.add_argument("--record",type=Path)
    args=cli.parse_args()
    if args.emit:
        args.emit.write_text(circuit(args.modulus,args.base,args.method),encoding="utf8")
    else:
        evidence=experiment(args.modulus,args.base,args.method,args.shots,args.seed,args.backend,args.verify)
        text=json.dumps(evidence,indent=2,allow_nan=False)
        if args.record: args.record.write_text(text+"\n",encoding="utf8")
        print(text)
