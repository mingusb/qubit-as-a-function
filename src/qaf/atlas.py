"""Independent small-matrix constructions used in the historical and mathematical examples."""
import math
import numpy as np
from .qaf_core import I,X,Y,Z

PAULI=np.stack([X,Y,Z])
TETRA=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)


def bloch(rho):
    return np.trace(PAULI@rho,axis1=1,axis2=2).real


def from_bloch(r):
    return (I+np.einsum("k,kab->ab",r,PAULI))/2


def sic(rho):
    return (1+TETRA@bloch(rho))/4


def from_sic(p):
    return from_bloch(3*np.asarray(p)@TETRA)


def discrete_wigner(rho):
    points=(I+np.einsum("ik,kab->iab",TETRA*np.sqrt(3),PAULI))/2
    weights=np.trace(points@rho,axis1=1,axis2=2).real/2
    return weights,np.einsum("i,iab->ab",weights,points)


def kirkwood_dirac(rho,basis_a,basis_b):
    # Basis vectors are columns. K_ab = <b|a><a|rho|b>.
    return (basis_b.conj().T@basis_a).T*(basis_a.conj().T@rho@basis_b)


def bargmann(coeff,z):
    return sum(a*z**j/math.sqrt(math.factorial(j)) for j,a in enumerate(coeff))


def majorana_root(psi):
    # A homogeneous degree-one section a*u+b*v vanishes at [b:-a].
    a,b=psi
    return np.array([b,-a])


def choi(kraus):
    return sum(np.outer(v,v.conj()) for k in kraus for v in [k.T.reshape(-1)])


def choi_apply(j,rho):
    d=len(rho)
    return np.einsum("ij,iajb->ab",rho,j.reshape(d,-1,d,len(j)//d))


def matrix_function(h,f):
    values,vectors=np.linalg.eigh(h)
    return (vectors*f(values))@vectors.conj().T


def cubic_svt(a):
    d=len(a)
    ident=np.eye(d)
    root=lambda v:np.sqrt(np.maximum(v,0))
    u=np.block([[a,matrix_function(ident-a@a.conj().T,root)],
                [matrix_function(ident-a.conj().T@a,root),-a.conj().T]])
    z=np.diag([1]*d+[-1]*d)
    return (u@z@u.conj().T@z@u)[:d,:d]


def two_step(intervention):
    from .qaf_core import BUILTINS
    cnot=BUILTINS["CX"].unitary()
    plus=np.array([1,1])/np.sqrt(2)
    joint=np.kron(np.outer(plus,plus),np.diag([1,0]))
    joint=cnot@joint@cnot.conj().T
    joint=sum(k@joint@k.conj().T for op in intervention for k in [np.kron(I,op)])
    joint=cnot@joint@cnot.conj().T
    return np.trace(joint.reshape(2,2,2,2),axis1=0,axis2=2)


def field_grid(width):
    # Dirichlet box on (0,1), hbar=m=1, interior points only.
    n=2**width
    dx=1/(n+1)
    h=(2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1))/(2*dx**2)
    psi=np.sin(np.pi*np.arange(1,n+1)*dx)
    psi/=np.linalg.norm(psi)
    return float(psi@h@psi)


def qft_steps(n,inverse=False):
    gates=[]
    for j in range(n):
        gates.append(("H",[j]))
        for k in range(j+1,n):
            gates.append((f"ctrl(Phase[pi/{2**(k-j)}])",[k,j]))
    gates.extend(("SWAP",[j,n-1-j]) for j in range(n//2))
    if inverse:
        gates=[(f"adj({op})",wires) for op,wires in reversed(gates)]
    return "\n".join(f"apply {op} on "+",".join(map(str,wires))+";" for op,wires in gates)


def order_finding():
    lines=["qubits 7;"]
    for k in range(3):
        factor=pow(2,2**(2-k),15)
        lines.append(f"gate M{k}[4](y,x) = eq(x,15)*eq(y,15)+(1-eq(x,15))*eq(y,mod({factor}*x,15));")
    lines.append("state(b) = eq(b,1);")
    for k in range(3):
        lines.extend([f"apply H on {k};",f"apply ctrl(M{k}) on {k},3,4,5,6;"])
    from .qaf_style import format_source
    return format_source("\n".join([*lines,qft_steps(3,True),"show probs;"])+"\n")
