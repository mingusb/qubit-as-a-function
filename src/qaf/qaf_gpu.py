"""GPU-first lowering: compile scalar functions and evolve device-resident programs."""
from dataclasses import dataclass
from contextvars import ContextVar
from functools import lru_cache
import hashlib
from pathlib import Path

import cupy as cp
import numpy as np

from .interpreter import (ANGLES, ARITH, COMBINATORS, CONSTANTS, FUNCTIONS, Syntax, instructions,
                 local_names, needs_density, pair, parser)
from .qaf_core import BUILTINS, TOL

captured_sources = ContextVar("qaf_captured_sources", default=None)
replay_module = ContextVar("qaf_replay_module", default=None)


@lru_cache(maxsize=1)
def module():
    source = Path(__file__).with_name("qaf_kernels.cu").read_text()
    return cp.RawModule(code=source, options=("--std=c++17",))


def launch(name, out, *args):
    threads = 128
    blocks = (out.size + threads-1) // threads
    native = replay_module.get() or module()
    native.get_function(name)((blocks,), (threads,), args)


PRELUDE = """
#include <cupy/complex.cuh>
using C = complex<double>;
__device__ bool natural(C a) {
    return a.imag()==0 && a.real()>=0 && a.real()<9007199254740992.0
           && floor(a.real())==a.real();
}
__device__ C bit_value(C a, C b) {
    if (!natural(a) || !natural(b)) return C(nan(""),0);
    if (b.real()>=53) return C(0,0);
    return C(((unsigned long long)a.real() >> (int)b.real()) & 1ULL,0);
}
__device__ C mod_value(C a, C b) {
    if (!natural(a) || !natural(b) || b.real()==0) return C(nan(""),0);
    return C(fmod(a.real(),b.real()),0);
}
"""


def emit(node, names):
    from .qaf_language import current
    lib=current.get()
    kind, *args = node
    if kind == "literal":
        v = complex(args[0])
        if not np.isfinite(v):
            raise ValueError("literal exceeds double precision range")
        return f"C({v.real:.17e},0)"
    if kind == "variable":
        name = args[0]
        if name in names:
            return names[name]
        if name not in CONSTANTS:
            raise ValueError(f"unknown scalar {name}")
        return "C(0,1)" if name == "i" else "C(3.14159265358979323846,0)"
    if kind in ARITH:
        if kind=="/":
            return f"qdiv({emit(args[0],names)},{emit(args[1],names)},fuel)"
        return f"({emit(args[0],names)}{kind}{emit(args[1],names)})"
    if kind=="field":
        return f"({emit(args[0],names)}).v_{args[1]}"
    if kind=="method":
        vals=["fuel","depth+1"]+[emit(x,names) for x in args[2]]
        return f"({emit(args[0],names)}).m_{args[1]}({','.join(vals)})"
    if kind=="compare":
        op,a,b=args
        aa,bb=emit(a,names),emit(b,names)
        if op not in ("==","!="): aa,bb=f"qreal({aa},fuel)",f"qreal({bb},fuel)"
        return f"C(({aa}{op}{bb})?1:0,0)"
    fn, nodes = args
    if lib and fn in lib.classes:
        return "Q_"+fn+"{"+",".join(emit(n,names) for n in nodes)+"}"
    if lib and fn in lib.routines:
        return "q_"+fn+"("+",".join(["fuel","depth+1"]+[emit(n,names) for n in nodes])+")"
    if fn not in FUNCTIONS or len(nodes) != FUNCTIONS[fn][0]:
        raise ValueError(f"unknown scalar function or wrong argument count: {fn}")
    a = [emit(n,names) for n in nodes]
    if fn == "eq":
        return f"C(({a[0]}=={a[1]})?1:0,0)"
    if fn == "cis":
        return f"exp(C(0,1)*{a[0]})"
    cuda_fn = {"bit":"bit_value","mod":"mod_value","sqrt":"thrust::sqrt"}.get(fn,fn)
    return f"{cuda_fn}({','.join(a)})"


def library_code(lib):
    def ty(t): return "C" if t in (None,"num") else "Q_"+t
    def params(d): return ",".join(["int &fuel","int depth"]+
                                  [ty(p.type)+" v_"+p.name for p in d.params])
    def signature(d,owner=None,local=False):
        name=("m_" if owner else "q_")+d.name
        if owner and not local: name="Q_"+owner+"::"+name
        return "__device__ __noinline__ "+ty(d.result)+" "+name+"("+params(d)+")"+(" const" if owner else "")
    def block(body,env):
        lines=[]
        for s in body:
            lines.append("if (--fuel<0) return {};")
            if s.kind=="return": lines.append("return "+emit(s.value,env)+";")
            elif s.kind in ("let","set"):
                if (s.name in env)!=(s.kind=="set"): raise ValueError("invalid local binding")
                rhs=emit(s.value,env)
                line=("auto " if s.kind=="let" else "")+"v_"+s.name+"="+rhs+";"
                lines.append(line)
                env[s.name]="v_"+s.name
            elif s.kind in ("if","while"):
                lines += [s.kind+"(qreal("+emit(s.condition,env)+",fuel)!=0) {",
                          "if (--fuel<0) return {};"]
                lines+=block(s.yes if s.kind=="if" else s.body,env.copy())+["}"]
                if s.kind=="if" and s.no: lines+=["else {"]+block(s.no,env.copy())+["}"]
            else: raise ValueError("staging statement inside a device routine")
        return lines
    code=["__device__ double qreal(C z,int &fuel) { if(z.imag()!=0 || !isfinite(z)) {fuel=-1;return 0;}return z.real();}",
          "__device__ C qdiv(C a,C b,int &fuel) {if(b==C(0,0)){fuel=-1;return C(0,0);}return a/b;}"]
    if not lib: return "\n".join(code)
    for c in lib.classes.values():
        code+=["struct Q_"+c.name+" {"]
        code += [ty(p.type)+" v_"+p.name+";" for p in c.fields]
        code += [signature(d,c.name,True)+";" for d in c.methods]+["};"]
    code += [signature(d)+";" for d in lib.routines.values()]
    items=[(d,None) for d in lib.routines.values()]
    items += [(d,c.name) for c in lib.classes.values() for d in c.methods]
    for d,owner in items:
        env={p.name:"v_"+p.name for p in d.params}
        if owner: env["self"]="(*this)"
        code += [signature(d,owner)+" {","if(depth>32 || fuel<0) {fuel=-1; return {};}"]
        code += block(d.body,env)+["fuel=-1; return {};","}"]
    return "\n".join(code)


def grid_function(expr, variables, rows, cols):
    from .qaf_language import current
    names = dict(zip(variables, ("C((double)y,0)", "C((double)x,0)")))
    code = "int fuel=4096, depth=0; value = " + emit(expr,names) + "; if(fuel<0) value=C(nan(\"\"),0);"
    preamble=PRELUDE+"\n"+library_code(current.get())
    name = "qaf_expr_" + hashlib.sha256((code+preamble).encode()).hexdigest()[:16]
    fn = cp.ElementwiseKernel("int64 y, int64 x", "complex128 value", code, name,
                              preamble=preamble)
    values = fn(rows, cols)
    capture = captured_sources.get()
    if capture is not None:
        for signature,compiled in fn.cached_codes.items():
            capture[name+"_"+hashlib.sha256(str(signature).encode()).hexdigest()[:8]]=compiled
    if not bool(cp.all(cp.isfinite(values))):
        raise ValueError("nonfinite result in device scalar expression")
    return values


@dataclass(frozen=True)
class DeviceOp:
    matrices: object
    is_gate: bool

    @property
    def width(self):
        return self.matrices.shape[-1].bit_length()-1

    def unitary(self):
        if not self.is_gate:
            raise ValueError("a phase-specified gate is required")
        return self.matrices[0]


def gate(matrix):
    return DeviceOp(cp.ascontiguousarray(matrix[None], dtype=cp.complex128), True)


def seq(first, second):
    if first.width != second.width:
        raise ValueError("sequential operations must have equal widths")
    d = 2**first.width
    matrices = (second.matrices[None,:] @ first.matrices[:,None]).reshape(-1,d,d)
    return DeviceOp(cp.ascontiguousarray(matrices), first.is_gate and second.is_gate)


def builtins():
    # Only fixed literal matrices enter from the host; user kernels are generated on device.
    return {k:DeviceOp(cp.asarray(np.stack(v.matrices)),v.is_gate) for k,v in BUILTINS.items()}


def resolve(node, names):
    name = node.name
    if node.angles is not None:
        if name not in ANGLES or len(node.angles) != ANGLES[name][0]:
            raise ValueError("unknown parametrized gate or wrong arity")
        zero = cp.zeros((), dtype=cp.int64)
        raw = [grid_function(v, (), zero, zero) for v in node.angles]
        if any(bool(cp.any(v.imag != 0)) for v in raw):
            raise ValueError("angles must be real")
        theta, *other = [v.real for v in raw]
        if name == "Phase":
            return gate(cp.diag(cp.stack((cp.ones_like(theta),cp.exp(1j*theta)))))
        c, s = cp.cos(theta/2), cp.sin(theta/2)
        if name == "U":
            p, l = other
            return gate(cp.stack((c, -cp.exp(1j*l)*s, cp.exp(1j*p)*s,
                                  cp.exp(1j*(p+l))*c)).reshape(2,2))
        return gate(c*names["I"].unitary() - 1j*s*names[name[-1].upper()].unitary())
    if node.operands is None:
        if not isinstance(names.get(name), DeviceOp):
            raise ValueError(f"unknown gate or channel {name}")
        return names[name]
    ops = [resolve(a,names) for a in node.operands]
    if isinstance(names.get(name), tuple) and len(ops)==1:
        before, after = names[name]
        return DeviceOp(seq(seq(before,ops[0]),after).matrices, False)
    if name not in COMBINATORS or len(ops)!=COMBINATORS[name][0]:
        raise ValueError(f"unknown constructor or wrong arity: {name}")
    first = ops[0]
    if name == "adj":
        return gate(first.unitary().conj().T)
    if name == "ctrl":
        a = first.unitary()
        mat = cp.eye(2*len(a), dtype=cp.complex128)
        mat[len(a):,len(a):] = a
        return gate(mat)
    second = ops[1]
    if name == "seq":
        return seq(first,second)
    if name == "tensor":
        mats = cp.stack([cp.kron(a,b) for a in first.matrices for b in second.matrices])
        return DeviceOp(mats, first.is_gate and second.is_gate)
    a,b = first.unitary(),second.unitary()
    if a.shape != b.shape:
        raise ValueError("order requires gates of equal width")
    mat = cp.zeros((2*len(a),2*len(a)),dtype=cp.complex128)
    mat[:len(a),:len(a)],mat[len(a):,len(a):] = (a,b) if name=="select" else (b@a,a@b)
    return gate(mat)


def definitions(nodes):
    names = builtins()
    for obj in nodes:
        if obj.name in names or obj.name in ANGLES or obj.name in COMBINATORS:
            raise ValueError(f"duplicate operation {obj.name}")
        if obj.kind == "super":
            a,b = resolve(obj.before,names),resolve(obj.after,names)
            if obj.middle != obj.slot or a.width != b.width:
                raise ValueError("invalid single-slot superchannel")
            names[obj.name] = (a,b)
            continue
        if obj.width < 1:
            raise ValueError("zero-width operation")
        indices = local_names(obj.row,obj.col)
        basis = cp.arange(2**obj.width, dtype=cp.int64)
        exprs = (obj.element,) if obj.kind=="gate" else obj.elements
        ops = cp.stack([grid_function(e, indices, basis[:,None], basis[None,:]) for e in exprs])
        gram = (ops.conj().transpose(0,2,1) @ ops).sum(axis=0)
        if not bool(cp.allclose(gram,cp.eye(len(basis)),rtol=0,atol=TOL)):
            raise ValueError("device gate unitarity or channel trace preservation failed")
        names[obj.name] = DeviceOp(ops,obj.kind=="gate")
    return names


def update(steps, state, width, bits, mixed):
    n = 2**width
    for kind,*args in steps:
        if kind=="mulmod":
            modulus,inverse,shift,mask,flag=args
            if modulus.bit_length()>32:
                raise ValueError("native modular permutation supports up to 32 target bits")
            out=cp.empty_like(state)
            launch("transform_modular",out,state,out,np.uint64(modulus),np.uint64(inverse),
                   np.int32(shift),np.uint64(mask),np.int32(flag),np.uint64(n),
                   np.uint64(state.size),np.int32(mixed))
            state=out
        elif kind=="apply":
            op,wires = args
            shifts = cp.asarray([width-1-q for q in wires],dtype=cp.int32)
            out = cp.empty_like(state)
            params = (state,out,op.matrices,shifts,np.int32(len(wires)))
            if mixed:
                launch("transform_density",out,*params,np.int32(len(op.matrices)),
                       np.uint64(n),np.uint64(out.size))
            else:
                launch("transform_wave",out,*params,np.uint64(n))
            state = out
        elif kind=="measure":
            q,c = args
            out = cp.empty_like(state)
            launch("measure_z",out,state,out,np.int32(width-1-q),np.int32(bits-1-c),
                   np.uint64(n),np.uint64(out.size))
            state = out
        elif kind=="repeat":
            for _ in range(args[0]):
                state = update(args[1],state,width,bits,mixed)
        else:
            c,value,body = args
            labels = cp.arange(state.shape[0], dtype=cp.int64)
            selected = (((labels >> (bits-1-c)) & 1)==value)[:,None,None]
            taken = update(body, cp.where(selected,state,0j),width,bits,True)
            state = taken + cp.where(selected,0j,state)
    return state


def evaluate(source, record_plan=None):
    from .qaf_language import prepare, response
    ast = prepare(source)
    width,bits = ast.width,ast.bits or 0
    if width < 1:
        raise ValueError("a register needs at least one qubit")
    names = definitions(ast.definitions)
    steps = instructions(ast.steps,names,width,bits,resolve)
    mixed = needs_density(steps)
    n = 2**width
    required = 2*16*((2**bits)*n*n if mixed else n)
    if required > cp.cuda.Device().mem_info[0]:
        raise MemoryError(f"device buffers require at least {required} bytes")
    labels = cp.arange(n,dtype=cp.int64)
    if ast.initial is None:
        wave = (labels==0).astype(cp.complex128)
    else:
        obj = ast.initial
        wave = grid_function(obj.amplitude,local_names(obj.index),labels,cp.int64(0))
        if not bool(cp.isclose(cp.sum(cp.abs(wave)**2),1,rtol=0,atol=TOL)):
            raise ValueError("the prepared state must have squared norm one")
    if mixed:
        state = cp.zeros((2**bits,n,n),dtype=cp.complex128)
        state[0] = wave[:,None]*wave.conj()[None,:]
    else:
        state = wave
    if record_plan is not None:
        record_plan(width,bits,mixed,steps,wave)
    out = update(steps,state,width,bits,mixed)
    if mixed:
        diagonal = cp.diagonal(out,axis1=1,axis2=2).real
        probs = diagonal.sum(axis=0)
    else:
        probs = cp.abs(out)**2
    if not bool(cp.isclose(probs.sum(),1,rtol=0,atol=TOL)):
        raise ArithmeticError("device probability drift exceeds tolerance")
    results = []
    for query in ast.queries:
        kind = query.kind
        if kind in ("amp","prob"):
            if len(query.word)!=width:
                raise ValueError("basis word must match register width")
            index = int(query.word,2)
            if kind=="amp" and mixed:
                raise ValueError("amplitude inspection requires a unitary-only program")
            value = pair(complex(out[index].item())) if kind=="amp" else float(probs[index])
        elif kind=="bits":
            if mixed:
                weights = diagonal.sum(axis=1).get()
                value = {(format(i,f"0{bits}b") if bits else ""):float(p)
                         for i,p in enumerate(weights) if p>0}
            else:
                value = {"0"*bits:1.0}
        elif kind in ("density","response"):
            rho = out.sum(axis=0) if mixed else out[:,None]*out.conj()[None,:]
            value = response(query,rho,width,cp) if kind=="response" else [[pair(z) for z in row] for row in rho.get()]
        else:
            value = probs.tolist()
        results.append(value)
    return out,results
