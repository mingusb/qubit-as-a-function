"""Classical definitions, typed records, staging, and the single quantum-function type."""
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace as NS
import operator
import math

from qaf import ARITH, CONSTANTS, FUNCTIONS, Syntax, integer, parser
from qaf_core import check_wires

current=ContextVar("qaf_classical_definitions",default=None)
REL={"==":operator.eq,"!=":operator.ne,"<":operator.lt,"<=":operator.le,
     ">":operator.gt,">=":operator.ge}


@dataclass
class Record:
    typename:str
    fields:dict


class Returned(Exception):
    pass


def real(x):
    z=complex(x)
    if z.imag or not math.isfinite(z.real):
        raise ValueError("expected a finite real number")
    return z.real


def named(items):
    mapping={x.name:x for x in items}
    if len(mapping)!=len(items): raise ValueError("duplicate declaration or parameter")
    return mapping


class Library:
    def __init__(self, declarations):
        self.names=named(declarations)
        if set(self.names)&(FUNCTIONS.keys()|CONSTANTS.keys()|{"num","self"}):
            raise ValueError("reserved declaration name")
        self.classes={n:d for n,d in self.names.items() if d.kind=="class"}
        self.routines={n:d for n,d in self.names.items() if d.kind=="def"}
        self.procs={n:d for n,d in self.names.items() if d.kind=="proc"}
        self.methods={n:named(d.methods) for n,d in self.classes.items()}
        for d in declarations:
            fields=d.fields if d.kind=="class" else d.params
            named(fields)
            if any((p.type or "num") not in {"num",*self.classes} for p in fields):
                raise ValueError("unknown declared type")

    def value(self,node,env,budget=None,depth=0):
        if budget is None: budget=[4096]
        kind,*a=node
        if kind=="literal": return a[0]
        if kind=="variable":
            if a[0] in env: return env[a[0]]
            if a[0] in CONSTANTS: return CONSTANTS[a[0]]
            raise ValueError(f"unbound variable {a[0]}")
        ev=lambda x:self.value(x,env,budget,depth)
        if kind in ARITH: return ARITH[kind](ev(a[0]),ev(a[1]))
        if kind=="compare":
            x,y=ev(a[1]),ev(a[2])
            if a[0] not in ("==","!="): x,y=real(x),real(y)
            return int(REL[a[0]](x,y))
        if kind in ("method","field"):
            obj=ev(a[0])
            if not isinstance(obj,Record): raise ValueError("a record is required")
            if kind=="field": return obj.fields[a[1]]
            method=self.methods[obj.typename][a[1]]
            return self.call(method,[ev(x) for x in a[2]],budget,depth+1,obj)
        name,nodes=a
        vals=[ev(x) for x in nodes]
        if name in FUNCTIONS:
            arity,fn=FUNCTIONS[name]
            if len(vals)!=arity: raise ValueError("wrong scalar argument count")
            return fn(*vals)
        if name in self.classes:
            return Record(name,self.bind(self.classes[name].fields,vals))
        if name not in self.routines: raise ValueError(f"unknown scalar routine {name}")
        return self.call(self.routines[name],vals,budget,depth+1)

    def bind(self,params,values):
        if len(params)!=len(values): raise ValueError("wrong argument count")
        for p,v in zip(params,values):
            actual=v.typename if isinstance(v,Record) else "num"
            if (p.type or "num")!=actual: raise ValueError(f"type mismatch for {p.name}")
        return dict(zip((p.name for p in params),values))

    def call(self,fn,values,budget,depth,owner=None,emitted=None):
        if depth>32: raise ValueError("recursion depth exceeds 32")
        env=self.bind(fn.params,values)
        if owner is not None: env["self"]=owner
        try:
            self.block(fn.body,env,budget,depth,emitted)
        except Returned as r:
            result=r.args[0]
            if fn.kind=="proc": raise ValueError("a procedure has no scalar result")
            actual=result.typename if isinstance(result,Record) else "num"
            if actual!=(fn.result or "num"): raise ValueError("return type mismatch")
            return result
        if fn.kind!="proc": raise ValueError(f"{fn.name} ends without returning")

    def block(self,body,env,budget,depth,emitted):
        def tick():
            budget[0]-=1
            if budget[0]<0: raise ValueError("classical evaluation budget exhausted")
        ev=lambda x:self.value(x,env,budget,depth)
        def nested(statements):
            local=env.copy()
            self.block(statements,local,budget,depth,emitted)
            for name in env:
                env[name]=local[name]
        for s in body:
            tick()
            if s.kind=="return": raise Returned(ev(s.value))
            if s.kind in ("let","set"):
                if (s.name in env)!=(s.kind=="set"):
                    raise ValueError(f"invalid {s.kind} of {s.name}")
                env[s.name]=ev(s.value)
            elif s.kind=="if":
                nested(s.yes if real(ev(s.condition)) else s.no or [])
            elif s.kind=="while":
                while real(ev(s.condition)):
                    tick()
                    nested(s.body)
            elif s.kind=="place" and emitted is not None:
                emitted.append(NS(kind="apply",operation=freeze_op(s.operation,ev),
                                  wires=[integer(ev(x)) for x in s.args]))
            elif s.kind=="mulmod" and emitted is not None:
                emitted.append(modular_step(s,ev))
            elif s.kind=="do" and emitted is not None and s.name in self.procs:
                self.call(self.procs[s.name],[ev(x) for x in s.args],budget,depth+1,emitted=emitted)
            else: raise ValueError("invalid action in this declaration")


def plain(name,operands=None):
    return NS(name=name,angles=None,operands=operands)


def freeze_op(op,ev):
    if op.angles is not None:
        return NS(name=op.name,angles=[("literal",real(ev(v))) for v in op.angles],operands=None)
    return plain(op.name,[freeze_op(x,ev) for x in op.operands] if op.operands is not None else None)


def modular_step(s,ev):
    return NS(kind="mulmod",factor=integer(ev(s.factor)),modulus=integer(ev(s.modulus)),
              lo=integer(ev(s.lo)),hi=integer(ev(s.hi)),
              control=None if s.control is None else integer(ev(s.control)))


def prepare(source):
    from qaf_layout import expand_layout
    from qaf_prefix import is_prefix,lower
    expanded=lower(source) if is_prefix(source) else expand_layout(source)[0]
    ast=parser().parse(expanded,semantics=Syntax())
    lib=Library(ast.library)
    current.set(lib)
    if ast.get("model"): return ast
    aliases={}
    retained=[]
    for decl in ast.definitions:
        if decl.kind=="fn":
            if decl.name in aliases: raise ValueError("duplicate function-qubit binding")
            check_wires(ast.width,[decl.wire])
            aliases[decl.name]=(decl.wire,decl.first,decl.second)
        else: retained.append(decl)
    def binding(ref):
        if isinstance(ref,int): return ref,plain("I"),plain("X")
        if ref not in aliases: raise ValueError(f"unknown function qubit {ref}")
        return aliases[ref]
    def expand(steps):
        out=[]
        for step in steps:
            if step.kind=="mulmod":
                out.append(modular_step(step,lambda v:lib.value(v,{})))
            elif step.kind=="invoke":
                wire,a,b=binding(step.ref)
                out.append(NS(kind="apply",operation=plain("select",[a,b]),wires=[wire,*step.wires]))
            elif step.kind=="do":
                if step.name not in lib.procs: raise ValueError("unknown circuit procedure")
                lib.call(lib.procs[step.name],[lib.value(x,{}) for x in step.args],[4096],0,emitted=out)
            elif step.kind in ("if","repeat"):
                fields=dict(step)
                fields["body"]=expand(step.body)
                out.append(NS(**fields))
            else: out.append(step)
        return out
    queries=[]
    for q in ast.queries:
        if q.kind=="response":
            wire,_,_=binding(q.ref)
            check_wires(ast.width,[wire])
            coeff=[real(lib.value(x,{})) for x in q.coeffs]
            if len(coeff)!=4: raise ValueError("effect needs four real Pauli coefficients")
            a,*v=coeff
            if a-math.hypot(*v)<-1e-12 or a+math.hypot(*v)>1+1e-12:
                raise ValueError("effect spectrum must lie in [0,1]")
            q=NS(kind="response",wire=wire,coeff=coeff)
        queries.append(q)
    return NS(width=ast.width,bits=ast.bits,definitions=retained,initial=ast.initial,
              steps=expand(ast.steps),queries=queries,library=lib)


def response(query,rho,width,xp):
    a,x,y,z=query.coeff
    e=xp.asarray([[a+z,x-1j*y],[x+1j*y,a-z]],dtype=xp.complex128)
    axis=query.wire
    tensor=rho.reshape((2,)*(2*width))
    order=[axis]+[q for q in range(width) if q!=axis]
    tensor=tensor.transpose(order+[q+width for q in order]).reshape(2,-1,2,2**(width-1))
    reduced=xp.trace(tensor,axis1=1,axis2=3)
    return float(xp.trace(e@reduced).real)
