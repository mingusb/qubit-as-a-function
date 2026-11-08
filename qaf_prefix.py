"""TatSu S-expressions with indentation as an alternative grouping operation."""
from functools import lru_cache
from pathlib import Path
import re
from contextvars import ContextVar
import tatsu


def is_prefix(source):
    line=next((v.strip() for v in source.splitlines() if v.strip() and not v.lstrip().startswith("#")),"")
    return line in ("syntax prefix","syntax lisp") or line.startswith("(")


@lru_cache(maxsize=1)
def grammar():
    return tatsu.compile(Path(__file__).with_name("prefix.ebnf").read_text())


def layout(source):
    out,offsets,indents=[],[0],[]
    pos=level=0
    def raw(a,b):
        out.append(source[a:b]); offsets.extend(range(a+1,b+1))
    def insert(s,i):
        out.append(s); offsets.extend([i]*len(s))
    for line in source.splitlines(keepends=True):
        end=pos+len(line)
        code=line.partition("#")[0].strip()
        pad=line[:len(line)-len(line.lstrip(" \t"))]
        col=len(pad.expandtabs(8))
        if code in ("syntax prefix","syntax lisp") and not out or (
                code in ("syntax prefix","syntax lisp") and all(not x.strip() or x.lstrip().startswith("#") for x in out)):
            out.append(" "*len(line.rstrip("\r\n"))+line[len(line.rstrip("\r\n")):])
            offsets.extend(range(pos+1,end+1)); pos=end; continue
        if code and level==0:
            while indents and col<=indents[-1][0]:
                if col==indents[-1][0] and pad!=indents[-1][1]:
                    raise ValueError("inconsistent indentation at the same prefix depth")
                insert(") ",pos); indents.pop()
            if indents:
                if indents[-1][2] is None: indents[-1][2]=(col,pad)
                if indents[-1][2]!=(col,pad): raise ValueError("align sibling prefix forms")
            if not code.startswith("("):
                raw(pos,pos+len(pad)); insert("(",pos+len(pad)); pos+=len(pad)
                indents.append([col,pad,None])
        raw(pos,end); pos=end
        level+=code.count("(")-code.count(")")
        if level<0: raise ValueError("unexpected closing parenthesis")
    if level: raise ValueError("unclosed parenthesis")
    if indents: insert("\n"+")"*len(indents),len(source))
    return "".join(out),offsets


class Reader:
    def group(self,ast): return list(ast.parts)
    def prefix_program(self,ast): return list(ast.forms)


def forms(source):
    def mutable(node):
        return [mutable(v) for v in node] if isinstance(node,(list,tuple)) else node
    return mutable(grammar().parse(layout(source)[0],semantics=Reader()))


OPS={"+","-","*","/","==","!=","<",">","<=",">="}
ANGLES={"U","Phase","Rx","Ry","Rz"}
arity=ContextVar("prefix_arity",default={})


def operands(tokens):
    def take(i):
        if i>=len(tokens): raise ValueError("missing prefix operand")
        item=tokens[i]; end=i+1
        count=arity.get().get(item,0) if isinstance(item,str) else 0
        if not count: return item,end
        result=[item]
        for _ in range(count):
            child,end=take(end); result.append(child)
        return result,end
    out=[]; i=0
    while i<len(tokens):
        value,i=take(i); out.append(value)
    return out


def atom(node):
    while isinstance(node,list) and len(node)==1: node=node[0]
    if not isinstance(node,str): raise ValueError(f"expected a name or literal: {node}")
    return node


def expr(node):
    if isinstance(node,str): return node
    if len(node)==1: return atom(node)
    name,*args=node
    if name=="pipe":
        value=expr(args[0])
        for call in args[1:]:
            method,*parameters=call
            value+=f".{method}("+", ".join(map(expr,parameters))+")"
        return value
    if name in arity.get(): args=operands(args)
    if name in OPS:
        if len(args)!=2: raise ValueError(f"{name} takes two arguments; group nested expressions")
        return f"({expr(args[0])} {name} {expr(args[1])})"
    if name=="neg": return f"(-{expr(args[0])})"
    if name=="field": return f"{expr(args[0])}.{atom(args[1])}"
    if name=="method":
        return f"{expr(args[0])}.{atom(args[1])}("+", ".join(map(expr,args[2:]))+")"
    return f"{name}("+", ".join(map(expr,args))+")"


def operation(node):
    if isinstance(node,str) or len(node)==1: return atom(node)
    name,*args=node
    if name in ANGLES: return name+"["+", ".join(map(expr,args))+"]"
    return name+"("+", ".join(map(operation,args))+")"


def scalar_tail(nodes):
    return expr(nodes[0] if len(nodes)==1 else nodes)


def render(form,staged=False):
    if not isinstance(form,list) or not form: raise ValueError(f"expected a QAF form: {form}")
    head,*args=form
    if head in ("class","def","proc"):
        name=atom(args.pop(0))
        args=list(args); params=[]; ret=""
        while args and isinstance(args[0],str): params.append(args.pop(0))
        if args and args[0] and args[0][0] in ("args","params"):
            params.extend(args.pop(0)[1:])
        elif args and isinstance(args[0],list) and args[0] and args[0][0] not in {
            "return","let","set","if","while","do","place","def","mulmod","returns",*OPS} and len(args)>1:
            # A conventional Lisp parameter list, e.g. (def sq (x) (* x x)).
            params.extend(args.pop(0))
        if args and args[0][0]=="returns": ret=" -> "+atom(args.pop(0)[1])
        p=", ".join(x if isinstance(x,str) else atom(x[-2])+": "+atom(x[-1]) for x in params)
        body=[]
        for i,x in enumerate(args):
            if head=="def" and i==len(args)-1 and x[0] not in ("return","if","while","set","let"):
                body.append("return "+expr(x)+";")
            else: body.append(render(x,True))
        return f"{head} {name}({p}){ret} {{ "+" ".join(body)+" }"
    if head in ("qubits","bits","qubo","ising"): return f"{head} {atom(args[0])};"
    if head=="state": return f"state({atom(args[0])}) = {scalar_tail(args[1:])};"
    if head in ("let","set"): return f"{head} {atom(args[0])} = {scalar_tail(args[1:])};"
    if head=="return": return "return "+scalar_tail(args)+";"
    if head=="do": return "do "+atom(args[0])+"("+", ".join(map(expr,args[1:]))+");"
    if head in ("if","while"):
        yes=args[1:]; no=[]
        if yes and yes[-1][0]=="else": no=yes[-1][1:]; yes=yes[:-1]
        return head+" ("+expr(args[0])+") { "+" ".join(render(x,True) for x in yes)+" }"+(
            " else { "+" ".join(render(x,True) for x in no)+" }" if no else "")
    if head in ("apply","place"):
        wires=", ".join(map(expr,args[1:]))
        return head+" "+operation(args[0])+" on "+("("+wires+")" if head=="place" else wires)+";"
    if head=="mulmod":
        a,n,lo,hi,*ctrl=args
        return f"mulmod {expr(a)} modulo {expr(n)} on ({expr(lo)}, {expr(hi)})"+(
            " by "+expr(ctrl[0]) if ctrl else "")+";"
    if head=="fn":
        return f"fn {atom(args[0])} at {atom(args[1])} = {operation(args[2])} | {operation(args[3])};"
    if head=="invoke": return "invoke "+atom(args[0])+" on "+", ".join(map(atom,args[1:]))+";"
    if head=="measure": return f"measure {atom(args[0])} -> c[{atom(args[1])}];"
    if head=="if_bit": return f"if c[{atom(args[0])}] == {atom(args[1])} {{ "+" ".join(render(x) for x in args[2:])+" }"
    if head=="repeat": return f"repeat {atom(args[0])} {{ "+" ".join(render(x) for x in args[1:])+" }"
    if head=="gate": return f"gate {args[0]}[{args[1]}]({args[2]}, {args[3]}) = {scalar_tail(args[4:])};"
    if head=="channel": return f"channel {args[0]}[{args[1]}]({args[2]}, {args[3]}) = kraus {{ "+", ".join(map(expr,args[4:]))+" };"
    if head=="super": return f"super {args[0]}({args[1]}) = {operation(args[2])} >> {args[1]} >> {operation(args[3])};"
    if head=="show":
        what,*a=args
        if what in ("amp","prob"):
            word=atom(a[0]).strip('"')
            return f'show {what} "{word}";'
        if what=="response": return f"show response {atom(a[0])} ["+", ".join(map(expr,a[1:]))+"];"
        if what in ("svm","network","fidelity"):
            size=1 if what=="fidelity" else 2
            tail=" penalty "+atom(a[-1]) if what=="network" else ""
            return f"show {what} "+", ".join(expr(o).replace(" ","") for o in a[:size])+tail+";"
        return "show "+" ".join(map(atom,args))+";"
    if head=="hilbert":
        terms=[]
        for term in args:
            if term[0]!="term": raise ValueError("Hilbert expansions consist of term forms")
            re_,*imag,vector=term[1:]
            im=" imag "+atom(imag[0]) if imag else ""
            terms.append(f"{atom(re_)}{im} * "+expr(vector).replace(" ","")+";")
        return "hilbert { "+" ".join(terms)+" }"
    if head in ("linear","pair","offset"):
        return head+" "+", ".join(map(atom,args[:-1]))+" = "+expr(args[-1])+";"
    if head=="network":
        name,*layers=args
        return "network "+name+" { "+" ".join(" ".join(map(atom,x))+";" for x in layers)+" }"
    raise ValueError(f"unknown prefix form {head}")


def lower(source):
    data=forms(source)
    signatures={k:2 for k in OPS|{"eq","mod","bit","field"}}
    signatures.update({k:1 for k in ("neg","sin","cos","cis","sqrt","exp","conj")})
    for f in data:
        if f[0] in ("def","class"):
            params=[]
            for x in f[2:]:
                if isinstance(x,str): params.append(x)
                elif x[0] in ("args","params"): params.extend(x[1:]); break
                else: break
            if params: signatures[f[1]]=len(params)
    token=arity.set(signatures)
    try: return "\n".join(render(f) for f in data)
    finally: arity.reset(token)
