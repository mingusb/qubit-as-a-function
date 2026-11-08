"""Print QAF as indentation-delimited prefix forms suitable for publication."""
from qaf import Syntax,parser
from qaf_layout import expand_layout
from qaf_prefix import forms,is_prefix,lower


def expression(node):
    head,*a=node
    if head=="literal": return str(a[0])
    if head=="variable": return a[0]
    if head=="call": return [a[0],*map(expression,a[1])]
    if head=="method": return ["method",expression(a[0]),a[1],*map(expression,a[2])]
    if head=="field": return ["field",expression(a[0]),a[1]]
    if head=="compare": return [a[0],expression(a[1]),expression(a[2])]
    return [head,expression(a[0]),expression(a[1])]


def op(node):
    if node.angles is not None: return [node.name,*map(expression,node.angles)]
    if node.operands is not None: return [node.name,*map(op,node.operands)]
    return node.name


def declarations(n):
    params=n.fields if n.kind=="class" else n.params
    names=[["type",p.name,p.type] if p.type else p.name for p in params]
    items=[["args",*names]] if any(isinstance(x,list) for x in names) else names
    if n.kind=="class": return [n.kind,n.name,*items,*map(declarations,n.methods)]
    result=[["returns",n.result]] if n.result else []
    return [n.kind,n.name,*items,*result,*map(step,n.body)]


def step(s):
    k=s.kind
    if k in ("set","let"): return [k,s.name,expression(s.value)]
    if k=="return": return [k,expression(s.value)]
    if k in ("if","while") and s.condition is not None:
        body=s.yes if k=="if" else s.body
        return [k,expression(s.condition),*map(step,body),*([["else",*map(step,s.no)]] if s.no else [])]
    if k=="place": return [k,op(s.operation),*map(expression,s.args)]
    if k=="do": return [k,s.name,*map(expression,s.args)]
    if k=="mulmod":
        return [k,*map(expression,[s.factor,s.modulus,s.lo,s.hi]),*([expression(s.control)] if s.control else [])]
    if k=="apply": return [k,op(s.operation),*map(str,s.wires)]
    if k=="invoke": return [k,str(s.ref),*map(str,s.wires)]
    if k=="measure": return [k,str(s.wire),str(s.bit)]
    if k=="if": return ["if_bit",str(s.bit),str(s.value),*map(step,s.body)]
    if k=="repeat": return [k,str(s.count),*map(step,s.body)]
    raise ValueError(k)


def definition(d):
    if d.kind=="fn": return ["fn",d.name,str(d.wire),op(d.first),op(d.second)]
    if d.kind=="super": return ["super",d.name,d.slot,op(d.before),op(d.after)]
    entries=[d.element] if d.kind=="gate" else d.elements
    return [d.kind,d.name,str(d.width),d.row,d.col,*map(expression,entries)]


def series_atom(t): return [t.kind,str(t.a),*([str(t.b)] if t.b is not None else [])]


def tree(text):
    if is_prefix(text): return forms(text)
    a=parser().parse(expand_layout(text)[0],semantics=Syntax())
    statements=list(map(declarations,a.library))
    if a.model=="hilbert":
        statements.append(["hilbert",*[["term",str(t.real),*([str(t.imag)] if t.imag else []),series_atom(t.atom)] for t in a.terms]])
        for q in a.queries:
            args=[str(q.rank)] if q.kind=="tail" else [series_atom(q.first)]
            if q.second is not None: args.append(series_atom(q.second))
            if q.cost is not None: args.append(str(q.cost))
            statements.append(["show",q.kind,*args])
        return statements
    if a.model=="network":
        return [["network",a.name,*[[l.kind,*([str(l.width)] if l.width is not None else [])] for l in a.layers]]]
    if a.model:
        statements.append([a.model,str(a.width)])
        for t in a.terms:
            labels=[str(x) for x in (t.a,t.b) if x is not None]
            statements.append([t.kind,*labels,expression(t.value)])
        return statements
    statements.append(["qubits",str(a.width)])
    if a.bits is not None: statements.append(["bits",str(a.bits)])
    statements+=list(map(definition,a.definitions))
    if a.initial: statements.append(["state",a.initial.index,expression(a.initial.amplitude)])
    statements+=list(map(step,a.steps))
    for q in a.queries:
        statement=["show",q.kind]
        if q.kind in ("amp","prob"): statement.append(q.word)
        if q.kind=="response": statement += [str(q.ref),*map(expression,q.coeffs)]
        statements.append(statement)
    return statements


def show(tree,level=0):
    if isinstance(tree,str): return ["    "*level+tree]
    def flat(expr):
        if isinstance(expr,str): return [expr]
        if expr[0] in ("method","pipe"): return None
        parts=[flat(child) for child in expr]
        return None if any(x is None for x in parts) else [v for p in parts for v in p]
    if tree[0] in ("return","let","set","state") and isinstance(tree[-1],list):
        tail=flat(tree[-1])
        if tail is not None and len(" ".join(tail))<=65:
            return ["    "*level+" ".join([*tree[:-1],*tail])]
    split=next((i for i,x in enumerate(tree) if isinstance(x,list)),len(tree))
    line="    "*level+" ".join(tree[:split])
    return [line,*[line for x in tree[split:] for line in show(x,level+1)]]


def pretty(source):
    roots=tree(source)
    lines=["syntax prefix",""]
    for i,form in enumerate(roots):
        if i and (form[0] in ("def","proc","class","qubits") or roots[i-1][0] in ("def","proc","class")):
            lines.append("")
        lines.extend(show(form))
    result="\n".join(lines)+"\n"
    before=lower(source) if is_prefix(source) else expand_layout(source)[0]
    parse=lambda text:parser().parse(text,semantics=Syntax())
    if parse(before)!=parse(lower(result)): raise ValueError("prefix rendering changed the program")
    return result
