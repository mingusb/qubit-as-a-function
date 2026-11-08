"""Exact, finite gate-word translation. Each emitted operation has a named ideal matrix."""
from dataclasses import dataclass
import hashlib
import json

from .interpreter import Syntax, parser
from .qaf_core import check_wires


@dataclass
class Portable:
    width: int
    bits: int
    total: int
    gates: list
    queries: list
    source_hash: str


def ccx(a,b,c):
    # Exact Clifford+T decomposition, checked independently on all eight columns.
    return [("h",c),("cx",b,c),("ti",c),("cx",a,c),("t",c),("cx",b,c),
            ("ti",c),("cx",a,c),("t",b),("t",c),("h",c),("cx",a,b),
            ("t",a),("ti",b),("cx",a,b)]


def lower(source):
    from .qaf_language import prepare
    ast = prepare(source)
    if getattr(ast,"model",None) or ast.width<1:
        raise ValueError("gate certification requires a positive-width circuit")
    n,m = ast.width,ast.bits or 0
    if ast.initial is not None or ast.definitions:
        raise ValueError("the exact portable profile accepts gate preparation and named primitive gates")
    if any(q.kind not in ("prob","probs","bits") for q in ast.queries):
        raise ValueError("this target certifies measurement statistics")
    if any(q.kind=="prob" and len(q.word)!=n for q in ast.queries):
        raise ValueError("basis word must match register width")
    gates = []
    total = n+m

    def fresh():
        nonlocal total
        total += 1
        return total-1

    def append(word,guard):
        for op,*w in word:
            if guard is None:
                gates.append((op,*w))
            elif op=="x":
                gates.append(("cx",guard,w[0]))
            elif op=="cx":
                gates.extend(ccx(guard,*w))
            elif op=="z":
                gates.extend([("h",w[0]),("cx",guard,w[0]),("h",w[0])])
            else:
                raise ValueError(f"no exact controlled rule registered for {op}")

    def physical(op,w):
        if op.angles is not None:
            raise ValueError("continuous-angle gate lacks an exact portable encoding in this profile")
        if op.operands is not None:
            if op.name=="select" and len(op.operands)==2 and len(w)==2:
                left,right=[physical(x,w[1:]) for x in op.operands]
                if not left and right==[("x",w[1])]: return [("cx",*w)]
            if op.name=="seq" and len(op.operands)==2:
                return physical(op.operands[0],w)+physical(op.operands[1],w)
            if op.name=="ctrl" and len(op.operands)==1 and len(w)==2:
                child = physical(op.operands[0],w[1:])
                if child==[("x",w[1])]:
                    return [("cx",*w)]
            raise ValueError(f"no exact portable constructor for {op.name}")
        if op.name in ("I","H","X","Y","Z","S","T") and len(w)==1:
            return [] if op.name=="I" else [(op.name.lower(),*w)]
        if op.name in ("CX","CZ","SWAP") and len(w)==2:
            a,b=w
            if op.name=="CX": return [("cx",a,b)]
            if op.name=="CZ": return [("h",b),("cx",a,b),("h",b)]
            return [("cx",a,b),("cx",b,a),("cx",a,b)]
        raise ValueError(f"uncertified operation or wrong wire count: {op.name}")

    def block(steps,guard=None):
        for step in steps:
            if step.kind=="apply":
                w = tuple(step.wires)
                check_wires(n,w)
                append(physical(step.operation,w),guard)
            elif step.kind=="repeat":
                for _ in range(step.count): block(step.body,guard)
            elif step.kind in ("measure","if"):
                if step.bit >= m:
                    raise ValueError("classical bit outside register")
                record = n+step.bit
                if step.kind=="measure":
                    check_wires(n,[step.wire])
                    history = fresh()
                    # The history wire starts at zero. Move the old record there.
                    append([("cx",record,history),("cx",history,record),
                            ("cx",step.wire,record)],guard)
                else:
                    if step.value not in (0,1):
                        raise ValueError("invalid bit value")
                    flag = fresh()
                    if step.value==0: gates.append(("x",record))
                    append([("cx",record,flag)],guard)
                    if step.value==0: gates.append(("x",record))
                    block(step.body,flag)
            else:
                raise ValueError(f"no exact portable gate lowering for {step.kind}")
    block(ast.steps)
    return Portable(n,m,total,gates,ast.queries,hashlib.sha256(source.encode()).hexdigest())


LABELS = {"h":"H","x":"X","y":"Y","z":"Z","s":"S","t":"T",
          "si":"DAGGER S","ti":"DAGGER T","cx":"CNOT"}


def quil(program):
    width = program.width+program.bits
    lines = [f"DECLARE ro BIT[{width}]"]
    lines += [LABELS[name]+" "+" ".join(map(str,wires)) for name,*wires in program.gates]
    lines += [f"MEASURE {i} ro[{i}]" for i in range(width)]
    return "\n".join(lines)+"\n"


def read_quil(text):
    import re
    lines = text.splitlines()
    if not lines or not (header:=re.fullmatch(r"DECLARE ro BIT\[(\d+)\]",lines[0])):
        raise ValueError("unsupported Quil readout declaration")
    width = int(header[1])
    reverse = {"H":"h","X":"x","Y":"y","Z":"z","S":"s","T":"t",
               "DAGGER S":"si","DAGGER T":"ti","CNOT":"cx"}
    word,readouts = [],[]
    for line in lines[1:]:
        if match:=re.fullmatch(r"MEASURE (\d+) ro\[(\d+)\]",line):
            readouts.append((int(match[1]),int(match[2])))
        elif not readouts and (match:=re.fullmatch(r"(DAGGER [ST]|CNOT|[HXYZST]) (\d+)(?: (\d+))?",line)):
            op = reverse[match[1]]
            wires = tuple(int(w) for w in match.groups()[1:] if w is not None)
            if len(wires)!=(2 if op=="cx" else 1):
                raise ValueError("incorrect Quil arity")
            word.append((op,*wires))
        else:
            raise ValueError(f"unrecognized or misplaced Quil instruction {line}")
    return width,word,readouts


def trace_word(word,n):
    # Equal per-wire histories allow only swaps of gates on disjoint wires.
    histories=[[] for _ in range(n)]
    for instruction in word:
        op,*w=instruction
        check_wires(n,w)
        for q in w: histories[q].append(tuple(instruction))
    return histories


def certify(program,word,readouts):
    count=program.width+program.bits
    if (trace_word(program.gates,program.total)!=trace_word(word,program.total)
        or sorted(readouts)!=[(i,i) for i in range(count)]):
        raise ValueError("target gate word, wire map or readout differs from QAF")
    digest=hashlib.sha256(json.dumps([word,readouts],separators=(",",":")).encode()).hexdigest()
    return {"source_sha256":program.source_hash,"gate_readout_sha256":digest,
            "ideal_profile":"QAF Clifford+T with terminal records","primitive_count":len(word)}


def certify_quil(program,text):
    count,word,writes=read_quil(text)
    if count!=program.width+program.bits:
        raise ValueError("Quil readout width mismatch")
    cert=certify(program,word,writes)
    cert["quil_sha256"]=hashlib.sha256(text.encode()).hexdigest()
    return cert
