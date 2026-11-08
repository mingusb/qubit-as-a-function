"""Qubit-as-a-Function command line, TatSu front end and reference evaluator."""
import argparse
import cmath
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import operator
from pathlib import Path

import numpy as np
import tatsu
from tatsu.exceptions import FailedParse

from qaf_core import (BUILTINS, Branch, Operation, P0, P1, Wave, X, Y, Z, select,
                      check_wires, close, controlled, density, euler, evolve,
                      gate, ordered, rotation, sequence, tensor)


class Syntax:
    def uint(self, text):
        return int(text)

    def word(self, text):
        return text[1:-1]

    def literal(self, text):
        value = float(text) if any(c in text for c in ".eE") else int(text)
        return ("literal", value)

    def variable(self, name):
        return ("variable", name)

    def call(self, ast):
        return ("call", ast.fn, tuple(ast.args))

    def negative(self, ast):
        return ("-", ("literal", 0), ast.child)

    def sum(self, ast):
        node = ast.first
        for item in ast.rest:
            node = (item.sign, node, item.term)
        return node

    product = sum

    def expr(self, ast):
        return ("compare",ast.relation,ast.first,ast.second) if ast.relation else ast.first

    def atom(self, ast):
        result=ast.base
        for s in ast.suffix:
            result=("method",result,s.name,tuple(s.args)) if s.open else ("field",result,s.name)
        return result

    def program(self, ast):
        ast.body["library"]=ast.library
        return ast.body


@lru_cache(maxsize=1)
def parser():
    grammar = Path(__file__).with_name("qaf.ebnf").read_text(encoding="utf-8")
    return tatsu.compile(grammar)


ARITH = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


def integer(value):
    z = complex(value)
    if z.imag or not math.isfinite(z.real) or not z.real.is_integer() or z.real < 0:
        raise ValueError("expected a nonnegative integer")
    return int(value.real if isinstance(value, complex) else value)


def bit(value, position):
    return (integer(value) >> integer(position)) & 1


FUNCTIONS = {
    "sqrt": (1, cmath.sqrt), "exp": (1, cmath.exp), "cis": (1, lambda x: cmath.exp(1j*x)),
    "sin": (1, cmath.sin), "cos": (1, cmath.cos), "conj": (1, lambda x: complex(x).conjugate()),
    "eq": (2, lambda a, b: int(a == b)), "bit": (2, bit),
    "mod": (2, lambda a, b: integer(a) % integer(b)),
}
CONSTANTS = {"pi": math.pi, "i": 1j}


def scalar(node, names=()):
    from qaf_language import current
    if current.get() is not None:
        lib=current.get()
        return lambda env:lib.value(node,env)
    kind, *args = node
    if kind == "literal":
        return lambda env: args[0]
    if kind == "variable":
        name = args[0]
        if name in names:
            return lambda env: env[name]
        if name in CONSTANTS:
            return lambda env: CONSTANTS[name]
        raise ValueError(f"unknown scalar name {name}")
    if kind == "call":
        name, arguments = args
        if name not in FUNCTIONS or FUNCTIONS[name][0] != len(arguments):
            raise ValueError(f"unknown scalar function or wrong argument count: {name}")
        fn = FUNCTIONS[name][1]
        parts = [scalar(a, names) for a in arguments]
        return lambda env: fn(*(a(env) for a in parts))
    left, right = (scalar(a, names) for a in args)
    return lambda env: ARITH[kind](left(env), right(env))


def local_names(*names):
    if len(names) != len(set(names)) or set(names) & CONSTANTS.keys():
        raise ValueError("bind distinct variable names other than pi and i")
    return names


def angle(node):
    z = complex(scalar(node)({}))
    if z.imag or not math.isfinite(z.real):
        raise ValueError("gate angles must be finite real numbers in radians")
    return z.real


@dataclass(frozen=True)
class Wrapper:
    before: Operation
    after: Operation

    def __call__(self, slot):
        op = sequence(sequence(self.before, slot), self.after)
        # A superchannel accepts channels, including unitary channels. Its result
        # is exposed as a channel, so coherent control cannot inspect a Kraus phase.
        return Operation(op.matrices)


COMBINATORS = {"seq": (2, sequence), "tensor": (2, tensor), "select": (2,select), "ctrl": (1, controlled),
               "adj": (1, lambda a: gate(a.unitary().conj().T)), "order": (2, ordered)}
ANGLES = {"Rx": (1, lambda t: rotation(X,t)), "Ry": (1, lambda t: rotation(Y,t)),
          "Rz": (1, lambda t: rotation(Z,t)), "U": (3, euler),
          "Phase": (1, lambda t: gate(np.diag([1, np.exp(1j*t)])))}


def operation(node, names):
    name = node.name
    if node.angles is not None:
        if name not in ANGLES or len(node.angles) != ANGLES[name][0]:
            raise ValueError(f"unknown angle gate or wrong arity: {name}")
        return ANGLES[name][1](*(angle(x) for x in node.angles))
    if node.operands is not None:
        operands = [operation(arg, names) for arg in node.operands]
        if name in COMBINATORS and len(operands) == COMBINATORS[name][0]:
            return COMBINATORS[name][1](*operands)
        if isinstance(names.get(name), Wrapper) and len(operands) == 1:
            return names[name](operands[0])
        raise ValueError(f"unknown operation constructor or wrong arity: {name}")
    if not isinstance(names.get(name), Operation):
        raise ValueError(f"expected a declared gate or channel: {name}")
    return names[name]


def definitions(nodes):
    names = BUILTINS.copy()
    for node in nodes:
        if node.name in names or node.name in ANGLES or node.name in COMBINATORS:
            raise ValueError(f"duplicate operation name: {node.name}")
        if node.kind == "super":
            if node.middle != node.slot:
                raise ValueError("the middle operation must be the bound channel slot")
            before, after = operation(node.before, names), operation(node.after, names)
            if before.width != after.width:
                raise ValueError("superchannel input and output wrappers need matching widths")
            names[node.name] = Wrapper(before, after)
            continue
        if node.width < 1:
            raise ValueError("an operation needs at least one qubit")
        indices = local_names(node.row, node.col)
        d = 2**node.width
        entries = (node.element,) if node.kind == "gate" else node.elements
        matrices = []
        for entry in entries:
            fn = scalar(entry, indices)
            matrices.append([[fn({node.row: y, node.col: x}) for x in range(d)]
                             for y in range(d)])
        names[node.name] = Operation(tuple(matrices), is_gate=node.kind == "gate")
    return names


def instructions(nodes, names, width, bits, resolve=operation):
    result = []
    for node in nodes:
        if node.kind in ("if", "measure") and not 0 <= node.bit < bits:
            raise ValueError("classical bit outside the declared register")
        if node.kind == "mulmod":
            n,a=node.modulus,node.factor
            if (n<2 or math.gcd(a,n)!=1 or node.hi-node.lo+1!=(n-1).bit_length()
                    or not 0<=node.lo<=node.hi<width):
                raise ValueError("mulmod needs a coprime multiplier and a correctly sized register")
            check_wires(width,[*range(node.lo,node.hi+1)]+([] if node.control is None else [node.control]))
            shift=width-1-node.hi
            flag=-1 if node.control is None else width-1-node.control
            result.append(("mulmod",n,pow(a,-1,n),shift,(1<<(node.hi-node.lo+1))-1,flag))
        elif node.kind == "apply":
            op = resolve(node.operation, names)
            wires = tuple(node.wires)
            check_wires(width, wires)
            if len(wires) != op.width:
                raise ValueError("operation width and wire count disagree")
            result.append(("apply", op, wires))
        elif node.kind == "measure":
            check_wires(width, [node.wire])
            result.append(("measure", node.wire, node.bit))
        else:
            body = instructions(node.body, names, width, bits, resolve)
            if node.kind == "if":
                if node.value not in (0, 1):
                    raise ValueError("a classical bit is zero or one")
                result.append(("if", node.bit, node.value, body))
            else:
                result.append(("repeat", node.count, body))
    return result


def execute(steps, branches):
    for kind, *args in steps:
        if kind == "mulmod":
            n,inv,shift,mask,flag=args
            labels=np.arange(2**branches[0].wave.width,dtype=np.int64)
            local=(labels>>shift)&mask
            active=local<n
            if flag>=0: active &= (labels&(1<<flag))!=0
            source=np.where(active,labels^((local^(local*inv%n))<<shift),labels)
            branches=[Branch(b.weight,Wave.from_values(b.wave.values()[source]),b.bits) for b in branches]
        elif kind == "apply":
            op, wires = args
            branches = [child for b in branches for child in evolve(b, op.matrices, wires)]
        elif kind == "measure":
            wire, bit_index = args
            branches = [child for b in branches
                        for child in evolve(b, (P0,P1), (wire,), bit_index)]
        elif kind == "repeat":
            count, body = args
            for _ in range(count):
                branches = execute(body, branches)
        else:
            index, value, body = args
            branches = [child for b in branches
                        for child in (execute(body, [b]) if b.bits[index] == value else [b])]
    return branches


def pair(z):
    return [float(z.real), float(z.imag)]


def needs_density(steps):
    for kind,*args in steps:
        if kind in ("if","measure") or (kind=="apply" and not args[0].is_gate):
            return True
        if kind=="repeat" and args[0] and needs_density(args[1]):
            return True
    return False


def observe(query, branches, width):
    if query.kind in ("amp", "prob"):
        if len(query.word) != width:
            raise ValueError("basis word must match register width")
        label = int(query.word, 2)
        if query.kind == "amp":
            if len(branches) != 1:
                raise ValueError("amplitude needs a single trajectory; use density for an ensemble")
            return pair(branches[0].wave(label))
        return sum(b.weight * abs(b.wave(label))**2 for b in branches)
    if query.kind == "probs":
        return sum(b.weight * abs(b.wave.values())**2 for b in branches).tolist()
    if query.kind == "bits":
        distribution = {}
        for b in branches:
            word = "".join(map(str, b.bits))
            distribution[word] = distribution.get(word, 0.0) + b.weight
        return distribution
    return [[pair(z) for z in row] for row in density(branches)]


def run(source):
    from qaf_language import prepare, response
    syntax = prepare(source)
    width, bits = syntax.width, syntax.bits or 0
    if width < 1:
        raise ValueError("a register needs at least one qubit")
    names = definitions(syntax.definitions)
    steps = instructions(syntax.steps, names, width, bits)
    if any(q.kind=="amp" for q in syntax.queries) and needs_density(steps):
        raise ValueError("amplitude inspection requires a unitary-only program")
    labels = range(2**width)
    if syntax.initial is None:
        values = [int(x == 0) for x in labels]
    else:
        state = syntax.initial
        fn = scalar(state.amplitude, local_names(state.index))
        values = [fn({state.index: x}) for x in labels]
    initial = Branch(1.0, Wave.from_values(values), (0,) * bits)
    branches = execute(steps, [initial])
    if not close(sum(b.weight for b in branches), 1.0):
        raise ArithmeticError("register probability drift exceeds tolerance")
    results = [response(q,density(branches),width,np) if q.kind=="response" else
               observe(q, branches, width) for q in syntax.queries]
    return branches, results


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    args = cli.parse_args()
    try:
        _, results = run(args.source.read_text(encoding="utf-8"))
        print(json.dumps(results, indent=2, allow_nan=False))
    except (ValueError, ArithmeticError, FailedParse, OSError) as error:
        cli.exit(1, f"{args.source}: {error}\n")
