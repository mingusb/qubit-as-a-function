"""Exact squared amplitudes, overlaps and tails of two-mode series on ℓ²(N²)."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction as Q


@dataclass(frozen=True)
class Atom:
    kind: str
    a: int | Q
    b: int | Q = 0

    @classmethod
    def parse(cls, obj):
        if obj.kind == "basis":
            return cls("basis", obj.a, obj.b)
        a, b = Q(obj.a), Q(obj.b or 0)
        if max(abs(a), abs(b)) >= 1:
            raise ValueError("geometric ratios require absolute value below one")
        return cls(obj.kind, a, b)

    def at(self, i, j):
        if self.kind == "basis":
            return Q((i, j) == (self.a, self.b))
        if self.kind == "diagonal":
            return self.a**i if i == j else Q(0)
        return self.a**i * self.b**j


def geometric(ratio, rank):
    return (1 if rank is None else 1-ratio**rank) / (1-ratio)


def gram(left, right, rank=None):
    """Inner product inside the rank-square projection, or in the full space."""
    if right.kind == "basis":
        left, right = right, left
    if left.kind == "basis":
        if rank is not None and max(left.a, left.b) >= rank:
            return Q(0)
        return right.at(left.a, left.b)
    if left.kind == right.kind == "product":
        return geometric(left.a*right.a, rank)*geometric(left.b*right.b, rank)
    ratio = left.a*right.a
    for atom in (left, right):
        if atom.kind == "product":
            ratio *= atom.b
    return geometric(ratio, rank)


def inner(left, right, rank=None):
    real = imag = Q(0)
    for a, b, first in left:
        for c, d, second in right:
            g = gram(first, second, rank)
            real += (a*c+b*d)*g
            imag += (a*d-b*c)*g
    return real, imag


def norm2(state, rank=None):
    return inner(state, state, rank)[0]


def fidelity(left, right):
    re, im = inner(left, right)
    return (re**2+im**2)/(norm2(left)*norm2(right))


def parse_state(ast):
    state = [(Q(t.real), Q(t.imag or 0), Atom.parse(t.atom)) for t in ast.terms]
    if norm2(state) <= 0:
        raise ValueError("zero series has no normalized state")
    return state


def single(source):
    return [(Q(1), Q(0), Atom.parse(source))]


def sine_kernel(c):
    """Infinite-width two-layer sine NTK on unit inputs. c is their inner product."""
    d = Decimal(c.numerator)/Decimal(c.denominator)
    positive, negative = (d-1).exp(), (-d-1).exp()
    return ((1+d)*positive-(1-d)*negative)/2


def network(state, positive, negative, cost):
    if cost <= 0:
        raise ValueError("soft-margin penalty must be positive")
    u, v = fidelity(state,positive), fidelity(state,negative)
    k = fidelity(positive,negative)
    if k == 1:
        return {"expression":"0","decimal":"0"}
    # Extra significant digits distinguish nearly identical support vectors.
    with localcontext() as ctx:
        ctx.prec = 45 + len(str(k.denominator))
        c = Decimal(cost.numerator)/Decimal(cost.denominator)
        alpha = min(c, 1/(1-sine_kernel(k)))
        value = alpha*(sine_kernel(u)-sine_kernel(v))
    return {"expression":f"min({cost},1/(1-k({k})))*(k({u})-k({v}))",
            "decimal":format(value,".20g")}


def evaluate(ast):
    state = parse_state(ast)
    result = []
    for q in ast.queries:
        if q.kind == "tail":
            v = 1-norm2(state, q.rank)/norm2(state)
        elif q.kind == "fidelity":
            v = fidelity(state, single(q.first))
        elif q.kind == "network":
            answer = network(state,single(q.first),single(q.second),Q(q.cost))
            result.append({"network":answer})
            continue
        else:
            pos, neg = single(q.first), single(q.second)
            denom = 1-fidelity(pos, neg)
            if denom == 0:
                raise ValueError("opposite training labels on the same ray")
            v = (fidelity(state, pos)-fidelity(state, neg))/denom
        result.append({q.kind: str(v)})
    return result


def register_source(ast, rank):
    """Emit a normalized finite QAF circuit using the exact projected Gram norm."""
    if rank < 1:
        raise ValueError("projection rank must be positive")
    state = parse_state(ast)
    mass = norm2(state, rank)
    if mass <= 0:
        raise ValueError("projection has zero support")
    width = max(1, (rank-1).bit_length())
    terms = []
    for re, im, atom in state:
        if atom.kind == "basis":
            formula = f"eq(row,{atom.a})*eq(col,{atom.b})"
        elif atom.kind == "diagonal":
            formula = f"eq(row,col)*power(({atom.a}),row)"
        else:
            formula = f"power(({atom.a}),row)*power(({atom.b}),col)"
        terms.append(f"(({re})+i*({im}))*({formula})")
    series = "+".join(terms)
    # Decode both integer coordinates explicitly from the MSB-first bit word.
    row = "+".join(f"{2**j}*bit(b,{j+width})" for j in range(width))
    col = "+".join(f"{2**j}*bit(b,{j})" for j in range(width))
    from .qaf_style import format_source
    return format_source(f"def power(x,n) {{ let v=1; while (n>0) {{ set v=v*x; set n=n-1; }} return v; }}\n"
            f"def packet(b) {{ let row={row}; let col={col}; "
            f"return (row<{rank})*(col<{rank})*({series})/sqrt({mass}); }}\n"
            f"qubits {2*width};\nstate(b) = packet(b);\nshow probs;\n")
