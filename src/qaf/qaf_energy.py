"""An energy-function workload with CUDA, CPU and D-Wave Ocean translations."""
from collections import defaultdict

from .interpreter import angle


def coefficients(ast):
    if ast.width < 1:
        raise ValueError("an energy model needs at least one variable")
    linear = {i:0.0 for i in range(ast.width)}
    pairs, offset = defaultdict(float), 0.0
    for term in ast.terms:
        v = angle(term.value)
        if term.kind == "offset":
            offset += v
            continue
        if term.a >= ast.width or (term.kind=="pair" and term.b >= ast.width):
            raise ValueError("variable outside energy model")
        if term.kind == "linear":
            linear[term.a] += v
        elif term.a==term.b:
            if ast.model=="ising":
                offset += v
            else:
                linear[term.a] += v
        else:
            pairs[tuple(sorted((term.a,term.b)))] += v
    return linear,dict(pairs),offset


def exhaustive(ast, xp):
    h,j,offset = coefficients(ast)
    if ast.width >= 63:
        raise ValueError("exact enumeration exceeds the integer index range")
    n = 2**ast.width
    labels = xp.arange(n,dtype=xp.int64)
    energy = xp.full(n,offset,dtype=xp.float64)
    def value(k):
        x = (labels >> (ast.width-1-k)) & 1
        return 2*x-1 if ast.model=="ising" else x
    for k,a in h.items():
        energy += a*value(k)
    for (k,l),a in j.items():
        energy += a*value(k)*value(l)
    best = float(energy.min())
    indices = xp.flatnonzero(energy == best).tolist()
    return {"minimum":best,"words":[format(i,f"0{ast.width}b") for i in indices]}


def ocean(ast, sampler=None, reads=100):
    import dimod
    h,j,offset = coefficients(ast)
    variable = dimod.SPIN if ast.model=="ising" else dimod.BINARY
    model = dimod.BinaryQuadraticModel(h,j,offset,variable)
    actual_pairs = {tuple(sorted(edge)):v for edge,v in model.quadratic.items()}
    if (dict(model.linear)!=h or actual_pairs!=j or model.offset!=offset
            or model.vartype!=variable):
        raise ValueError("Ocean energy function differs from the source model")
    if sampler is None:
        from dwave.system import DWaveSampler, EmbeddingComposite
        import os
        options = {"solver":os.environ["QAF_DWAVE_SOLVER"]} if os.getenv("QAF_DWAVE_SOLVER") else {}
        with DWaveSampler(**options) as device:
            samples = EmbeddingComposite(device).sample(model,num_reads=reads)
            samples.resolve()
    else:
        samples = sampler.sample(model)
    best = samples.first
    if abs(float(best.energy)-float(model.energy(best.sample)))>1e-9:
        raise ValueError("returned energy disagrees with the source polynomial")
    word = "".join("1" if best.sample[k]>0 else "0" for k in range(ast.width))
    return {"best_observed_energy":float(model.energy(best.sample)),"word":word,
            "reads_returned":int(samples.record.num_occurrences.sum())}
