"""QAF entry point. Remote sampling uses a named, explicitly configured target."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

from .interpreter import Syntax,parser,run
from .qaf_language import prepare


def cuda_available():
    if importlib.util.find_spec("cupy") is None: return False
    import cupy
    try: return cupy.cuda.runtime.getDeviceCount()>0
    except cupy.cuda.runtime.CUDARuntimeError: return False


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source",type=Path)
    cli.add_argument("--backend",default="auto",choices=[
        "auto","cuda","cpu","torch","braket","braket-local","google","cirq-local","azure","dwave","ocean-local"])
    cli.add_argument("--target",default=os.getenv("QAF_TARGET",""))
    cli.add_argument("--shots",type=int,help="request samples from the selected provider")
    cli.add_argument("--export",choices=["quil","qasm3","ionq"])
    cli.add_argument("--emit-register",type=int,metavar="R",
                     help="emit a normalized R by R projection of an infinite Hilbert state as QAF")
    args=cli.parse_args()
    if args.shots is not None and args.shots<=0: cli.error("shots must be positive")
    source=args.source.read_text(encoding="utf8")
    ast=prepare(source)
    if getattr(ast,"model",None) == "network":
        if args.shots or args.export or args.emit_register is not None:
            cli.error("neural declarations export through qaf_torch.py")
        from .qaf_torch import emit_python
        print(emit_python(ast),end="")
        return
    if getattr(ast,"model",None) == "hilbert":
        from . import hilbert
        if args.shots or args.export or args.backend not in ("auto","cuda","cpu"):
            cli.error("use a finite register for physical sampling of a Hilbert series")
        if args.emit_register is not None:
            print(hilbert.register_source(ast,args.emit_register),end="")
        else:
            print(json.dumps({"backend":"exact rational series","result":hilbert.evaluate(ast)},indent=2))
        return
    if args.emit_register is not None: cli.error("--emit-register needs a Hilbert series")
    energy=bool(getattr(ast,"model",None))
    backend=args.backend
    if backend=="auto":
        chosen=os.getenv("QAF_QPU","")
        if args.shots and chosen in ({"dwave"} if energy else {"azure","braket","google"}):
            backend=chosen
        else: backend="cuda" if cuda_available() else "cpu"
    if args.export:
        from .qaf_cloud import ionq_payload,lower,qasm3,quil,certify_quil
        p=lower(source)
        if args.export=="quil":
            text=quil(p); certify_quil(p,text); print(text,end="")
        elif args.export=="qasm3": print(qasm3(p),end="")
        else: print(json.dumps(ionq_payload(p),indent=2))
        return
    if energy:
        from .qaf_energy import exhaustive,ocean
        if backend=="cuda":
            import cupy
            result=exhaustive(ast,cupy)
        elif backend=="cpu":
            import numpy
            result=exhaustive(ast,numpy)
        elif backend=="ocean-local":
            import dimod
            result=ocean(ast,dimod.ExactSolver())
        elif backend=="dwave" and args.shots: result=ocean(ast,reads=args.shots)
        else: cli.error("energy model needs CUDA, CPU, Ocean, or D-Wave with --shots")
    elif backend=="cuda":
        from .qaf_gpu import evaluate
        result=evaluate(source)[1]
    elif backend=="cpu": result=run(source)[1]
    elif backend=="torch":
        import torch
        from .qaf_torch import circuit_wave
        from .qaf_language import response
        from .interpreter import pair
        with torch.no_grad(): state=circuit_wave(source)
        probabilities=state.abs().square()
        result=[]
        for q in ast.queries:
            if q.kind=="probs": value=probabilities.tolist()
            elif q.kind=="bits": value={"0"*(ast.bits or 0):1.0}
            elif q.kind in ("amp","prob"):
                if len(q.word)!=ast.width: raise ValueError("basis word must match width")
                i=int(q.word,2)
                value=pair(complex(state[i].item())) if q.kind=="amp" else float(probabilities[i])
            else:
                rho=torch.outer(state,state.conj()).cpu().numpy()
                import numpy as np
                value=response(q,rho,ast.width,np) if q.kind=="response" else [[pair(z) for z in row] for row in rho]
            result.append(value)
    else:
        if backend in ("dwave","ocean-local"): cli.error("the annealing target requires an energy function")
        if not args.shots: cli.error("provider sampling requires --shots")
        if backend in ("azure","braket","google") and not args.target: cli.error("supply the provider target")
        from .qaf_cloud import lower,sample
        result=sample(lower(source),backend,args.target,args.shots)
    print(json.dumps({"backend":backend,"result":result},indent=2,allow_nan=False))


if __name__=="__main__":
    main()
