"""Build one source as executable CUDA plus a lossless, kernel-checked Lean program."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_plan(folder, width, bits, mixed, steps, wave):
    arrays={"initial":wave.get()}
    def encode(steps):
        result=[]
        for kind,*args in steps:
            if kind=="apply":
                op,wires=args
                matrix=op.matrices.get()
                name=digest(matrix.tobytes()+str(matrix.shape).encode())[:24]
                arrays[name]=matrix
                args=[name,op.is_gate,list(wires)]
            elif kind in ("if","repeat"): args[-1]=encode(args[-1])
            result.append([kind,*args])
        return result
    plan={"qubits":width,"bits":bits,"mixed":mixed,"steps":encode(steps)}
    (folder/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    np.savez_compressed(folder/"tables.npz",**arrays)


def replay(folder):
    import cupy as cp
    from .qaf_gpu import DeviceOp,replay_module,update
    plan=json.loads((folder/"plan.json").read_text())
    with np.load(folder/"tables.npz",allow_pickle=False) as file:
        arrays={name:cp.asarray(file[name]) for name in file}
    def decode(rows):
        steps=[]
        for kind,*args in rows:
            if kind=="apply": args=[DeviceOp(arrays[args[0]],args[1]),tuple(args[2])]
            elif kind in ("if","repeat"): args[-1]=decode(args[-1])
            steps.append((kind,*args))
        return steps
    wave=arrays["initial"]; mixed=plan["mixed"]
    if mixed:
        state=cp.zeros((2**plan["bits"],len(wave),len(wave)),dtype=cp.complex128)
        state[0]=wave[:,None]*wave.conj()[None,:]
    else: state=wave
    native=cp.RawModule(code=(folder/"kernels.cu").read_text(),options=("--std=c++17",))
    token=replay_module.set(native)
    try: return update(decode(plan["steps"]),state,plan["qubits"],plan["bits"],mixed)
    finally: replay_module.reset(token)


def build(source,dest):
    import cupy as cp
    from .qaf_gpu import captured_sources,evaluate
    from .lean_bridge import convert
    if dest.exists(): raise FileExistsError(f"choose a new build directory: {dest}")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as staging:
        folder=Path(staging)/"program"
        folder.mkdir()
        original=folder/"program.qaf"
        shutil.copyfile(source,original)
        text=original.read_bytes().decode("utf8")
        shutil.copyfile(Path(__file__).with_name("qaf_kernels.cu"),folder/"kernels.cu")
        capture={}
        token=captured_sources.set(capture)
        try:
            live,_=evaluate(text,lambda *a:save_plan(folder,*a))
        finally: captured_sources.reset(token)
        if capture:
            (folder/"expressions").mkdir()
            for name,code in capture.items():
                (folder/"expressions"/(name+".cu")).write_text(code,encoding="utf8")
        echoed=replay(folder)
        error=float(cp.max(cp.abs(echoed-live)))
        if not bool(cp.array_equal(echoed,live)):
            raise ValueError(f"exported CUDA execution differs from source execution by {error}")
        convert("to-lean",original,folder/"Program.lean")
        recovered=Path(staging)/"recovered.qaf"
        convert("from-lean",folder/"Program.lean",recovered)
        if recovered.read_bytes()!=original.read_bytes():
            raise ValueError("Lean round trip changed the input")
        record={"source_sha256":digest(original.read_bytes()),"lean_roundtrip":"byte-identical",
                "cuda_replay_max_difference":error,"cuda_state_shape":list(live.shape),
                "device":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                "files":{str(p.relative_to(folder)):digest(p.read_bytes())
                         for p in sorted(folder.rglob("*")) if p.is_file()}}
        (folder/"manifest.json").write_text(json.dumps(record,indent=2)+"\n")
        folder.rename(dest)
    return record


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source",type=Path); cli.add_argument("output",type=Path)
    args=cli.parse_args()
    print(json.dumps(build(args.source,args.output),indent=2))
