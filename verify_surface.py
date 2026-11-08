"""Check equivalent surface spellings, including byte-preserving imports and formatting."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from lean_bridge import concrete,convert,spell
from qaf import run
from qaf_language import prepare
from qaf_prefix import lower,forms
from qaf_cert import lower as portable,quil


root=Path(__file__).resolve().parent
standard="""def sign(b) { return 1 - 2 * b; }
qubits 1;
state(b) = sign(b) / sqrt(2);
show probs;"""
space="""def sign(b)
  return 1 - 2 * b
qubits 1
state(b) = sign(b) / sqrt(2)
show probs"""
tab=space.replace("  return","\treturn").replace("def sign(b)","def sign(b):")
prefix="""syntax prefix
def sign b
    return - 1 * 2 b
qubits 1
state b / sign b sqrt 2
show probs"""
lisp="""(def sign (b) (- 1 (* 2 b)))
(qubits 1)
(state b (/ (sign b) (sqrt 2)))
(show probs)"""


def qread(name): return (root/"examples/prefix"/(name+".qaf")).read_text(encoding="utf8")


def main():
    outputs=[]
    for form in (standard,space,tab,prefix,lisp):
        for ending in ("\n","\r\n"):
            source=(form+ending+"# trailing comment").replace("\r\n","\n").replace("\n",ending)
            assert spell(concrete(source))==source
            _,results=run(source)
            np.testing.assert_allclose(results,[[.5,.5]])
            outputs.append(results)
    assert lower("syntax prefix\nqubits 1\nstate b\n  / 1 sqrt 2\nshow probs")
    try: prepare("def f(x)\n    let a=x\n   return a\nqubits 1")
    except ValueError: pass
    else: raise AssertionError("bad sibling indentation accepted")
    expected=(root/"evidence/azure-bell.json").read_text()
    live=(root/"evidence/azure-bell-submitted.qaf").read_bytes()
    assert hashlib.sha256(live).hexdigest()==json.loads(expected)["certificate"]["source_sha256"]
    assert quil(portable(live.decode()))==quil(portable(qread("bell_state")))
    from factoring import circuit
    assert qread("factoring")==circuit(143,2,"shortlog")
    executed=[]
    for name in ("callable","recursion","algebra","functions","grover","bell_state","bell","teleport","order15"):
        ref=run((root/"examples"/(name+".qaf")).read_text(encoding="utf8"))[1]
        obs=run(qread(name))[1]
        assert ref==obs
        executed.append(name)
    import hilbert
    for name in ("hilbert","wave_packet","nonlinear"):
        ref=prepare((root/"examples"/(name+".qaf")).read_text(encoding="utf8"))
        assert hilbert.evaluate(ref)==hilbert.evaluate(prepare(qread(name)))
        executed.append(name)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        source=folder/"sample.qaf"; dest=folder/"Program.lean"; restored=folder/"restored.qaf"
        for s in (tab,lisp,prefix):
            source.write_bytes((s+"\n# comments survive\n").replace("\n","\r\n").encode("utf8"))
            convert("to-lean",source,dest); convert("from-lean",dest,restored)
            assert restored.read_bytes()==source.read_bytes()
    record={"five_spellings_times_two_line_endings":len(outputs),"prefix_reference_examples":executed,
            "checked_Lean_byte_roundtrips":3,"historical_Quil_unchanged":True}
    (root/"evidence/surface.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))


if __name__=="__main__": main()
