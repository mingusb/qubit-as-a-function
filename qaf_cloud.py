"""Lower physical QAF programs to a terminal-measurement circuit for cloud providers."""
import os
import numpy as np

from qaf_cert import Portable, certify, certify_quil, lower, quil


def braket_circuit(program):
    from braket.circuits import Circuit
    circuit = Circuit()
    for name,*wires in program.gates:
        getattr(circuit,"cnot" if name=="cx" else name)(*wires)
    circuit.measure(list(range(program.width+program.bits)))
    word,measurements=[],[]
    for item in circuit.instructions:
        token=item.operator.name.lower()
        wires=tuple(map(int,item.target))
        if token=="measure": measurements.append((wires[0],item.operator._target_index))
        else: word.append(({"cnot":"cx"}.get(token,token),*wires))
    certify(program,word,measurements)
    return circuit


def google_circuit(program,device=None):
    import cirq
    qubits = cirq.LineQubit.range(program.total)
    circuit = cirq.Circuit()
    gates={"h":cirq.H,"x":cirq.X,"y":cirq.Y,"z":cirq.Z,"s":cirq.S,"t":cirq.T,
           "si":cirq.S**-1,"ti":cirq.T**-1,"cx":cirq.CNOT}
    for name,*wires in program.gates:
        circuit.append(gates[name](*(qubits[q] for q in wires)))
    for i in range(program.width+program.bits):
        circuit.append(cirq.measure(qubits[i],key=f"m{i}"))
    inverse={v:k for k,v in gates.items()}
    word,writes=[],[]
    for op in circuit.all_operations():
        ws=[q.x for q in op.qubits]
        if cirq.is_measurement(op): writes.append((ws[0],int(cirq.measurement_key_name(op)[1:])))
        else: word.append((inverse[op.gate],*ws))
    certify(program,word,writes)
    if device is not None:
        # Map logical wires by graph embedding. Replace CX with exact H-CZ-H.
        # No approximate device synthesis is accepted by this profile.
        import networkx as nx
        graph=nx.Graph()
        graph.add_nodes_from(range(program.total))
        graph.add_edges_from((a,b) for name,a,b in [e for e in word if e[0]=="cx"])
        match=nx.algorithms.isomorphism.GraphMatcher(device.metadata.nx_graph,graph)
        mapping=next(match.subgraph_monomorphisms_iter(),None)
        if mapping is None: raise ValueError("device has no embedding for this interaction graph")
        placement={q:p for p,q in mapping.items()}
        # Cirq's exact half-turn convention gives these dyadic rotations.
        # H and Y may pick up a common phase, irrelevant for a full gate word's readout.
        angles={"h":(.5,1,-.5),"x":(1,0,0),"y":(1,0,.5),"z":(0,1,0),
                "s":(0,.5,0),"si":(0,-.5,0),"t":(0,.25,0),"ti":(0,-.25,0)}
        single={op:cirq.PhasedXZGate(x_exponent=x,z_exponent=z,axis_phase_exponent=a)
                for op,(x,z,a) in angles.items()}
        expected=[]
        native=cirq.Circuit()
        for item in circuit.all_operations():
            w=[placement[q.x] for q in item.qubits]
            if item.gate==cirq.CNOT:
                a,b=w
                if cirq.CZ(a,b) in device.metadata.gateset:
                    cz=[cirq.CZ(a,b)]
                else:
                    candidates=[(cirq.ISWAP**.5,"s"),(cirq.ISWAP**-.5,"si")]
                    pair=next(((g(a,b),phase) for g,phase in candidates if g(a,b) in device.metadata.gateset),None)
                    if pair is None: raise ValueError("device lacks a certified CZ or square-root iSWAP")
                    entangler,phase=pair
                    # Conjugating the exchange interaction by X on a cancels YY.
                    # H on both wires then maps the residual XX rotation to ZZ.
                    cz=[single["h"](a),single["h"](b),single["x"](a),entangler,
                        single["x"](a),entangler,single["h"](a),single["h"](b),
                        single[phase](a),single[phase](b)]
                emitted=[single["h"](b),*cz,single["h"](b)]
            elif cirq.is_measurement(item):
                emitted=[item.with_qubits(*w)]
            else:
                emitted=[single[inverse[item.gate]](*w)]
            expected.extend(emitted)
            native.append(emitted,strategy=cirq.InsertStrategy.NEW)
        if list(native.all_operations())!=expected:
            raise ValueError("native Cirq word differs from the certified decomposition")
        device.validate_circuit(native)
        return native
    return circuit


def ionq_payload(program):
    entries = []
    for name,*q in program.gates:
        if name=="cx":
            entries.append({"gate":"cnot","control":q[0],"target":q[1]})
        else:
            entries.append({"gate":name,"target":q[0]})
    decoded=[("cx",v["control"],v["target"]) if v["gate"]=="cnot" else
             (v["gate"],v["target"]) for v in entries]
    certify(program,decoded,[(i,i) for i in range(program.width+program.bits)])
    return {"qubits":program.total,"circuit":entries}


def qasm3(program):
    lines = ["OPENQASM 3.0;",'include "stdgates.inc";',f"qubit[{program.total}] q;",
             f"bit[{program.width+program.bits}] out;"]
    for name,*qs in program.gates:
        name={"ti":"tdg","si":"sdg"}.get(name,name)
        wires = ", ".join(f"q[{q}]" for q in qs)
        lines.append(f"{name} {wires};")
    for i in range(program.width+program.bits):
        lines.append(f"out[{i}] = measure q[{i}];")
    return "\n".join(lines)+"\n"


def project(program, joint):
    """Words are in QAF logical order, register first, then classical bits."""
    n = program.width
    probs, bits = np.zeros(2**n),{}
    total = sum(joint.values())
    if total<=0:
        raise ValueError("provider returned no samples")
    for word,p in joint.items():
        probs[int(word[:n],2)] += p/total
        c = word[n:n+program.bits]
        bits[c] = bits.get(c,0)+p/total
    return [probs.tolist() if q.kind=="probs" else bits if q.kind=="bits"
            else float(probs[int(q.word,2)]) for q in program.queries]


def sample(program,backend,target,shots):
    from collections import Counter
    if shots<1:
        raise ValueError("shots must be positive")
    run_id = None
    if backend in ("braket","braket-local"):
        circuit = braket_circuit(program)
        if backend=="braket":
            from braket.aws import AwsDevice
            device = AwsDevice(target)
        else:
            from braket.devices import LocalSimulator
            device = LocalSimulator()
        task = device.run(circuit,shots=shots)
        run_id = str(task.id)
        result = task.result()
        ids = list(result.measured_qubits)
        joint = Counter("".join(str(int(row[ids.index(i)]))
                                for i in range(program.width+program.bits))
                        for row in result.measurements)
    elif backend in ("google","cirq-local"):
        import cirq
        if backend=="google":
            import cirq_google
            proc = cirq_google.get_engine(project_id=os.environ["GOOGLE_CLOUD_PROJECT"]).get_processor(target)
            circuit = google_circuit(program,proc.get_device())
            sampler = proc.get_sampler()
        else:
            circuit,sampler = google_circuit(program),cirq.Simulator(seed=19)
        result = sampler.run(circuit,repetitions=shots)
        keys = [f"m{i}" for i in range(program.width+program.bits)]
        joint = Counter("".join(str(int(result.measurements[k][i,0])) for k in keys)
                        for i in range(shots))
    else:
        from azure.identity import DefaultAzureCredential
        from azure.quantum import Workspace
        ws = Workspace(resource_id=os.environ["AZURE_QUANTUM_WORKSPACE_RESOURCE_ID"],
                       location=os.environ["AZURE_QUANTUM_WORKSPACE_LOCATION"],
                       credential=DefaultAzureCredential())
        device = ws.get_targets(name=target)
        if target.startswith("rigetti."):
            from azure.quantum.target.rigetti import Result
            text=quil(program)
            certificate=certify_quil(program,text)
            job=device.submit(input_data=text,name="QAF certified circuit",shots=shots)
            job.wait_until_completed()
            rows=Result(job)["ro"]
            joint=Counter("".join(str(int(x)) for x in row) for row in rows)
            return {"backend":backend,"target":target,"job":job.id,"status":job.details.status,
                    "shots":len(rows),"counts":dict(joint),"queries":project(program,joint),
                    "certificate":certificate}
        if not target.startswith("ionq."):
            raise ValueError("unsupported Azure target format")
        job = device.submit(circuit=ionq_payload(program),shots=shots,name="qaf-sampling")
        run_id = job.id
        histogram = job.get_results()["histogram"]
        joint = Counter()
        for key,p in histogram.items():
            # IonQ interprets qubit 0 as the least-significant bit of its integer.
            word = format(int(key),f"0{program.total}b")[::-1]
            joint[word[:program.width+program.bits]] += p
    return {"backend":backend,"job":run_id,"shots":shots,"queries":project(program,joint)}
