"""Check a variational wavefunction, parameter redundancy, and the same factoring circuit."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from qaf.qaf_language import prepare
from qaf.qaf_torch import BasisAmplitudes,circuit_wave,compile_network,emit_python,normalize


ROOT=Path(__file__).resolve().parents[2]
device="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(20261108)
torch.set_default_dtype(torch.float64)
spec=prepare((ROOT/"examples/wave_net.qaf").read_text())
model=compile_network(spec).to(device)
env={}
exec(compile(emit_python(spec),"<QAF PyTorch export>","exec"),env)
exported=env[spec.name].to(device)
exported.load_state_dict(model.state_dict())
n=40
x=(torch.arange(1,n+1,device=device)/(n+1)).reshape(-1,1)
torch.testing.assert_close(model(x),exported(x))
off=torch.ones(n-1,device=device)
h=(2*torch.eye(n,device=device)-off.diag(1)-off.diag(-1))*(n+1)**2/2
h=h.to(torch.complex128)
def energy():
    raw=model(x)
    psi=normalize(torch.complex(raw[:,0],raw[:,1])*(x*(1-x)).flatten())
    return torch.vdot(psi,h@psi).real
optimizer=torch.optim.LBFGS(model.parameters(),max_iter=180,line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10,tolerance_change=1e-14)
def closure():
    optimizer.zero_grad(); loss=energy(); loss.backward(); return loss
initial=float(energy().detach())
optimizer.step(closure)
final=float(energy().detach()); reference=2*(n+1)**2*math.sin(math.pi/(2*(n+1)))**2
if not reference-1e-10<=final<min(initial,reference+.002): raise AssertionError("variational energy")
from qaf.qaf_gpu import evaluate
path=ROOT/"examples/prefix/factoring.qaf"
source=path.read_text(encoding="utf8")
network_qubit=BasisAmplitudes([2,0],device)
torch.testing.assert_close(network_qubit(),BasisAmplitudes([1,0],device)())
wave=circuit_wave(source,device,network_qubit)
cuda,_=evaluate(source)
diff=float(np.max(abs(wave.detach().cpu().numpy()-cuda.get())))
if diff>=1e-12: raise AssertionError("Shor backend amplitude difference")
result={"pytorch":torch.__version__,"device":torch.cuda.get_device_name(0) if device=="cuda" else device,
        "grid_points":n,"initial_energy":initial,"variational_energy":final,
        "exact_grid_energy":reference,"continuum_energy":math.pi**2/2,
        "shor_amplitudes_compared":wave.numel(),"max_amplitude_difference":diff}
(ROOT/"docs/evidence/neural.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
