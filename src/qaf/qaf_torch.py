"""Compile neural QAF to PyTorch and execute unmodified pure circuits on network-defined inputs."""
import argparse
from pathlib import Path
import torch
from torch import nn


class Sine(nn.Module):
    def forward(self,x):
        return torch.sin(x)


ACTIVATIONS={"tanh":nn.Tanh,"relu":nn.ReLU,"sigmoid":nn.Sigmoid,"sine":Sine}


def layers(ast):
    if ast.model!="network" or ast.layers[0].kind!="input" or ast.layers[0].width<1:
        raise ValueError("a neural program starts with a positive input width")
    previous=ast.layers[0].width
    for layer in ast.layers[1:]:
        if layer.kind=="linear" and layer.width>0:
            yield "Linear",(previous,layer.width)
            previous=layer.width
        elif layer.kind in ACTIVATIONS:
            yield ACTIVATIONS[layer.kind].__name__,()
        else: raise ValueError("each following layer needs a supported activation or positive width")


def compile_network(ast):
    return nn.Sequential(*(getattr(nn,name,Sine)(*args) for name,args in layers(ast)))


def emit_python(ast):
    parts=list(layers(ast))
    body="from torch import nn\n"
    if any(n=="Sine" for n,_ in parts):
        body+="import torch\n\nclass Sine(nn.Module):\n    def forward(self, x):\n        return torch.sin(x)\n"
    body+=f"\n{ast.name} = nn.Sequential(\n"
    for name,args in parts:
        body+="    "+("" if name=="Sine" else "nn.")+name+"("+", ".join(map(str,args))+"),\n"
    return body+")\n"


def normalize(amplitudes):
    norm=torch.linalg.vector_norm(amplitudes)
    if not bool(torch.isfinite(norm)) or not bool(norm>0):
        raise ValueError("a nonzero, finite amplitude function is required")
    return amplitudes/norm


class BasisAmplitudes(nn.Module):
    """Learnable complex basis function, evaluated and normalized over all labels."""
    def __init__(self, values, device):
        super().__init__()
        self.weight=nn.Parameter(torch.as_tensor(values,dtype=torch.complex128,device=device).clone())

    def forward(self,labels=None):
        wave=normalize(self.weight)
        return wave if labels is None else wave[labels]


def local_gate(wave, gate, wires, width):
    order=list(wires)+[q for q in range(width) if q not in wires]
    undo=[order.index(q) for q in range(width)]
    dims=(2,)*width
    blocks=wave.reshape(dims).permute(order).reshape(len(gate),-1)
    return (gate@blocks).reshape(dims).permute(undo).reshape(-1)


def circuit_wave(source,device=None,input_qubit=None):
    from .interpreter import definitions,instructions,needs_density,scalar
    from .qaf_language import prepare
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    ast=prepare(source); n=ast.width
    steps=instructions(ast.steps,definitions(ast.definitions),n,ast.bits or 0)
    if needs_density(steps): raise ValueError("PyTorch amplitude backend requires a pure circuit")
    if ast.initial:
        if input_qubit is not None: raise ValueError("source already defines a preparation")
        initial=ast.initial
        fun=scalar(initial.amplitude,[initial.index])
        values=[fun({initial.index:i}) for i in range(2**n)]
        v=torch.as_tensor(values,dtype=torch.complex128,device=device)
        if abs(float(torch.linalg.vector_norm(v))-1)>1e-10: raise ValueError("source state needs unit norm")
        wave=v
    else:
        initial=input_qubit if input_qubit is not None else BasisAmplitudes([1,0],device)
        q=initial()
        if tuple(q.shape)!=(2,) or not bool(torch.isclose(torch.vdot(q,q).real,q.real.new_tensor(1),rtol=0,atol=1e-10)):
            raise ValueError("one normalized two-amplitude input is required")
        wave=q
        for _ in range(n-1): wave=torch.kron(wave,q)
    labels=torch.arange(2**n,device=device)
    def evolve(wave,block):
        for kind,*args in block:
            if kind=="apply":
                op,wires=args
                gate=torch.tensor(op.unitary().copy(),dtype=torch.complex128,device=device)
                wave=local_gate(wave,gate,wires,n)
            elif kind=="mulmod":
                modulus,inverse,shift,mask,flag=args
                if modulus.bit_length()>31: raise ValueError("PyTorch modular integer range exceeded")
                local=(labels>>shift)&mask
                active=local<modulus
                if flag>=0: active=active&((labels&(1<<flag))!=0)
                index=torch.where(active,labels^((local^(local*inverse%modulus))<<shift),labels)
                wave=wave[index]
            elif kind=="repeat":
                for _ in range(args[0]): wave=evolve(wave,args[1])
        return wave
    return evolve(wave,steps)


if __name__=="__main__":
    from .qaf_language import prepare
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source",type=Path); cli.add_argument("--emit",type=Path)
    args=cli.parse_args()
    code=emit_python(prepare(args.source.read_text(encoding="utf8")))
    if args.emit: args.emit.write_text(code,encoding="utf8")
    else: print(code,end="")
