"""Finite-register quantum mechanics in the computational basis (wire 0 is leftmost)."""
from dataclasses import dataclass
from typing import Callable

import numpy as np


TOL = 1e-10
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = (X + Z) / np.sqrt(2)


def close(actual, expected):
    return np.allclose(actual, expected, rtol=0, atol=TOL)


def check_wires(width, wires):
    if len(set(wires)) != len(wires) or any(q < 0 or q >= width for q in wires):
        raise ValueError(f"wires must be distinct indices in 0..{width - 1}: {wires}")


@dataclass(frozen=True)
class Wave:
    width: int
    at: Callable[[int], complex]

    def __call__(self, label):
        if not 0 <= label < 2**self.width:
            raise ValueError("basis label outside register")
        return self.at(label)

    def values(self):
        return np.fromiter((self(x) for x in range(2**self.width)), complex)

    def split(self):
        """Two unnormalized prefix slices, including a possible zero slice."""
        return tuple(self.values().reshape(2, -1))

    @classmethod
    def join(cls, left, right):
        if np.shape(left)!=np.shape(right) or np.ndim(left)!=1:
            raise ValueError("two equal-length prefix slices are required")
        return cls.from_values(np.concatenate((left,right)))

    @classmethod
    def from_values(cls, values):
        data = np.array(values, dtype=complex, copy=True)
        n = data.size.bit_length() - 1
        if n < 1 or data.shape != (2**n,) or not np.isfinite(data).all():
            raise ValueError("expected a finite amplitude for each register label")
        if not close(np.vdot(data, data), 1):
            raise ValueError("state amplitudes must have squared norm one")
        data.setflags(write=False)
        return cls(n, data.__getitem__)


def contract(wave, kernel, wires):
    """Evaluate the local kernel against the input amplitude function."""
    check_wires(wave.width, wires)
    shifts = [wave.width - 1 - wire for wire in wires]
    size = 2**len(wires)
    if kernel.shape != (size, size):
        raise ValueError("kernel dimension and target wires disagree")
    offsets = [sum(((label >> (len(shifts)-1-j)) & 1) << shift
                   for j, shift in enumerate(shifts)) for label in range(size)]
    active = sum(1 << shift for shift in shifts)
    result = np.empty(2**wave.width, dtype=complex)
    for base in range(result.size):
        if base & active:
            continue
        labels = [base | offset for offset in offsets]
        inputs = [wave(label) for label in labels]
        result[labels] = kernel @ inputs
    return result


@dataclass(frozen=True)
class Operation:
    matrices: tuple
    is_gate: bool = False

    def __post_init__(self):
        ops = tuple(np.array(a, dtype=complex, copy=True) for a in self.matrices)
        if not ops or ops[0].ndim != 2:
            raise ValueError("a channel needs at least one square Kraus matrix")
        d = ops[0].shape[0]
        if d < 2 or d & (d-1) or any(a.shape != (d,d) or
                not np.isfinite(a).all() for a in ops):
            raise ValueError("Kraus matrices must be finite and square with qubit dimension")
        if self.is_gate and len(ops) != 1:
            raise ValueError("a gate has one unitary matrix")
        if not close(sum(a.conj().T @ a for a in ops), np.eye(d)):
            raise ValueError("gate unitarity or channel trace preservation failed")
        for a in ops:
            a.setflags(write=False)
        object.__setattr__(self, "matrices", ops)

    @property
    def width(self):
        return self.matrices[0].shape[0].bit_length() - 1

    def unitary(self):
        if not self.is_gate:
            raise ValueError("this construct needs a gate with a specified phase")
        return self.matrices[0]


def gate(matrix):
    return Operation((matrix,), is_gate=True)


def sequence(first, second):
    if first.width != second.width:
        raise ValueError("sequential operations must have equal widths")
    return Operation(tuple(b @ a for a in first.matrices for b in second.matrices),
                     first.is_gate and second.is_gate)


def tensor(first, second):
    return Operation(tuple(np.kron(a, b) for a in first.matrices for b in second.matrices),
                     first.is_gate and second.is_gate)


def controlled(op):
    unitary = op.unitary()
    d = len(unitary)
    matrix = np.eye(2*d, dtype=complex)
    matrix[d:, d:] = unitary
    return gate(matrix)


def ordered(first, second):
    """Controlled order for two specified same-width gate implementations."""
    a, b = first.unitary(), second.unitary()
    if a.shape != b.shape:
        raise ValueError("ordered gates must have the same width")
    d = len(a)
    result = np.zeros((2*d, 2*d), dtype=complex)
    result[:d, :d], result[d:, d:] = b @ a, a @ b
    return gate(result)


def rotation(pauli, angle):
    return gate(np.cos(angle/2)*I - 1j*np.sin(angle/2)*pauli)


def select(first,second):
    a,b=first.unitary(),second.unitary()
    if a.shape!=b.shape: raise ValueError("function alternatives must have the same width")
    d=len(a)
    result=np.zeros((2*d,2*d),complex)
    result[:d,:d],result[d:,d:]=a,b
    return gate(result)


def euler(theta, phi, lam):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return gate([[c, -np.exp(1j*lam)*s],
                 [np.exp(1j*phi)*s, np.exp(1j*(phi+lam))*c]])


P0, P1 = np.diag([1, 0]), np.diag([0, 1])
BUILTINS = {label: gate(mat) for label, mat in {
    "I": I, "X": X, "Y": Y, "Z": Z, "H": H, "S": np.diag([1,1j]),
    "T": np.diag([1,np.exp(1j*np.pi/4)]), "SWAP": np.eye(4)[[0,2,1,3]]
}.items()}
BUILTINS.update(CX=controlled(BUILTINS["X"]), CZ=controlled(BUILTINS["Z"]),
                Dephase=Operation((P0, P1)), Reset=Operation((P0, X @ P1)))


@dataclass(frozen=True)
class Branch:
    weight: float
    wave: Wave
    bits: tuple


def evolve(branch, matrices, wires, measured_bit=None):
    children = []
    for outcome, matrix in enumerate(matrices):
        values = contract(branch.wave, matrix, wires)
        scale = float(np.max(np.abs(values)))
        if scale == 0:
            continue
        scaled = values / scale
        norm = float(np.linalg.norm(scaled))
        probability = (scale * norm)**2
        weight = branch.weight * probability
        if weight == 0:
            raise ArithmeticError("a nonzero branch weight underflowed")
        bits = list(branch.bits)
        if measured_bit is not None:
            bits[measured_bit] = outcome
        wave = Wave.from_values(scaled / norm)
        children.append(Branch(weight, wave, tuple(bits)))
    return children


def density(branches):
    return sum(b.weight * np.outer(v, v.conj())
               for b in branches for v in [b.wave.values()])
