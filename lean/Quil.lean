import QAF
namespace QAF.Quil

inductive Gate where
  | h | x | y | z | s | t | si | ti
  deriving DecidableEq
inductive Instruction where
  | one (g : Gate) (wire : Nat)
  | cx (control target : Nat)
  | read (wire bit : Nat)
  deriving DecidableEq
inductive Opcode where
  | H | X | Y | Z | S | T | daggerS | daggerT
  deriving DecidableEq
inductive Line where
  | gate (op : Opcode) (wire : Nat)
  | cnot (control target : Nat)
  | measure (wire bit : Nat)
  deriving DecidableEq

def op : Gate → Opcode
  | .h => .H | .x => .X | .y => .Y | .z => .Z
  | .s => .S | .t => .T | .si => .daggerS | .ti => .daggerT
def unop : Opcode → Gate
  | .H => .h | .X => .x | .Y => .y | .Z => .z
  | .S => .s | .T => .t | .daggerS => .si | .daggerT => .ti
def emitLine : Instruction → Line
  | .one g q => .gate (op g) q
  | .cx a b => .cnot a b
  | .read a b => .measure a b
def readLine : Line → Instruction
  | .gate g q => .one (unop g) q
  | .cnot a b => .cx a b
  | .measure a b => .read a b

theorem read_emit (i : Instruction) : readLine (emitLine i) = i := by
  cases i with
  | one g _ => cases g <;> rfl
  | cx | read => rfl
theorem emit_read (l : Line) : emitLine (readLine l) = l := by
  cases l with
  | gate g _ => cases g <;> rfl
  | cnot | measure => rfl

def wordsIso : Iso (List Instruction) (List Line) where
  forward := List.map emitLine
  backward := List.map readLine
  left_inv xs := by induction xs <;> simp_all [read_emit]
  right_inv xs := by induction xs <;> simp_all [emit_read]

theorem semantics {A : Type} (meaning : List Instruction → A) (w : List Instruction) :
    meaning ((w.map emitLine).map readLine) = meaning w :=
  congrArg meaning (wordsIso.left_inv w)

end QAF.Quil
