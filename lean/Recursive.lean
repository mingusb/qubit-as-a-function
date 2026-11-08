import QAF
namespace QAF.Recursive

inductive Word : Nat → Type where
  | nil : Word 0
  | cons : Bool → Word n → Word (n+1)

def split {A : Type} {n : Nat} (f : Word (n+1) → A) (b : Bool) (s : Word n) : A :=
  f (.cons b s)

def join {A : Type} {n : Nat} (g : Bool → Word n → A) : Word (n+1) → A
  | .cons b s => g b s

theorem join_split {A : Type} {n : Nat} (f : Word (n+1) → A) : join (split f) = f := by
  funext s
  cases s
  rfl

theorem split_join {A : Type} {n : Nat} (g : Bool → Word n → A) : split (join g) = g :=
  rfl

def prefixIso (A : Type) (n : Nat) : Iso (Word (n+1) → A) (Bool → Word n → A) :=
  ⟨split, join, join_split, split_join⟩

end QAF.Recursive
