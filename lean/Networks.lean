import QAF
namespace QAF.Networks

def sameOutput {P S : Type} (prepare : P → S) : Setoid P where
  r p q := prepare p = prepare q
  iseqv := ⟨fun _ => rfl, Eq.symm, Eq.trans⟩

abbrev Reachable {P S : Type} (prepare : P → S) := {s : S // ∃ p, prepare p = s}

noncomputable def weightClasses {P S : Type} (prepare : P → S) :
    Iso (Quotient (sameOutput prepare)) (Reachable prepare) where
  forward := Quotient.lift (fun p => ⟨prepare p, p, rfl⟩) (fun _ _ h => Subtype.ext h)
  backward s := Quotient.mk (sameOutput prepare) s.property.choose
  left_inv q := by
    induction q using Quotient.inductionOn with
    | _ p =>
      apply Quotient.sound
      exact (show ∃ p', prepare p' = prepare p from ⟨p,rfl⟩).choose_spec
  right_inv s := by
    apply Subtype.ext
    exact s.property.choose_spec

-- The rays of two nonzero real-coordinate outputs (1,0) and (2,0) agree.
def proportional (v w : Nat × Nat) := v.1 * w.2 = v.2 * w.1
example : (1,0) ≠ ((2,0) : Nat × Nat) ∧ proportional (1,0) (2,0) := by
  unfold proportional
  decide
end QAF.Networks
