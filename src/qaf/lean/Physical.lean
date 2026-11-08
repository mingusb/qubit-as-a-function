import QAF
namespace QAF.Physical
open QAF

/- Transport works for every amplitude domain and predicate. Instantiating P
   with squared norm = 1 restricts both sides without choosing a preferred ray. -/
def admittedIso {A : Type} (P : (Bool → A) → Prop) :
    Iso {v : A × A // P (vectorToFunction v)} {f : Bool → A // P f} where
  forward v := ⟨vectorToFunction v.val, v.property⟩
  backward f := ⟨functionToVector f.val, by
    have h := (qubitCoordinateIso A).right_inv f.val
    change P (vectorToFunction (functionToVector f.val))
    change vectorToFunction (functionToVector f.val) = f.val at h
    rw [h]; exact f.property⟩
  left_inv v := Subtype.ext ((qubitCoordinateIso A).left_inv v.val)
  right_inv f := Subtype.ext ((qubitCoordinateIso A).right_inv f.val)

/- A ray setoid on admitted functions pulls back to the corresponding relation
   on coordinate pairs. This includes the complex unit-phase relation. -/
def pulled {A B : Type} (e : Iso A B) (s : Setoid B) : Setoid A where
  r a b := s.r (e.forward a) (e.forward b)
  iseqv := ⟨fun a => s.iseqv.refl (e.forward a), s.iseqv.symm, s.iseqv.trans⟩

def mapQ {A B : Type} {s : Setoid A} {t : Setoid B} (f : A → B)
    (law : ∀ a b, s.r a b → t.r (f a) (f b)) : Quotient s → Quotient t :=
  Quotient.lift (fun a => Quotient.mk t (f a)) (fun a b h => Quotient.sound (law a b h))

def onQuotients {A B : Type} (e : Iso A B) (s : Setoid B) :
    Iso (Quotient (pulled e s)) (Quotient s) where
  forward := mapQ e.forward (fun _ _ h => h)
  backward := mapQ e.backward (by
    intro a b h; change s.r (e.forward (e.backward a)) (e.forward (e.backward b))
    rw [e.right_inv,e.right_inv]; exact h)
  left_inv := by
    intro q; induction q using Quotient.inductionOn with
    | _ a => exact congrArg (Quotient.mk (pulled e s)) (e.left_inv a)
  right_inv := by
    intro q; induction q using Quotient.inductionOn with
    | _ b => exact congrArg (Quotient.mk s) (e.right_inv b)

/- The ideal core certifies physical register transformations, including
   trace-preserving instruments when A is a classical-quantum register.
   Verifying each primitive's mathematical premise is a separate obligation. -/
structure PhysMap (A : Type) (valid : A → Prop) where
  apply : A → A
  preserves : ∀ a, valid a → valid (apply a)

inductive Expr (A : Type) (valid : A → Prop) where
  | prepare (v : A) (proof : valid v)
  | transform (op : PhysMap A valid) (input : Expr A valid)
  | choose (choice : Bool) (yes no : Expr A valid)

def eval {A : Type} {valid : A → Prop} : Expr A valid → A
  | .prepare v _ => v
  | .transform op rest => op.apply (eval rest)
  | .choose true yes _ => eval yes
  | .choose false _ no => eval no

theorem preservation {A : Type} {valid : A → Prop} (e : Expr A valid) :
    valid (eval e) := by
  induction e with
  | prepare _ h => exact h
  | transform op rest ih => exact op.preserves _ ih
  | choose b _ _ ih₁ ih₂ => cases b <;> assumption
end QAF.Physical
