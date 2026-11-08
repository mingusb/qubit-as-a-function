import QAF
namespace QAF.Algebra

/- Exactly the field laws used here. The results hold for any field, in particular
   for the complex field, with its ordinary addition, negation and multiplication. -/
structure F (K : Type) where
  zero : K
  one : K
  add : K → K → K
  mul : K → K → K
  neg : K → K
  inv : K → K
  addAssoc : ∀ a b c, add (add a b) c = add a (add b c)
  addComm : ∀ a b, add a b = add b a
  addZero : ∀ a, add a zero = a
  addNeg : ∀ a, add a (neg a) = zero
  mulAssoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  mulComm : ∀ a b, mul a b = mul b a
  mulOne : ∀ a, mul a one = a
  distrib : ∀ a b c, mul a (add b c) = add (mul a b) (mul a c)
  cancelInv : ∀ a b, b ≠ zero → mul (mul a (inv b)) b = a

abbrev Amp (K : Type) := Bool → K
variable {K : Type} (k : F K)
def plus (f g : Amp K) : Amp K := fun b => k.add (f b) (g b)
def minus (f g : Amp K) : Amp K := fun b => k.add (f b) (k.neg (g b))
def times (f g : Amp K) : Amp K := fun b => k.mul (f b) (g b)
def quotient (f g : Amp K) (_ : ∀ b, g b ≠ k.zero) : Amp K :=
  fun b => k.mul (f b) (k.inv (g b))
def compose {A B C : Type} (g : B → C) (f : A → B) := fun x => g (f x)

theorem comp_assoc {A B C D : Type} (h : C → D) (g : B → C) (f : A → B) :
    compose (compose h g) f = compose h (compose g f) := rfl
theorem comp_id {A B : Type} (f : A → B) :
    compose id f = f ∧ compose f id = f := ⟨rfl,rfl⟩
theorem plus_comm (f g : Amp K) : plus k f g = plus k g f :=
  funext fun b => k.addComm (f b) (g b)
theorem plus_assoc (f g h : Amp K) : plus k (plus k f g) h = plus k f (plus k g h) :=
  funext fun b => k.addAssoc (f b) (g b) (h b)
theorem plus_zero (f : Amp K) : plus k f (fun _ => k.zero) = f :=
  funext fun b => k.addZero (f b)
theorem sub_self (f : Amp K) : minus k f f = fun _ => k.zero :=
  funext fun b => k.addNeg (f b)
theorem sub_add_cancel (f g : Amp K) : plus k (minus k f g) g = f := by
  funext b
  change k.add (k.add (f b) (k.neg (g b))) (g b) = f b
  rw [k.addAssoc, k.addComm (k.neg (g b)), k.addNeg, k.addZero]
theorem times_comm (f g : Amp K) : times k f g = times k g f :=
  funext fun b => k.mulComm (f b) (g b)
theorem times_assoc (f g h : Amp K) : times k (times k f g) h = times k f (times k g h) :=
  funext fun b => k.mulAssoc (f b) (g b) (h b)
theorem times_one (f : Amp K) : times k f (fun _ => k.one) = f :=
  funext fun b => k.mulOne (f b)
theorem distribution (f g h : Amp K) :
    times k f (plus k g h) = plus k (times k f g) (times k f h) :=
  funext fun b => k.distrib (f b) (g b) (h b)
theorem quotient_cancel (f g : Amp K) (nz : ∀ b, g b ≠ k.zero) :
    times k (quotient k f g nz) g = f :=
  funext fun b => k.cancelInv (f b) (g b) (nz b)

def unitReal (f : Amp Int) : Prop := f false * f false + f true * f true = 1
theorem zero_has_no_ray : ¬ unitReal (fun _ => 0) := by unfold unitReal; decide
theorem ones_not_unit : ¬ unitReal (fun _ => 1) := by unfold unitReal; decide
example : unitReal (fun b => if b then 0 else 1) := by unfold unitReal; decide

/- Restrict any usual operation to a state space. The witness records that both
   the operation's inputs and its output meet the declared domain conditions. -/
def restrict {A : Type} (P : A → Prop) (op : A → A → A)
    (f g : Subtype P) (ok : P (op f.val g.val)) : Subtype P := ⟨_,ok⟩
theorem restrict_value {A : Type} (P : A → Prop) (op : A → A → A)
    (f g : Subtype P) (ok : P (op f.val g.val)) :
    (restrict P op f g ok).val = op f.val g.val := rfl
end QAF.Algebra
