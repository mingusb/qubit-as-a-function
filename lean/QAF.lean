import Lean

namespace QAF

/- Grammar productions retain their terminals, including concrete source trivia. -/
inductive Source where
  | token : String → Source
  | empty : Source
  | cons : Source → Source → Source
  | rule : String → Source → Source
  deriving Repr, Inhabited, Lean.ToJson, Lean.FromJson

/- The public Lean embedding. Users may construct these terms with ordinary Lean
   definitions, abstraction and recursion before evaluating a closed program. -/
inductive Term where
  | literal : String → Term
  | endArgs : Term
  | args : Term → Term → Term
  | applyRule : String → Term → Term
  deriving Repr, Inhabited

def encode : Source → Term
  | .token s => .literal s
  | .empty => .endArgs
  | .cons a b => .args (encode a) (encode b)
  | .rule r a => .applyRule r (encode a)

def decode : Term → Source
  | .literal s => .token s
  | .endArgs => .empty
  | .args a b => .cons (decode a) (decode b)
  | .applyRule r a => .rule r (decode a)

theorem decode_encode (s : Source) : decode (encode s) = s := by
  induction s <;> simp_all [encode, decode]

theorem encode_decode (t : Term) : encode (decode t) = t := by
  induction t <;> simp_all [encode, decode]

def render : Source → String
  | .token s => s
  | .empty => ""
  | .cons a b => render a ++ render b
  | .rule _ a => render a

structure Parsed (text : String) where
  tree : Source
  spells : render tree = text

theorem text_roundtrip {s : String} (p : Parsed s) :
    render (decode (encode p.tree)) = s := by rw [decode_encode, p.spells]

theorem semantic_roundtrip {A : Type} (meaning : Source → A) (p : Source) :
    meaning (decode (encode p)) = meaning p := by rw [decode_encode]

structure Iso (A B : Type) where
  forward : A → B
  backward : B → A
  left_inv : ∀ x, backward (forward x) = x
  right_inv : ∀ y, forward (backward y) = y

def programIso : Iso Source Term := ⟨encode, decode, decode_encode, encode_decode⟩

/- Independent of quantum coefficients: the coordinate map is an isomorphism
   for any field of amplitudes. Normalization/phase use the same coordinates. -/
def vectorToFunction {A : Type} (v : A × A) : Bool → A
  | false => v.1
  | true => v.2
def functionToVector {A : Type} (f : Bool → A) : A × A := (f false, f true)

def qubitCoordinateIso (A : Type) : Iso (A × A) (Bool → A) where
  forward := vectorToFunction
  backward := functionToVector
  left_inv v := by cases v; rfl
  right_inv f := by funext b; cases b <;> rfl

def emit : Term → String
  | .literal s => "(.literal " ++ reprStr s ++ ")"
  | .endArgs => "(.endArgs)"
  | .args a b => "(.args " ++ emit a ++ " " ++ emit b ++ ")"
  | .applyRule r a => "(.applyRule " ++ reprStr r ++ " " ++ emit a ++ ")"

#print axioms decode_encode
#print axioms encode_decode
#print axioms text_roundtrip
end QAF
