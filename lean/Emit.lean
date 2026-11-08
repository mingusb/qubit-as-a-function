import QAF
open Lean QAF
def main : IO Unit := do
  let text ← (← IO.getStdin).readToEnd
  let j ← IO.ofExcept (Json.parse text)
  let tree ← IO.ofExcept (fromJson? (α := Source) j)
  let value := encode tree
  IO.println "import QAF\nopen QAF\nset_option maxRecDepth 100000\nset_option maxHeartbeats 0"
  IO.println ("def program : Term := " ++ emit value)
  IO.println ("theorem source_spelling : render (decode program) = " ++ reprStr (render tree) ++ " := by rfl")
  IO.println "theorem lean_roundtrip : encode (decode program) = program := encode_decode program"
  IO.println "def main : IO Unit := IO.println (Lean.toJson (decode program)).compress"
