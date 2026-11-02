import Std

/- Executable Float formulation of the transferred operator and shared phase
   address. UNCOMPILED here: no Lean executable is installed. This is not an
   IEEE/AD proof and does not formalize the full native TAC/SSA graph. -/
namespace Collect011

abbrev Vec (n : Nat) := Fin n → Float
abbrev Mat (m n : Nat) := Fin m → Fin n → Float

def dot {n : Nat} (a b : Vec n) : Float :=
  (List.finRange n).foldl (fun acc i => acc + a i * b i) 0

def matvec {m n : Nat} (a : Mat m n) (x : Vec n) : Vec m :=
  fun i => dot (a i) x

def direction {n : Nat} (p : Mat n n) (k : Vec n) : Vec n :=
  let z := matvec p k
  let leverage := max (dot k z) 0.00000001
  fun i => z i / leverage

def writeMemory {v n : Nat} (s : Mat v n) (p : Mat n n)
    (k : Vec n) (value : Vec v) (gain : Float) : Mat v n :=
  if gain == 0 then s else
  let old := matvec s k
  let u := direction p k
  fun i j => s i j + gain * (value i - old i) * u j

def writePrecision {n : Nat} (p : Mat n n) (k : Vec n)
    (gain : Float) : Mat n n :=
  if gain == 0 then p else
  let z := matvec p k
  let leverage := dot k z
  let candidate := fun i j => p i j - gain * z i * z j / (1 + gain * leverage)
  fun i j => (candidate i j + candidate j i) * 0.5

def observe (clock count : Nat) : Nat := clock + count
theorem observe_partition (clock a b : Nat) :
    observe (observe clock a) b = observe clock (a + b) := by
  simp [observe, Nat.add_assoc]

/- Content/role linear maps, tanh/cos/sin, soft bank routing, sigmoid admission,
   support gate, native banks/rotary/SSA/FFN and AD remain OPEN in Lean. Python
   actual-source tests cover specific executions only. No full-graph claim. -/
end Collect011
