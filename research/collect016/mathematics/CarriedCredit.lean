import Mathlib

/- COLLECT016 local exact-real definitions, NOT a full neural-source proof.
   StopGradient below models a declared local AD pair, not the ordinary
   derivative of the real identity function. The full hard router uses a
   straight-through surrogate; no classical hard-argmax derivative is asserted.
   Whole-model Lean / IEEE / VJP equivalence remain OPEN.
-/
namespace COLLECT016
abbrev Vector (n : Nat) := Fin n → ℝ
abbrev Matrix (v k : Nat) := Fin v → Fin k → ℝ
def currentValue {h v : Nat} (route gain : Vector h)
    (value : Fin h → Vector v) : Fin h → Vector v :=
  fun i j => route i * gain i * route i * value i j
def jointLoss (natural task answer balance : ℝ) : ℝ :=
  natural + (1/4 : ℝ)*task + answer + (1/100 : ℝ)*balance
def stopGradientPair (valueAndTangent : ℝ × ℝ) : ℝ × ℝ :=
  (valueAndTangent.1, 0)
theorem stopGradient_value (x : ℝ × ℝ) :
    (stopGradientPair x).1 = x.1 := rfl
theorem stopGradient_declared_tangent (x : ℝ × ℝ) :
    (stopGradientPair x).2 = 0 := rfl
structure CausalSequenceGraph (d s : Nat) where
  start : Vector s
  step : Vector s → Vector d → Vector s
  predict : Vector s → Vector d → Vector 256
def carry {d s : Nat} (g : CausalSequenceGraph d s)
    (observations : List (Vector d)) : Vector s :=
  observations.foldl g.step g.start
end COLLECT016
