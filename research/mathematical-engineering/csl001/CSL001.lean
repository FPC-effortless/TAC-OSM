/-!
CSL-001 limited mathematical reference (NOT full-source formal verification).
Exact real-field algebra only, distinct from fp32 rounding, Torch autodiff,
role routing, masked data, tool authority, or any trained capability.
Lean build/axiom audit: OPEN until independently compiled in PNDS.
-/
import Mathlib

namespace CSL001

variable {α : Type*} [LinearOrderedField α]

def retentionStep (decay old observed : α) : α :=
  decay * old + (1 - decay) * observed

theorem retentionStep_zero (old observed : α) :
    retentionStep (0 : α) old observed = observed := by
  simp [retentionStep]

theorem retentionStep_one (old observed : α) :
    retentionStep (1 : α) old observed = old := by
  simp [retentionStep]

theorem retentionStep_fixed (decay observed : α) :
    retentionStep decay observed observed = observed := by
  dsimp [retentionStep]
  ring

/-! Boundedness under 0≤decay≤1 and 0≤old,observed≤1. -/
theorem retentionStep_nonneg (decay old observed : α)
    (hd0 : 0 ≤ decay) (hd1 : decay ≤ 1)
    (ho : 0 ≤ old) (hz : 0 ≤ observed) :
    0 ≤ retentionStep decay old observed := by
  dsimp [retentionStep]
  apply add_nonneg (mul_nonneg hd0 ho)
  exact mul_nonneg (sub_nonneg.mpr hd1) hz

end CSL001
