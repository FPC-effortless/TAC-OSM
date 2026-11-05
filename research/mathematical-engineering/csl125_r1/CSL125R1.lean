/-
CSL-125-R1 source-side formal *candidate*, not a kernel-checked proof record.
The exact real-variable state updates below match the Python numerator/mass
recurrence. This module has NOT been compiled or axiom-audited and does NOT
prove full PyTorch equivalence, floating-point bounds or retrieval capability.
-/
import Mathlib
namespace CSL125R1

def numStep (alpha address value numerator : ℝ) : ℝ :=
  alpha * numerator + (1-alpha) * address * value

def massStep (alpha address mass : ℝ) : ℝ :=
  alpha * mass + (1-alpha) * address

/- If numerator is no greater than mass and all updates are convex, the
upper bound holds for one recurrent step. This is a proof candidate; the
actual Lean build has NOT run in the current execution environment. -/
theorem step_upper (alpha address value numerator mass : ℝ)
    (hAlphaNonneg : 0 ≤ alpha) (hAlphaLeOne : alpha ≤ 1)
    (hAddress : 0 ≤ address) (hValue : value ≤ 1)
    (hState : numerator ≤ mass) :
    numStep alpha address value numerator ≤ massStep alpha address mass := by
  unfold numStep massStep
  have hLeft := mul_le_mul_of_nonneg_left hState hAlphaNonneg
  have hCoeff : 0 ≤ (1-alpha)*address :=
    mul_nonneg (sub_nonneg.mpr hAlphaLeOne) hAddress
  have hRight : ((1-alpha)*address)*value ≤ (1-alpha)*address := by
    nlinarith [mul_nonneg hCoeff (sub_nonneg.mpr hValue)]
  linarith

/- Dual sign bound. Both statements are local exact-real inequalities,
not an online-state representational sufficiency theorem. -/
theorem step_lower (alpha address value numerator mass : ℝ)
    (hAlphaNonneg : 0 ≤ alpha) (hAlphaLeOne : alpha ≤ 1)
    (hAddress : 0 ≤ address) (hValue : -1 ≤ value)
    (hState : -mass ≤ numerator) :
    -massStep alpha address mass ≤ numStep alpha address value numerator := by
  unfold numStep massStep
  have hLeft := mul_le_mul_of_nonneg_left hState hAlphaNonneg
  have hCoeff : 0 ≤ (1-alpha)*address :=
    mul_nonneg (sub_nonneg.mpr hAlphaLeOne) hAddress
  have hRight : -((1-alpha)*address) ≤ ((1-alpha)*address)*value := by
    nlinarith [mul_nonneg hCoeff (sub_nonneg.mpr hValue)]
  linarith

theorem parameter_cap : 123699534 ≤ 125000000 := by decide

end CSL125R1
