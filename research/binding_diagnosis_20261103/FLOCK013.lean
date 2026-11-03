import Mathlib

/-! Partial exact-real formulation of the zero-gain adjoint repair.
    Native observation-to-output graph is inherited from the pinned COLLECT018
    source and its partial typed formulation. Full-model correspondence,
    hard-argmax differentiation, SPD/IEEE error and global AD proofs are OPEN.
    These lemmas do not imply trainability or retrieval capability. -/
noncomputable section
namespace FLOCK013

abbrev Vec (n : Nat) := Fin n → ℝ
abbrev Mat (v k : Nat) := Fin v → Fin k → ℝ

def candidate {v k : Nat} (previous : Mat v k) (residual : Vec v)
    (direction : Vec k) (gain : ℝ) : Mat v k :=
  fun i j => previous i j + gain * residual i * direction j

def gainAdjoint {v k : Nat} (residual : Vec v) (direction : Vec k)
    (upstream : Mat v k) : ℝ :=
  ∑ i, ∑ j, upstream i j * residual i * direction j

theorem zero_gain_forward {v k : Nat} (m : Mat v k) (r : Vec v)
    (d : Vec k) : candidate m r d 0 = m := by
  funext i j
  simp [candidate]

theorem exact_gain_increment {v k : Nat} (m : Mat v k) (r : Vec v)
    (d : Vec k) (g e : ℝ) (i : Fin v) (j : Fin k) :
    candidate m r d (g+e) i j - candidate m r d g i j = e*r i*d j := by
  simp only [candidate]
  ring

def legacyGainCredit (gain residual direction upstream : ℝ) : ℝ :=
  if gain = 0 then 0 else upstream*residual*direction

def repairedGainCredit (residual direction upstream : ℝ) : ℝ :=
  upstream*residual*direction

theorem legacy_zero_gain_credit (r d u : ℝ) : legacyGainCredit 0 r d u = 0 := by
  simp [legacyGainCredit]

theorem positive_gain_credit_unchanged (g r d u : ℝ) (h : g ≠ 0) :
    legacyGainCredit g r d u = repairedGainCredit r d u := by
  simp [legacyGainCredit, repairedGainCredit, h]

theorem nonzero_boundary_counterexample : repairedGainCredit 2 1 1 = 2 := by
  norm_num [repairedGainCredit]

theorem masked_chain_stays_zero (r d u : ℝ) : repairedGainCredit r d u * 0 = 0 := by
  simp

end FLOCK013
