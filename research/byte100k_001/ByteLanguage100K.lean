/- Partial arithmetic/offset contract; not PyTorch graph, IEEE or AD refinement. -/
import Mathlib

namespace ByteLanguage100K

def blockParameters : ℕ :=
  2*48 + 3*48 + 4*48*8 + 4*8 + 48*48 + 2*4*8 + 3*48*96 + 2 + 48*4 + 4

def attentionParameters : ℕ := 48 + 3*48*48 + 1

def modelParameters : ℕ := 256*48 + 4*blockParameters + 2*attentionParameters + 48

theorem parameter_count : modelParameters = 99050 := by
  norm_num [modelParameters, blockParameters, attentionParameters]

theorem parameter_band : 95000 ≤ modelParameters ∧ modelParameters ≤ 105000 := by
  norm_num [modelParameters, blockParameters, attentionParameters]

def inputOffset (start position : ℕ) : ℕ := start + position

def targetOffset (start position : ℕ) : ℕ := start + position + 1

theorem target_follows_input (start position : ℕ) :
    inputOffset start position + 1 = targetOffset start position := by
  rfl

theorem target_inside_source (start stop length position : ℕ)
    (hstart : start < stop) (hstop : stop < length)
    (hposition : position < stop - start) :
    targetOffset start position < length := by
  dsimp [targetOffset]
  omega

end ByteLanguage100K
