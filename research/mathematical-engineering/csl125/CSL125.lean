/-
  CSL-125: partial Lean 4 formal record. Typed assumptions and propositions
  apply ONLY to discrete causal-window membership and a fixed integer budget.
  No whole-model forward/backward, learned-memory, floating-point stability,
  tokenizer or training claim is proved here.
-/
namespace CSL125

def legalKey (query key window : Nat) : Prop :=
  key ≤ query ∧ query < key + window

theorem legalKey_no_future (q k w : Nat) (h : legalKey q k w) : k ≤ q := h.1

theorem legalKey_window_bound (q k w : Nat) (h : legalKey q k w) : q < k + w := h.2

theorem fixed_count_within_cap : 123699534 ≤ 125000000 := by decide

end CSL125