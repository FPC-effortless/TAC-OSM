# GRAPH018 — primary-protected gradient projection, engineered but NOT TRAINED

Date: 2026-10-28. System-1/language TAC family, stacked on GRAPH017 #546. **Status: verified mathematical software candidate, not a new predictive model result.** GRAPH017 failed the h1 guardrail and h4/h16 matched practical references. That failure is preserved and must not be retrospectively rescored as a win.

## Grounded failure motivating the repair
GRAPH017 native next-byte NLL 2.550268 vs original native TAC 2.537438 on authentic previously EXPOSED Vim DEV; h4 3.507245 vs FIT-trigram 2.675518; h16 3.573829 vs FIT-trigram 3.195463. Five complete original checkpoint/optimizer/RNG seed replays. Read-only real FIT gradient inspection at each seed's **first original sample batch** shows opposing shared recurrent-gate task gradients: mean cosine for 1-vs-4 at native_gate.weight **−0.36645**, 1-vs-16 **−0.09678**; different feature layers are positive. Five authentic original FIT minibatches provide a very narrow observation; this does NOT establish that the conflict caused the failed neural horizon scores.

## Repaired optimization law — constrained auxiliary gradients (not novel)
For Euclidean main-objective gradient `g1` and auxiliary `gh`, apply per-active-tensor
`gh' = gh − min(0, gh·g1)/||g1||² * g1` when `||g1||² > epsilon`; use `gh'=gh` otherwise. Then form `g = g1 + 0.15*g4' + 0.15*g16'`.
In **exact arithmetic** with nonzero main gradient: `g1·gh'=max(0,g1·gh)`, and for nonnegative weights `g1·g >= ||g1||²`. Therefore the direction `−g` cannot be an ascent direction for the main differentiable loss *at that parameter position in Euclidean first-order Taylor analysis*. This is a first-order geometrical result, not a bound on actual loss after applying AdamW, gradient clipping, momentum, finite step-size, decoupled weight decay, nonconvex curvature or numerical errors.

This adaptation is rooted in published gradient surgery/PCGrad (Yu et al., NeurIPS 2020) and is **not proprietary novel theory**. See https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html.

## Completed implementation diagnostics
- Versioned function `primary_guard` validates same-shaped floating finite tensors and handles zero-main direction and negative/positive dot product.
- `protected_backward` computes all three source-causal task gradients on a shared graph, projects horizon auxiliaries *per learned tensor*, populates `.grad` without taking an optimizer step, and exposes counts of detected opposing gradient components.
- Six **constructed** mathematical unit tests PASS for orthogonality, nonconflicting gradients, zero main gradient, rejection of malformed input, full adversarial multimodal gradients and finite shared/separate heads. These tests are NOT trained model intelligence.
- Five original GRAPH017 512,634-weight frozen checkpoints were loaded read-only and projected on the first corresponding authentic FIT mini-batch. All five satisfy `g1·g_protected ≥ ||g1||²` within numeric tolerance on the complete active learned parameter vector, with removed opposing components across h4/h16 (seed-dependent). Original model weights and all prior DEV scores remained unchanged.
- Real FIT evidence is **gradient-only**, not a new forecast skill. The user-owned archive and lawful source were locally verified, but **the checkpoint binaries remain outside Git/LFS** so REPO_COLD_REPLAY=BLOCKED.

## Scientific next training gate (not run)
A prospective new fit would need a *new preregistration* with full authentic FIT, exact original trained seed/model/source hashes, real observed h1/4/16 horizon denominators, physical resource/FLOP cost, a source-based FIT reference and an already-frozen one-step guardrail. Any new official score after the old Vim/CMake/Python exposures is development-only. No blind tuning on those exposed documents. Better aim for transferable, source-authenticated task and long-horizon outcomes, not merely another 128-byte corpus. Need original trained binary Git replay before scientific promotion.

**Current status:** MATH_SOFTWARE=PASS; GENUINE_FIRST_ORDER_FIT_DIAGNOSTIC=PASS; GRAPH018_TRAINED_NEURAL_CAPABILITY=NOT_RUN; INDEPENDENT_NEW_DATA=NOT_RUN; ADAMW_LOSS_GUARANTEE=NONE; REPO_COLD_REPLAY=BLOCKED; LEAN_FP32_FULL_PROOF=OPEN; CUSTOMER_UTILITY=UNKNOWN.
