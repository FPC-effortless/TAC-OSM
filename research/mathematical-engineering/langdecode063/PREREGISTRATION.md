# LANGDECODE063 — pre-registered shared upper input, inference-only

## Source hypothesis and mathematical change
The original `RecurrenceOnlyUpperBlock.read_from_prior(z,h)` and `observe(z,h)` independently calculate exactly the same
`u = tanh(input_proj(input_norm(z)))`. Original `IncrementalTAC043._advance` calls both sequentially on the same `z,h`, in eval/no hooks. With a pure deterministic input projection and no mutation between calls, one calculated `u` can be reused for both original calls.

Prewrite read (original order): `gate = sigmoid(read_gate(cat(u,h)))`; `refined = z + sigmoid(gain) * output(tanh(u + gate*h))`.
Then the original state write: `alpha = .995 * sigmoid(forget(u))`; `next_upper = alpha*h + (1-alpha)*tanh(offer(u))`.
**No** new weights/buffers, no hidden-state change, no position encoding change, no reordered algebra within these expressions. The original decoder remains untouched. The candidate source is a distinct `SharedUpperTAC063` subclass of the original validated decoder.

## Falsification protocol
Parent source: original `research/mathematical-engineering/langstream043/incremental.py`, source-pinned 499623-parameter original seed1701 FIT320 checkpoint, original licensed CPython3.13 original FIT owner, 128/512 bytes and 16 original greedy next tokens, entirely **inference** only.

A. Constructed FP32/FP64, native/cap_all, two stream owners, causal write masks, direct upper reference and bidirectional snapshots. Check exact all nine state tensors and logits.
B. Original checkpoint, source/optimizer/RNG SHA receipts. `BatchedValidationTAC057` versus `SharedUpperTAC063` compare bitwise incremental logits and all nine states, frozen full growing-prefix max abs <= 3e-5, same argmax and snapshots. FP32 3904B/owner. No architecture novelty or model quality claim.
C. Fair CPU FP32 eager batch1, torch intraop2/interop1; two paired warmups, seven paired measured repeats alternating arm order, same original 16 generated tokens, inherited state validation included. **Acceptance only if** both 128/512 prompts see median >=5% improvement and candidate/original p95 <=1.10, alongside ALL parity passes. Otherwise archive NEGATIVE/INCONCLUSIVE; don't retune gates. Single-host acceptance cannot promote to production: independently repeated host and complete PNDS mathematical/source bridge still needed.

Prior LANGDECODE062 was negative (+4.49%/+4.35% median, 128B p95 ratio 1.1223). It is not the baseline of this controlled test; use the original decoder.

## Formalization / evidence status
Reciprocal PNDS exact operator/Lean4+Python reference and source-pin: **OPEN**, to be linked from a separate PNDS draft. General IEEE, AD, complete trained graph and source equivalence: **OPEN**. Restricted causal actual checkpoint replay supports engineering equivalence only. Original ~500K model is historical and not retrained; prospective 100K parity-fit constraints do not apply to this no-fit diagnostic.
No protected TEST, no Kaggle, GPU minutes zero, 0 training updates, draft only, no merge or checkpoint promotion. The workflow may succeed while `all_latency_gates_pass=false`; that is a negative experiment, not permission to ship.
