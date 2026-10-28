# TAC-LANG-FLOW-500K-001: prospective 500K-scale wave/CED language feasibility

Date: 2026-10-28. Status: **local experimental candidate / 3 of 5 planned fits complete**, not qualified superiority, not verified as whole-model Lean/PyTorch refinement. Research within the existing six-family architecture. This report does not revise or upgrade the original 100K checkpoint results.

## Prospective policy

Compulsory new target is **500,000 active learned parameters**, allowed **475,000–525,000**, and new comparative models must differ in active learned count by <=1% with exactly matched authentic data, objectives, optimizer and training FLOPs where a fair architecture superiority claim is intended. Legacy 95K–105K experiments retain original budgets and records. All original checkpoint/model/optimizer/RNG bytes must be replayable from Git before model promotion.

## Actual locally executed candidate

Source: `tac_wave_ced_500k.py` SHA256 `acf80cb9051b7f2d2c3900d1676eed212890208cd5213088c6751f9cfd24392c`. 128-character vocabulary, 48-dimensional learned embedding, 88-dimensional causal feature-prefix sum, native 8x8 matrix, two 16x8 aligned heads, eight 2D damped wave modes and six 16D persistent source-only concept seeds. **520 floating recurrent values (2080 nominal bytes)**, excludes weights, metadata, transient scan tensors and solver workspace. 605-wide SwiGLU language readout and 6x6 quadratic postscan concept equilibrium. Python/PyTorch live `model.parameters()` count is **499,290**, and all 499,290 learned coordinates participated in the first actual backward graph (non-None gradients).

The input-only schedules are right-affine and scanned chronologically through a Hillis–Steele implementation, O(log T) time *depth* with O(T log T) work on the memory/wave scans. This is **not a proven fast complete model or a streaming-carrying Torch implementation**. The normal forward begins with zero state at each segment. The source `start` argument is not a valid carried-session state initializer; any nonzero-offset runtime deployment requires an actual stateful streaming implementation and new correspondence checks. Native/aligned writes are masked; basis/waves/concepts still process current observed tokens, an intentional separate admission boundary requiring formalization.

### Math

Damped transport: z_{j,t+1}=rho_{j,t} R(omega_j) z_{j,t}+B_j e_t with 0<rho<1, R orthogonal.

Two independent affine rectangular memory read/write schedules and original native memory: `M'=MP+Q`, `P=alpha(I-erase*k*k.T)`, `Q=write*v*k.T`. If erase=write the source returns to corrective gated delta; do not claim arbitrary independently trained gates preserve retrieval. Read memory at previous prefix only, no current-write self-read.

Persistent concept seeds: `C^p_next=(1-g) C^p + g tanh(W_c e)` with learned source-only coordinate gates, scan-parallel; concept working state is separate from this persistent state.

After the scan, directed concept similarities are normalized and symmetrized with nonnegative weights; Laplacian `L=D-A` is PSD. Compute `C*=[(1.25)I+0.25 L]^-1(G+0.25 C^p_previous)`. This uniquely minimizes `0.5||C-G||²+0.125 Tr(C.T L C)+0.125||C-C^p_previous||²`. This is a predictive-consistency quadratic, **not yet a fully identified active-inference free-energy model**. Next-token logits come from a 605-wide SwiGLU plus pooled concepts; categorical softmax is normalized by construction.

## Executed checks

`python -m unittest -v test_wave_500k.py` **9/9 constructed software groups PASS**: tensor count, full backward graph, FP32/FP64 affine scan vs chronological serial, autograd scan gradients, independent concept-seed scalar scan, future-token nonleakage, memory non-write byte preservation, all-refused associative writes, symmetric graph PSD, ordinal code's known 155,701 periodic alias.

Real text: authentic Vim9.1 manual, **30 FIT chapters, 6 previously exposed DEV chapters, all 36 original SHA256 hashes verified** against archived TAC-LANG-CURRENT-005 protocol. New isolated context128/batch8/120-update next-char-only AdamW fits (no history auxiliary target). FIT-only smoothed bigram measured on identical DEV windows. Eight fixed windows from each of six DEV files, 6,144 character predictions per completed seed. Completed seeds: 1701, 1702, 1703. Seeds 1704 and 1705 did not complete within this execution, **do not count them as trained**.

| completed seed | original DEV character perplexity ↓ | FIT bigram ↓ |
|---|---:|---:|
| 1701 | 13.254243 | 15.472 |
| 1702 | 12.497735 | 15.472 |
| 1703 | 12.412341 | 15.472 |

A separate `replay_completed.py` loaded all three original model/checkpoint bytes without optimization, checked SHA256 and exactly reproduced scores within 1e-8; each checkpoint includes optimizer and Python/NumPy/Torch and sampling RNG state. The binary files are in **this local research archive, not GitHub**, leaving the user's compulsory in-repository independent replay gate **OPEN**. This is not a fair comparison with 500K GRU/SSM/Transformer and not an independent unseen test; the exposed documents were already used in prior work. The comparison to bigram is only a practical reference, not matched learned parameter/compute architecture superiority.

## Research decision and outstanding gates

**Source behavior/architecture viability observed; no promotion.** The larger wave/CED full graph trains and beats the fitted bigram on a limited previously exposed real text diagnostic, with all three completed local checkpoint replays. It has not established long-distance factual retrieval, calibrated language predictions, efficient end-to-end streaming, 500K superiority, or new Lean formal correctness.

1. Commit tested Torch source, original binary checkpoints, optimizer and RNG, and source manifests into TAC; use clean checkout cold replay in CI. Do not silently change this source because its SHA and checkpoint computations are fixed.
2. Complete PNDS source-pinned exact complete token-to-categorical Lean model, wave/energy formal lemmas, independent Python/Torch FP32/FP64/AD correspondence, and finite-precision / optimizer proof obligations. New PNDS 500K scalar lemmas are scoped mathematics, not whole-program proofs.
3. Fix the explicit streaming-carry boundary in a *new version ID* and test actual chunk/resume equivalence. Current code is a zero-state per-window trainer.
4. Train prospective 500K SSM, GRU and attention controls on truly identical source samples, objectives and full training compute, plus strong next-token ngrams. Reserve a new independent multi-document TEST for research promotion.
5. Test authentic 512+ character and subword contexts for **specific past fact recovery and downstream prediction improvement**, not just retrospective position-related loss changes.