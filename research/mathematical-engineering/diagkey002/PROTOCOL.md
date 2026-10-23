# DIAGKEY002 — separate WRITE / RETAIN / READ, then retest original frozen model short-delay ladder

**NO new trained neural models. NO original real TEST. Synthetic unit and mechanism diagnostics only.** This is a response to the failed-original DIAGKEY001 16/64/256 key/value association recall pilot: all native recurrent/GRU controls approximately 3.125% chance, transformer around 99%. The first five-seed original training lacked required live-budget optimizer guards and had ~20.9% relative measured partial FLOP spread; do NOT promote either success or global failure from that unequal-compute pilot. Original pilot bytes remain under `research/replay-bundles/DIAGKEY001`. Guarded DIAGKEY001_GUARDED approval is not yet verified; bots may trigger GitHub Actions `action_required` awaiting human approval, not technical run failure. No ability to approve workflow from connector currently.

## Stage 1 — exact arithmetic oracle before changing native architecture

On 8-dimensional orthonormal keys and onehot values, WRITE exactly four independent associations via the existing rank-one delta error update: `M <- M + (v-Mk)k^T`. Then query a randomly selected association after 0,1,2,4,8,16,64,256 DISTRACTOR writes. Predeclared four conditions:

1. **Correct write mask (oracle)**: never update any of the four protected target keys. This is nonlearned, reveals exactly if 8x8 memory math can retain four orthogonal mappings.
2. **Unfiltered distractor write**: update all random keys, including colliding original keys, measuring retention failure.
3. **Whole-matrix L2 normalization**: exactly the global norm-bounding operator used by native recurrence, with unfiltered writes; report true read amplitude and accuracy.
4. **Query-time overwrite**: overwrite the queried association with zero before read, testing that action type controls write admission; output may collapse even at zero distractors.

Run 1000 randomized source-only software oracle cases for each condition and delay. Exact answer accuracy, Wilson95 intervals, correct-value read amplitude, read-vector L2 error. The protected-key oracle **knows the target keys**; this is not an implementable learned attention policy or architectural victory. Any positive arithmetic only verifies correctness of the idealized rank-one update.

## Stage 2 — re-score ORIGINAL model weights before new optimizer updates

Load exact unchanged original five seeds × four DIAGKEY001 old pilot models (20 Git-owned original binary weights/optimizer/RNG and SHA) and evaluate a *new* stateless, answer-free 0/1/2/4/8/16/64/256 distractor ladder. Same four-write, 32-value original task, 512 new deterministic case IDs per delay/seed/arm. Report case accuracy, original five seed directions, chance3.125%, proper NLL and Wilson95 intervals. **Critical limitation: these fitted models saw only 16,64,256 during gradient training; 0,1,2,4,8 are out-of-training-distribution and cannot establish that a model trained from scratch on 0 cannot learn.** Short delays characterize existing fitted-model failure only.

Native model gradient integrity at observed write tokens and query: hook the original encoder output and obtain first four write token gradient norm, last query token gradient norm, norms of actual `write_map`, `query`, `fast_map` parameter gradients, finite check at delays 0,1,4,16,64,256. A nonzero gradient is necessary for learning the pathway but is NOT a measure of successful key/value binding.

## Stage 3 — rigor needed before training again

Do **not** immediately add a new gated delta-cell, hardcoded warmup, or optimizer sweep. Stage 1 must verify arithmetic and identify exactly which write/update admits unwanted tokens. Stage 2 describes pretrained-model responses but fails training/compute parity for superiority. A new **owner-registered ~100k-parameter 5-seed** zero-distractor curriculum comparing TAC, GRU, noassoc, and exact-8x8 delta positive control must freeze 0→1→2→4→8→16 difficulty, train/update compute, cumulative exposure, equal inputs, and meaningful param/FLOP parity before training. If native TAC cannot solve a correctly trained and instrumented zero-distractor condition, report that failure instead of spending compute on 256 distractors. If it can solve zero but fails as distractors grow, inspect write-gate/normalization effects and train controlled retention variants. Do not change the original real ME073 checkpoint or infer real performance from these synthetic fixtures.

Physical/enterprise performance: only authentic independently scored world events with controlled equal ~100k architectures can qualify TAC-OSM, with real information-time constraints, proper END-aware scoring and independent generalization checks.
