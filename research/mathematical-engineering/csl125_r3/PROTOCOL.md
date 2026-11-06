# CSL-125-R3: risk-conditioned revision, no-fit preregistration

Status: SOURCE-ONLY, NOT TRAINED, NOT CALIBRATED, NOT MERGE/PROMOTION READY.

## Parent integrity and dependency

- Source parent: TAC-OSM PR #714 (itself stacked on R2 #713, R1 #712, CSL125 #711, CSL001 #710).
- R3 changes no existing file or R2 memory-update arithmetic; the failing 11/84 gradient rows of #714 remain blocking. R3 is a *new variant*, not a repair of those failures.
- Model ceiling: 125,000,000 unique learned parameters including token embedding/tied output and every auxiliary head. At default width=768/rank=32, R3 adds 50,689 parameters to R2's 123,822,574 = **123,873,263**. R3 must be checked with meta-device construction before any fit.
- The 32,768-piece byte-fallback tokenizer remains untrained; no model optimizer step is authorized.

## Hypothesis and limitations

R2's learned admission suppresses some irrelevant writes, but an output can still be wrong. R3 makes a *separate*, shared, trainable prediction of whether the base next-token argmax is wrong and proposes a bounded, low-rank residual to the final feature vector. One small head is reused at the 2, 8 and 20 exits. The corrector does not write to persistent state, invoke tools, autonomously retry, or confer epistemic authority; it is not a general-purpose verifier. Critically, without additional evidence the model may not know when its own predictions are wrong. Its risk score must be independently calibrated before it can govern inference budgets.

Let h_r be the normalized hidden state at rung r, w_r = sigmoid(u^T h_r + b), and δ_r = tanh(W_down SiLU(W_up h_r)), with rank k=32. The revised hidden proposal is h'_r = h_r + g_max w_r δ_r, where g_max=0.25. Thus, for each coordinate, |h'_r-h_r| <= 0.25 in real arithmetic. Both base and revised logits use the exact same counted tied output matrix.

Teacher-forced training computes per-token base and revised NLL, a risk BCE target from the detached base prediction's correctness, and a positive penalty whenever a revision worsens a token already correct under the base model. Ground-truth next tokens are targets **only in the loss**, never inputs to forward. The loss is a design candidate, not evidence of calibrated error detection or error correction.

## Preregistered experiment order

1. **Math/code qualification:** independently check the R2 scan cancellation and stationary gradients; retain original failing ledgers. R3 shape/causality/stream-identity/mask/leakage tests on CPU, full-size meta parameter cap. Test source-vs-parent equality of every base output and memory state with identical weights. No accuracy promotion.
2. **Data and budget gate:** authenticate rights and fixed FIT/DEV/held-out TEST; fit BPE **only with FIT**; register optimizer/LR/scheduler, exact token budget and source spans, seed list, devices, checkpoint restoration and independent evaluator; complete FAIR_COMPARISON plus reviewer gate and capped GPU budget.
3. **Mechanism screen (small first):** 1M and 10M CSL-R2 unchanged vs R3 vs R3 error gate constant vs R3 shuffled error score vs R3 correction disabled vs equal-extra-parameter FFN control. Use >=5 paired seeds, matched training tokens/data, equal budget/optimizer, and held-out exactly-once tests. Verify architecture-specific persistent identity binding and selective write protection separately; no oracle gate.
4. **Credible controls and scaling:** same tokenizer/training data/updates and matched parameter + compute envelopes against trained GQA Transformer and competent DeltaNet/SSM. Scale to 125M only after smaller model demonstrates robust paired improvement and budget approval.
5. **Actual selective compute:** after error-head calibration on DEV, evaluate early-exit stopping on *fresh independent prompts*, reporting full prefill+decode cost including risk scoring and any replay. Do not claim per-stream 2->8->20 switching efficiency; existing state ownership prohibits depth changes without replay. Never use an uncalibrated error head to execute or accept tools.

## Mandatory metrics

- Base and revised held-out token NLL and perplexity per rung/seed, mean, median, range, paired deltas and 95% intervals.
- Correction accounting: false->true / false->false / true->false / true->true, net fix minus break, error-classifier Brier, ECE, precision/recall, AUPRC, selective-risk and error-detection coverage; detection calibration on DEV only.
- Functional tests on genuine two-key role binding, replacements, untouched facts, absent facts, long-distractor retention, distribution shift, and conflict/contradiction. Report all failures.
- Both training and inference: parameter counts, active parameters, FLOPs/task and token, wall time, token/s, peak RAM/VRAM, p50/p95, quantized model bytes and actual mobile power where measured. Include reruns/replays, verifier and correction overhead. Device and precision pinned.
- Pareto dominance definition: compare accuracy/task utility at **both** fixed parameter/memory and fixed all-in compute. A model is Pareto improved only when no metric is worse beyond the preregistered uncertainty and >=1 materially improves. No 'intelligence per parameter' claims from random-weight tests or unequal corpora.

## Positive and negative falsifiers

- If risk scores are not calibrated, if gate shuffling or disabling revision doesn't reduce quality, if correct->wrong breakage neutralizes wrong->right fixes, or if data/evidence leakage drives the gain: NOT SUPPORTED.
- If R3 simply imitates adding a slightly larger FFN, or trained Transformer/SSM dominates at comparable budget: no novel efficiency or architecture claim.
- If any numerical/gradient audit that affects the trained path remains unresolved: training and 125M scale promotion blocked.
- Corrector is **read-only on persistent memory**. Verified retrospective memory correction, authority management and persistent evidence provenance are separate future research, not implemented here.


## Optional candidate-only revision: compute screen (source, not measured speed)

R3 also implements \`candidate_repair_topk=K\`, where the parent base
logits select K candidate IDs and the correction residual only re-scores the
associated K embedding rows. This does **not** construct a revised vocabulary
distribution and does **not** find a target excluded from the base top-K. The
option fails closed if K is not an integer in [1,min(128,V)] or the base logits
are disabled; it cannot authorize generation, acceptance or external actions.

For V=32768, d=768, rank=32 and K=32, first-order per-token extra dense
linear/gather dot-product MAC counts are:

- Full revised projection: V*d + 2*d*rank + d = **25,215,744 MAC**.
- Candidate-only revised scores: K*d + 2*d*rank + d = **74,496 MAC**.
- Pure operator-count ratio for the added revision path: ~338.5x smaller.

These are **analytical MAC estimates**, exclude base V*d projection, top-K
selection/comparisons, nonlinearity, softmax, kernel launch, data movement,
memory bandwidth and all earlier model layers. Do not call this a 338.5x
inference speedup. Report actual tokens/s, p50/p95, RSS/VRAM, kernel dispatch
and overall compute in a separately qualified hardware experiment.

Additional falsifier: if the true answer is absent from base top-K, this path
is structurally incapable of recovering it. Report top-K oracle recall and
candidate-only net fixes minus breakages for K in {4,16,32,128} separately
from unrestricted revised logits and never score a truncated K distribution
as though it were full-vocabulary NLL.
