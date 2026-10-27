# LATFUSE010 — original physically observed event write/read attribution

**October 27, 2026. Read-only original genuine-source forensic complete and CI passed. All source-trained 99,390-parameter checkpoints unchanged. No architecture superiority, long-horizon factual recall or independent generalization established.**

## Original authentication and scope

[Original genuine physically observed UCI501 true-source CI #40449654017](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40449654017) PASS. Recovered exact historical source ZIP SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8. All FOUR original LATFUSE008 source-trained checkpoints (seeds80701/80702 × associative historic query versus fast/slow-only historic query) match original training checkpoint SHA before/after; every original weight fixed, zero optimizer/model updates. All original 1,696 FP32 recurrent values+16 occupancy flags=6,800 nominal bytes. Original normalizer is from source history preceding March1 2014, not future May sensor readings. Exactly first FOUR genuine physical event/ordinary pairs from existing registered 24-pair source-locked May–June2014 internal cohort; same source monitoring station, same hour-of-day, authentic raw PM2.5 high >=150 ug/m3 versus ordinary <=35 ug/m3, original physically observed source hourly histories. Entire original 2015 DEV numeric, protected2016–17 numeric, and unobserved original outcomes never accessed.

All statements are **source-matched internal forensic only**; no new model fit, no new scientific controlled generalization gate, and no evidence of a learned robotic action/counterfactual policy.

## Three new independently defined measurements

1. **Actual source event WRITE key/value gradients.** With model learned parameters frozen, source observation at genuine anchor t becomes a differentiable input to the UNCHANGED original learned encoder/step. Capture actual key_map(u_t) and val_map(u_t) activations by hooks at the real observed event step; run original real hourly suffix and actual native historical-time query. Read source PM2.5 high/ordinary event truth ONLY AFTER the model predicts; backprop true binary BCE into captured original event activations and observed source input (activation gradients, NOT optimizer gradients to learned key/val parameters). The no-association event-query control must have no event-BCE gradient into past key/value despite regular ordinary forecast retaining associative memory.
2. **Original trained historical QUERY to event WRITE-key alignment.** Compute exact original 2×8D normalized learned event write keys under genuine observation masks and elapsed-time station inputs; compute exact original model past-time6D known calendar + present fast/slow belief query, normalized per head, with original trained read router weights. Observe headwise cosine, read-weighted signed/abs cosine, physical high/ordinary context. Original write key is a diagnostic target reference and NEVER supplied as an event query input. High-dimensional random angular geometry, small source cohort and learned routes limit interpretation.
3. **ONLY one original source event-step matrix change lesion.** Duplicate stream immediately after source observed anchor t, changing associative matrix back to its immediate pre-update value only; identical legitimate fast/slow/entity/occupancy state and identical subsequent physical source observations/masks/elapsed. Compare native predicted event probability and source-pair ranking after16/64/128/512 authentic hourly updates. The bypass removes original anchor write AND original anchor retention/overwrite of earlier facts, so it is an exact local matrix **step difference**, NOT surgical factual deletion or an actual alternate world.

## Original source actual result, two original trained associative-query seeds mean

| Real source delay | Event key activation grad norm | Event value activation grad norm | Historical read/write weighted absolute cosine | Mean ABS predicted event-probability difference after exact event-step matrix bypass | Original source paired ranking (same after lesion) |
|---:|---:|---:|---:|---:|---:|
| 16h | 0.0009775 | 0.0002406 | 0.2980 | **0.0008361** | 25.0% |
| 64h | 0.0007671 | 0.0001474 | 0.2691 | **0.0003921** | 25.0% |
| 128h | 0.0005588 | 0.0000998 | 0.3345 | **0.0002872** | 25.0% |
| 512h | 0.0001055 | 0.0000140 | 0.3058 | **0.0000422** | 25.0% |

The source event label BCE gradients are CONNECTED to original event-time key AND value **activations** across real16–512h source suffix. They are small in native units, but absolute scales cannot be compared directly to unrelated module parameters without Jacobian normalization. Event-time source INPUT gradient norm associated model ~0.000440/0.000326/0.000227/0.0000348; actual source event post-write associative STATE gradient~0.03084/0.02522/0.01772/0.00309. These numbers distinguish a mathematically reachable matrix state from the later BCE loss's low sensitivity to any single source event writer activation.

At original L=64h (ONLY trained source lag) original native historical write key and learned query average weighted absolute cosine **0.269**, loosely similar to chance angular similarity scale for two unstructured8D unit vectors. This **does not mathematically prove misalignment**; useful true event address would require source-conditioned same-event versus nuisance controls and decoder relevance, not merely achieving a high cosine score. The model's original event prediction changes by just 0.000392 average probability when its exact event write-step matrix update is bypassed; source true high/ordinary pair ranking remains25.0% in this deliberately tiny four-pair forensic cohort. Attribution of the prediction's failure exclusively to key geometry is therefore premature; event-encoding relevance, read-head bottleneck, directional overwrites and only32 original AdamW optimizer steps remain distinct possible causes.

## Original non-associative historical event query control

Both original fast/slow-only query-model seeds showed **zero connection** of true event BCE gradient to original event key_map / val_map activations and original after-write associative state at every delay (expected). Physical source EVENT INPUT gradients were NONZERO via the actual fast/slow recurrence at16–512h, confirming the source observation drives a distinct legitimate read path. Its historic event logit is bitwise invariant to bypassing the event-only associative matrix update after identical source suffix, while its ordinary source six-hour forecast STILL uses the associative memory. It is a clean historical query isolation control, NOT a totally memoryless world model.

## Mathematically scoped original event-step difference

For this fitted two-head source-conditional delta update:

M_t = α_t M_{t-1} + β_t (v_t - α_t M_{t-1} k_t) k_t^T.

The local exact original step bypass state uses M_t^{skip}=M_{t-1}, so ΔM_t=(α_t-1)M_{t-1} + β_t(v_t-α_t M_{t-1}k_t)k_t^T; all other state fields remain identical. Conditional on identical subsequent genuinely observed source inputs, and because the actual model's writer gates/fast-slow/entity state do not read old M, future difference evolves as ΔM_{t+L} = ΔM_t × ∏_{i=t+1}^{t+L} α_i(I-β_i k_i k_i^T) for each head. This is the same local affine difference transport numerically checked at original FP32 in LATFUSE005. It does not establish information-theoretic source event capacity, exact IEEE floating point/AD refinement in Lean, causal physical intervention, or calibrated probability-law correctness.

## Next experiment and gates

**Priority 1: authentic source-conditioned memory use rather than generic gate occupancy.** Falsify whether a meaningful source event-specific **historical age/identity cue** and training objective actually aligns a source-relevance-based writer with later legal queries. Six cyclic historic-clock features do not explicitly encode lag interval, a unique original sensor event identifier, or absolute elapsed time since observation. A candidate should use known requested lag/event-query metadata ONLY if truly available at inference; past recorded sensor value remains a training target, never an oracle query argument.

**Priority 2: isolate useful write supervision from reformatting the writer** under original FIXED256-cell associative storage, unchanged 1,696 FP32+16 bool state and ~100k params. First train/query-align read-only baseline, then separately source-dependent beta admission, contrasting actual historical-event-value utility against fast/slow/no-association, source-station/clock/last-value shortcuts. Do not claim improvement just because keys are orthogonal or writes are uniform.

**Priority 3: prerequisites before NEW qualification-grade training:** New source time/station/entity split, genuine independent source outcomes, non-transductive normalizer, meaningful old-event labels and source identity, same documented samples/optimizer/seeds/horizon and training target, at least2 seeds, <=1% active/deployed param and **independently verified FULL forward/AD/optimizer FLOP parity**, original checkpoint/optimizer/RNG/source/code SHA and clean CI replay; protected2016–17 numeric untouched absent source authorization. Current original exploratory real FIT remains a negative result, cannot be retrofitted into a successful model by tuning exposed 2014 eval.

**Conclusion:** native past-event gradients reach BOTH actually learned write-key and write-value activations, but their event-specific effect on the trained historical output is tiny and learned read/write angular alignment is weak/unverified. Treat next fix as **useful information writing and source-grounded address supervision**, not indiscriminate retention or more latent cells. Keep PR DRAFT/UNMERGED; original model work unchanged.
