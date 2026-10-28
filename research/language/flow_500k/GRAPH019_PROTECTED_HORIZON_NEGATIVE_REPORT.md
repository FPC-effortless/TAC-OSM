# GRAPH019 — full five-seed protected-horizon TAC training: capability gain FAIL

**2026-10-28. Frozen preregistration commit 2722ff6b505d013c63263597052e243c39a72118 preceded any GRAPH019 fit.** System-1/language family, not a new seventh variant. All GRAPH009–018 evidence stays unchanged.

## What actually trained
- Native model: exact GRAPH017 WaveCED500K + 1/4/16 future heads, **512,634 learned trained and deployed parameters**; original 499,290-parameter TAC and trained GRAPH017 comparison frozen.
- One new prospective fit strategy, *not* new model algebra: compute each original h1/4/16 categorical CE gradient on the real 129-byte FIT source window, project per-tensor h4/h16 gradient only if its dot product with h1 gradient is negative, combine `g1+.15*g4'+.15*g16'`, clip1, then apply AdamW lr=.001. No training-time modifications to model predictions or source owner metadata. This is related prior PCGrad-style optimization and not an original mathematical novelty.
- 5/5 seeds 1701–1705 fully trained on original authentic Vim9.1 30 FIT chapters, same source SHA for original 36 publisher chapters, same **120×8 recorded FIT sequence sampled per seed** as the original WaveCED checkpoint, context128. Legitimate authentic original six DEV chapters already previously EXPOSED.
- Original model, optimizer, torch/NumPy/Python/sampler RNG, seed and FIT sample SHA preserved in all five locally saved checkpoint binaries. All original checkpoint SHA computed and five no-fitting cold local reload/score checks passed; **Git/LFS repo cold replay remains BLOCKED**.

## Complete five-seed authentic EXPOSED DEV result

| NLL ↓ | Original TAC | GRAPH017 joint multi-horizon | GRAPH019 primary-protected |
| --- | ---: | ---: | ---: |
| 1 ahead | **2.537438** | 2.550268 | 2.550477 |
| 4 ahead | no matching direct head | 3.507245 | 3.507317 |
| 16 ahead | no matching direct head | 3.573829 | 3.573877 |

At exact same h4/16 original real targets the stronger practical FIT-only 3-gram baselines scored **2.675518** and **3.195463**, respectively. Graph019 remains markedly worse. Exact evaluation targets per seed h1=6144, h4=6000, h16=5424. Report each source/horizon separately; do not treat all overlapping characters and five seeds as independent real cases.

| Seed | GRAPH019 h1 | GRAPH019 h4 | GRAPH019 h16 | h1 delta vs GRAPH017 |
| --- | ---: | ---: | ---: | ---: |
| 1701 | 2.591302 | 3.493648 | 3.572632 | −0.000295 |
| 1702 | 2.537933 | 3.510360 | 3.535697 | −0.000527 |
| 1703 | 2.531459 | 3.534477 | 3.671065 | −0.000281 |
| 1704 | 2.528488 | 3.495183 | 3.530032 | +0.002405 |
| 1705 | 2.563201 | 3.502918 | 3.559962 | −0.000257 |

Mean GRAPH019 NLL h1 +0.000209 *worse* than GRAPH017, and h4/h16 also slightly worse. Four seeds have very small h1 gains but seed1704 regresses enough to erase the mean. This does not meet the precommitted original-TAC one-step non-regression gate (original h1+0.01 threshold) and fails both future-horizon FIT reference gates.

Training 120-step elapsed times with triple-gradient protection were 27–31 seconds per seed on local PyTorch eager CPU, compared with ~14–16 seconds per GRAPH017 simple joint backward seed, an approximately 1.8–2× observed training overhead. Time is operational cost evidence, NOT a full-FLOP-matched head-to-head.

## Mathematical interpretation and next source-data strategy
GRAPH018 had guaranteed an **exact-real Euclidean first-order gradient inner-product fact**, not an actual AdamW finite-step validation score. All five frozen real FIT gradient tests satisfied that algebra. **GRAPH019 empirically fails to convert that algebra into actual one-step or future-horizon gain**, showing no basis to promote projected gradients as intelligence.

A source-only FIT-trained history model with 21k contexts predicts h4/h16 substantially better after sufficient exposure than the new neural heads after 120 optimizer updates. Further tests should **not** be shallow loss-weight sweeps on the same 6 previously exposed source files; instead freeze sufficient data-exposure budget and evaluate on independently authorized operational timeseries or source-specific facts with observable consequences. Quantify calibration, complete clean-original checkpoint replay and actual new-world ability, not just source-text NLL.

**Gates:** REAL_ORIGINAL_SOURCE=PASS locally, 5/5 NEW_MODEL_FIT_AND_REPLAY=PASS locally, GRAPH019_1STEP_GUARD=FAIL, GRAPH019_H4_FIT_REFERENCE=FAIL, GRAPH019_H16_FIT_REFERENCE=FAIL, REPO_COLD_REPLAY=BLOCKED (trained original binary bytes outside Git), NEW_SEALED_TEST=NOT_RUN, ARCHITECTURAL_SUPERIORITY=NOT_ESTABLISHED, CUSTOMER_VALUE=NOT_MEASURED, WHOLE_PNDS_LEAN_IEEE_AD_PROOF=OPEN. No promotion, no merge or deployment.
