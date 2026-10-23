# ME076 — Actual original robot-trained System 1 · NEGATIVE predictive result

**Status: five original authentic-source models FIT, checkpoint-replay PASS; development capability improvement FAIL.** October 23, 2026. This is not a new SOTA result, live policy model or independent real-world demonstration. [Draft PR #360](https://github.com/FPC-effortless/TAC-OSM/pull/360), unmerged.

## What actually ran

- Five **whole native System1FastModel checkpoints**, **100,476 actual trainable/deployed learned parameters each**, seeds **76001–76005**, 160 guarded genuine real FIT AdamW updates/seed.
- Official BodenAI RW-RL-HIL 14D real robot source, **CC BY 4.0**; original previously exposed source FIT 160 episodes checksum `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`, original per-source metadata and human/policy authority validated. No original CAL, prior sealed TEST or protected new tissue-box task opened.
- Deterministic episode-wide partition from *previously exposed FIT*: 120 episodes for model and normalization training, 40 episode-disjoint REPORT, eight per task. **This is exposed DEVELOPMENT evidence** despite episode segregation, **not an untouched TEST**. Original optimizer/RNG, source identity and every serialized model/prediction/checkpoint SHA in Git.
- Training: 8 factual observed state frames, 8 previous issued actions (lag1, original observed source), 1/15s nominal timestamp grid, 160 updates batch12, half movement-enriched training samples; no runtime human authority leakage or future labels in inference. The physical motion label is used only to balance *training* or to stratify *post-outcome* evaluation.
- Evaluation: **960 registered factual next-state predictions per model** over 40 original episodes; 84 genuinely moving, 876 near-stationary. Real task/file ending is not verified task success.

## Core results: actual physical 14D next-state MAE

| Cohort | Observed real queries/seed | Last-state persistence | FIT-only servo (available lag1) | TAC native fast-only (five-seed mean) |
| --- | ---: | ---: | ---: | ---: |
| ALL real eligible future states | 960 | **0.068668** | 0.097457 | **0.080158** |
| Actual motion-rich (threshold >=0.025 mean physical movement) | 84 | 0.743241 | **0.704517** | **0.748786** |
| Near-stationary | 876 | **0.003983** | 0.039245 | **0.016043** |

**0/5 original trained seeds beat last-state persistence for all cases or motion-rich cases.** Registered primary all-case and moving >=5% MAE improvement gates **FAIL**. No demonstrated greater world-state predictive intelligence or physical reflex benefit. The system beats servo on the *all-case mixture* but servo is worse than persistence there; do not cherry-pick the weaker reference.

Original seed-wise all-case native MAE: `76001 0.083078`, `76002 0.076604`, `76003 0.080977`, `76004 0.080320`, `76005 0.079813`. All share identical **960 factual queries**, so five seeds are **not** 4,800 independent cases. Genuine original episode count is **40**, across five source tasks.

### No-fit repeatability and negative-result uncertainty

- [Original five fitted weights, optimizer/RNG, live budgets, 4,800 raw real predictions](../../replay-bundles/ME076/): Git-owned original artifacts, each model checkpoint hash verified.
- [Independent clean-checkout original replay](../../replay-bundles/ME076/INDEPENDENT_ORIGINAL_REAL_REPLAY.json): **PASS all 5 original checkpoint/score restores**, zero refits and unchanged original source.
- [Original full live parameter+optimizer+training budget final certificate](FINAL_COMPARISON.json): verified **PASS** for all five original frozen 160-update model receipts; no comparison-compute parity or architecture-superiority claim.
- [Post-hoc real-data diagnostic](POSTHOC_REAL_SCORE_DIAGNOSTIC.json): pooled native **+0.011490** MAE worse than persistence, 5,000 resampled **40-episode cluster bootstrap 95% interval +0.011116 to +0.011862**. For 84 actual motion-rich queries across 37 original episodes, excess native error **+0.005545** (post-hoc 95% CI **+0.004138 to +0.007020**). This is a **post-hoc original real exposed-development interval**, not a prospective independent scientific promotion test. It conditions on original model-seed collection and source authority/sampling limitations.
- Historical per-task check: native exceeds persistence on **all five** original task-defined overall MAE cohorts (8 real report episodes/task). Negative result generality is limited to those tasks and that original source.
- The original source/forecast, actual physical label and live receipts are unchanged by this report.

### Why the deficit matters, without unsupported attribution

Under the publisher's autonomous-trajectory definition, only **8.75%** of the registered REPORT queries have meaningful physical movement under the 0.025 diagnostic. Original training deliberately balanced ~50% meaningful movement despite ~8.86% natural movement prevalence among its 190,911 eligible historical training windows. Overestimating movement in stationary states is a *plausible* explanation for the near-stationary error, but the study did not isolate the memory, sampler or loss as uniquely causal. We must not declare the 96D fast state “broken” or claim any global recurrent-world-model impossibility.

The model only received previously recorded lag1 issued action qpos, not verified physical motor torque; real actuator delay and causal commands remain unqualified. Output is next-state density, **not** independent safety permission, learned takeover hazard, human benefit, verified business completion or low-latency deployment.

### Compulsory next *capability* experiment, NOT an attribution series

Register **a new integrated model** (separate prospectively frozen real experiment, 95–105k trainable/deployed parameters) with: last-observation persistence anchor, learned factually observed *movement-occurrence* gate, movement-conditioned residual/scale and honest risk/abstention diagnostics. Train only on authentic FIT original outcomes. Conditional probability and residual must be based on known-at-time observations/actions; future movement label can be used for FIT supervision but never at inference. Avoid outcome-enriched sampling bias or correctly correct for the actual source base rate. Score proper next-state NLL and balanced/actual cohort movement error together; penalize unnecessary corrections.

Use the historical ME076 model as the **frozen same-task developmental reference** and strongest original last-state/servo practical reference. A genuine `INDEPENDENT_GAIN` needs a newly preregistered genuinely untouched real source/task and case-cluster confidence, and all original seed checkpoints+replays in Git. The protected originally unopened physical task may not be sampled without an explicit prospective target authorization.

**Investor claim impact:** physical advisory System 1 is **not product ready**. Engineering, real training and faithful replay are demonstrated; measured practical predictive advantage, verified action utility, policy safety, submillisecond latency, compact 25k deployment and commercial ROI remain unqualified.

## Reproduction

The original dataset/source and split identity are in [SOURCE_QUALIFICATION.json](SOURCE_QUALIFICATION.json). Fit entrypoint `scripts/me076_train_system1_real.py fit` is **historical and checkpoint-overwrite protected**. Reproduce scores without any new fitting via its `replay` mode and the clean-checkout CI. Recalculate published posthoc results without data download via `python scripts/me076_analyze_original_scores.py`. Runtime validation `python scripts/validate_research_parity.py --manifest research/mathematical-engineering/me076/FINAL_COMPARISON.json --phase final`.

**No original physical outcomes or previous trained weights were altered to produce a more favorable headline.**
