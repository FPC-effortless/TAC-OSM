# ME070 — long-history enterprise TAC-memory qualification and ME069 factual audit

**Registered on 2026-10-23, BEFORE any ME070 fit, raw BPI2018 cohort read, or ME070 heldout score.** This is a source-qualification and retrospective-diagnostic work package, **not** trained evidence or an authorization to open heldout targets.

## Precisely what ME069 proved, and failed

- On authentic BPI2012, matched native TAC and GRU share categorical/time encoders, the same FIT transition prior, residual/mixture head and waiting-time head. Both beat the empirical transition baseline. TAC **does not beat GRU**: TEST log loss .122626 vs .119182, with a case-paired CI of the difference entirely above zero; top-one accuracy ~94.77% each; TAC is ~2.56x slower to train. Do not label the shared forecasting gain TAC-specific memory usefulness.
- BPI2012 contains 262,200 events / 13,087 cases (~20 events per case); at-most-32-event reset windows **cannot establish a 64/256-event retention advantage**. ME069's midpoint prediction position was selected using hindsight total case length, so not a deployable online query-time scheduler.
- The existing original ME069 per-case score bundle permits **paired accuracy error overlap, exact McNemar per seed, median/p90 interval widths and coverage** without training. It does NOT contain next-event label identities or complete lower/upper waiting-time endpoint/true waiting-time tuples. Consequently lifecycle-triviality and interval scores **cannot be reconstructed from stored scores alone**. Do not fabricate them. Source-licensed raw-data replay must be separately qualified if needed.
- The replay records show 10 original committed checkpoints and at least one successful independently recorded GitHub replay. Two other GitHub executions failed fixed waiting-score numerical tolerances. Record exact attempt counts, environment hashes and failure rates; never claim universal numeric determinism.

## CPU-fairness schema: explicit resolution of ME055 vs ME069

**Governing rule**: \`research/REAL-WORLD-EVIDENCE-RULES.md\`, prospective machine-checkable \`tac-osm.parity.v1\`, and the owner's **2026-10-22 CPU amendment**. Fair comparison matches trainable parameters (<1% difference), authentic data/splits/exposure, training and tuning COMPUTE ceilings/receipts and evaluation, NOT elapsed CPU runtime. ME055's older compulsory +/-5% elapsed-CPU limit was a **superseded administrative acceptance condition of that historical study**; preserve its original old-gate failure rather than retroactively calling it a scientific invalidation. ME069 consumes 800 paired steps and common supported-operator estimated FLOPs within ~0.284%; the **full hardware FLOP count is still uncertified**. TAC's 35.251s vs GRU's 13.777s is a meaningful ~2.56x TRAINING EFFICIENCY LOSS and must be reported. Matching CPU time instead of equal examples/steps/compute would be an invalid respecification.

## Stronger authentic long-history source: qualification BEFORE training

Primary candidate: **BPI Challenge 2018** (official original DOI https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972), authentic EU agricultural payment application events, original 4TU usage terms. Published descriptive reports state 43,809 traces, ~2,514,266 events, **maximum 2,973 events per trace** (https://www.mdpi.com/2076-3417/9/11/2368); that maximum says nothing about how many independent eligible 256-event cases survive honest temporal source splits. Source download/hash/terms are **not yet independently verified in ME070**. Other real event logs are alternatives only after measuring source-specific histories and rights (e.g. original BPI2015/2017; no synthetic stand-in).

Execute:
\`\`\`bash
python scripts/me070_qualify_long_xes.py --input /licensed/data/original-BPIC18.xes.gz \
  --output research/mathematical-engineering/me070/QUALIFIED_SOURCE.json
\`\`\`
This parses the **actual** user-obtained XES(.gz) without committing raw identifiers/records and reports original SHA256, case counts, case-length distribution, lifecycle counts, negative/nonmonotonic timestamp rejection, chronological 60/20/20 case-start splits, crossing-case exclusions, and independent long-history support at 64/128/256 events. It must yield >=50 eligible FIT and >=25 eligible TEST **distinct actual cases with a next outcome after 256 past events**. Otherwise \`STOP_NO_LONG_HISTORY_COHORT\`: do not train ME070 and choose another original real dataset. Require sufficient unrelated entities/time/site coverage; thresholds are only minimum viability, not proof of statistical power. Preserve exact original publisher file identity/license and split hash; later training must not peek at TEST labels. Reject unsupported unverified actions/costs; this is factual next-event/time forecasting ONLY.

## Diagnostic 1: exact accuracy overlap and timing interval properties — now executable

\`\`\`bash
python scripts/me070_audit_me069.py \
  --input research/replay-bundles/ME069/REAL_EVALUATION.json \
  --output research/mathematical-engineering/me070/EXPOSED_ME069_DIAGNOSTIC.json
\`\`\`

For EACH original pair of the five seeds compute native-only correct, GRU-only correct, both correct, both wrong, exact two-sided McNemar p on the **same case/prefix**, accuracy difference, mean log loss, and native/GRU width median/p90 and proportions below 24h/above 168h. Report DEV and TEST separately, conspicuously \`POST-HOC EXPOSED TEST\`; do not claim new heldout evidence or multiply 2,618 cases by five seeds as independent observations.

Only after legal original source replay, quantify next-target lifecycle START→COMPLETE and structural-closure frequencies, per-class conditional entropy, rare branch performance and a grouped event target separating activity from lifecycle. PRE-REGISTER a subsequent untouched source/test split if this motivates a new learned score; do not posthoc change ME069 scores.

## One integrated NEW fitted comparison: only after source and FAIR_COMPARISON preflight PASS

**Research hypothesis**: native TAC fast/slow/associative memory preserves factual information useful for predicting case evolution at >=64 and >=256 actual observed events better than an equally trained GRU with the **same** encoder, decoder, priors, calibration pathway and current-event features.

In a single preregistered integrated experiment, compare:
- Original native TAC, strong matched GRU, and **TAC with associative writes/reads removed** but its parameter count restored by meaningful alternative state capacity, without merely adding unused parameters. Match model FLOPs budgets and train/selection access for the causal ablation claim.
- **No-neural-correction mixture**, empirical first-order prior, FIT-only second-/third-/fifth-order smoothed n-gram process histories and simple observed-prefix duration quantile priors as *clearly marked unmatched practical references* (no padding simple baselines with meaningless learned parameters).
- **A real process-monitoring specialist** with a published source implementation, such as an LSTM-style predictive process-monitoring sequence model (Tax et al. CAiSE 2017) and/or SuTraN 2024; include a genuinely matched smaller trained implementation if eligible, otherwise mark as an informative unmatched practical competitor. Reuse prior art, do not rediscover n-gram backoff or conformal calibration.

Predict **original factual next event** and **actual observed next-event wait** from only a real observed prefix at its contemporaneous event time, never from midpoint chosen using eventual case length. Keep case-level splits and event-time/observation-time availability honest. Train on multiple history budgets 32/64/128/256 **only when source support is qualified**; compare on the **same eligible subset** when attributing a horizon gain (report coverage differences separately). Truncate >256 only for the bounded experiment, and label it bounded rather than lifetime memory. Provide rare-event, unchanged/changed transition and lifecycle-stratified scores, plus drift and heldout site/time performance.

Primary preregistered falsifier (for each true-history >=64 arm, *not* retrospectively optimized on an exposed TEST):
- **Reject 64-event TAC memory-advantage claim** if native TAC fails to beat the matched GRU by >=1% relative case-mean next-event log loss, upper endpoint of a case-clustered bootstrap 95% CI for TAC–GRU <0, and >=4/5 paired seed wins. If 64 fails, do NOT promote a general TAC memory benefit, even if an exploratory 256 result differs; 256 may still be investigated under a separately justified hypothesis.
- For **256-event superiority**, independently require the same gates on true histories with >=256 observed events, and better performance than strong n-gram/forecasting references. Memory ablation must demonstrate positive *incremental contribution* under matched capacity/compute; GRU improvement alone is not native memory evidence.
- Evaluate negative log loss / Brier / entropy per activity and lifecycle, conditioned on event difficulty. Confidence/accuracy uncertainty clustered by **case**, seeds as paired repeat fits; prevent leakage through shared original case targets.
- Timing: report **median and p90** 90%-interval width, Winkler interval score (uses actual lower, upper, target), pinball losses .05/.5/.95 and strictly post-FIT conformal calibration when feasible, compared to FIT conditional empirical-quantile baseline. Require proper calibration **and** sufficiently narrow useful intervals; no arbitrary target of "hours" without underlying irreducible process uncertainty.
- Record exact numeric replay tolerances justified from floating-point error analysis, deterministic op/thread setting, and **N=10 repeated clean-checkout replay attempts** across at least one fixed runner image and a second declared runner, with PASS count and first discrepant index/delta. Divergence under registered tolerance = exception, not rewritten success.

**No training until**: authentic source eligibility PASS, dataset SHA and licence logged, new \`research/mathematical-engineering/me070/FAIR_COMPARISON.json\` passes \`scripts/validate_research_parity.py --phase preflight\`, 5 paired seeds and original data exposure budgets are registered, real source outcomes and available feature time-cuts proven, and in-repo storage budget (Git-tracked/LFS) adequate for every model/seed. Once fitted, preserve all original checkpoints, optimizer/RNG state, source/split receipts and score arrays under \`research/replay-bundles/ME070\`; freeze originals in Git before touching final heldout scores, complete clean-checkout restore and scored replay; audit final resource receipts. CPU time is a separate efficiency outcome. Never rely only on expiring CI artifacts.

**Stop/reject, not redesign around TEST**: if GRU remains better at 64 and associative ablation is equivalent, record **no native TAC memory benefit** on this task. If 256-event eligible cases do not exist in sufficient real numbers, record \`SOURCE_UNQUALIFIED\`. If replay diverges, record \`REPLAY_BLOCKED\`. If a model achieves lower log loss yet no operator outcome is measured, record forecasting benefit only, not enterprise decision intelligence.

## Woyengi boundary

Until the independent real forecasting and calibration gates pass, all outputs are research advisory only: no customer SLA labels fabricated from time quantiles, no authorization to execute an action, no claims of counterfactual treatment effect or user-relevant savings. Woyengi authoritative state, observed-at timestamps, permissions, execution receipts and independent feedback remain external responsibilities; never train on postdecision facts that were not observed at query time.

## References

- ME069 frozen experiment: https://github.com/FPC-effortless/TAC-OSM/pull/348 (trained vs GRU failure, source-score and replays)
- 4TU original BPIC18: https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972
- Long-trace descriptive study: https://www.mdpi.com/2076-3417/9/11/2368
- Tax et al., predictive business process monitoring, CAiSE 2017: https://arxiv.org/abs/1612.02130
- SuTraN, 2024: https://doi.org/10.1109/ICPM63005.2024.10680671
- Predictive process monitoring review 2024: https://link.springer.com/article/10.1007/s44311-024-00002-4
- TAC repository science gates: research/REAL-WORLD-EVIDENCE-RULES.md and research/REPOSITORY-REPLAY-AND-CHECKPOINT-RULE.md (on master-research-plan; cherry-pick governance before ME070 fits).
