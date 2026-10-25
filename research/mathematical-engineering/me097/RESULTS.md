# ME097 original real-data development result

All five seeds passed the frozen numerical criteria and independent original replay. **Exposed-DEV capability gate PASS; independent-site, architecture-superiority and control qualification OPEN.**

| Metric | Persistence | Five-seed mean | Error reduction |
| --- | ---: | ---: | ---: |
| mse | 0.000467679 | 0.000301391 | 35.56% |
| late_mse | 0.000438049 | 0.000261363 | 40.33% |
| moving_mse | 0.002507034 | 0.001496158 | 40.32% |

Full original per-seed scores are in `research/replay-bundles/ME097/RESULTS.json`.

59 original FIT episodes,472 64-frame windows;13 previously exposed DEV episodes,104 windows. Scoring includes6,656 next-state window positions per seed, potentially overlapping source records. Five seeds do not multiply the number of independent episodes. Only a single requested dataset/task is qualified at development level. No reserved test evaluation occurred.

Each 95,874-parameter model received256 updates ×8windows ×64 positions =131,072 supervised exposures, with no tuning sweep. Repeated training windows are not new independent observations. Inputs use x[t] and recorded action[t-1], FIT-only normalization/delta scale. Source CC and exact publisher Apache-2.0 bytes are in ../me096/SOURCE_README.md. Mean uses standardized MSE, not certified physical-unit risk or safety.

Architecture: shared six-family core with a separately learned contextual query, read-before-write associative retrieval and direct retrieved-delta head. It retains the original input-only controlled affine schedule. Balanced batch scan and carried streaming state match; no seventh model or six independent trained heads. Normalization does not force useful alignment; this experiment does not attribute gains to memory.

No controlled comparison with the old 93,324-parameter recipes: different parameter counts, lagged-action alignment, windows and4×updates. Improvements against a persistence forecast demonstrate learned development prediction, not superiority of the new architecture over the previous TAC or another trained architecture.

Provenance: prereg b6510508; prefit loader path correction78c8a0ba occurred before ANY optimizer update. Original five-model freeze1901d395 preceded scoringaaa58ad5; scoring now compares every original byte with committed Git before running. Independent clean-checkout replay locally and GitHub run40163334508 PASS; original forecasts reproduce within1e-5/1e-6 and all score JSON values reproduce exactly.

All10 ME096/097 original fits remain preserved. ME096 numeric thresholds passed but its attempted Git add failed under sparse checkout and scoring continued, violating archive order: **ME096 full gate BLOCKED**, no retroactive repair. ME097 was separately preregistered with NEW seeds and unchanged architecture/data/objective/budget/criteria, not a hidden tuning retry. ME095 remains4originals+1interrupted/noREPORT; old failures remain unchanged.

Verification:947 local tests PASS; all5 fitted models match independent PNDS serial full neural equations on first2 real FIT windows (FP32 max prediction discrepancy1.1921e-7; max gradient discrepancy4.6567e-9). Lean graph padding, pre-write retrieval and query-nonmutation statements compile with no axioms. Full concrete neural/finite-precision Lean refinement remains OPEN; no whole-network proof, benchmark-wide intelligence or causality claim.

Operational relevance: a read-only predictive advisory candidate. Customer/pilot/decision benefit, autonomous policy, Woyengi deployment and ROI NOT_MEASURED. All PRs remain draft and unmerged.

Dedicated study-named budget-audit workflows were added after fitting when the
repository integration check found their naming/hooks missing. They restore
original pre-update runtime guards and saved tensors; they do not train or
retroactively constitute prefit CI. The launcher manifest preflight and actual
live guards did run before updates. This integration exception remains recorded.
