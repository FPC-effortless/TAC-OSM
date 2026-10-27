# HISTORY-QUERY016 — query-diverse historical-content supervision

Internal research, not independently qualified. This preregistration and exact
RECIPE.json precede source profiling and every new fit. Continues PR495's negative
frozen-model specificity result; preserves all previous data, models and failures.

Question: did the original single-lag objective fail to supervise timestamp-specific
memory? Reuse the unchanged source-timed original associative and fastslow models
(fd201207), 99,390 parameters each, 6,800B original state. Train both from scratch
with seeds26016/26017, identical source order, 64 AdamW updates x16 windows, FP32
single CPU thread, no tuning. The only training change is eight observed-mask
historical-content targets per window, rather than one. Forecast loss unchanged.

Original UCI501 physical observations, CC-BY-4.0, exact original archive hash in
RECIPE. Read only pre2015 source numerical rows, normalize using preMarch2014,
causal caches and original replay. No synthetic performance claims. Source target
values and observation masks enter losses only after outputs. All eight historical
clocks are legal past timestamps; no target-derived memory key or read selection.

Train March1–April18 2014, station12h grid. Primary July1–August15 exclusive 2014,
station96h grid; secondary original May1–June10. Source-only presence eligibility
requires H64 PM25 and at least one forecast target observed, as in LATFUSE018.
Freeze all identities, orders and hashes before nonzero optimizer steps. All
cohorts are exposed FIT: do not claim virgin holdout or independent generalization.

Primary gate for BOTH seeds: positive within-window centered skill; >=5% lower
all-query MAE vs target-free timestamp-average collapse, circular wrong clocks,
and matched fastslow; >=5% H64 MAE gain vs zero-matrix read. Each station-bootstrap
95% mean gain lower bound strictly positive (4000 draws,seed15015). Zero-read is
an out-of-distribution intervention; regional station dependence remains. Preserve
partial successes and failures, counts/masks, normalization units and all windows.
This gate is internal diagnostic usefulness, never architecture certification.

Compute router: GitHub CPU, <=45 minutes, zero GPU budget. Before fitting, inventory
all ATen signatures (all tensor inputs/outputs, no truncation) across actual forward,
backward, gradient clipping and AdamW at lr0. First and warm states, each arm/seed;
weights bitwise unchanged. Common math operators cancel; bound every differing
arithmetic operator against common matmul work. Unknown differing arithmetic or
>1% conservative gap BLOCKS. No claim of exact hardware FLOPs or equal CPU time.
Commit source-only frozen plan/profile/comparison files before fitting. At every
actual step require the identical first/warm signature fingerprint. Final parity
gate before assessment; original weights+optimizer+RNG+source/order/losses retained
in a new bundle; fresh-process source/checkpoint evaluation replay required.

PNDS: inference graph unchanged. Revised loss gets Lean and executable Python
formulations and reciprocal source mapping before fit. Full original graph, IEEE
and AD equivalence remain OPEN. No protected TEST, promotion, safety/reward
certification, external signatures or external qualification in this experiment.

Acceptance: preregistered budget passes, all four fits/replay retained, honest
primary/secondary results and exact blockers in Issue472 and a draft PR. If the
budget fails, retain the inventory and stop before fit rather than relax fairness.
