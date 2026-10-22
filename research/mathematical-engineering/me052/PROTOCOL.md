# ME052 ORIGINAL real-robot joint world-model training: PREREGISTERED

Freeze this document on GitHub **before first model fit**. Source-only previous studies
verify the full original training corpus. This is the actual experiment,
not a resource survey or architecture proposal.

Scientific question: Can TAC native predictive state improve actual measured
robot physical-state transition forecasting and next operator-takeover prediction,
relative to an established trained dual-feedback GRU world model?

## Real source, no simulated performance
Publisher: BodenAI/RW-RL-HIL-Dataset, exact original source Git revision
9290f5a96dc91ea58256e81d9ce2dd9621890b51.
Task `scoop_cat_litter_into_trash_can`. Published LeRobot original Parquet
episodes carry 14-dimensional real observed robot joint states, 14-D real
executed commands, timestamps and real operator-control flags.
Immutable original 5-consecutive-ID block partitions registered previously:
FIT 325 episodes / DEV 80 / TEST 100. Publisher episodes.jsonl metadata does
contain intervention summaries; those are forbidden for stratification,
threshold setting and model selection.

## Matched budget (user's requirements, not invented CPU equality)
Architectures: existing integrated RealRobotPredictiveState TAC, 37,985
trainable parameters and fully learned GRURobotWorldModel(hidden_size=the
original nearest-parameter count found by PR325), 38,295 parameters.
Difference 0.81%, below 1%. No unused capacity padding.

Same actual source FIT episode IDs, same in-episode 256 real-frame history,
same 14 observed and 14 action channels and known current authority,
same next physical-state & new human-takeover targets, same policy-control
mask, exact same normalized genuine source inputs (FIT-only mean/std),
same AdamW optimizer, lr 0.001, wd0.0001, grad clip norm 1,
exact same per-seed real window sample schedule and stochastic random
sampling indices, batch size 2, 80 updates per model, truncated backprop
through last 8 actual recurrent frames after 256 real-frame burn-in.
Five predeclared paired seeds: 52001, 52002, 52003, 52004, 52005.
Same maximum *allocated* training compute envelope, matched updates/data
presentations/tuning (zero tuning trials). Record actual wall/CPU time and
available profiler FLOP counts as efficiency/compute diagnostics, not a
requirement for identical CPU runtime. Neither architecture may receive
more optimizer steps, episodes, sequence tokens, feature channels or DEV
model-selection access to rescue unfavorable results.

For each update select 2 actual FIT windows: one uniformly sampled eligible
FIT window, one actual FIT window whose last 8 outcome targets include a
REAL recorded operator onset (if authentic source supplies at least 10 such
FIT windows; else abort, not fabricate). This is balanced FIT-only
oversampling to train rare genuine onset supervision, never used for
heldout scoring. Both models see precisely those same original examples,
same order, and same random-number sampling state.

The only supervised training targets are real x_(t+1) observed AFTER a logged
action and real human takeover at t+1, defined only if robot was policy-
controlled at t. Per-step loss `MSE(normalized predicted physical state,
measured normalized next state) + 0.2 * masked_BCE(real new takeover)`.
Missing human-control labels do not get assigned false negatives. Real
physical feedback arrives only after the original prediction; 14D physical
correction and masked onset correction are both learned and backpropagated
through final 8 true sequential steps. No synthetic rewards or success.
This experiment predicts observational policy trajectories, not causal
effects of interventions or verified robotic task completion.

## Frozen actual scoring and outcome access
All 10 authentic FIT trained weights and full optimizer/RNG/checkpoint hashes
frozen BEFORE first frame-level DEV/TEST parquet is opened. No hyperparameter
changes after DEV/TEST revelation. Heldout normalized physical MAE/MSE and
real-unit physical MAE, baseline persistence absolute next-state MAE, and
masked operator onset logloss/AP where enough positives; same episode-
level prediction positions and same actual labels for all models.
Score original future positions after 256 authentic history frames,
every 16th target frame per heldout episode. Do not use feedback from future
frames before their actual arrival. Provide performance by seed and paired
episode-bootstrap intervals if sample support allows. No macro
task-success/reward, causal policy win or System2 claim.

For reliability, compare across 5 matched seeds. Primary capability claim
requires TAC to improve physical MAE relative to GRU in >=4/5 paired seeds,
and aggregate episode-cluster 95% bootstrap CI for GRU minus TAC MAE > 0.
All other claims are descriptive. If TAC loses or matches, publish unchanged.
Train-checkpoint/source/sample-order and exact replay checks are compulsory.
Original user data never uploaded.

## Why this is not rediscovery
Recurrent replay is established (R2D2); action-conditioned world-model learning
(DreamerV3, TD-MPC2) and human corrective learning (HG-DAgger) are established.
This experiment tests *the integrated native TAC-specific recurrent/
associative relevance geometry* against a capable similarly sized recurrent
physical world model. The novelty claim is intentionally withheld pending
real external benefit.

## Fairness documentation
CPU execution time need not match. Both receive SAME fixed training updates,
data, learning objective and compute *budget allocation*. Actual consumption
may differ and is disclosed; do not claim exact same FLOPs if unavailable.
The legacy validator demands exactly matched *measured FLOPs*, which is
additional to the user-requested fixed budget. This trial MUST identify it as
a separate nonqualified criterion if exact FLOPs are not independently
established: do not fake FLOPs or certify a stronger head-to-head claim
than supported. Code and results will explain each concrete match/mismatch.

This protocol may not be changed after first fit. Corrections allowed only
for evident runner bugs BEFORE any DEV/TEST data revelation and must be
fully audited. No promotion or merge of architecture based only on numerical
test success.
