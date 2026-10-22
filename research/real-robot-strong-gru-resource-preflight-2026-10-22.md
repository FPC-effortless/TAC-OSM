# Integrated TAC robot world-model: established recurrent comparator and REAL prefit audit

**Registered 2026-10-22. Resource/sourcing diagnostic only; not ME-trained models.**
Prior merged PR323 physically verified 325 original FIT robot episodes and
354,027 measured frames, including 376 factual human-takeover onsets; PR324
integrated native action-conditional next-state and masked onset-feedback.
The compulsory FAIR_COMPARISON.json is NOT complete and no training is
authorized until its genuine parameter/data/training-budget parity passes.
Different CPU execution times are permissible and informative.

## What we are NOT trying to rediscover

Recurrent action-state learning, learned corrective updates and model-based
physical prediction are established. We do not interpret a successful GRUCell
or source checksum as new intelligence. We reuse R2D2 recurrent replay,
[DreamerV3](https://doi.org/10.1038/s41586-025-08744-2),
[TD-MPC2](https://arxiv.org/abs/2310.16828), and
[HG-DAgger](https://arxiv.org/abs/1710.02840).
We do not convert native TAC to a state-space model.

The **single** future falsifiable question is whether the native persistent
fast/slow/associative and corrective state leads to better held-out real
physical predictions after 256 real observations and authentic corrective
events, compared with a full recurrent world model under identical compute,
data and evaluation, *not* whether GRUs can process sequences.

## Genuine strong comparator

`GRURobotWorldModel` is a learned GRUCell observation recurrent state plus
GRUCell learned feedback transition. Like native `RealRobotPredictiveState`
it receives exactly a 14D measured current joint observation, a 14D
ACTUALLY executed action, and known *current* human/policy control
authority. Both predict the next 14D measured physical state and the
human-onset binary label. Both assimilate the actual next-state physical
error AFTER prediction and mask onset feedback when it is undefined or
unavailable. Both use the identical native factual loss constructor.
Neither has privileged true next-state or next operator-control label
in the prediction input. Neither has a reward/cost head or scripted policy.

GRU hidden dimension is a MEANINGFUL architectural capacity dial; scan
integer widths 32..64 before any model fit. Do not add unused parameters,
dummy FLOPs, artificial sleeps, or extra data to make results look equal.
Choosing a nearest parameter count still DOES NOT pass compute parity.

## Source and instrumentation before training

The resource pilot downloads only ONE original training episode chosen by the
already preregistered train-only hash/split. Source revision:
`9290f5a96dc91ea58256e81d9ce2dd9621890b51`; task
`scoop_cat_litter_into_trash_can`. Use the SAME 256 contiguous real
frames and actually executed commands, and real next-state observations
AFTER each original action to update the belief. Both models receive
the same 256-event recurrent state history. At the end of that history,
trace the last 16 state transitions with gradients through BOTH prediction
and correction heads, with no optimizer step or model weights fitted.

Important distinction: the scalar used solely to exercise backward
operator coverage is a function of the model outputs, NOT a real heldout
prediction metric; it does not count as evidence for physical intelligence.

Report actual trainable/deployed parameter counts, measured CPU time on
one pinned AVX2 core, raw operator receipts, original source parquet
SHA, and estimated matmul/conv FLOPs. The profiler contract itself is
scientifically limited: official
[PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
states that `with_flops` estimates FLOPs for specific operators
(mainly matmul and 2D convolution), not all recurrent elementwise,
fused GRU and backward operations. Thus partial FLOP counts cannot be
passed off as fully measured training FLOPs. **The preflight must remain
FALSE** unless additional validated full-operator accounting is available.

This audit is intentionally not a comparison win or model fit. The
demonstrated resource mismatch (if any) is a falsification of training
readiness; no cherry-picked reinterpretation can authorize original training.

## Next one real matched integrated experiment: BLOCKED until quantified

Prerequisite one: source-controlled manifest with a real original FIT
dataset SHA and immutable episode-ID and sample-order receipts, independent
dev/test frozen IDs and label availability, all real 256 input histories.
Prerequisite two: two fully specified scientifically strong architectures
with **active and deployed parameters within 1%**, exactly equal original
FIT examples/order/steps/tokens/tuning, and *fully validated* FLOP equality
within the declared compute budget, with CPU reported for efficiency on the same original hardware/data
shape. Use 5 matched original seeds 52001..52005.
Prerequisite three: independent original evaluator and identical physical
metrics, full recurrent belief after physical correction and human feedback.
The TEST parquet episodes remain unopened until checkpoints and fitting
protocol are frozen.

No training begins just because same declared budget ceilings are listed.
Parameter parity and an incomplete profiler are insufficient.
Postfit actual training-budget/FLOP usage receipts, diagnostic CPU time,
and exact checkpoint/source replay remain
required before any TAC versus recurrent-model comparison.
This robot dataset contains no independently verified success/cost/reward,
so even a future valid physics-prediction gain cannot be presented as
successful cost-sensitive action selection or causal robot policy benefit.


**Scientific-rule clarification (2026-10-22):** CPU wall-clock/process time
is an efficiency measurement and is never required to match between
architectures. The compulsory fair-comparison requirements are model
parameters, identical real training data/exposure, matched training/tuning
budgets and equal evaluation conditions; see the latest amendment to
`research/REAL-WORLD-EVIDENCE-RULES.md`. Historical CPU observations are
preserved but do not make a comparison scientifically invalid by themselves.
