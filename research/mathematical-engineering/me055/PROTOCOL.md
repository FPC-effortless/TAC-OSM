# ME-055 protocol

## Capability question

Can a bounded, action-conditioned TAC predictive state produce accurate
16-step physical trajectories and calibrated intervention risk on an
independently selected real robot task, under the same data and resources as a
strong GRU world-model control?

The work builds on established recurrent burn-in (R2D2), jointly trained world
models (DreamerV3), and multi-step action-conditioned latent dynamics
(TD-MPC2). Servo residual modeling and bounded state updates are engineering
foundations. The unresolved TAC-specific question is whether the fused
fast/slow relevance trace improves authentic multi-step prediction and later
action selection.

## Frozen source contract

- Publisher: BodenAI RW-RL HIL Dataset.
- Immutable revision: `9290f5a96dc91ea58256e81d9ce2dd9621890b51`.
- Task: `clean_stains_off_tabletop`, selected without outcome inspection.
- Split: deterministic contiguous blocks; 228 FIT / 76 DEV / 80 TEST episodes.
- TEST remains sealed until every registered checkpoint is frozen.
- Prediction origins begin under robot control. Physical targets and logged
  human actions after takeover are censored.

## Resource gate

Before training, both arms must differ by at most 1% in trainable and deployed
parameters, 1% in profiled update FLOPs, and 5% in measured optimizer CPU.
They must also share training observations, ordering, update count, examples,
history, rollout horizon, optimizer schedule, tuning privileges and evaluation.

The first frozen candidate failed CPU parity at 13.175%. Consequently the
training protocol is not activated and no accuracy thresholds are evaluated.

## Prior work

- Kapturowski et al., *Recurrent Experience Replay in Distributed
  Reinforcement Learning (R2D2)*: recurrent burn-in and replay-state handling.
  https://openreview.net/forum?id=r1lyTjAqYX
- Hafner et al., *Mastering Diverse Domains through World Models (DreamerV3)*:
  joint learned dynamics for control. https://arxiv.org/abs/2301.04104
- Hansen et al., *TD-MPC2*: scalable multi-task latent action-conditioned
  dynamics and planning. https://arxiv.org/abs/2310.16828

