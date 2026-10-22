# Genuine robot-feedback source qualification — October 22, 2026

Status: source-only audit. The integrated native TAC predictive-state model has NOT
been trained, evaluated on this source or demonstrated intelligent behavior.

## The actual scientific question
Can a native TAC agent retain useful causal-order information over real
environmental observations, predict the physical result of logged executed
actions, update its belief on genuine feedback and improve future actions
against a fully matched established recurrent model? This is not another
retrieval/head micro-variation and uses known recurrent replay and
action-conditioned dynamics research instead of rediscovering them.

## Authentic dataset eligibility

Candidate: BodenAI RW-RL Human Intervention Dataset:
https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset

The publisher documents 82.23 hours of real robot interactions: 3,347
distinct episode trajectories, 4,440,155 original state/action frames,
3,068 episodes with recorded human intervention and 279 with none,
across nine actual tasks. LeRobot v2.1 files have episode-local timestamps,
14-dimensional measured robot joint state, 14-dimensional executed actions,
a Boolean intervention flag and the human-versus-policy control authority.
CC-BY-4.0, subject to attribution and ordinary downstream data diligence.

CRITICAL documented absence: the published release DOES NOT provide
scalar rewards, episode-level verified success labels, done flags or
individual causal counterfactual action outcomes. Human takeovers are
operator judgments and not proof of physical failure. Intervention absence
is NOT success. Data collection is selection biased.

## Publisher-file audit protocol: zero model training

The new audit script fetches a SHA-pinned publisher repository revision at
runtime. It downloads ONLY source metadata plus one genuine episode parquet
from the real scoop_cat_litter_into_trash_can task. It checks the actual original
episode counts, IDs and split boundaries, original source SHA256,
strictly increasing real times, state/action 14-channel dimensions,
policy/human authority and intervention metadata agreement.

Only aggregate counts, hashes and source commit ID are archived. Raw original
episode trajectory frames/videos are NEVER uploaded. This one-episode audit
is a minimal first source eligibility test, not a full scientific cohort
audit, and it does not expose or score any independent heldout labels.
A future study needs a FULL pinned-source archive and independent real splits.

## Three capability lanes (never combine unrelated scores)

1. Real longitudinal predictive-state lane: train on actual executed
   observation/action/next-observation triples at 1, 16, 64 and 256 *real*
   frame horizons, and predict time-to-human-intervention BEFORE its onset.
   Each model receives the same physical state/action/feedback stream.
2. Real verification lane: consume observed next-state evidence and
   genuine takeover events to correct the persistent belief, and evaluate
   subsequent prediction changes and damage to already verified facts.
   Distinguish a next-state residual from a binary takeover outcome and
   do not use future feedback at prediction time.
3. Real decision-value lane: BLOCKED on this dataset because actual
   policy returns, verified success, costs and action assignment probabilities
   are unavailable. OBD can be explored separately for *one-step* OPE, but
   does NOT supply a persistent entity identity.

## Compulsory comparison before fitting any model

No model architecture-training or superiority comparison may begin until
the existing strict parity validator preflight passes: identical genuine
FIT/DEV/TEST observations, source SHA, training row/sequence order, all
inputs/labels, optimizer steps, training exposure, active and deployed
parameters (within 1%), actual prefit profiler FLOPs (within 1%),
CPU-seconds (within 5%), tuning privileges and source-verified evaluation
conditions. Minimum five matched seeds for this prospective capability
program. Postfit measured resources and genuine-model replay must pass
the separate mandatory final gate. No new reward labels, extra synthetic
world tasks, padding of unused model parameters, or sleeps to fake compute.
No training is requested or approved by this source-audit-only branch.

## Prior established research, not TAC novelty

- R2D2: https://willdabney.com/publication/r2d2/
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828
- HG-DAgger: https://arxiv.org/abs/1710.02840
- Real Robot HIL card: https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset
- Open Bandit Dataset: https://arxiv.org/abs/2008.07146

This source is promising for learning sequential physical-state prediction,
but still lacks the measured action utility needed to qualify autonomous
decision intelligence or real-world causal control.
