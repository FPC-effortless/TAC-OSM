# Integrated TAC world-state research: authentic physical episodes and sealed cohorts
Registered 2026-10-22 BEFORE the first cohort acquisition under this protocol.
SOURCE-ONLY study: not an architecture fit, optimization pass, or ME-numbered comparison.

## Eliminate already-solved research questions
Do NOT reinvent recurrent replay, offline autoregressive sensor forecasting,
intervention imitation, state-space recurrence, time-series masks, or world-model
pretraining. Use established R2D2 burn-in/replay, DreamerV3 and TD-MPC2
action-conditioned transition learning, and HG-DAgger correction boundaries as
KNOWN engineering foundations. Their reproduction is not TAC novelty.
The single remaining original question is whether the native TAC relevance-
conditioned and correction-aware persistent state improves genuine physical
prediction *and downstream verified decisions* over an equally resourced
established recurrent predictive-state model.

Relevant prior work:
- R2D2 recurrent replay and stale-state correction:
  https://willdabney.com/publication/r2d2/
- DreamerV3 world-model learning:
  https://arxiv.org/abs/2301.04104
- TD-MPC2 action-conditioned latent representation:
  https://arxiv.org/abs/2310.16828
- HG-DAgger human takeover learning:
  https://arxiv.org/abs/1710.02840

## Real data and fixed independent source cohorts

- Original real-world dataset: BodenAI RW-RL-HIL:
  https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset
- Immutable publisher data revision:
  9290f5a96dc91ea58256e81d9ce2dd9621890b51
- Task directory actually verified by CI:
  scoop_cat_litter_into_trash_can.
- Original metadata: **505 physically logged episodes**, original episode IDs
  0..504; no inferred session/person identifiers.
- Selection algorithm depends ONLY on episode ID, immutable source revision,
  task and a block size of 5 adjacent episode IDs. Sort 101 blocks by SHA256 of
  (revision,task,protocol version,block ID), reserve first 65 blocks for FIT,
  next 16 for DEV and final 20 for sealed TEST.
- Fixed numbers: FIT=325, DEV=80, TEST=100 ORIGINAL episodes.
  No episode or adjacent 5-ID block crosses cohort. No stratification on
  intervention labels, rewards, future state, duration of a takeover, or test
  quality. Episode lengths are used solely for deterministic eligibility and
  counts. There is no evidence episode IDs are actual chronological capture
  order or unique sessions; this is **episode-block-disjoint**, not
  cross-operator/cross-session/calendar-future validation.
- Source cohort selection does NOT read DEV/TEST parquet. Published
  episodes.jsonl happens to CONTAIN intervention summaries, but those fields
  are deliberately ignored when determining the splits. This metadata was
  already publicly released, so claiming that no TEST labels exist outside
  the experiment would be inaccurate. The original DEV/TEST *frame level*
  state, executed actions and authority masks remain unopened until the
  authorized final evaluation. Original source hashes, all membership IDs,
  and per-history opportunity counts are archived.
- This source qualification checks the entire 505-episode publisher metadata
  but fetches **20 FIT episode parquet files only**, selected from FIT solely
  by preregistered SHA256 hash and >=257 real frame eligibility. It verifies
  all frames within those 20 physical episodes: real 14D state/action,
  15fps monotonic timestamps, episode-local frame index, human-policy
  authority, and exact original metadata intervention ranges. No source
  interpolation or generated events are accepted.

### Engineered real-time prediction and verification interface

For each observed episode and real frame t with sufficient preceding data,
the model may consume ONLY measured observations up to and including t,
actually executed actions up to t-1, the current logged action a_t and known
control authority at t. Then it must predict measured next physical state
x_(t+1) and whether the operator takes over in the subsequent frame. The
next-state target is physically observed, not hypothetical. Intervention
onset is defined only while policy was in control at t; when a human is
already controlling, takeover-onset supervision is missing/masked rather
than mislabeled zero. The model must predict BEFORE arrival of x_(t+1) or
the next operator flag. Verification can consume those genuine outcomes only
AFTER their arrival. Same-episode real recurrent history of 16,64,256 frames
is supported; H means **genuine sequential inputs**, not same-user false
coordinate tokens. All model arms MUST receive exactly the same frames and
action availability at the same timestamps.

One-step logged-action next physical-state prediction is supervised.
Multi-step physical predictions at h>1 have unobserved future actions when
queried at t; any such outcome is behavior-policy continuation forecasting,
NOT a uniquely identified causal effect of a_t. This corpus has NO verified
task success/return, operator motive or logged intervention propensity; it
cannot establish action-policy intelligence or commercial payoff alone.
No inferred humanoid skill, independent physics efficacy or causal control
can be claimed from a learned takeover signal.

### Full-capability future training preflight — not launched by this PR

Do not train until (a) fresh original cohort and raw hashes are complete,
(b) test episodes and exact targets are sealed, (c) TAC versus a documented
recurrent predictive-state baseline have active and deployed parameter parity
<=1%, (d) same FIT examples in same order, 256 real event exposure,
identical optimizer steps, learning/tuning privileges and test IDs, (e)
measured prefit profiler FLOP parity <=1% AND CPU time <=5%, (f) both actual
postfit budgets, checkpoints, samples and test receipts pass the compulsory
`scripts/validate_research_parity.py` validator, with five preregistered seeds.
NO padding with useless weights, dummy FLOPs, sleep time or unconsumed budgets.
No ad-hoc micro-study is an acceptable substitute.

External practical persistence/constant-motion and last-state predictors
should still be scored as explicitly UNMATCHED reference models. Primary
scientific metrics once training qualifies: next physical state 14D error,
calibration by action authority and independent episode, human takeover
prediction PR-AUC/calibration and latency-to-takeover, belief-prediction
damage over 16/64/256 real event histories, and effect of later verified
feedback on independent next predictions. Original real physical intervention
labels are operator judgment, not verified failures or real economic costs.
No heldout metrics or trained agent behavior have yet been produced.

## Source-only qualification outcomes gate

Pass only if metadata rows=505, splits exactly 325/80/100, source SHA and
membership manifests retained, all future TEST and DEV parquet unaccessed,
20 authenticated original FIT parquets audited, real 256-prior-frame windows
possible across each split, and test suite catches label leakage, source
identity corruption, action ownership and true frame ordering. If any
condition fails, preserve evidence and STOP. No model experiments and no
success claims until the actual parity gate passes.

Reproducible publisher-file command:

```bash
python -m scripts.qualify_real_robot_cohort --private-cache /tmp/tac-robot-private \
  --receipt /tmp/tac-cohort-receipt.json
```
Original robot frames are never committed or exported by the CI workflow;
only anonymized episode IDs, aggregate counts, source hashes and no-model-fit
receipts are archived.
