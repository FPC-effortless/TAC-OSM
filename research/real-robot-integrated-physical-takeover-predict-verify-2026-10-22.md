# Native TAC-OSM robot world-model integration: dual factual feedback

October 22, 2026 — **source engineering, untrained**. Builds upon merged
PR320 native fast/slow associative belief and PR323 original complete training
source (325 FIT robot episodes, 354,027 physical frames). This is the
integrated physical prediction/verification interface required before the one
meaningful full world-model training study, not another 1-variable memory
ablation. No architecture-comparison or scientific performance claim.

## Hard problem already identified by real source audit

A robot next frame has TWO distinct labels:
- measured physical next state `x_(t+1) ∈ R^14`, regardless of human/policy
  control at t, and
- observed **new** human takeover at `t+1`, *defined only if policy is
  controlling at t*. When operator is already controlling, no new takeover
  event is defined, so the binary supervised target must be masked, not
  filled with "no takeover."

The existing native PredictiveStateAgent had one scalar outcome-verification
channel when instantiated for takeover, which could not also assimilate
real physical next-state residuals. Mixing these targets would discard the
14D real physics and inject false negatives during operator intervention.

## Integrated architecture (still the native recurrent TAC)

`RealRobotPredictiveState` composes the original
`PredictiveStateAgent(context_size=14, action_size=15, outcome_size=1)`.
The action input is 14 real commanded joint actions plus a single Boolean
**current** control-authority flag already known at frame t, NOT future
intervention. Its native fast32/slow32, 8×8 associative state,
relevance-conditioned reads and recurrence remain intact.

Before any feedback:
```
z_t^observe = native.observe(z_(t-1), real_x_t)
Q_t(a_t) = native.predict(z_t^observe, real_x_t,
                         [logged_a_t, current_human_authority])
        = [takeover_logit, predicted_real_x_(t+1)]
```
The candidate query NEVER changes the belief and neither later measured
state nor later human intervention appears in any prediction input. The two
separate supervised objectives will use only factual outcomes on the actual
logged action:
`L_physical = MSE(predicted_x_(t+1), observed_x_(t+1))` and
`L_takeover = masked_BCE(predicted_takeover_logit,
                         real_new_takeover_(t+1))`.
No universal loss mixture or weights have been chosen on a revealed heldout
cohort; train-only joint-state scaling and task weights must be predeclared in
a full matched-resource training protocol.

AFTER verified real feedback:
- existing native verification assimilates new-takeover residual ONLY where
  independently labelled (policy at t, next frame observed);
- a separate fully neural physical-feedback head retains the entire
  14-dimensional physical residual; it updates fast/slow and associative
  belief with per-coordinate bounded corrections, and never applies an
  update if the physical result is missing; NaN for genuinely missing source
  labels is safely masked BEFORE subtraction;
- both feedback channels use the original executed action, not a hypothetical
  candidate; a subsequent observation/prediction can train true
  consequence-of-correction skill through a real later state loss.

The native takeover update is coordinate bounded by 0.25 per event, and the
physical residual update by 0.25, giving an upper bound of 0.5 of total
coordinate correction per event. These facts are software properties, not
a proof of stable long-horizon intelligence.

## Capabilities NOT established

No human rollout or action-reward optimization is implemented. The source
has no independently verified scalar task reward, success or policy
propensity. Human intervention is human judgment, not proof of physical
failure. The model has NO matched GRU world-model comparison, NO trained
checkpoint, no real state-forecast metrics, no autonomous action policy,
no language or general System-2 reasoning. The masks, bound, predict-before-
feedback and numerical differentiability checks are **software-only**;
they do not count as evidence the network learned the world.

## Next single major scientific training experiment (still BLOCKED)

Run one integrated full-data experiment only after a separate registered
`FAIR_COMPARISON.json` verifies equal active/deployed parameter count (≤1%
difference), equal original real FIT identities/chronology/sampling, exact
feature and outcome timing, equal optimizer steps/token exposures, five seeds,
validation-only selection, prefit measured FLOPs (≤1% difference) and CPU
(≤5% difference) using exact real FIT example shape, original matched
hardware, sealed TEST, and postfit measured resource/parity replay.

The competing established recurrent world model must have exactly the same
two real heads and postfeedback availability, not a weaker baseline
deliberately missing state or takeover data. Compare on truly held-out
physical next-state accuracy and after-real-feedback prediction quality.
Evidence for practical causal decision intelligence still requires a
SEPARATE genuinely measured reward/verified success source. If costs
cannot be matched prefit, STOP without fitting rather than explaining
unfairness after results.

Existing prior-art foundations (not claims of novelty):
[R2D2](https://willdabney.com/publication/r2d2/),
[DreamerV3](https://doi.org/10.1038/s41586-025-08744-2),
[TD-MPC2](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html),
[HG-DAgger](https://arxiv.org/abs/1710.02840).
