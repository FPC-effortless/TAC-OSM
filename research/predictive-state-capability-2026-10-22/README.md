# Integrated predictive-state candidate — 2026-10-22

**Implemented, untrained; no capability improvement or architecture superiority
established. Not an executed ME-052 experiment.** Governance PR #318 is included
in this draft branch, without merging it into master. Earlier results and
weights are unchanged.

The engineering objective is one native predictive-state model that observes,
queries real actions, predicts observed consequences, and incorporates later
verification. This implements a prospective interface for that objective;
model quality and the value of its state remain unmeasured. It does not implement
an actor/critic, counterfactual planner or qualified System 2.

## Executable architecture

[`PredictiveStateAgent`](../../src/tac_osm/predictive_state_agent.py) retains the
native nonlinear fast32/slow32 plus 8x8 associative error-corrected state from
ME-038/ME-051, without importing the old forecasting experiment stack. The
original source is pinned to commit
`65a2d549b9297ca923db121b9acb9725090fb3e2`,
`src/tac_osm/me038_innovation_forecast.py:InnovationMemory`.
The new API has distinct operations:

1. `observe(state, context)` updates belief from available observations.
2. `predict(state, context, candidates)` scores all action feature vectors and
   predicts their next observations. Candidate queries do not mutate belief.
3. `verify(state, executed_action, outcome, preceding_prediction, available)`
   learns a bounded correction when an outcome actually arrives. Missing
   outcomes, including NaNs behind a false mask, leave that entity's state
   bit-identical. Predicted outcome must be in the measured outcome units;
   binary logits must be converted to probabilities by the caller.

The current outcome head returns factual outcome logits. A next-observation
head is present but cannot be trained on unlinked impressions. No policy reward
or causal effect is implied by having an action argument. Context/action feature
construction, preprocessing and loss contracts must be fixed by the authentic
source before training. The caller is responsible for correctly routing entity
IDs, chronological events and delayed outcomes. Routing is not implemented or
claimed to have been validated on longitudinal real data in this change.

Example geometry (unit tests only): 5 context features, 3 action features, one
outcome: **25,184 trainable/deployed parameters**, **512 float32 state bytes per
entity**, excluding model weights, tensors, allocator/runtime and routing
metadata. This is not a matched comparator count or measured latency result.

## Mathematical specification

For the inherited cell, normalized addresses obey `||k||_2 < 1` and:

```
M_ret = rho M,                       .95 < rho < .9999
M' = M_ret + beta (v - M_ret k) k^T,  0 < beta < 1
f' = f + g (tanh(W_u u + W_f f + W_m M q) - f)
s' = s + g_s (tanh(s + innovation(f', s, M' q)) - s)
```

Observation-only fast/slow updates are convex combinations of their old state
and bounded proposals. The verification extension is:

```
e = tanh(y_observed - stop_gradient(y_predicted))
[delta_f, delta_s] = .25 sigmoid(W_g[z,a,e]) * tanh(W_c[z,a,e])
[z_f', z_s'] = [z_f, z_s] + available * [delta_f, delta_s]
M' = M + available * .25 (tanh(V e) - M k_a) k_a^T
```

Each fast/slow verification coordinate changes by at most .25 per call.
This is a **local correction bound**, not a contraction proof for the coupled
network or a guarantee against destructive forgetting. Correction usefulness
requires future real targets; it is not established by finite gradients.

Future registered training must use measured targets only: observed outcome
likelihood and next-observation error on genuinely linked transitions. There is
no old-reward reconstruction objective masquerading as mutual information,
no unobserved reward target and no propensity-difference 'counterfactual'.

## Source qualification: OBD rejects entity-memory use

The [publisher's schema](https://huggingface.co/datasets/zozonext/open-bandit/blob/main/README.md)
and [source implementation](https://github.com/st-tech/zr-obp/blob/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obp/dataset/real.py)
provide impressions, contexts, item choices, positions, clicks and propensities.
They do not document a stable user/session ID. Timestamps and anonymized feature
collisions do not establish entity continuity. Neither a closed feedback loop
nor causal changes in future world state can be inferred from these rows.

The actual publisher example bytes were downloaded and verified:
[OBD_SOURCE_AUDIT.json](OBD_SOURCE_AUDIT.json). SHA-256:
`7168295b6e0a9eabcf3392320a5dd434e542b68e705d5cd9491499af589812f1`.
Only the header was parsed; reward values were not scored or used for fitting.
These public example rows are engineering-source material, **not a sealed test**.
The raw human records are not committed here.

Reproduce (curl URL is in the receipt):

```sh
python scripts/audit_bandit_source.py /path/to/publisher-all.csv \
  --output /tmp/obd-audit.json --purpose entity_memory
# expected exit 2: BLOCK_ENTITY_MEMORY_STUDY
```

OBD remains a separate candidate for one-step OPE. It would require a registered
position-aware policy contract, supported actions, cross-fitted nuisance models,
SNIPS and doubly robust estimates, effective sample size/weight diagnostics and
uncertainty accounting for training and evaluation splits. Those estimators and
policy training are **not implemented by this change**. The proposed absolute
+.02 policy-value gate has not been adopted without an outcome-scale/power
justification; the proposed .15-bit MI gate has no implemented valid estimator.

[SCANIA v3](https://researchdata.se/en/catalogue/dataset/2024-34/3) is independently
confirmed as authentic longitudinal operational data. Its official source
supports observed repair-risk prediction, not identifiable repair efficacy,
logging-policy propensities or actual measured maintenance costs. No SCANIA raw
files were acquired by this change. Its released source must be acquired,
checksum-verified and qualified before an entity-state fit; absent targets will
not be invented or borrowed from OBD.

## Prior art and the unresolved capability question

- [R2D2, ICLR 2019](https://willdabney.com/publication/r2d2/) already studies
  recurrent replay, representational drift and state staleness. A future
  training implementation must handle burn-in/replay; this candidate does not
  supply that trainer and makes no discovery claim about recurrent replay.
- [DreamerV3](https://arxiv.org/abs/2301.04104) already jointly learns world
  models and behavior from imagined outcomes. This candidate is not Dreamer,
  does not run imagined policies, and has not demonstrated comparable control.
- [Open Bandit Dataset and Pipeline](https://arxiv.org/abs/2008.07146) establishes
  a real logged-bandit/OPE benchmark. It does not turn anonymous impressions
  into individual state trajectories.

The unresolved TAC question remains whether its native relevance-conditioned
state and verification updates improve real future prediction **and supported
real decision utility** over an established equally resourced model. This
implementation enables that question; it does not answer it. The combination
of existing ideas is not evidence of a novel algorithmic contribution.

## Execution status and next gate

- No optimizer steps, trained checkpoints, architecture comparisons or new
  validation/test scores. Generated test tensors are numerical fixtures only.
- [PARITY_DRAFT.json](PARITY_DRAFT.json) is deliberately incomplete and fails
  the compulsory validator. It contains no fabricated parameters or profiler
  measurements for a rival. It is not a preregistered fit permission.
- Do not begin ME-052 training until an authentic linked dataset supports the
  chosen targets and a complete `FAIR_COMPARISON.json` passes preflight. Use at
  least five matched seeds. Parameter/data/exposure/FLOP/CPU/tuning parity and
  final measured receipts remain compulsory. Padding, sleeps and unused weights
  to manufacture parity are forbidden.
- New data must have a reserved cohort registered before label inspection.
  Previously exposed MovieLens, KuaiRec and forecasting tests are not fresh.
- This source-only candidate is available for further engineering. No merge,
  promotion, deployment or intelligence/latency claim is requested.
