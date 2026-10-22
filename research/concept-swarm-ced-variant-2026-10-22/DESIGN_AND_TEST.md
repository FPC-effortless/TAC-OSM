# Persistent Concept-Swarm CED — TAC-OSM variant (2026-10-22)

**Status: SOURCE-ONLY MECHANISM CHECK, NO TRAINED CAPABILITY RESULT.**
This is an integrated exploratory variant of the native model, not an extension
to the synthetic-development scorecard, not a new ME fit, and not a replacement
for historical checkpoints. No change to scientific governance or parity thresholds.

## Hypothesis

A learned persistent concept population, with relevance-weighted information
movement and shared-weight recursive causal decoding, will predict authentic
future world states better than equally-resourced established recurrent
predictors *when the dataset really contains entity-continuous observations*.
The hypothesis is falsified if it fails real heldout predictive/calibration
gates under the registered fair-comparison protocol. A decoder that changes
its output between iterations is only an engineering property.

## Implemented end-to-end model boundary

The implementation is at
[src/tac_osm/concept_swarm_ced.py](../../src/tac_osm/concept_swarm_ced.py).
It composes the native PredictiveStateAgent (fast32/slow32/8x8 innovation
memory) and adds a learned persistent concept tensor of shape B x N x D.

- observe(belief, **past/present available** context): native recurrence
  updates once; relevance-weighted concept messages and a bounded/gated
  concept update persist across real events. The model itself does **not**
  resolve real entity identities: the caller must provide valid entity routing.
- predict(belief, context, real candidate actions): native factual outcome
  and next-context heads run once; temporary concepts are seeded from stored
  concepts, action and global state; a shared, repeated decoder exchanges
  concepts, produces multilevel outcome refinements and predicts possible
  future concept embeddings. Candidate query and decoder iterations NEVER
  write the persistent belief.
- The decoder exposes an untrained decision-score head and an untrained
  information-gain-score head. These are not calibrated value/epistemic
  estimates and do **not** implement a qualified policy or full variational
  active inference. Decision use requires real, attributable logged
  actions/rewards and an independently validated OPE or online protocol.
- verify(belief, **actual executed action**, observed outcome, **prior
  prediction**, available mask): native bounded feedback correction plus
  locally bounded concept correction. Missing labels leave every entity's
  state exactly unchanged, even with NaNs. No imagined rollout can commit.
- Future concept predictions have **no validated semantic labels**. An
  eventual future-state loss must be linked to real later observations using
  chronology-safe latent alignment, and evaluated on external outcomes,
  not merely by agreement with the model's own representation.

Native recurrence is retained; no architecture conversion to a Transformer
or state-space model. The relevance-weighted concept interaction is an
attention-like neural message-passing operator, not novel attention, and is
not an imitation of collective-intelligence evidence.

## Architecture in equations

Concept communication, using shared weights and N concept slots:

    A = softmax(Q(C) K(C)^T / sqrt(d) + B_relation)
    M = A V(C)
    C_obs = C + .25 sigmoid(G_obs(C,g)) *
                        (tanh(F_obs(C,M,g)) - C)

Temporary action-conditioned decoder, with shared weights at all K levels:

    q_0 = tanh(C_obs + anchors + action_embedding + global_embedding)
    q_(k+1) = q_k + (.75 / K) sigmoid(G_dec(q_k,M_k,g)) *
                        (tanh(F_dec(q_k,M_k,a,g)) - q_k)
    predicted_y_k = predicted_y_native + R_y(pool(q_k))
    predicted_next_obs = native_next_obs + R_o(pool(q_K))

Verification concept correction with real measured feedback only:

    e = tanh(y_observed - stopgrad(y_predicted_pre_feedback))
    C_verified = C + available * .125 *
                      sigmoid(G_verify(C,e)) * tanh(F_verify(C,e))

These are **bounded one-step updates**, not proofs of global convergence,
multi-step stability, information preservation, causality or intelligence.
The refinement trace is observability instrumentation, not a training result.

## Mechanism tests actually executed locally

Command on a local overlay with an exact transcription of the current
native predictive_state_agent.py:

    PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -m pytest -q tests/test_concept_swarm_ced.py

Result: **10 passed** on CPU / PyTorch 2.10.0+cpu. Tests are shape,
read-only query, action order, persistent observation, no-feedback identity,
bounded correction, gradient connectivity, batch entity separation, finite
130-event stream, and untrained save/load replay. Randomized numerical
fixtures are not world simulation and provide **zero evidence of predictive
capability**. Full repository CI has to run independently on the PR.

Representative shape example (5 context values, 3 action values, 1 outcome,
6 concept slots x 16 coordinates, four shared decoder iterations):

- Original native model: **25,184** active parameters (reference only).
- New variant: **32,701** active parameters.
- New extra persistent concept coordinates: 96 per externally routed entity.
- This **does not** satisfy the mandatory 1%-parameter parity threshold:
  a native/GRU comparator must be properly matched and independently profiled
  before any trained head-to-head comparison.

No training, real-world fit, final-test access, out-of-policy value estimate,
registered parity manifest, or heldout improvement occurred in this variant.

## Research qualification: do not bypass

The repository's existing
[real-data and fair-comparison rules](../REAL-WORLD-EVIDENCE-RULES.md)
and [capability-first reset](../CAPABILITY-FIRST-RESEARCH-RESET-2026-10-22.md)
remain mandatory. This file is **not** FAIR_COMPARISON.json and does not
satisfy preflight. A formal first fit must:

1. Independently acquire and checksum an authentic, entity-continuous source,
   review license, chronology, censoring, actions and outcomes. SCANIA v3 is
   preferred for longitudinal observational forecasting; it **does not**
   establish causal intervention effects or measured decision costs.
2. Pre-register source hashes, split/entity IDs, target contracts, horizons,
   leakage guard, source-verified calibration and baseline acceptance metrics.
   Reserve heldout entities/outcomes before first fit.
3. Preflight at least five matched seeds comparing this full variant with a
   GRU/recurrent control, current native model and practical persistence/last
   observation references (latter unmatched). Match **active trainable
   parameters to <=1%**, identical real training sample order and
   presentations, fixed optimizer/tuning budgets, measured training FLOPs
   <=1% when claiming fully fair, and identical evaluation contracts. CPU
   execution time is reported, *not* forced to be equal.
4. Train shared objectives only where genuine measured targets exist:
   observed future, observed action outcome, realized future-state alignment.
   Train value and information scores only if real attributable reward and
   information target evidence exists. No synthetic success, fictitious
   repairs, hidden future state or fabricated alternative actions.
5. Independently rerun post-fit resource receipts and replay frozen
   checkpoints; compare future prediction, calibration, long-history
   retention, robustness and *where identifiable* decision benefit.

## Prior art reused instead of rediscovered

- Next-Concept Modeling, arXiv:2610.14162: multiscale generative
  representations and coarse-to-fine decoding, not established world models.
- Deep Equilibrium Models, NeurIPS 2019, https://arxiv.org/abs/1909.01377:
  iterative learned state refinement; no novel claim for looping alone.
- DreamerV3, Nature 2025, https://doi.org/10.1038/s41586-025-08744-2:
  world-state prediction, imagined rollouts and control are established;
  this variant has not demonstrated them.
- R2D2, ICLR 2019, https://willdabney.com/publication/r2d2/:
  long-history recurrent replay/hidden-state staleness are established.
- ME044, https://github.com/FPC-effortless/TAC-OSM/pull/309:
  actual six-step multiscale TAC innovation decoding and real operational
  forecasts already attempted; no need to rediscover repeated decoding.
- CED DEV010, https://github.com/FPC-effortless/TAC-OSM/pull/166:
  earlier learned-memory CED experiments did not demonstrate durable quality.

**Unanswered capability:** Does concept-specific, persistent, relevance-
conditioned recursion improve authentic future outcomes and grounded
decisions under a fair matched budget? No answer from this source-only work.
