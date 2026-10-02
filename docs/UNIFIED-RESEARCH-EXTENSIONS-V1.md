# Unified native-model extensions from 2026 research

## Active information acquisition

Identifiability is sometimes an environmental property rather than a property
of the encoder. When the current observation does not determine which latent
state matters, the system may need to actively probe the environment before
committing to a structural representation.

Introduce an optional learning-time stage:

`I_t -> U_t -> Probe_t -> I_(t+1) -> D_t`

where `U_t` estimates uncertainty/equivalence classes and `Probe_t` selects a
bounded information-gathering action. A probe is allowed to change the world
only through the declared environment interface and is never given hidden
truth. The decision to probe and the benefit of the information gained are
separately measured.

Recent structured world-model research reports a closed loop in which active
probing collects informative trajectories and structured world-model learning
distills task-sufficient state from them. This maps directly onto the
identifiability problem exposed by REP-001. This prior art motivates the
mechanism; it does not transfer its empirical results to TAC-OSM. 

## Multi-projection future prediction

Do not force `Z_t` to preserve one chosen rendering of the future. Instead use a
shared future latent with several registered prediction targets:

`T(Z_t,a_t) -> (Z'_(t+1), Y_text, Y_image, Y_audio, Delta_structure)`

when the modality is available.

The key architectural constraint is that these targets are post-action
supervision only. They must not enter the pre-action route representation.

Recent multimodal world-action research uses multiple projections of the same
future—visual, semantic, geometric and interaction projections—to constrain a
shared action-conditioned model. The useful principle for TAC-OSM is to test
whether several future consequences force the latent to preserve structure
that a single reconstruction objective would discard. 

This must not be confused with the earlier TAC-OSM positive-view experiment:
that experiment averaged gradients over multiple noisy positive realizations
within one representation objective and was harmful in that configuration.
Here the intervention is different: multiple **future target modalities** are
co-predicted from a shared latent, with no per-view gradient-selection heuristic.

## Action sufficiency as an explicit gate

A latent can predict value or observation consequences while still collapsing
states that require different actions. Therefore the representation gate now
requires action sufficiency in addition to representability, identifiability,
non-collapse and prediction.

The action-sufficiency test should ask whether two latent-equivalent states can
require different optimal executable actions under the registered environment.
If yes, the representation is insufficient regardless of reconstruction quality.

## Memory evaluation

Memory must be evaluated on continuous interaction trajectories, not only on
isolated query/answer recall. The canonical future-memory endpoint is:

`performance_after_raw_observation_removal`

measured after a verified experience has been stored and after the original
observation has become inaccessible.

This aligns with current long-horizon agent-memory evaluation, which emphasizes
state/action/observation trajectories and experience reuse rather than dialogue
recall alone. 

## Research consequence

The unified architecture is therefore not only a passive learner. During
training and in environments that permit information-gathering actions, it may
actively acquire observations that make downstream structure learning possible.
That provides a principled response to the REP-001 identifiability barrier
without leaking the answer into the learner.
