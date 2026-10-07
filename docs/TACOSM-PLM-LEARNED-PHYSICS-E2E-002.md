# TACOSM-PLM-LEARNED-PHYSICS-E2E-002

Status: PRE-REGISTERED — NOT YET MEASURED

## Objective

Demonstrate a fully learned multimodal history-dependent physical-control
controller on the same synthetic benchmark used by E2E-001.

E2E-001 was integrity-valid but failed the matched-history endpoint: the
physics-prior arm reached 0.7217 mean accuracy while matched-history exact
accuracy was only 0.0020. The successor therefore targets the diagnosed
representation bottleneck before optimizing the physical-law effect.

## Registered model change

The only architecture change is generic information preservation in the
sensor encoders:

- the image encoder retains a learned spatial feature map before projection;
- the audio encoder retains learned local temporal/frequency features before
  projection;
- the text encoder, multimodal fusion, recurrent temporal state, canonical
  latent, and direct action path remain learned;
- no particle state, object slot, entity identifier, address rule, or fixed
  executor is introduced.

Global-average pooling is removed from image and audio encoders. This is a
generic neural representation change, not a physical or task-specific prior.

## Physics-law boundary

The optional physics arm receives only:

1. Hamilton canonical equations
   dq/dt=dH/dp and dp/dt=-dH/dq;
2. conservation of the learned Hamiltonian H on passive observation intervals.

No physical-state labels, object ontology, masses, entity identities, target
actions, rewards, or test outcomes enter the prior.

## Primary decision

The no-prior learned arm must achieve:

- mean held-out action accuracy >= 0.70;
- minimum seed accuracy >= 0.55;
- matched-history exact accuracy >= 0.70.

The physics-prior arm is secondary. A registered physical-law benefit requires
delta >= 0.05 with a paired-bootstrap lower bound > 0.

## Integrity

Five seeds are preregistered. Training and evaluation episode keys must be
disjoint. Evaluation fingerprints must differ across seeds and match between
arms within each seed. No checkpoint, seed, hyperparameter, or metric selection
is permitted after seeing results.

The benchmark generator is semantically unchanged from E2E-001; only its prior
manifest metadata is corrected so it exactly matches the registered
Hamiltonian-law boundary.

## Interpretation

A positive result is bounded to learned synthetic multimodal physical control.
It does not establish general physical reasoning, real-world multimodal
competence, general memory, learned semantic addressing, or scaling.

A negative result is a model-specific negative on this benchmark and does not
establish architectural impossibility.
