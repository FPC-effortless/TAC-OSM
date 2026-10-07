# TACOSM-PLM-LEARNED-PHYSICS-E2E-001

Status: MEASURED — VALID NEGATIVE; PRIMARY END-TO-END GATE FAILED

## Objective

Build the first genuinely learned end-to-end multimodal PLM controller in the
repository without hard-coded entity addressing, operator execution, Boolean
truth tables, or answer-bearing modality channels.

The environment is a two-particle elastic physical system. The model receives a
six-step history consisting of image particle positions, audio speed magnitude,
and text goal coordinates. Velocity direction is deliberately omitted from the
current audio surface, so temporal history is required to infer motion direction.

The model contains a learned multimodal encoder, learned recurrent state, and
learned action head. It has no explicit entity IDs and no fixed executor.

## Physical-law prior

Only the following privileged structure is allowed:

1. Hamilton's canonical equations dq/dt = dH/dp and dp/dt = -dH/dq.
2. Conservation of the learned Hamiltonian H on passive observation intervals.

The prior is applied only to the model's own predicted generic canonical latent trajectory. No state ground truth, future reward, target action, or test outcome enters the prior loss.

Two arms are measured:
data-only learning and the same learned model with the registered physics prior.

## Scientific questions

1. Can the fully learned model solve held-out physical control?
2. Does it use history when current multimodal observations are identical but the
   correct actions differ?
3. Does adding only physical-law priors improve the end-to-end result?

## Interpretation boundary

A positive result demonstrates only a bounded learned synthetic physical-control
capability. It does not establish general physical reasoning, real-world
multimodal competence, semantic memory, learned semantic addressing, or scaling.

Earlier E2E-008 Boolean results are not used for model selection.

## Preregistration amendment A1

Before the confirmatory measurement, a construct audit found that the initial
implementation allowed the action head to bypass the predicted physical-state
pathway. That would have made a physics-prior result difficult to interpret:
the prior could regularize an auxiliary state head without necessarily
regularizing the representation used by action selection.

A1 therefore binds the predicted physical state directly into the learned
action head and adds a focused gradient-dependency gate. This amendment was
made before any valid measurement artifact existed and does not alter the
benchmark, held-out split, seeds, optimizer, training budget, physical-law
prior, or decision rules.

Workflow run 37645169621 corresponds to the pre-amendment implementation and
is instrument-invalid before measurement. No result from that run is
scientific evidence.

## Preregistration amendment A2

A second construct audit found that the A1 implementation still encoded an
explicit two-particle, eight-dimensional physical-state ontology. That violates
the experiment's stricter rule that only physical laws, rather than task-specific
physical object structure, may be privileged.

A2 replaces that ontology with a generic learned canonical latent and a learned
scalar Hamiltonian. The physics arm is regularized only by Hamilton's canonical
equations and conservation of the learned Hamiltonian on passive intervals.

The benchmark, held-out split, seeds, optimizer, training budget, end-to-end
thresholds, and attribution decision rules remain unchanged.

Runs 37645169621 and 37645649621 are instrument-invalid and must not contribute
scientific evidence.
## Preregistration amendment A3

A benchmark audit found that deterministic audio contained a fixed phase offset based on particle order. This was unnecessary and could create an implicit object-order channel.

The benchmark generator is therefore amended to make deterministic audio permutation-invariant to particle order. The generator version is now `learned-physics-e2e-v2-no-object-order-audio` and is pinned before the next measurement. No model, physics prior, endpoint, seed, or decision rule changes.
## Preregistration amendment A4

A final causal-boundary audit found that the generic canonical latent was still auxiliary to the action head. A4 binds the final canonical latent directly into learned action selection and adds a focused gradient-dependency gate. No benchmark, prior law, seed, optimizer, budget, endpoint, or decision rule changes.

Superseded runs 37645169621, 37645649621, 37646974646, and 37647273293 are instrument-invalid and must not be interpreted.


## Follow-up disposition

E2E-001 is retained as valid negative capability evidence. The next experiment should address the information bottleneck in the generic learned sensor encoders and preserve the same benchmark/leakage boundary. No result from this run may be used for checkpoint or hyperparameter selection.
