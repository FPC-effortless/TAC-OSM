# TACOSM-PLM-LEARNED-PHYSICS-E2E-001

Status: PRE-REGISTERED — NOT YET MEASURED

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

1. linear momentum conservation on collision-free observation intervals;
2. kinetic-energy conservation on collision-free observation intervals;
3. kinematic consistency between displacement and average velocity.

The prior is applied only to the model's own predicted physical-state trajectory.
No state ground truth, future reward, target action, or test outcome enters the
prior loss.

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
