# TACOSM-PLM-TEMPORAL-PERSISTENCE-002 — Result

Status: **VALID MEASUREMENT**

Workflow: `37637018951`  
Artifact: `11490971609`  
Measured commit: `a5129dc4f8e241de44902fba9e1edcb817856f57`

## Question

Does retaining a q1-independent q2 entity across the action boundary improve q2 when the secret needed by q2 is masked from the post-boundary observation?

## Registered design

q1 operates on entity 0. The q2 secret is stored in entity 1 and q2 queries entity 1. After the boundary, the secret-bearing text token is masked. The carry arm retains entity 1 and writes the masked observation only to scratch entity 2. The fresh arm discards all state and rebuilds entity 1 only from the masked post-boundary observation.

The two arms use the same q2 query, the same post-boundary observation, the same multimodal encoders, the same residual-linear state write mechanism, and the same fixed XOR execution.

## Integrity

All preregistered gates passed:

- q1 and q2 target distinct entities.
- q1's outcome depends only on q1-entity bits.
- The q1 secret-interference gate is exactly zero.
- The q2 secret is masked after the boundary.
- Carry retains the secret-bearing entity; fresh reconstructs q2 from the masked observation.
- Evaluation secret/visible combinations are exactly balanced.
- Train/evaluation semantic overlap is zero.
- Evaluation fingerprints are distinct across seeds.
- No post-run model/seed/checkpoint/hyperparameter selection occurred.
- Independent artifact gates passed.

## Result

| Metric | Result |
|---|---:|
| Carry q2 accuracy | 1.0000 |
| Fresh q2 accuracy | 0.5000 |
| Carry − fresh | +0.5000 |
| Hierarchical paired bootstrap 95% CI | [0.48167, 0.51800] |
| Registered materiality threshold | +0.1000 |
| Primary criterion | PASS |

Every one of the five seeds produced carry = 1.0000 and fresh = 0.5000.

## Interpretation

This is bounded evidence that the retained state carries information that is load-bearing for a later decision when the same information is no longer present in the current observation.

It is **not** evidence that the system has MTSK. The implementation measured here uses the existing `ExplicitEntityState` with residual-linear writes. There is currently no multi-timescale state mechanism in this experiment.

It also does not establish semantic memory, general long-horizon memory, learned addressing, operator discovery, scaling, or real-world multimodal competence.

## Next scientific question

Does a multi-timescale adaptive state kernel improve retention of a slowly changing fact while preserving responsiveness to a rapidly changing fact, relative to a single adaptive timescale at the same hidden width?

That question requires its own preregistration and leakage-controlled multimodal benchmark; it must not be inferred from this result.
