# TACOSM-PLM-INTEGRATED-E2E-003 — Confirmatory Result and Audit

## Disposition

Result: NOT ESTABLISHED on the registered primary capability criterion.

The corrected benchmark generator was independently implemented after the E2E-001/E2E-002 generator invalidation. The current run passed the full repository and benchmark gate before executing the three scientific arms.

Authoritative workflow: 37069959733
Experiment head: d638ce29a828084d6cdcaa0fd38fd70b5cb89d53
Combined artifact: 11254731258
Combined artifact SHA-256: 6f9cfb2cfc0054f84155a383b003b5cd686f02c9f82bbbccc55b3c80f891f4c5

## Primary endpoint

Explicit-both q2 accuracy: 0.5200

Five seed results: 0.5475, 0.4925, 0.5075, 0.5425, 0.5100

Seed-bootstrap 95% interval: [0.5020, 0.5385]

Registered primary threshold: 0.80.

Minimum seed: 0.4925, above the registered 0.40 per-seed floor.

Therefore the primary criterion fails on capability level, not because of a missing seed.

## Address-path diagnostics

All three arms produced the same five seed values:

| Arm | Mean q2 |
|---|---:|
| explicit write | 0.5200 |
| explicit read | 0.5200 |
| explicit both | 0.5200 |

The registered 0.05 materiality threshold is not met for either single-path comparison or the explicit-both interaction.

Explicit-read target-slot attention is exactly 1.0 by construction. Explicit-write write-target address accuracy is 1.0. Explicit-both has both structural addresses fixed.

Supplying correct structural state addresses therefore does not recover capability.

## Controls

For explicit-both, all paired controls exactly match normal performance:

| Condition | Mean q2 |
|---|---:|
| normal | 0.5200 |
| no memory | 0.5200 |
| shuffled image | 0.5200 |
| text only | 0.5200 |
| image only | 0.5200 |
| audio only | 0.5200 |

All paired effects are exactly zero.

## Mechanistic interpretation

The result shifts the immediate blocker upstream of persistent-state addressing.

The measured chain is representation -> persistent state -> query-conditioned read -> CASM -> action -> verifier -> state update.

Correct structural addressing did not change q1 or q2 accuracy. The current implementation has no guaranteed shared content interface between the representation's supervised bit head and the CASM bit head: the representation bit loss trains one projection from z, while CASM independently learns another projection from persistent state.

This is the next falsifiable representation/execution-interface hypothesis; it is not itself treated as a result.

## Scientific limits

This does not establish impossibility of PLM, persistent state, multimodal fusion, or CASM; learned semantic addressing; natural language, vision, or audio competence; arbitrary learned operator synthesis; or scaling behavior.

E2E-003 is the first valid integrated benchmark in this sequence. Earlier E2E-001 remains provenance-only because its query-target generator was invalid.

## Next falsifiable experiment

Hold the corrected temporal benchmark fixed while replacing the unconstrained hidden-state-to-CASM interface with a typed state-content interface.

Compare the current hidden-state interface, a supervised 12-bit typed state consumed directly by CASM, and an oracle payload state. The purpose is to separate representation failure, state-content interface failure, and CASM/execution failure.

No threshold, seed, or hyperparameter may be selected from the E2E-003 outcome.