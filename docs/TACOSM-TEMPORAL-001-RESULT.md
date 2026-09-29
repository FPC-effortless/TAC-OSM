# TACOSM-TEMPORAL-001 — Result

Run: GitHub Actions 36509506303
Measurement commit: `ebb1dad73edd79c94bac97af6ac42c7d8729af1f`
Contract fingerprint: `765ac1e41631d975`
Artifact: `tacosm-temporal-001` (artifact 11008367442)
Configuration: H=64, seeds 0–4, delays k={1,2,4,8,16,32}, 100 probes per seed/arm/delay.
Pre-read causal boundary gate: **PASS for all 30 seed×delay cells**.

## Primary endpoint

| delay k | carry | reset | corrupt |
|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0100 | 0.0000 |
| 2 | 1.0000 | 0.0100 | 0.0000 |
| 4 | 1.0000 | 0.0100 | 0.0000 |
| 8 | 1.0000 | 0.0100 | 0.0000 |
| 16 | 1.0000 | 0.0100 | 0.0000 |
| 32 | 1.0000 | 0.0140 | 0.0000 |

`read_available` was 1.0 for carry and corrupt, 0.0 for reset at every delay. The router saw 64 candidates at each read boundary.

## Interpretation

The registered comparison is causal: the same task stream and read boundary
are used, while reset removes the readable state and corrupt alters it at the
read boundary. Carry remains at 1.0 through 32 intervening decision
boundaries.

This supports a **bounded TAC-OSM mechanism claim**: in this synthetic
state-conditioned control, an explicitly written vector can survive enforced
decision boundaries and causally change the later decision.

It does not establish semantic long-horizon memory, learned persistent
representation, or the full PLM claim. The state representation is still an
explicit vector store and the router is a deterministic control.

Amendments A1 and A2 were applied before the successful result. The earlier
A1 run failed before producing an artifact because the generator requires at
least two candidates; neither failed attempt is used as evidence.
