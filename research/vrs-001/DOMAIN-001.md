# VRS-001 Benchmark-Private Domain Description — Rover Dynamics v1

**Classification: benchmark-private. Never provide this file to the external representation proposer.**

This file contains benchmark construction details used by the runner and verifier.
The proposer-facing description is frozen separately at
`research/vrs-001/PROPOSER-DOMAIN-001.md`.

State: (power, grip, heat, wear, terrain_affinity), all in [0,1].
Comparison vector: (power, grip, 1-heat, 1-wear, terrain_affinity).
Task: (required_power, roughness, volatility, terrain).
Exact task score: fixed weighted L1 similarity to
(required_power, roughness, 1-volatility, 0.8, terrain).

Actions: drive, climb, cool, repair. Each deterministically changes physical
state variables. Safety is defined from post-action power, grip, heat, and wear.

The representation experiment tests whether an externally synthesized semantic
geometry can compress states for task selection while remaining behaviorally
valid under fixed future action schedules. The exact score and safety rule in
this file are evaluator-side information and are not proposal inputs.
