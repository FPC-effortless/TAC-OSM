# TACOSM-PLM-FULL-LEARNED-SWILA-OPT-002

Status: **PRE-REGISTERED — awaiting measurement**

## Purpose

This diagnostic is a direct follow-up to the completed
TACOSM-PLM-FULL-LEARNED-SWILA-001 capability failure. It separates three
questions without changing the benchmark or architecture:

1. What does the exact architecture do before any optimizer updates?
2. Did the registered 220-step training protocol actually move the model?
3. Is the preregistered Hamiltonian prior materially constraining that movement?

## Arms

- **untrained** — exact model construction at each registered seed, zero
  optimizer updates.
- **trained_full** — exact full learned training protocol, including the
  physics prior weight 0.03.
- **trained_physics_off** — exact same training protocol with only the physics
  prior weight set to 0.0.

The three arms share the same deterministic train/evaluation generation
functions. Evaluation fingerprints must match within seed across all arms and
differ across seeds.

## Frozen protocol

- five seeds: 0–4
- 72 training episodes per action
- 36 evaluation episodes per action
- 80 history pairs
- 220 training steps
- batch size 8
- AdamW, learning rate 0.001, weight decay 1e-4
- d_model 96, four heads, six MTSK experts
- CDL top-k 2
- four CASM operators, depth 3, runtime top-k 1
- identical benchmark generator and observation schema as the completed
  full-learned run

No checkpoint, seed, hyperparameter, metric, or arm will be selected after
measurement.

## Registered diagnostic rules

A trained arm counts as showing a training signal when its mean total loss over
steps 201–220 is at least 10% below its mean over steps 1–20 and its parameters
changed.

A trained arm counts as exceeding its exact untrained floor when its mean action
accuracy is at least 0.05 above the untrained arm.

The physics prior is considered a material constraint only when
physics-off action accuracy is at least 0.05 above the full arm, under the same
integrity conditions.

These are diagnosis rules, not capability thresholds.

## Expected interpretation boundary

If the untrained and trained arms are indistinguishable and the trained
parameters do not change, the negative result is instrument/optimization
invalid.

If training clearly changes the model but capability remains low, the failure
is a genuine optimization/generalization problem rather than evidence that the
weights were never used.

If physics-off materially improves learning or capability, the Hamiltonian prior
becomes a concrete optimization suspect and should be removed or repaired in
the next construction experiment.

None of these outcomes establishes or refutes general PLM capability, learned
long-horizon persistence, multimodal real-world competence, or scaling.
