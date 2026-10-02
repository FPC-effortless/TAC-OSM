# VRS-001 Proposer-Facing Domain Description — Rover Dynamics v1

This file is the only domain description permitted to the external representation
proposer.

## State

A rover state has five continuous variables in [0,1]:

- power: available energy reserve
- grip: traction reserve
- heat: thermal load
- wear: mechanical wear
- terrain_affinity: compatibility with the current terrain

Higher power, grip, and terrain_affinity indicate more reserve/compatibility.
Lower heat and wear indicate a healthier physical state.

## Actions

The rover may take one of four actions:

- drive: consumes power and increases heat and wear
- climb: consumes power and grip and increases heat and wear
- cool: slightly consumes power while reducing heat
- repair: consumes some power while reducing wear

The action dynamics are deterministic.

## Task descriptions

Each task is described by four public attributes:

- required_power
- roughness
- volatility
- terrain

The attributes describe what a successful task demands from the rover and its
environment. The proposer may use these public descriptions when constructing
query coordinates.

## Representation objective

Construct a continuous multidimensional representation whose coordinates are
interpretable semantic features of rover/task state. The representation should
preserve distinctions that matter for choosing useful rover states for tasks
and should remain meaningful when the rover state changes after actions.

The representation must be expressed through explicit features with values on
a common unit interval [0,1]. Do not rely on task outcomes, rewards, target
identities, benchmark scores, truth tables, verifier outputs, or post-action
evaluation results.
