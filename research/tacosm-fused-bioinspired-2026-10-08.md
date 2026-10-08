# TAC-OSM Fused Bio-Inspired Research Chain

Date: 2026-10-08

## Scope

This branch implements a new experimental TAC-OSM treatment combining two transferable biological principles: distributed local feedback and long-range coordination from fly nervous-system research, plus multi-timescale internal memory and experience-dependent priming from bacterial systems.

The implementation does not copy a biological connectome, neuron topology, iron chemistry, or ribosomal machinery.

## Core hypothesis

A persistent computational agent may benefit from the following sequence:

native representation -> persistent fact state -> multi-timescale experience dynamics -> learned regime state -> sparse module recruitment -> independent local computation -> action/outcome -> post-hoc verification -> conservative experience write -> future routing/computation.

Canonical loop:

S_t -> R_t -> G_t -> CDL_t -> C_t(local) -> A_t -> O_t -> V_t -> M_(t+1) -> S_(t+1)

## Module contracts

### Persistent fact state

The existing key-addressed PersistentStore remains the authoritative world/task state. It is intentionally separate from experience dynamics so a memory-state ablation does not silently erase world facts unless that is the explicit fact-state arm.

### MultiScaleMemory

Owns independent fast/medium/slow global states, optional local state for each computational module, priming, surprise, volatility, and read-time intervention controls.

### RegimeState

A small learned predictor and latent projection using only observable current input plus persisted history. The hidden current target enters only after the environment transition.

### SparseCoordinator

Recruits local computational modules. Sparse mode applies coarse query bucketing before local ranking. Dense/all/single/disabled modes are controls. Module IDs can be hard-lesioned while leaving the model and other modules unchanged.

### SpecialistPool

Each module owns parameters and a cached feature trace. Modules can be independently trained, inspected, shared-parameter coupled, replaced, or lesioned. Adaptive halting is separately measurable.

### FusionVerifier

Verification is post-action. Final and path modes are separate. With the commit gate enabled, unsuccessful or unverified outcomes cannot become authoritative experience writes. Repair is bounded and forward-looking; it cannot rewrite historical outcomes using the gold action.

## Research predictions

1. Same present / different past: identical current query and candidate information can yield different internal routing or computation after different histories.
2. Useful memory: history dependence should improve subsequent outcomes rather than merely changing hidden state.
3. Temporal spectrum: fast state should decay faster than slow state under washout.
4. Sparse recruitment: sparse coordination should evaluate fewer module candidates and execute fewer modules than dense coordination, subject to a capability constraint.
5. Specialization: independent module parameters and usage should diverge when tasks support specialization; shared-parameter ablation tests whether that divergence matters.
6. Graceful degradation: module lesions should produce an interpretable capability curve without hidden fallback to all modules.
7. Verification: verifier gating should suppress false authoritative writes. It is not credited as a capability improvement unless capability improves independently.
8. Adaptive computation: hard inputs should consume more refinement/halting work than easy inputs if adaptive execution is useful.

## Ablation matrices

The executable catalog in src/tac_osm/fusion/ablation.py defines:

- component: full, fact-state removed, experience memory removed, local/global memory removed, priming removed, dense/single/all coordination, shared operators, fixed halting, verifier variants, repair off;
- memory: 1/2/4 timescales, local/global switches, reset, shuffle, random, corruption, wrong-module;
- coordination: sparse k=1/2/4, dense, single, all, disabled;
- priming: on/off and memory-off control;
- specialization: independent/shared;
- halting: adaptive/fixed;
- verification: none/final/path, commit-gate off, repair off;
- fact state: reset/shuffle/random/corruption/wrong-key/no-write/disabled;
- integrity: control, dynamic-reset, same-present/different-past, no-future-outcome control;
- lesion: each of modules 0..7 disabled individually.

## Measurement rules

Capability metrics and mechanism metrics are separate.

Capability: aggregate accuracy, per-family accuracy, success count, confidence intervals over seeds.

Routing: module candidate set size, selected module count, routing work, selection histogram.

Execution: number of module executions, active candidates processed, adaptive halting depth.

Memory: state norm by timescale, priming norm, volatility/surprise, washout decay.

Verification: pass rate, failed constraints, repair rate, false-commit count.

Specialization: parameter divergence, module usage concentration, module-specific accuracy where the environment permits it.

A candidate-count reduction is never a positive result by itself.

## Integrity boundary

WorldEnvironment creates the hidden outcome only after routing and local execution. Task.public() omits the target action. None of the forward modules receives target_action, gold_index, outcome, reward, or success before selection.

Existing TAC-OSM representability and leakage principles remain upstream controls. A fused result cannot be interpreted if the task is not uniquely solvable by the intended relation or if the routing hypothesis class cannot express the intended relation.

## Same-present / different-past test

Construct two agents that receive the exact same current Query and candidate set but have undergone different historical regimes. Record:

- selected modules;
- module score vectors;
- selected candidate;
- memory state before current computation;
- regime context before current computation.

Then compare future task success. A valid memory effect requires a history-dependent computation change and an outcome benefit on held-out current tasks. The current hidden target is never used to condition the forward path.

## Falsification conditions

The fused mechanism should be considered unsupported when:

- resetting or scrambling dynamic memory does not change the predicted history effect;
- sparse routing has no capability-preserving advantage over dense routing at equal module capacity;
- lesions have no causal effect despite apparent module specialization;
- verifier gating cannot reduce false writes;
- apparent historical effects disappear when current observable inputs are exactly matched;
- improvements are obtained only by benchmark-specific tuning or leakage.

## Provenance requirements

Every confirmatory result should record repository, branch, commit, benchmark version, experiment suite, configuration, seed, command, artifact path, and model/component manifests. Frozen controls should not be overwritten when later runs differ.

## Interpretation

This is an architectural experiment, not a claim that biological mechanisms are required for intelligence. The biological studies motivate which computational mechanisms to test; they do not validate TAC-OSM. Positive findings must be tied to the ablated mechanism and negative findings must first be localized to representation, addressing, computation, verification, or benchmark integrity.
