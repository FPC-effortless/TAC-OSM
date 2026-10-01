# TACOSM-LEARN-REP-004

Status: PREREGISTERED — result pending.

## Purpose

REP-003 showed that the explicit-program representation is sufficient for the semantic dependency task, but the outcome-trained learner is reward-starved in several seeds. REP-004 tests only the training-time exploration hypothesis.

Nothing about the candidate representation, semantic task, graph encoder, learning rule, or evaluation protocol changes.

## Intervention

During training only, the action is selected with epsilon-greedy exploration:

`epsilon(step) = 0.30 * max(0, 1 - step/500)`.

With probability epsilon, the selected candidate is sampled uniformly from the eight candidates; otherwise the router's greedy Top-1 action is taken.

The learning rule still receives the actually selected action and only updates on verified success. Evaluation is greedy with epsilon = 0.

These constants are fixed to the values already registered in TACOSM-LEARN-001. No search over epsilon or schedule length is permitted.

## Protocol

- seeds: 0, 1, 2, 3, 4;
- training episodes: 512 per seed;
- held-out episodes: 256 per seed;
- candidates: 8;
- same REP-003 semantic program task;
- same 7 -> 16 tanh -> 8 candidate encoder;
- same 5 -> 8 query encoder;
- same pairwise hard-negative success update;
- learning rate 0.01;
- margin 0.1;
- exact explicit graph executor.

## Conditions

A. Analytic semantic matcher.
B. Baseline deterministic learner from REP-003.
C. No-learning control.
D. Epsilon-greedy exploration intervention.
E. Oracle.

## Primary endpoint

Held-out Top-1 recall of D versus B, using the same held-out tasks and seeds.

Secondary diagnostics:

- training successes;
- number of parameter updates;
- mean target rank;
- hard-negative margin;
- exact execution integrity;
- actual number of exploratory decisions.

## Falsifiers

- D does not improve held-out Top-1 over B;
- D's gains disappear under the registered multi-seed aggregation;
- exploration changes execution correctness;
- any C5 evidence or prior experimental artifact is modified.

An improvement in D would support the narrower claim that training-time exploration mitigates reward scarcity for this learner. It would not establish that the optimizer is generally adequate.

## Non-claims

REP-004 does not establish semantic generalization beyond the REP-003 task, persistent-state memory, sublinear retrieval, selective hardware computation, PLM/PNDS validity, or real-world capability.

REP-003, REP-002, REP-001, and C5 evidence remain frozen historical measurements.