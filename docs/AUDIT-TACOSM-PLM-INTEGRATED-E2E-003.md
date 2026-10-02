# Audit — TACOSM-PLM-INTEGRATED-E2E-003

## Pre-measurement disposition

**READY FOR CONFIRMATORY EXECUTION**

### Benchmark validity

The benchmark is independently generated in
`src/tac_osm/integrated_e2e_benchmark.py`.

The generator requires:
- three distinct entities per episode;
- q1 target = entities[0];
- q2 target = entities[1];
- eight held-out operator/index compositions;
- held-out compositions absent from TRAIN_COMBOS.

Before measurement, the focused test suite asserts these properties.

### Leakage

Entity identity is structural metadata and is available before action.

No answer, target payload, environment outcome, or evaluation correctness is
supplied before action.

The verifier receives environment outcome only after action.

### Statistical protocol

Five fixed seeds 0–4 are retained.

Each seed receives exactly 300 training steps and 400 corrected held-out
evaluation episodes.

No arm, seed, or hyperparameter is selected after observing another arm.

### Parallel execution

The three arms are run as independent GitHub Actions matrix jobs and aggregated
afterward. Parallelism changes wall-clock execution only; it does not alter the
registered seed set, episode generator, optimization schedule, or evaluation.

### Claim boundary

No result may be interpreted as real-world multimodal understanding, learned
semantic addressing, autonomous operator discovery, or asymptotic scaling.
