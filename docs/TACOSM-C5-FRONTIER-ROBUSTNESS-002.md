# TACOSM-C5-FRONTIER-ROBUSTNESS-002

Status: PRE-REGISTERED.

## Question

Does the product-key search surface retain a fixed configuration that preserves at least 80% of exhaustive reference capability across M={128,256,512} while scoring no more than 20% of the state population on average, on independent seeds 10-19?

## Why this experiment

C5-004 produced a valid count-conserving frontier on seeds 0-4. The subsequent independent replication on seeds 5-9 failed to reproduce the accepted (16,6) configuration at M=512; (32,8) emerged as the independent-seed frontier.

This experiment does not retune from those outcomes. It re-evaluates the entire 14-configuration registered search surface on a new ten-seed set, 10-19.

## Registered protocol

- M = {128, 256, 512}
- H = 256
- K = 32
- 14 product-key factor-size/beam configurations, unchanged from C5-004
- seeds = {10,11,12,13,14,15,16,17,18,19}
- 100 evaluation steps per seed/configuration/M
- capability-retention floor = 0.80
- exact count-conserving pooling across all ten seeds
- state-distinct downstream executor

## Primary decision

For each M/configuration pair:

capability_retention = summed selective successes / summed exhaustive successes.

A fixed configuration is eligible only if it meets the 0.80 floor at all three M levels.

Among eligible fixed configurations, the reported frontier is the one with minimum mean states_scored_over_M, using the same tie-break order as C5-004.

The second decision question is whether any such robust fixed configuration also satisfies mean(states_scored/M) <= 0.20.

## Integrity boundary

No factor size, beam, capability floor, M level, task stream, teacher, executor, or evaluation count is altered after seeing the result.

Each cell retains the explicit seed identifier so the ten-seed artifact can be audited at both the cell and pooled levels.

## Interpretation boundary

A positive result would strengthen robustness of the bounded product-key capability/computation observation on this synthetic workload.

A negative result would not prove product-key addressing invalid. It would show that the currently registered search surface does not yield a stable fixed capability-constrained sparse configuration across the tested seed sets.

The experiment does not establish semantic memory, universal sparse retrieval, sublinear asymptotics, or hardware speedup.

## Reproduction

Smoke: python scripts/run_c5_frontier_robustness_002.py --smoke

Full: python scripts/run_c5_frontier_robustness_002.py

Artifact: artifacts/TACOSM-C5-FRONTIER-ROBUSTNESS-002.json

Full-run marker: [run-c5-frontier-robustness-002-full]

Full measurement trigger: commit message [run-c5-frontier-robustness-002-full] is the registered measurement trigger.
