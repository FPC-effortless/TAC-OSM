# TACOSM-PLM-INTEGRATED-E2E-001 — Confirmatory Result and Audit

## Disposition

**Result:** NOT ESTABLISHED on the registered synthetic integrated capability criterion.

**Authoritative workflow:** 37062142549
**Experiment head:** ca74399b713931abe639de3b76bd859585635b26
**Artifact:** 11251182171
**Artifact SHA-256:** 03990654df7409ac69cf87012152376bdf1572341fa0ea1f0cc56c18eefad24e

## Software and protocol gates

The focused Torch gate passed. The full repository suite passed 807 tests before the scientific benchmark ran. The benchmark and artifact upload both completed successfully.

## Primary result

Seed q2 accuracies:

- seed 0: 0.5475
- seed 1: 0.4925
- seed 2: 0.5075
- seed 3: 0.5425
- seed 4: 0.5100

Mean q2 accuracy: **0.5200**

Deterministic five-seed bootstrap 95% interval: **[0.5020, 0.5385]**

Registered primary threshold: **0.80**.

The primary criterion therefore failed.

The minimum seed result is 0.4925, so the registered per-seed floor of 0.40 was satisfied. This was not a single-seed failure.

## Controls

All paired q2 controls produced exactly the same mean as the integrated arm:

| Condition | Mean q2 |
|---|---:|
| Integrated | 0.5200 |
| No memory | 0.5200 |
| Image shuffled | 0.5200 |
| Text only | 0.5200 |
| Image only | 0.5200 |
| Audio only | 0.5200 |

Therefore memory drop, image-alignment drop, and all three unimodal gaps are exactly 0.0000. Their registered seed-bootstrap intervals are [0.0, 0.0].

## Persistent-state diagnostic

Mean target-slot attention was **0.060937**.

There are 16 state slots, so uniform attention is **1/16 = 0.0625**.

The observed attention is effectively uniform. This is consistent with failure to learn target-specific persistent-state addressing.

This is a localization diagnostic, not a separately preregistered hypothesis test of attention uniformity.

## Operator diagnostic

Operator-selection accuracy was **1.0000** for every seed.

This does not establish meaningful operator routing because the operator identity is explicitly supplied by the query and is supervised during training. The result shows that the selector can reproduce the requested operator label, not that it can infer or autonomously select the required computation.

## Trivial baseline

The registered theoretical query-only operator-prior baseline is **0.625**.

The integrated model's 0.5200 is below that baseline.

## Scientific interpretation

The preregistered negative branch applies: the integrated architecture does not meet the preregistered primary criterion on this synthetic benchmark; diagnostics are retained for localization.

The evidence localizes the failure toward the representation/state-address interface rather than the explicit operator selector:

1. persistent-state removal has no effect;
2. multimodal image misalignment has no effect;
3. all unimodal controls match the integrated arm;
4. target-slot state attention is effectively uniform;
5. operator selection is perfect only because the operator is directly given.

This does not establish that PLM, persistent state, multimodal fusion, CASM, or end-to-end training are impossible in general.

## Post-run audit findings

The frozen contract contains duplicate secondary endpoint names: operator_selection_accuracy and target_memory_attention. The runner did not use those names dynamically, and the primary endpoint remained unique, so this is a clerical contract-quality defect rather than a benchmark invalidation. Future contract validation should enforce endpoint-name uniqueness.

The artifact's internal provenance.git_commit is the pull-request merge ref a0da4351158a516382001c4c68405a86f60f3685, which explicitly merges experiment head ca74399b713931abe639de3b76bd859585635b26 into master. The workflow artifact metadata separately identifies the experiment head. Future runners should record both the event SHA and the PR head SHA plus workflow run ID.

The result is not voided because the benchmark ran only after the focused Torch gate, the full repository suite, train/evaluation separation checks, and information-flow checks passed, and the artifact was uploaded with a fixed digest.

## Next research hypothesis

The next experiment should not tune learning rates or thresholds to recover 0.80. It should explicitly diagnose the persistent-state address interface:

**Can a shared write/read address representation restore target-specific persistent-state retrieval while holding the rest of E2E-001 fixed?**

The clean diagnostic should include oracle-write and oracle-read controls plus a shared learned address-space arm, with the same seeds and held-out benchmark.