# DEV-023: coordinate isolation helps; only the tested bundle passes reliability

**Complete and audited. Primary isolation-only reliability claim failed.**
The masked-write/read plus entity-local-aging arm passed prediction and physical
development gates in both training regimes across five fresh seeds. Official
overall TAC/CASM capability and prior memory-attribution gates remain failed.
October 16, 2026.

## Frozen evidence and verification

- [Complete run 38839933638](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38839933638),
  frozen scientific source `18c71f8c808c594a17ca92b87520dd159cc1a6af`.
  Regression/smoke, all **60 training jobs**, exact separate-process
  prediction/action/cost replay, and the independent stdlib audit passed.
- **1,086 tests passed** locally, including 131 contract/factor checks. All six
  two-step smoke/replay cells and a full engineering-cell audit also passed.
- [Complete evidence artifact 12021705289](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38839933638/artifacts/12021705289),
  `DEV-023-COMPLETE-AUDITED-NOT-CONFIRMATORY`, 43,202,076 bytes.
  ZIP SHA-256: `1eda6f44bee91d03262f6b751681d731a5192fb3ec612a3ed55c125ec79cc8fb`.
  Contains all checkpoints, training fingerprints, raw predictions/actions/
  receipts/physical outcomes, replay certificates and summary.
- [Committed audited JSON](DEV-023-AUDITED-SUMMARY.json) retains all per-seed
  metrics, fixed gate flags, factorial contrasts, interactions and strata.
  Local download verification checked all 60 checkpoint/replay/source hashes
  and reproduced raw-row metrics and the aggregate summary byte-for-byte.
  The full history/physical audit ran on GitHub; local verification did not
  repeat that entire regeneration pass. Neither audit retrains all models.
- The [pre-execution protocol](DEV-023-ENTITY-AGING-PROTOCOL.md) and
  [contract](../contracts/DEV-023.json) are frozen before outcomes. Their
  pre-results status is archival; this separate report contains the findings.

Six architectures × five seeds (43/59/71/89/107) × two regimes produced 60
checkpoints and 69,120 held-out model-episode evaluations. These reuse **5,760
unique public histories** across architectures/regimes. Pairs, repeated model
evaluations and multiple prefixes are not independent replications. Each
checkpoint has 96 primary H512 mature cancellation episodes; costs cover all
288 H512 episodes, including the other registered roles and deadlines.

## Factorial prediction result

All four TAC models begin with identical tensors, **8,674 trainable parameters**
and 768 state scalars. Learned decay and erase remain enabled and trainable in
every arm. Only write/read coordinate masks and the scope of decay differ.

| Write/read coordinates | Aging scope | Short H512 | Long H512 |
|---|---|---:|---:|
| Full learned keys | Global | 50.00% | 50.00% |
| Public entity masks | Global | 75.42% | 84.17% |
| Full learned keys | Entity-local | 50.21% | 50.00% |
| Public entity masks | Entity-local | **98.54%** | **93.75%** |
| Structured schema41 reference | Its original slot dynamics | 98.33% | 98.125% |
| Memoryless reference | No historical state | 50.00% | 50.00% |

The **primary** masked-global minus native-global mean gain was **25.42 points
short** and **34.17 points long**. Exploratory paired five-seed df4 95% intervals
were **[2.05, 48.78]** and **[7.75, 60.58]** percentage points. Both passed the
registered mean-gain/positive-lower-bound test, but that is only part of the
registered sufficiency criterion.

Masked-global seed accuracies were **95.83/72.92/93.75/57.29/57.29%** short and
**100/100/77.08/50/93.75%** long. Three short seeds and two long seeds failed the
80% floor; long seed89 also had zero paired-state-swap accuracy drop. Thus the
primary **coordinate-isolation prediction success flag is false in both
regimes**, as is its combined prediction-and-cost flag. A positive mean effect
does not license reliable isolation-only recovery.

Masked-local accuracies were **100/100/100/93.75/98.96%** short and
**97.92/94.79/93.75/90.63/91.67%** long. Every seed passed the primary accuracy
floor and the >=10-point swap-drop criterion. The smallest observed drop was
81.25 points. State reset remained exactly at the balanced 50% primary baseline.

## Secondary effects and interaction

All intervals below are exploratory df4 intervals in percentage points,
without multiplicity adjustment. They cannot replace the failed primary
reliability claim.

| Fixed contrast | Short mean [interval] | Long mean [interval] |
|---|---|---|
| Masking, with global aging | +25.42 [2.05, 48.78] | +34.17 [7.75, 60.58] |
| Masking, with local aging | +48.33 [45.09, 51.58] | +43.75 [40.21, 47.29] |
| Local aging, with full keys | +0.21 [-0.37, 0.79] | 0.00 [0.00, 0.00] |
| Local aging, with masked keys | +23.13 [1.79, 44.46] | +9.58 [-14.50, 33.67] |
| Mask × local-aging interaction | +22.92 [1.64, 44.19] | +9.58 [-14.50, 33.67] |

The short-regime interaction is consistent with local aging helping when
coordinates are isolated. The long-regime interaction is inconclusive: local
aging reduced accuracy in three of five individually strong masked-global
seeds while helping the other two substantially. Do not describe local aging
as universally beneficial or both components as mathematically necessary.

## Physical control: keep prediction and economic gates separate

Costs include failure×4, service×1 and inspection×0.1, divided by eight entities.
Rates are per entity. Values are five-seed means; gates also check every seed.

| Arm | Regime | Selected cost | Model-only cost | Failure | Unnecessary service | Prediction gate | Selected physical gate |
|---|---|---:|---:|---:|---:|---|---|
| native_global | Short | 1.00378 | 1.00339 | 0.40% | 47.95% | Fail | Fail |
| native_global | Long | 1.00087 | 1.00087 | 0.12% | 48.52% | Fail | Fail |
| masked_global | Short | 0.53835 | 0.54271 | 0.22% | 1.52% | Fail | Pass |
| masked_global | Long | 0.53839 | 0.54045 | 0.00% | 2.06% | Fail | Pass |
| native_local | Short | 0.99932 | 0.99948 | 0.06% | 48.55% | Fail | Fail |
| native_local | Long | 1.00000 | 1.00000 | 0.00% | 48.80% | Fail | Fail |
| masked_local | Short | **0.54149** | 0.54167 | **0.91%** | **0.19%** | **Pass** | **Pass** |
| masked_local | Long | **0.53543** | 0.53594 | **0.13%** | **1.70%** | **Pass** | **Pass** |
| schema41 | Short | 0.53256 | 0.53255 | 0.25% | 1.30% | Pass | Fail |
| schema41 | Long | 0.53543 | 0.53524 | 0.48% | 0.90% | Pass | Fail |
| memoryless | Short | 1.00000 | 1.00000 | 0.00% | 48.80% | Fail | Fail |
| memoryless | Long | 1.00000 | 1.00000 | 0.00% | 48.80% | Fail | Fail |

**Only masked-local passed the combined prediction-and-selected-cost gate**, in
both regimes. Masked-global's low aggregate costs do not repair its failure on
critical causal predictions. Schema41 passed prediction and model-only physical
gates in both regimes; its selected gate failed solely because mean selected
cost exceeded model-only cost by approximately **0.00000868 short** and
**0.00018229 long**. The frozen exact inequality is preserved despite those
small increases. All six controller modes and fixed-threshold frontiers remain
in the raw evidence; no threshold was chosen from held-out performance.

## Held-out length extrapolation

Mature cancellation accuracy at H1024, outside the trained history lengths:

| Arm | Short | Long |
|---|---:|---:|
| native_global | 50.00% | 50.00% |
| masked_global | 50.83% | 70.63% |
| native_local | 50.00% | 50.00% |
| masked_local | 97.08% | 85.63% |
| schema41 | 92.71% | 93.54% |
| memoryless | 50.00% | 50.00% |

Masked-local's H1024 long per-seed values were **79.17/88.54/85.42/94.79/80.21%**.
Its strong H512 result does not establish uniformly reliable length transfer;
no new H1024 success gate is invented here. The task still has exactly eight
entities. There is no entity-count generalization or complete CASM evaluation.

## What this changes scientifically

DEV-021's contraction mechanism and DEV-022's failed retention-only rescue stay
unchanged. This fresh factorial shows a mean benefit from **joint write/read
coordinate isolation**, but not consistent recovery from that intervention
alone. Entity-local decay application with full learned keys also did not
recover prediction. Their combination was the only TAC variant meeting all
tested H512 development gates across these five-seed recipes.

The coordinate factor still masks writing and reading together. It does not
identify which operation matters more. Global gates are learned, so a positive
masked-global result would not mean decay factors stayed at their initialized
values. End-to-end optimization can change along with memory dynamics; equal
parameter counts do not eliminate that distinction.

The successful corner reproduces DEV-022's entity architecture exactly, but
uses new initialization/training/evaluation namespaces and five seeds. Different
means or gate outcomes across studies are not evidence of a new code improvement
or a pure training-history effect. Regimes differ in episode/label/receipt counts.
No fair parameter/FLOP superiority claim over schema41 follows.

The [DEV-024 proposal](DEV-024-STRUCTURED-WRITER-ATTRIBUTION-PROPOSAL.md) asks
whether learning the writer contributes beyond fixed random memory dynamics
inside this successful structured variant. That question must be answered
before attributing the improvement to a learned TAC memory mechanism. The
proposal is unexecuted; prior official attribution and overall gates stay failed.
Keep all research branches draft and unmerged.
