# DEV-022: retention alone did not recover causal prediction

**Complete and audited; primary retention hypothesis did not pass.**
October 16, 2026. The bundled entity-addressed/local-aging arm passed the
combined development gate under short training only. Official overall TAC/CASM
capability and prior memory-attribution gates remain failed.

## Frozen execution and evidence

- [Original complete run 38836296472](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38836296472),
  scientific source `1177096611cfa1f4f579bca71c8f4079a70cb1dc`.
  Regression/smoke, **all 60 trained checkpoint cells**, separate-process exact
  prediction/action/cost replay, and the independent stdlib audit passed.
- **1,069 tests passed** locally; all ten two-step smoke/replay cells passed.
- [Complete evidence artifact 12020092061](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38836296472/artifacts/12020092061),
  `DEV-022-COMPLETE-AUDITED-NOT-CONFIRMATORY`, 45,100,101 bytes.
  ZIP SHA-256: `85c0ae20d681c45cfa33990eb6d226569b3c858400471a0232d8fb6563e7d602`.
  Includes every checkpoint, training fingerprint list, raw prediction/action/
  outcome row, exact replay certificate, and audited summary.
- [Committed audited summary](DEV-022-AUDITED-SUMMARY.json) retains per-seed
  metrics, costs, fixed gates, role/history strata and all retention contrasts.
  After download, all 60 checkpoint/replay/source hashes were rechecked locally;
  raw-row metrics and the aggregate summary were reproduced byte-for-byte.
  The full history regeneration and physical audit ran on GitHub; local
  verification recomputed metrics/aggregation rather than repeating that entire
  history-regeneration pass. Neither audit independently reruns all training.
- The [fixed protocol](DEV-022-RETENTION-PROTOCOL.md) and
  [contract](../contracts/DEV-022.json) predate scientific execution. Old
  DEV-018/020/021 checkpoints and findings were not edited or fine-tuned.

Ten architectures × three seeds × two training regimes produced 60 checkpoints
and **69,120 held-out model-episode evaluations**. These reuse 3,456 unique public
evaluation histories across architectures/regimes; model episodes and paired
partners are not independent replications. Primary H512 evaluation has 96
balanced mature cancellation episodes per checkpoint. Physical control is
evaluated on all 288 H512 episodes per checkpoint, not just the primary subset.

## Primary prediction result

Mean pre-sensing primary accuracy over seeds 43/59/71:

| Arm | Short | Long |
|---|---:|---:|
| Unchanged native TAC | 50.00% | 50.00% |
| No decay | 50.00% | 50.00% |
| No erase | 50.00% | 50.00% |
| Neither decay nor erase | 50.00% | 50.00% |
| Entity-partitioned write/read + entity-local aging | **100.00%** | **88.19%** |
| Entity treatment + no decay | 100.00% | 78.47% |
| Shared writer/query key projection | 50.00% | 50.00% |
| Shared key projection + no decay | 49.65% | 50.00% |
| Structured schema41 reference | 94.79% | 98.61% |
| Memoryless reference | 50.00% | 50.00% |

The primary `no_decay - native` contrast was **0 percentage points in every
seed and both regimes**. Its exploratory paired df2 interval was [0, 0] on
these observed seed deltas; this is not an equivalence test over all recipes.
Removing erase as well also failed to recover the endpoint. None of the four
registered retention contrasts passed its positive-gain criterion.

For the entity treatment, per-seed H512 accuracies were **100/100/100%** short
and **82.29/95.83/86.46%** long. Runtime paired-state swaps reduced accuracy by
at least 64.58 percentage points in each of these six cells. This is evidence
that the modified predictor uses its historical state on this task, not a
claim that its native writer learned an independently isolated mechanism.

Without decay, the entity arm's long-regime mean was 9.72 percentage points
lower. That three-seed contrast has an exploratory interval of approximately
[-30.80, +11.35] percentage points, so consistent degradation is not established.
It provides no evidence of a reliable retention-removal benefit.

## Physical decisions and strict gate outcomes

Costs below include failure×4, service×1 and inspection×0.1, normalized by eight
entities. Failure and unnecessary-service columns are rates per entity.

| Arm | Regime | Selected cost | Failure | Unnecessary service | Prediction + swap gate | Physical gate |
|---|---|---:|---:|---:|---|---|
| Native | Short | 0.99659 | 0.41% | 48.12% | Fail | Fail |
| Native | Long | 1.00722 | 0.85% | 47.83% | Fail | Fail |
| No decay | Short | 1.00017 | 0.09% | 49.46% | Fail | Fail |
| No decay | Long | 1.00000 | 0.00% | 49.71% | Fail | Fail |
| Entity treatment | Short | **0.53223** | **0.81%** | **0.46%** | **Pass** | **Pass** |
| Entity treatment | Long | 0.53150 | 0.13% | 1.74% | Pass | Fail |
| Entity + no decay | Short | 0.59281 | 2.75% | 0.46% | Pass | Fail |
| Entity + no decay | Long | 0.75767 | 1.04% | 21.59% | Fail | Fail |
| Schema41 | Short | 0.52436 | 0.16% | 1.66% | Pass | Fail |
| Schema41 | Long | 0.53304 | 0.85% | 0.45% | Pass | Fail |

Gates apply to **every seed**, not these means alone. The physical gate also
requires mean selected-inspection cost to be no greater than model-only cost.
This last requirement accounts for the failures of entity-long,
entity-no-decay-short and schema41 in both regimes, despite their low absolute
failure/service costs. Their selected-minus-model-only mean cost differences
are approximately **+0.00415, +0.000072, +0.000058 and +0.000347**, respectively.
These are small charged-cost increases, not reasons to relax a frozen gate.
Entity-short's mean selected cost was slightly below its model-only cost.

The entity-no-decay long arm also failed absolute criteria: seed59 unnecessary
service rate was 46.66%, and its cost was 0.97092 per entity; seed71 unnecessary
service rate was 15.32%. Aggregate means cannot hide those seed failures.
All other TAC controls and the memoryless reference failed both prediction and
physical gates; all their values and the full fixed-threshold frontiers remain
in the evidence. **Only entity-short passed the combined development gate.**

## Held-out length extrapolation

Descriptive mature cancellation accuracy at H1024, never used for training:

| Arm | Short | Long |
|---|---:|---:|
| Native / no decay / shared keys | 50.00% | 50.00% |
| Entity treatment | 93.06% | 75.00% |
| Entity + no decay | 92.71% | 67.01% |
| Schema41 | 83.68% | 94.10% |

The long entity treatment's per-seed H1024 accuracies were 61.46/92.71/70.83%.
Its H512 success therefore cannot be advertised as uniform length robustness.
Entity count remains fixed at eight; no variable-entity transfer was tested.

## Scientific interpretation and next question

DEV-021 identified a genuine decay-driven contraction of paired differences.
DEV-022 now shows that disabling that contraction **alone did not produce usable
native prediction** under the tested full training recipe. Retention is not a
sufficient remedy here. Removing erase or sharing key projections also did not
rescue the primary endpoint. This does not establish that those mechanisms are
irrelevant under other recipes or that additive write interference is absent.

The successful entity treatment changes three things together: write-coordinate
isolation, read-coordinate isolation and which state coordinates age on an
unrelated event. It retains learned decay within the addressed slice. Its result
supports investigating that bundle, not attributing success solely to public
addressing or proclaiming that decay should never be used. Active parameter
counts also differ for disabled gates/shared keys; there is no parameter/FLOP
matched superiority claim over schema41 or a complete CASM result.

The [DEV-023 proposal](DEV-023-ENTITY-AGING-PROPOSAL.md) separates coordinate
partitioning from local versus global aging before choosing another redesign.
It is **unexecuted**. This adaptive synthetic study has only three seeds;
regimes also differ in episode/label/receipt exposure. All new development
findings must retain those limits and the failed primary retention result.

## Execution corrections and recovery provenance

Initial local regressions caught missing repository contract-schema fields.
They were added before formal execution without changing the scientific recipe
or thresholds. The full corrected suite passed.

One original cell (seed59, short, no-erase) had a long dependency-install delay
before training. An [isolated retry, run 38837538504](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38837538504),
started under byte-identical scientific sources. The original runner recovered
and completed; the retry then matched its trained parameter hash, raw rows and
summary exactly. The retry's redundant full audit was cancelled after that
check. **The authoritative complete artifact contains only the 60 cells from
original run 38836296472; it does not mix selected results from different runs.**
Duplicate full-matrix workflow runs 38837501552 and 38837538545 were cancelled
during their initial checks. Later scientific executions now require explicit
dispatch after the initial PR-opening run, so result-documentation edits do not
silently retrain the matrix. No research branch was merged.
