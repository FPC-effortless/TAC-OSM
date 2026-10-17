# DEV-040A — Fresh-seed frozen-head loss and probe controls

**Completed adaptive-development pilot; not the full DEV-040 convergence study.**

Single primary: bounded-MTSK balanced-minus-uniform MLP accuracy on mature reissued cancellation. Mean gain **+28.12 pp**; exploratory paired three-seed 95% t(df=2) interval **[-18.94, +75.19]**. Prespecified primary criterion met: **False**.

## Research decision

**Do not promote the balanced head as a reliable or economically better controller.** It achieves perfect mature target-wait classification in these trials, but the primary uncertainty criterion fails, calibrated cost increases from 0.531539 to 0.563043 per entity, and failures increase from 32 to 375 over 27,648 paired entity-world decisions. Unnecessary services fall from 386 to 228. This is the wrong tradeoff for the requested reliability objective. No evaluation sensing is exercised.

The prediction endpoint scores mature target-wait labels; physical decisions depend on wait/service probabilities for every entity, including pending worlds. Perfect endpoint accuracy therefore does not establish calibrated probabilities or correct counterfactual service decisions. This experiment does not locate the remaining policy errors more precisely.

The focused trained probes all reach 100%, while query-only and current-observation controls remain at 50%. Resetting the original model's entire recurrent state reduces its reissued score to 50%; swapping paired states usually reverses its predictions. These support use of historical state. However, reset-writer probes average 97.92% for bounded MTSK and 100% for masked-local TAC and schema41; even fully random schema41 reaches 100%. Learned recurrent weights are therefore not necessary for this probe performance on this cohort. Joint encoder/adapter training, supplied routing and random-reservoir retention remain viable explanations. A reset-writer intervention is not a matched retraining comparison.

Shuffled-label probes average near chance but vary sharply by seed, including 98.96% for masked-local TAC and 97.92% for schema41 at seed4002, and 0% for masked-local TAC at seed4003. One paired-label randomization per seed is an unstable null on this simple task, and its average must not be called a passed selectivity test. Preserve these outcomes; use repeated, prespecified permutations in subsequent probe work.

Next, test a **random frozen writer with matched encoder/adapter/head retraining** against the trainable writer, alongside an encoder-matched memoryless control. Train and score pending, mature and service-counterfactual predictions explicitly, and require a prespecified cost/failure gate. Preserve the current negative result. The full convergence-controlled DEV-040 comparison remains unregistered and unrun.

## Every fresh seed, primary architecture

| Seed | Original | Uniform MLP | Balanced MLP | Uniform linear | Balanced linear |
|---|---:|---:|---:|---:|---:|
| 4001 | 95.31% | 82.81% | 100.00% | 81.77% | 100.00% |
| 4002 | 50.00% | 50.00% | 100.00% | 51.04% | 100.00% |
| 4003 | 87.50% | 82.81% | 100.00% | 70.31% | 100.00% |

No head was selected from these evaluation outcomes. The primary is the prespecified MLP loss contrast; the linear/probe/architecture comparisons are secondary development diagnostics. Three seeds cannot establish population reliability.

## Same-cohort predictive and physical outcomes

All values below are three-seed means. Costs and rates use all eight entities across 1,152 evaluation worlds per seed. Prediction endpoints use 768 mature targets, 384 cancellation targets and 192 reissued targets per seed/architecture. These paired worlds share a synthetic generator and are not independent training replications.

| Architecture | Head | Mature | Cancellation | Reissued | Fixed cost/entity | Calibrated cost/entity | Calibrated failure rate | Calibrated unnecessary service |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bounded_mtsk | original | 86.85% | 79.60% | 77.60% | 0.586082 | 0.561415 | 0.39% | 3.57% |
| bounded_mtsk | mlp_uniform | 88.59% | 77.17% | 71.88% | 0.532986 | 0.531539 | 0.12% | 1.40% |
| bounded_mtsk | mlp_balanced | 100.00% | 100.00% | 100.00% | 0.613317 | 0.563043 | 1.36% | 0.82% |
| bounded_mtsk | linear_uniform | 86.15% | 73.26% | 67.71% | 0.554941 | 0.552337 | 0.48% | 2.39% |
| bounded_mtsk | linear_balanced | 98.65% | 97.31% | 100.00% | 0.647569 | 0.579789 | 1.84% | 1.06% |
| masked_local | original | 97.66% | 95.31% | 95.31% | 0.537326 | 0.534758 | 0.17% | 1.56% |
| masked_local | mlp_uniform | 96.79% | 93.58% | 94.79% | 0.527705 | 0.527452 | 0.14% | 0.92% |
| masked_local | mlp_balanced | 100.00% | 100.00% | 100.00% | 0.545356 | 0.540907 | 0.88% | 0.04% |
| masked_local | linear_uniform | 95.01% | 90.02% | 86.28% | 0.537326 | 0.535120 | 0.07% | 1.88% |
| masked_local | linear_balanced | 100.00% | 100.00% | 100.00% | 0.545356 | 0.543077 | 0.92% | 0.14% |
| schema41 | original | 100.00% | 100.00% | 100.00% | 0.543945 | 0.537218 | 0.45% | 0.97% |
| schema41 | mlp_uniform | 98.18% | 96.35% | 96.01% | 0.525137 | 0.524812 | 0.10% | 0.77% |
| schema41 | mlp_balanced | 100.00% | 100.00% | 100.00% | 0.540039 | 0.533601 | 0.61% | 0.10% |
| schema41 | linear_uniform | 95.92% | 91.84% | 94.27% | 0.535409 | 0.534650 | 0.06% | 1.87% |
| schema41 | linear_balanced | 100.00% | 100.00% | 100.00% | 0.545356 | 0.542824 | 0.90% | 0.18% |

Fixed threshold=.25; calibrated thresholds minimize labeled cost on the separate calibration split, with the exact grid and tie rule frozen before evaluation. Cost=4*failures+services+.1*inspections. No evaluation inspection occurs, so this cannot demonstrate active sensing. Calibration changes actions, not the reported argmax prediction accuracy.

| Shared reference | Cost/entity | Failure rate | Unnecessary service |
|---|---:|---:|---:|
| always | 1.000000 | 0.00% | 48.59% |
| never | 2.056424 | 51.41% | 0.00% |
| public | 0.514106 | 0.00% | 0.00% |

Bounded-MTSK calibrated balanced-minus-uniform MLP cost/entity: **+0.03150318**; preset mean nonincrease rule: **False**. This is separate from the primary accuracy criterion. Lower cost does not imply fewer failures; always-service has zero failures in this simulator.

Schema41 supplies eight entity-addressed 5D slots plus one global scalar; GRU dynamics are learned. Slot allocation and entity routing are hand-coded. Its performance tests that structural prior, not schema discovery.

## Frozen-feature probe controls

Logistic probes are trained only on 96 mature reissued head-training worlds and scored on 192 fresh worlds per seed. The first four probe columns use the existing 44D pre-head feature interface; query-only uses 14 public features and current-event-plus-query uses 30. Standardization uses training features only.

| Architecture | Trained | Fully random | Reset writer | Shuffled training labels | Query only | Current event + query |
|---|---:|---:|---:|---:|---:|---:|
| bounded_mtsk | 100.00% | 70.49% | 97.92% | 44.79% | 50.00% | 50.00% |
| masked_local | 100.00% | 72.57% | 100.00% | 52.26% | 50.00% | 50.00% |
| schema41 | 100.00% | 100.00% | 100.00% | 50.35% | 50.00% | 50.00% |

| Seed | Architecture | Trained probe | Random probe | Difference (pp) | Shuffled labels |
|---|---|---:|---:|---:|---:|
| 4001 | bounded_mtsk | 100.00% | 60.42% | +39.58 | 65.10% |
| 4001 | masked_local | 100.00% | 64.06% | +35.94 | 57.81% |
| 4001 | schema41 | 100.00% | 100.00% | +0.00 | 43.23% |
| 4002 | bounded_mtsk | 100.00% | 59.90% | +40.10 | 56.77% |
| 4002 | masked_local | 100.00% | 58.33% | +41.67 | 98.96% |
| 4002 | schema41 | 100.00% | 100.00% | +0.00 | 97.92% |
| 4003 | bounded_mtsk | 100.00% | 91.15% | +8.85 | 12.50% |
| 4003 | masked_local | 100.00% | 95.31% | +4.69 | 0.00% |
| 4003 | schema41 | 100.00% | 100.00% | +0.00 | 9.90% |

A positive trained probe supports accessible signal in that frozen representation. A strong random recurrent probe still demonstrates retention by a reservoir; it weakens attribution to training. Reset-writer keeps the trained encoder/query/adapter and replaces only recurrent weights, but creates a distribution shift and is not matched frozen-writer retraining. Shuffled-label performance is a finite-sample negative control, not automatically exactly 50%. Query/current controls must be exactly 50% on balanced identical-current paired inputs; the audit enforces this. None of these probes establishes full deployed reliability or a globally sufficient memory architecture.

## Protocol and integrity

| Constructor | Recurrent scalars (FP32) | Backbone parameters |
|---|---:|---:|
| bounded entity-local MTSK | 768 | 6,662 |
| masked-local TAC writer | 768 | 8,674 |
| learned schema41 | 41 | 5,932 |

These are actual constructor capacities, including the original head, not matched-parameter claims. Replacement MLP and linear heads contain 3,010 and 90 parameters respectively. No FLOP-equivalence claim is made.

This pilot trains three preserved constructors at three new seeds 4001/4002/4003, 216 B2/H512 backbone updates each, with matched public streams across architectures. The original DEV-036 archive was unavailable; these are fresh trials, not claims to reproduce its SGD. All memory/query/adapter/encoder parameters are frozen before fitting heads. Original heads remain untouched. MLPs warm-start from the original; linears start identically within their loss family. LR.002 versus.02 are fixed recipes, not a tuned capacity comparison.

Uniform head training supervises 13,824 bits. Balanced training puts half its total weight on 384 mature target-wait labels, equally divided by role, and half on the remaining labels. Ground-truth role/target metadata enter training weights only, never prediction features. Same-cohort head training/calibration/evaluation use 576/288/1,152 worlds and disjoint namespace/fingerprint sets. Fresh namespaces are not configuration-disjoint or unseen-identity tests.

Every seed fits all three architectures, 15 heads, 18 probes and calibrators before generating its evaluation cohort. Jobs run concurrently under code frozen before dispatch; this is not a globally sealed confirmatory study. The shared evaluation cohort across seeds means seed uncertainty is conditional on that cohort.

All 1,184 regression tests and the initial smoke pass. Separate-process replay regenerates all frozen train/calibration/evaluation features, refits 36 replacement heads and 54 probes exactly, and reproduces predictions, service decisions and physical outcomes. A separate stdlib audit regenerates labels/split census, threshold scores, actions, costs and all summaries. These establish consistency, not external independent training replication. No optimizer replay of the nine backbone training runs is claimed.

Frozen training source: `3ff9a7a9bcf1561f02cf94746e126fadf4d90de4`. [Successful CPU training/replay/audit run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38997852977). **Nine backbone checkpoints, 45 prediction heads, 54 probes; 10,368 representation-world rows and 51,840 head-world evaluations.** These are correlated views, not that many independent experiments.

[Complete raw checkpoint/feature/report archive](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38997852977/artifacts/12088257141): 109,834,861 bytes; ZIP SHA256 `00808f47c9b6dc1739597032898a9760e2183ee5f22a04d78848bd237b3a191b`. Raw seed archives were downloaded by the audit job; the complete audited archive was uploaded before aggregate results were printed. GitHub authentication may be required. Artifact retention expires 2027-01-15T03:11:53Z; this archive is outside the workspace, not a permanent DOI repository.

- [Compact audited analysis](../results/DEV-040A/analysis.json)
- [All 45 per-seed head rows](../results/DEV-040A/per-seed-heads.csv)
- [Evidence provenance](../results/DEV-040A/archive.json)
- [Frozen pilot protocol](../research/dev040/pilot-protocol.json)

## Scope and next decision

This pilot does not run the 24-trial search, train-to-convergence, 20-draw confirmation, parameter-matched GRU/delta comparison, configuration generalization or full CASM/CDL/OSM system. The bounded banked MTSK remains an adaptation with supplied entity indexing, not the unchanged original six-expert MTSK. Architecture parameters and state dimensions differ. Preserve all historical results and the unregistered full DEV-040 draft. Interpret the prespecified primary first; use the controls to decide whether the next intervention belongs in head optimization, representation learning or a larger converged comparison.
