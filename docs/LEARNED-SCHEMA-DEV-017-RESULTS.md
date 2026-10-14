# DEV-017 — Learned 41-scalar schema wins the task comparison; reliability and memory-learning gates fail

**REPLAY-VALIDATED ADAPTIVE DEVELOPMENT RESULT. DEV-017 TAC success: FALSE.**

The smaller learned schema architecture is substantially better at this task
under the fixed training budget. However, its frozen-GRU control nearly ties
it, one seed misses long-delay reliability, and selected sensing raises cost.
TAC-OSM stays at chance on the critical H512 pending-generation condition.
Do not promote this as a learned-memory-mechanism or reliable-control success.

## Audited provenance

[GitHub CPU run 38551851130](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38551851130)
**succeeded**: **1,029 tests**, suite provenance, six-arm smoke, three full
training seeds, saved-checkpoint forecast/sensing/cost replay and a separate
standard-library raw-evidence audit. Frozen training source:
`f3ae98365e45c23b5888216b4a126e3af1137c78`, stacked on DEV-016 #172.

**18 trained checkpoints, 31,104 model episodes** (1,728/checkpoint). Each
episode stores eight-entity pre-sensing forecasts, state reset/swap controls,
and no-sensing/selected/budget-matched-random sensing trajectories. These are
correlated controls, not extra independent seed replications. Primary at/after
due scoring uses 1,152 balanced target episodes per checkpoint. Each seed's
H512 long-pending endpoint has 128 target episodes: pending/reissued, delays
64/128, at/after due, eight opposite-past pairs per cell.

The raw auditor regenerated all target worlds/public fingerprints, checked
41/768-scalar budgets and exact public history lengths, authenticated all
18 files and frozen/learned component hashes, checked logits classifications,
rederived selected inspection choices from saved beliefs, checked actual public
receipts, causal outcomes and charged costs, and recomputed all summaries and
fixed scientific gates. Neural replay separately regenerated every row from
saved weights. Neither audit replays SGD or provides independent scientific
confirmation.

[Complete experimental evidence](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38551851130/artifacts/11886684731):
`DEV-017-COMPLETE-AUDITED-EVIDENCE-NOT-CONFIRMATORY`, artifact **11886684731**,
**82,088,432 bytes**, ZIP SHA-256
`9fad2ff62c511a6dc829808823a4706615906dd923ab1552395dc4ffd95983b1`.
Contains all checkpoints, raw seed reports and `analysis.json`.

| Seed | Original artifact ID | ZIP SHA-256 |
|---|---|---|
| 43 | 11887746043 | `d33daac32d9e0a3679660ac10ad65664ae9e9b48976e799ee878176ab549a9e8` |
| 59 | 11887995261 | `4dd7197b91f574c854d0781e3da63a52818779211fd3d89e2d1510e87ec1864f` |
| 71 | 11887407535 | `935009906c4549a41c960091601991f67d742ddf12e7d079026256180c292cef` |

## Precisely what was learned

Schema: eight public entity-indexed five-dimensional GRU slots plus one GRU
global scalar. State meanings are learned; no bit/generation/timer channels
or hand-coded tracking assignments exist in the learned forward pass.
Public entity addressing is an explicit architectural prior, not learned
address discovery. The frozen-schema control retains **random, fixed GRU
transitions**, while its encoder/readout learn; it still has recurrent memory.

TAC: actual 768-scalar delta memory, native CASM-aligned readout and sparse
conditional operators, with encoder-only/memory-only/reader/joint controls.
All arms start fresh. Initial tensors and total parameters match within schema
and TAC control groups. Each gets 384 updates of eight episodes and identical
current/wait/service outcome CE, before and after two common random training
inspections. No generation labels, hidden state, role or target mask enters
the model. Evaluation chooses its own inspections from learned beliefs.

Schema has **6,956 total parameters** (6,956 trainable; encoder control 6,626).
TAC has **41,456 total parameters**, including inherited unused frozen heads;
trainable counts: reader 32,476, encoder 33,416, memory 35,772, joint 36,712.
Different state layouts, addressing priors, readouts, parameter counts and work
prevent attributing the architecture difference to scalar count alone. These
are matched-update, not matched-FLOP or matched-wall-time comparisons.

## Pre-sensing future prediction

Means over three model/data seeds; primary is balanced target wait-future
accuracy at/after maturity, before any newly selected inspection.

| Arm | State scalars | Primary future | H512 long pending | Before-due future |
|---|---:|---:|---:|---:|
| Learned schema41 | 41 | **91.84%** | **76.04%** | 83.62% |
| Schema41, frozen GRUs | 41 | 91.12% | 73.96% | 84.66% |
| TAC reader-only | 768 | 53.82% | 50.00% | 48.55% |
| TAC encoder-only | 768 | 56.48% | 50.00% | 61.11% |
| TAC memory-only | 768 | 55.64% | 50.00% | 56.25% |
| TAC joint | 768 | 57.90% | **50.00%** | 67.88% |

Schema41 mature-role accuracy: retention **100%**, acquisition **100%**,
pending cancellation **88.19%**, reissued cancellation **79.17%**. Its before-due
score is lower, so its high primary average is not deadline-universal mastery.
TAC joint role accuracy: retention 52.55%, acquisition 76.27%, pending 51.62%,
reissued 51.16%. Gains are mostly recent-value acquisition; the critical older
generation-dependent future remains unsolved.

| Learned schema41 seed | Primary future | H512 long pending |
|---|---:|---:|
| 43 | 88.98% | **71.88%** |
| 59 | 92.19% | 80.47% |
| 71 | 94.36% | 75.78% |

All three primary scores exceed the preset 80% threshold, but seed 43 misses
the **75% per-seed** long-pending requirement. An average of 76.04% does not
rescue the all-seed gate. Every TAC arm scores exactly 50% on that long-pending
endpoint in all three seeds. The before/at/after and H512 conditions remain
in the report; no failed seed or condition was removed.

## Memory learning versus encoder adaptation

Paired three-seed exploratory Student-t 95% intervals, df=2:

| Primary forecast contrast | Mean gain | 95% interval |
|---|---:|---:|
| Schema41 − TAC joint | **+33.94 pp** | **[+13.06,+54.82]** |
| Schema41 − frozen-GRU schema | +0.72 pp | [-0.84,+2.28] |
| TAC joint − encoder-only | +1.42 pp | [-11.17,+14.01] |
| TAC memory-only − reader-only | +1.82 pp | [+0.83,+2.81] |

The schema architecture's advantage is a substantial development signal in
this task/budget, not proof that smaller state itself causes superiority.
Its frozen-GRU control nearly ties it: the evidence does not reliably establish
that training recurrent GRU weights is needed beyond encoder/readout learning
with fixed indexed recurrence. This is **not** an encoder with no memory.

TAC memory-only shows a small positive exploratory effect with a frozen event
encoder, but **1.82 pp** misses the preset materiality requirement of **5 pp**.
Joint versus encoder-only is uncertain and also below that requirement. The
memory-attribution gate therefore fails. No large hidden-state-budget or extra
downstream-decoder claim is supported.

Runtime state reset gives exactly **50%** primary accuracy for every arm.
Schema41 state swaps reduce accuracy from **91.84% to 8.16%**; TAC joint swaps
reduce it from 57.90% to 42.10%. This confirms decisions depend on stored past
information. It does **not** establish that learning recurrent parameters,
rather than encoder/readout learning on an indexed recurrent substrate, caused
the high schema score. Parameter interventions and runtime ablations answer
different causal questions.

## Physical cost and action-selected sensing

Actual cost is `4*failures + services + .1*inspections`, normalized per entity.
Rates below use all eight entities; pre-sensing critical-target forecasts are
reported separately above so these easier aggregate rates cannot hide pending
prediction failures. The controller uses an explicit uncertainty heuristic
over learned beliefs, not trained RL or optimal information acquisition.

| Arm | Selected cost/entity | H512 failure rate | H512 unnecessary-service rate |
|---|---:|---:|---:|
| Schema41 | **0.5278** | **0.00%** | **2.08%** |
| Schema41, frozen GRUs | 0.5276 | 0.00% | 2.08% |
| TAC reader-only | 1.2107 | 18.86% | 25.98% |
| TAC encoder-only | 1.0474 | 2.74% | 41.40% |
| TAC memory-only | 1.0343 | 2.53% | 44.29% |
| TAC joint | 0.9695 | 1.98% | **38.30%** |

TAC joint's relatively low failure rate is bought with excessive servicing;
its unnecessary-service fraction far exceeds the preset 10% limit, and its
H512 selected cost **0.9714** exceeds the .80 limit. Seed 71 effectively services
almost everything. Low failures alone would have given a misleading positive
interpretation.

| Policy for learned schema41 | Cost/entity | Failure rate | Unnecessary-service rate | Inspections/episode |
|---|---:|---:|---:|---:|
| No sensing | **0.5261** | 0.28% | 1.48% | 0 |
| Selected sensing | 0.5278 | **0.00%** | 2.08% | 0.328 |
| Budget-matched random sensing | 0.5303 | 0.26% | 1.55% | 0.328 |

Selected sensing is better than the matched random policy, but raises cost
versus no sensing by **0.00164/entity** while eliminating a small number of
failures. Extra servicing and inspection cost outweigh that benefit. The
preset no-cost-increase sensing gate fails even though the increase is small.
Do not reclassify it as success after observing the result.

For TAC joint: no-sensing cost **0.9354**, selected **0.9695**, random **0.9491**.
Chosen sensing is more costly than either comparison and does not solve the
pending-memory problem. Fresh inspections disclose current bits only; they
cannot directly reveal whether an earlier reset invalidated a future repair.

## Scientific decision

**All six arms fail the complete reliability-and-cost gate. TAC recurrent-memory
attribution and the overall DEV-017 TAC success gate are FALSE.**

Positive finding: a learned, publicly indexed 41-scalar recurrent architecture
learns far more of this maintenance task than the 768-scalar TAC variant at
the fixed budget, with much lower physical decision cost. Recurrent state
itself demonstrably carries historical information, as reset/swap show.

Still unproven: reliable forecast behavior in every seed at H512/long delays,
benefit from training recurrent transition weights beyond encoder/readout
adaptation, and cost-effective action-selected sensing. The frozen indexed
GRU control is especially important: preserve it, rather than presenting only
the learned baseline's high average as a memory-learning success.

Next controlled priorities: isolate explicit entity addressing from readout
and recurrence design, improve deadline/cancellation calibration, and compare
inspection policies that estimate the value of *current* observations rather
than treating uncertain future outcomes as necessarily observable. Freeze a new
protocol before any tuning. Results cannot be directly compared with DEV-016's
pretrained parents, different heads, task grid or action endpoint. No SOTA,
autonomous RL, full hardened-loop, external replication or compute-efficiency
claim follows. PR #173 stays draft and unmerged; DEV-015/016 remain intact.
