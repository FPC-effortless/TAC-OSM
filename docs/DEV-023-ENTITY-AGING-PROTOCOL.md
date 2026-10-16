# DEV-023: coordinate isolation versus local aging

**Fixed pre-execution adaptive-development protocol; no results yet.**
DEV-022 retained all original negatives: removing decay alone did not recover
native prediction. Its entity write/read/local-aging bundle succeeded under
short training but did not isolate its components. This experiment separates
two factors; it does not alter that prior evidence.

## Factorial and primary question

| Arm | Write/read key coordinates | Application of learned decay |
|---|---|---|
| native_global | Full learned keys | Entire state |
| masked_global | Public entity slice | Entire state |
| native_local | Full learned keys | Public entity slice only |
| masked_local | Public entity slice | Public entity slice only |

Every expert's 24 key coordinates contain eight public three-coordinate entity
slices. Masking applies to **both writing and reading**, not an isolated writer
or reader treatment. Local aging restricts only the multiplicative decay term.
In native_local, erase and additive writes still use all learned coordinates;
no accidental erase/write isolation is permitted. In masked_global, unrelated
coordinates still decay on every event. All decay, erase, value, assignment,
key, encoder and head parameters remain trainable. Four identical initial
8674-parameter tensors and 768-scalar states differ only in those operations.
The native_global and masked_local corners must exactly reproduce the DEV-022
native/entity implementations when given identical parameters and inputs.
Schema41 and memoryless remain positive/negative references.

**Primary comparison: masked_global minus native_global H512 mature causal
accuracy.** It asks whether coordinate isolation suffices while global learned
decay remains. The other three simple effects and the difference-in-differences
interaction are fixed secondary contrasts. A negative primary cannot be replaced
by a favorable secondary result. An interaction estimate is a trained recipe's
factorial response, not proof of a unique semantic mechanism.

## Fixed sampling and training

The [machine-readable contract](../contracts/DEV-023.json) registers five seeds
43/59/71/89/107, short and long regimes, six arms and 60 total cells. Initialization
uses seed+23000, the unchanged DEV-018 trainer receives seed+5000, and evaluation
receives seed+23000. Domain strings retain the old generator names but actual
seed namespaces are fresh. There is no held-out hyperparameter selection or
validation tuning; no validation split is consulted because the recipe is fixed.

Reuse 216 AdamW updates, lr0.002, decay0.0001, gradient clip1 and the common
pre/post-one-random-inspection supervision. Short H128/batch8 and long
H512/batch2 each expose 1024 pre-sensing public events per step. Their episode,
query-label and receipt counts differ, so a regime comparison is not solely a
sequence-length intervention. All arms share cases and receipts within cells.

Evaluation: H32/128/512/1024, delays32/128/512, four roles, before/at/after due,
four repetitions and opposite partners = 1152 episodes per checkpoint. The
primary is 96 H512 mature pending/reissued cancellation episodes. H1024 is
held-out length extrapolation. Report all history/role/deadline strata and
reset/paired-state-swap controls. Eight entities remain fixed; **no variable
entity-count generalization or full TAC/CASM capability claim is licensed**.

## Fixed prediction and economic decisions

For the primary coordinate-isolation claim, require mean accuracy gain >=5
percentage points, paired five-seed exploratory Student-t df4 95% lower bound
>0, and every masked_global seed accuracy >=80% and paired-state swap drop
>=10 points. Interval critical value is 2.7764451051977987. Secondary intervals
are exploratory and not multiplicity adjusted; zero/failed effects are retained.

Physical outcomes use the unchanged DEV-018 controller on all 288 H512 episodes
per checkpoint: model-only, one expected-value-selected inspection, matched
random inspection, do-nothing, always-service and public-history controller,
plus the fixed threshold frontier. Failure costs4, service1 and inspection0.1;
normalize total cost and rates per eight entities. Prediction commits before
any acquired receipt. Hidden labels/world state never enter model inference.

Report selected and model-only physical gates **separately**. Each requires
every seed failure <=5%, unnecessary services <=10%, cost per entity <=0.8.
The selected gate additionally retains DEV-022's exact mean selected cost <=
mean model-only cost requirement. No numerical tolerance or threshold may be
chosen after outcomes to turn a small charged-cost increase into a pass.
Combined prediction/selected-cost success and prediction/model-only-cost success
are separate reported flags; the primary combined claim uses selected cost.
Original official overall TAC and attribution gates remain failed regardless.

## Integrity and independent audit

Freeze source/protocol/contract hashes before scientific execution. Save model
checkpoints, actual active parameter counts, initial common component hashes,
training fingerprints, raw logits/actions/receipts/outcomes and summary. A
separate process must reproduce every prediction, action and cost exactly.
Audit all 60 cells; smoke/partial/mixed-source outputs cannot enter aggregation.

The stdlib auditor independently regenerates training/evaluation histories,
public labels, physical outcomes and action choices, verifies checkpoint/replay
provenance, and recomputes raw metrics and five-seed contrasts. It caches pure
regenerated references and physical outcomes across matched histories; every
saved row is still checked. Cache keys include the history identity and all
physical action/cost inputs. The audit does not independently retrain models.

All tests and six-arm two-step smoke/replay checks gate training. Dependency
installation has a bounded step timeout; any failed runtime job can be retried
without rerunning successful cells. Retrying an engineering failure does not
permit outcome-based model selection. Later runs use explicit dispatch so
documentation updates do not silently retrain the matrix. Keep draft/unmerged.
