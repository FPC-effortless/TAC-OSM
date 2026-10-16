# DEV-022: trained retention, causal prediction and physical cost

**Pre-execution adaptive-development protocol. No DEV-022 findings yet.**
Frozen prior: DEV-021 run 38834746064 showed much greater paired-state separation
when decay was removed from difference dynamics. It did not alter predictions.
DEV-022 trains full predictors from scratch; DEV-018/020/021 remain immutable.

## Fixed interventions

The registered ten arms, three seeds and two regimes are in
[`DEV-022.json`](../contracts/DEV-022.json). The primary causal comparison is
`no_decay - native`. Native uses the unchanged TAC writer. No-decay sets all
decay gates to exactly one throughout training/inference and freezes their
unused parameters. No-erase similarly sets erase to zero; `neither` disables
both. This complete 2x2 grid separates retention from key-side interference.

Two supplementary paired comparisons repeat the decay contrast while either:

- Partitioning each expert's 24 key coordinates into eight public entity slices
  of three coordinates. Writes and reads use only that entity's slice, and
  other entities' slices do not age on an unrelated update. This is an explicit
  write/read/local-aging bundle, not an isolated intervention on one mechanism.
- Applying the writer's key projection to encoded queries, sharing write/read
  parameters. Queries and event encodings still differ; this encourages common
  keys but does not guarantee semantically aligned addresses.

Schema41 and memoryless are positive/negative references. All TAC arms retain
the same 768-scalar layout and all initial tensor values. Common encoder, query
encoder and head initializations match across all ten arms. Disabled parameters
remain serialized but frozen; active parameter counts are reported. This is not
a parameter/FLOP-matched architecture-superiority experiment. No old checkpoint
is fine-tuned and no held-out result selects an arm, training length or threshold.

## Training and held-out evaluation

Reuse the unchanged DEV-018 public encoder, query/head interface, SGD recipe,
physical controller and independent cost functions. Use 216 AdamW steps at
0.002, weight decay 0.0001 and clip norm 1. Short history128/batch8 and long
history512/batch2 each see 1,024 pre-sensing events per update. Both pre/post
one common randomly chosen inspection losses supervise all entity/action
queries, as in DEV-018. The two regimes differ in episode/label/receipt counts;
their contrast is not solely sequence length.

Fresh namespaces: initialization uses seed+22000; unchanged trainer receives
seed+4000; evaluator receives seed+22000. Existing generator domains retain
their `dev018-train/eval` names but the actual seeds and histories are new.
All arms share cases/receipts within a cell. Train/test fingerprints must not
overlap. Train/eval seeds are regenerated independently by the stdlib auditor.

Evaluation covers H32/H128/H512/H1024, delays32/128/512, all four roles and
before/at/after deadlines, four repetitions with two opposite partners: 1,152
episodes per checkpoint. H1024 is held-out history-length extrapolation. The
primary endpoint remains 96 H512 mature pending/reissued cancellation episodes.
All endpoints are pre-sensing. State reset and paired-state swap controls must
be reported. Eight entities remain fixed; there is **no entity-count transfer
test or claim** in this experiment. Any subsequent entity-count study needs a
separate variable-size public boundary.

On all 288 H512 conditions per checkpoint, evaluate model-only service control,
one expected-value-selected inspection, budget-matched random inspection,
do-nothing, always-service and exact-public-history controllers. Charge every
failure (4), service (1), and inspection (0.1). Report normalized failure,
unnecessary-service and total-cost rates, and the fixed service-threshold
frontier. Raw action, receipt and outcome rows must allow independent replay.
The primary pre-sensing prediction cannot use any of these acquired receipts.

## Decision rules fixed before outcomes

The primary retention contrast is positive only if its mean accuracy gain is
at least five percentage points and the exploratory paired three-seed df2 95%
interval has lower bound greater than zero. Retention-only prediction recovery
also requires **every seed** of no-decay to reach 80% primary accuracy and a
ten-point drop under paired-state swap. The physical reliability gate requires
every seed to have selected failure <=5%, unnecessary services <=10%, cost per
entity <=0.8, and mean selected cost no greater than model-only cost. Report
prediction and physical gates separately and their conjunction explicitly.

Report all four retention contrasts, the decay/erase interaction and the
H1024/role/deadline strata; supplementary contrasts do not replace a failed
primary contrast. No multiple-comparison-adjusted confirmatory inference is
claimed. A successful combined entity/retention arm cannot be attributed solely
to retention. A failed no-decay arm does not prove no retention recipe can work.

## Integrity and publication

Hash source files and the protocol before execution. Each cell saves its trained
checkpoint, hashes, exact training fingerprints, raw predictions/actions/costs
and summaries. Reload the checkpoint in a separate process and require exact
prediction/action/cost replay. Independent stdlib audit regenerates histories,
labels and physical outcomes, checks actions from probabilities and public
receipts, verifies checkpoint/replay provenance, and recomputes summaries and
seed-level contrasts. It does not independently retrain models. Smoke runs are
untrained/two-step engineering checks and cannot enter the scientific audit.

Keep draft and unmerged. Even positive development prediction/cost gates do not
change official overall TAC/CASM capability or memory-attribution status.
