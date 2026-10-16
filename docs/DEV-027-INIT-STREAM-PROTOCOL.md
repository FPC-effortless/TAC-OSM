# DEV-027 frozen protocol: initialization by training-stream namespace

Adaptive follow-up to DEV-026. Its observed long failures persisted across grids,
but five-seed cohort intervals crossed zero. No initialization or training-data
cause has been established. Freeze this design before new training outcomes.

Twenty freshly trained models: four corners within each seed43/59/71/89/107.
A_A/A_B/B_A/B_B denote whole-model initialization A/B (offset23000/24000) and
training-stream A/B (offset5000/6000). Same masked-local computation,768-scalar
state,8,674 active parameters. Initial tensors must exactly match the archived
pre-training hashes in the bound initialization reference. Never load an old
trained checkpoint. Retrain both historical diagonals and both crossed corners.

All four corners for a seed run sequentially in one process on one runner, with
prospective order shuffled by Python Random('dev027-fixed-order:{seed}'). This
reduces runtime confounding between old diagonals and newly trained crosses.
Record CPU/runtime, order and block identity. Same216 AdamW updates, lr.002,
wd.0001, clip1; H512 batch2,221,184 public events and432 receipts/model. Training
namespace changes histories AND deterministic inspection choices; initialization
changes encoder, writer and reader together. These are bundled interventions.

Evaluate both known H512 grids (offset23000/24000),288 rows/grid/model,96 primary
rows each. Primary endpoint is mean pre-sensing mature-cancellation accuracy
across both grids. Compute paired-seed whole-model initialization B-minus-A and
training-stream B-minus-A main effects, averaging over the other factor. Each
is meaningful only if absolute mean>=5pp and exploratory95% t(df4) excludes0.
Report them independently; no claim that either must be responsible. Conditional
effects and interaction are secondary and cannot rescue failed primary effects.
No multiplicity adjustment; all evidence stays exploratory/adaptive.

Report every seed on each grid. Gates stay accuracy>=.8/swap drop>=.1 and selected
failure<=.05, unnecessary<=.10, cost<=.80, mean selected<=mean model-only. Pooling
accuracy for factorial analysis never hides grid-specific reliability or cost
failure. Keep zero-reset, swap, full physical outcomes and model-only controls.

Save checkpoints, all training fingerprints/component hashes/gradients, raw rows,
source hashes and exact separate-process replays. The independent stdlib auditor
regenerates training fingerprints, evaluation cases and physical costs. Require
all20 models/40 evaluation cells, identical initial hashes within initialization
pairs, identical histories/receipts by construction within stream pairs, and
same runtime block/order across four corners. No performance-based early exit.

Known namespaces/grids and adaptive study, only five seeds. This tests the long
recipe, not episode/receipt exposure or optimization hyperparameters separately.
Results may differ from historical diagonal training because numerical execution
can affect trajectories; those old checkpoints do not substitute for new corners.
Fixed-eight public-slot synthetic architecture; no native TAC/CASM recovery.
Use the remaining approximately30minute batch; no further matrix follows this one.
