# DEV-028 proposal: which initialization group drives learning sensitivity?

**Proposal only; no implementation, execution or findings.** DEV-027's registered
whole-model initialization effect passed its development rule: B-minus-A
-14.90pp, exploratory95% paired interval[-28.44,-1.35]. Its stream effect and
interaction did not pass. This supports a contributor under the tested recipe;
it neither proves initialization is the sole cause nor identifies a module.

Decompose the same A/B initial tensors into two exhaustive parameter groups:
(1) memory writer `.memory` (keys, values, erase/write/decay/assignment);
(2) all remaining encoder/reader/head parameters (event encoder, query encoder,
read key and prediction head). Cross A/B values for both groups, retaining the
masked-local graph,768 state scalars and8,674 active parameters. The all-A and
all-B initial states must reproduce DEV-027's bound pre-training hashes exactly.
Mixed states require registered per-component hashes. Never start from trained
weights. Initial decay weights/bias are identical across A/B in the current
source; a memory-group effect would not by itself identify initialized decay as
the cause. Writing/read-key alignment and optimization remain hypotheses.

Use both known training streams as nuisance contexts:4 module combinations x
2 streams x5 seeds =40 freshly trained models. Run the four combinations within
one seed/stream runtime block in prospective order. Both whole-model diagonals
are retrained; do not import historical final weights as controls. Pool stream
contexts within seed for the two primary module-group effects, retaining only
five paired seed units. Freeze thresholds, order, sources, component references,
exposure budget and independent audit before opening new outcomes.

Retain216 long updates, H512 batch2, AdamW.002/wd.0001/clip1, both evaluation
grids, reset/swap checks, per-seed failures, unnecessary maintenance and full
selected/model-only cost. A primary group effect requires magnitude>=5pp and
exploratory paired t(df4) interval excluding zero. Interactions and conditional
stream effects are secondary; they cannot rescue failed primaries. Preserve
every poor corner. A favorable initialization is not promoted as robust recovery.

This is adaptive decomposition on already-used seeds/namespaces, not fresh
confirmation. Any positive module finding needs fresh-seed replication before
architecture claims. A writer-group effect identifies a parameter group, not
storage/erase/decay separately. A nonmemory effect leaves encoder versus reader
unresolved. Fixed-eight synthetic development and native TAC overall failures
remain. Keep exact new-process checkpoint replay and independent raw-world/cost
audit. No DEV-028 matrix is launched in the completed batch.
