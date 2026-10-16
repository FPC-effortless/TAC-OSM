# DEV-026 proposal: numerically bounded frozen cross-evaluation

**Proposal and tested comparator only; no full experiment or findings.**
DEV-025's strict byte-identical diagonal protocol failed. It remains failed and
is not amended retroactively. One diagnosed cell had only float32 inference
rounding differences; the other failed cells need characterization. Do not
infer the source of the scientific long-regime discrepancy from partial data.

Before a successor run, freeze a new protocol and source commit. Retain the
same20 bound checkpoints, two H512 grids,40 crossed cells, zero training and
all scientific contrasts, costs and thresholds from DEV-025. Both known
diagonals remain known; no confirmatory claim is licensed. Checkpoint hashes
and parameter hashes must match exactly. Newly generated reports must still
replay exactly in a separate process under their own pinned runtime.

For comparison against archived diagonals only, permit explicit float32
rounding in logits, decision probabilities, estimated sensing gains and
hypothetical costs: relative tolerance1e-5, absolute tolerance1e-6. These are
numerical engineering bounds, not margins for model quality. Require exact
fingerprints, conditions, targets, predictions, reset/swap predictions,
inspection choices/receipts, services, physical failures, unnecessary service
and realized costs. Every other field stays exact; all numbers must be finite.
The prospective comparator `scripts/dev026_replay_numerics.py` has focused
checks that reject even tiny realized-cost changes, changed predictions,
nonfinite evidence and excess inference differences.

Record maximum numerical differences for every diagonal, not just a pass flag.
First characterize all20 diagonals before opening off-diagonal scientific
summaries. Any discrete or realized-cost change blocks the full claim; do not
select only the passing cells. The independent auditor must enforce the same
narrow numeric rule while regenerating physical outcomes and raw metrics.
Brier differences must be reported and remain exploratory; primary accuracy
and decision/cost criteria retain exact semantics. Initialization and training
cases remain bundled until a later matched initialization-by-data factorial.

DEV-026 execution is reserved for the next bounded batch. No more compute is
started in the initial about30minute batch after DEV-025's integrity failure.
