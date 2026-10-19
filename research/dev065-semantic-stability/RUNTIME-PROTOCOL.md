# DEV065 runtime batching — independent, weight-preserving experiment

Registered before runtime benchmarking on fresh main weights. This does not
change the frozen DEV065 training recipe, weights, gates or test selection.

Hypothesis: batching typed admission and FIFO ingestion reduces actual CPU time
compared with record-by-record `observe`, preserving exactly the same retained
events, presence mask, posterior, scores and actions. Model quality is unchanged.

Scope: valid one-hot source6 public events; no handwritten signal/noise rule.
Every admission is still evaluated by trained gate weights. Non-one-hot types
are rejected by this typed optimized adapter rather than silently rounded.
Missing/padded records use a presence mask. Batch sizes1/16, histories32/128/512/
1024, predeclared first seed96501, all six arms. First three warmups excluded;
twenty measured repetitions per case. Report CPU and wall times per batch,
end-to-end observation work, parameter bytes, retained tensor/storage bytes,
additional raw-event tensor bytes and the fact that every input is still scanned.
No physical DRAM, FLOPs, energy, C5 indexing or general learned selectivity claim.

Parity gate: exact buffer, presence and final output equality for every restored
checkpoint on a disjoint H128 public-input context; unit tests cover overflow,
multiple chunks, mixed masks, all-rejected/empty chunks and invalid type handling.
Portable first-seed sequential trajectories must match the original model's
record-by-record expected outputs. Only valid schema inputs are covered.

Engineering promotion requires all parity checks plus a median CPU ingestion
speedup ≥1.2 over the48 first-seed cases, and no case slower by more than10%.
Failure keeps the original adapter; do not change weights or remove bad cases.
One host is a narrow profile, not a production SLA or universal hardware claim.
Runtime profiling≤120CPU seconds, excluded from model fitting. The main study's
scientific and commercial gates remain independent of this optimization.
