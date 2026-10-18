# DEV062-BATCH — same-weight batched public observation intake

Register before comparison. DEV062-CACHE preserved predictions but was1.51x slower
on inclusive first-seed CPU timing and is NOT promoted. Hypothesis: per-record
Python/gate/GRU dispatch dominates this bounded workload. Batch only observations
already available before a decision; never wait for future records or change the
choice decoder, weights, FIFO64 or original raw/public information.

Three variants: unchanged base; batched intake + full history recomputation;
batched intake + incremental GRU. Learned type gate runs once per available packet,
closed packets do not copy memory, admitted records preserve exact order. Cached
updates run one packed GRU over the admitted packet, fallback on FIFO eviction.
No latest-state rule, planner, posterior filter, repair/STOP override or learned
retention claim. This is execution engineering, not an architecture quality gain.

Before benchmarking, disjoint tests cover empty/padded/noise-only histories,
interleaved reads, partial packets and eviction. For EACH batched variant compare
all20TAC primary recordings and all4 fixed firstseed92601 arms' complete recorded
cells, float64/thread1, atol/rtol1e-8 and exact choices/receipts/costs/outcomes.
Repeat separately for conditional R1 seed92801 weights if fitted; no transfer of
numeric qualification to new weights. Independent600CPU-second audit per recipe,
no automatic extension; preserve failures/incomplete coverage.

Time fixed firstseed/all4arms,8worlds per H32/128/512, float32/64, identical forced
public traces. Include all available initial/receipt/terminal observations and all
decisions, and report retained tensor bytes. Timers exclude simulation/filter,
JSON parsing and tensor creation; not an end-to-end production SLA or DRAM counter.
Base-batch promotion needs>=10% less mean totalfloat64TAC model CPU than original;
cache-batch promotion additionally needs>=10% less than base-batch. Both require
full exact conformance. Timing namespaces6263001 (original),6283001(R1).
Keep the first failed cache evidence. No training/test tuning and no inference
quality, memory-traffic, C5 or commercial ROI claim from faster tensor scheduling.
