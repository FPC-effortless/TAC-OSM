# DEV062-CACHE — same-weight incremental learned-state inference

Engineering registration before comparison, no fitting and no changed learned
policy. Preserve the original DEV062A model/export/evidence unchanged. Cache the
shared entity/global GRU states between admitted observations; retain FIFO64 and
rebuild after eviction. Read empty histories exactly as the original padded GRU.
Do not add latest-state tables, transition laws, repair rules or stopping overrides.
CDL/CASM, posterior/action heads and numeric choice decoder remain identical.

First qualify on disjoint mechanical tests (empty/noisy histories, padding,
interleaved receipts and overflow). Then compare every20TAC checkpoints on all
256primary rollouts and all four first-seed arms on validation/primary/secondary
recorded cells, source/checkpoint hashes fixed. Float64 thread1, per-value
atol/rtol1e-8, exact actions/receipts/costs/outcomes. Keep firstseed92601, not best.
No tolerance/decoder adjustment or weight tuning after parity failures. A failed
optimization is retained and not promoted. New audit cap600CPU seconds; incomplete
coverage forbids qualification, no automatic extension.

Measure baseline versus cache on fixed firstseed/all4arms, eight public worlds per
H32/128/512, bothfloat32/64. Charge every observation and decision call, report
per-call CPU and total episode model CPU separately; keep world/filter work out of
model timers. Require >=10% lower mean total model CPU in float64 TAC and exact
registered replay before recommending cache. Store state bytes and transient work
limitations; a cache can use more RAM and remain O(history) for observation intake.
No C5, physical-memory-traffic or production SLA claim. Qualification of equal
inference does not repair an unreliable learned policy or establish commercial ROI.
