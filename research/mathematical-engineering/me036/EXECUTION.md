# Execution notes

Original source/protocol freeze458c319f preceded all fits. Continuation/source
freeze9bb60592 preceded all continuation fitting and test scoring.

The container has2CPU quota units (`cpu.max`200000/100000). Every fit remains
single-threaded. A second single-thread process handles the independent Electricity
cohort while the coordinator handles ETT/Weather. The coordinator will be paused
before it enters Electricity, then resumed after that worker exits; it reuses only
completed fit records. No fit is duplicated or shares an in-progress checkpoint.
Optimizer/sampler functions, numerical settings, example counts, CPU ceilings and
selection rules are unchanged. Actual processCPU is charged per fit; resource
concurrency is2CPU processes, not a claim of exclusive physical-core allocation.
This is execution scheduling, not a new hyperparameter trial. No test scoring until
all models and ridge coefficients are frozen. Single-worker initial stage preserved.

After Weather completes, its released CPU handles Electricity H720 independently
while the first Electricity worker handles H192/H336. A guard pauses the first
worker before H720 if needed until H720's last fit finishes. All H720 fit records
are then reused; ridge coefficients are fitted only by the first worker. Maximum
concurrent fit workers remains2, with no duplicated fit or modified optimizer path.

After all selections and ridge coefficients freeze, the final evaluator verifies
ALL endpoint/coefficient hashes and executes one full test pass per model/dataset.
Independent datasets use at most2single-thread inference workers. Aggregation and
stride1 evaluation use the original registered scoring function unchanged. Saved
per-dataset completion receipts prevent accidental re-scoring on resume. Runtime
latency exports run afterward without concurrent training/evaluation workers.
