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
