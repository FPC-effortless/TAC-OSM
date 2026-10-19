# DEV067 — complete execution, failed individual calibration gate

**NO PROMOTION.** All 120 fresh candidate fits and 12 pilots completed within the
registered budget:5904.586 fitting/preparation/pilot CPU seconds. These comprise
60 fresh encoder fits and 6 pilot encoder fits; four TAC heads share each final TAC
encoder. Every encoder and actor received 4096 updates. Tests opened only after all
fresh fits completed, with a permanent exposure marker written first.

| Arm | Seeds passing each completion stratum | Seeds passing both calibration tails | Primary / secondary final faults | Primary mean cost |
|---|---:|---:|---:|---:|
| TAC moving / probability |20/20|16/20|0 /0|5.38807|
| TAC moving / log |20/20|16/20|0 /0|5.41442|
| TAC frozen / probability |20/20|16/20|0 /0|5.38953|
| **TAC frozen / log, primary** |20/20|16/20|0 /0|5.40984|
| **GRU frozen / log, validation-selected candidate** |20/20|9/20|0 /0|5.41446|
| No-CASM frozen / log |20/20|16/20|0 /0|5.41352|

Every arm achieved100% completion on every primary and secondary stratum across
all 20 seeds. The calibration rule required at least18individual seeds to satisfy
BOTH fixed-query p95 KL≤.05 and p99 KL≤.1. Neither the primary nor the preselected
candidate passes. No other arm is substituted after test inspection.

The primary has median p95 KL.01248 and median p99 KL.06934, but seeds96701,96703,
96711,96713 fail the p99 threshold (.1291,.1010,.1262,.1397). Its favorable median
does not qualify the recipe. All four TAC heads have exactly identical posterior
metrics and encoder hashes by design. Their16passing encoders are not64independent
successes. GRU's median p99 KL is .10116; no-CASM's is .07551.

## The registered mechanism hypotheses did not earn promotion

All completion contrasts are zero: the task is saturated. The required5percentage-
point completion gain is absent. Freezing's cost effect is−.00156, ordinary paired
seed bootstrap95interval[−.00478,+.00182], Holm p = 1.0. Log coordinates INCREASE cost
by .02333,[+.01300,+.03446], Holm p = .00266 across all six contrasts. The cost interaction
is inconclusive. These are20initializations on shared physical starts, not a
population estimate over independent real factories.

Primary minus GRU cost is−.00462,[−.01393,+.00420]; primary minus no-CASM is−.00368,
[−.01561,+.00889]. Noninferiority at the registered+.02 margin passes; superiority,
equivalence and CASM necessity do not follow. Common changes to data, precision,
losses, module clipping and training budget are confounded relative to DEV065.
The reliable execution improvement cannot be causally credited to freezing/logs.

On these same starts the known-law public reference completes100% at 5.47016 cost;
the privileged reference completes100% at 4.86031. They are bounded reference
policies, not proven globally optimal POMDP solvers. The public shortcut completes
68.75%with 112 final fault bits. Fixed/public/privileged cost gaps and all individual
metrics remain in statistics.json; no real economic saving follows.

## Final repair does not mean harmless execution

The primary completes5120primary rollouts with zero final faults and no missing
STOP. During those trajectories it makes1969repairs on already healthy target
entities and creates671new fault bits, subsequently repaired. The public reference
creates36new fault bits in 256 shared-start rollouts; privileged creates0. These
are action-level state changes, not counts of distinct independent machine failures.
Transient exposure and stopping precision/recall are retained separately.

Only22primary non-STOP decisions occur when the PUBLIC posterior is already
certain the world is healthy. Physically healthy actions under uncertain public
beliefs should not all be labeled stop blindness. No missing-STOP failure remains
on this primary test family; that does not certify unfamiliar worlds.

## Evidence and limitations

Local replay passed 132 checkpoints,54528 recorded trajectories,9514 native Veritas
actions,132 semantic-interface checks,132 batched-state checks and 264 clock
transformations. All28,442,346 compared numeric values match exactly on this host.
The first-seed portable adapters reproduce22logical decisions each. A separate
two-part smoke replay covers all 12excluded smoke checkpoints without overlap.
All 1477 regression tests pass. Remote replay is pending the new final-head checks;
parent PR225's seven successful checks do not certify this new evidence.

Fresh closed-loop predicted/exact/blurred substitutions completed92160mode-world
rollouts across 120 checkpoints in 671.291 CPU seconds. The primary and validation
candidate each pass every predicted/exact stratum for all 20 seeds, with 100%
completion and zero final faults. Primary exact substitution raises cost by .04854,
ordinary paired-seed95interval[+.04260,+.05448], rather than repairing the actor.
The intentionally blurred primary completes0% and leaves3014fault bits; those
damaged-model failures are not failures of the unmodified agent.

Fixed-query primary substitutions produce0identity flips,1010exact-posterior
flips and 6877 same-public-graph cyclic-shuffle flips across 10240 queries. The
shuffled donors have distinct public posteriors for7780queries;6633 of those flip.
This is semantic sensitivity, not execution regret or CASM necessity. Every
control is also evaluated. Exact beliefs are public-reference diagnostics and
never enter the exported controller.

One-host first-seed H1024/batch1 measurements give3.65msprimary decision CPU time
versus2.57ms GRU and 3.37 ms no-CASM; sequential ingestion costs39.51ms,27.26 msand
36.04 msrespectively. Primary parameter storage is 187568 bytes; retained unique
tensor storage23872bytes. All48timing cases are published. These are narrow
process measurements, not DRAM/FLOP/energy, C5, production-SLA or universal
hardware claims. The prior batching speedup is not reasserted for this study.

The full archive keeps
every raw/final EMA checkpoint, validation-only pilot, training-source provenance,
curve and negative calibration result. Both adapters export the predeclared FIRST
fresh seed 96701, which fails primary and GRU tail qualification; no best-seed swap.

Two entities, five choices, typed46fields, supervised admission, FIFO64, public
random16-vector identity keys/cosine binding,12 health×graph classes and known-law
public-posterior teachers remain structural priors. TAC uses two shared-weight
entity-ordered GRU passes over admitted records, attention and a dense mixture of
three neural MLP operators. It is not independently isolated per-entity writes,
symbolic CASM program discovery, sparse operator execution, C5 retrieval or
autonomous hidden-physics discovery. No analytical repair/STOP controller ships
inside the learned runtime.

Prior studies, failed checkpoints, DEV058export and MetroPT July B remain intact.
The overall maintenance-model goal is ACTIVE and NOT COMMERCIAL-QUALIFIED.
