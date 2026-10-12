# Retention protection DEV-005: matched development experiment

DEV-004 implicated memory competition and age: exact-address access barely
helped, isolated-target decoding was strong, and post-commit carry fell to
chance after strong competing writes. This experiment tests an actual learned
protection mechanism against an always-update control.

The complete development protocol is committed before model training. It is
not scientific preregistration or confirmatory evaluation. Earlier models,
experiments and results remain untouched.

## Mechanism

Each expert receives a learned scalar gate on its complete proposed delta-memory
transition, including forgetting/decay. The gate uses the current address/value,
that expert's existing read and read norm, and an expert identity indicator.
It cannot observe the later query, task role, history boundary, target position,
latent identity, label or future outcome. Recurrent state remains 768 scalars.

The always-update arm uses the same architecture and identical initial tensors,
but forces the transition gate to one and freezes its unused gate parameters.
All remaining parameters receive the same training batches and hyperparameters.
A third arm uses a constant sigmoid(2) gate, equal to the learned arm's initial
gate strength, with unused gate parameters frozen. This control was added before
any DEV-005 training or measurement to separate adaptive learning from uniform
slowing of updates. All three arms begin with identical parameter tensors.
The protection arm therefore has more actively trained parameters and extra
gate computation; this is not a parameter-count or FLOP-matched SOTA comparison.
The verifier is unused/frozen in this mechanism experiment. It does not inherit
the previously calibrated threshold or measure full verification/repair.

## Preventing the trivial freeze solution

Half the paired tasks query an earlier history item (retention); half query a
new intervening write (acquisition). Both use independent noisy views of hidden
latent addresses and the same visible event format. A model that simply refuses
new writes must fail acquisition, so report both endpoints and their tradeoff.

The generator exposes no latent identities or target metadata to either model.
Its independent latent generator remains development-family synthetic work,
and near-collision queries can be intrinsically ambiguous. Dense Euclidean
matching supplies an observable-only comparator, not a Bayes-optimal assertion.

Opposite-past partners differ in only the queried payload. Current query and
current bit are equal, and all other observations are equal. Reset has an exact
50% ceiling. Ignoring new writes also has a 50% ceiling for acquisition. That
control uses the evaluator's history boundary; it is not a learned runtime arm.

## Measurement

Three paired model seeds, 240 steps each, batch size eight. Training histories
are 32/128 with 4/16 intervening writes. Evaluation includes H=512 and D=32
length extrapolation, both task roles, two noise levels and unrelated/near-address
competitors. All arms receive exactly the same data and fixed training budget.

Raw evidence includes all per-pair predictions and controls, latent-free input
fingerprints, target age/role only on the evaluator side, actual per-write gate
traces, parameter/gradient diagnostics, checkpoints and source hashes. Replay
reloads both arms and regenerates every evaluation pair. Descriptive seed-level
intervals use three paired seeds and are exploratory.

Gate traces grow with sequence length for auditing; a fixed recurrent state is
not a claim of fixed total profiling memory. History ingestion remains linear
in the number of events. Report all negative findings and acquisition losses.
