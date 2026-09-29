# TACOSM-TEMPORAL-001 — Temporal persistence

**Status:** pre-registered; no confirmatory result yet.

## Question

When a value is written at time t and addressed only through persistent state,
does the system use that value correctly after k intervening decision
boundaries?

The experiment targets the temporal part of C1. The older v0 result did not
enforce this boundary because task construction could write the target and
read it immediately.

## Registered design

* H = 64 candidates.
* Seeds = 0, 1, 2, 3, 4.
* Delays k = 1, 2, 4, 8, 16, 32.
* 100 independent probes per seed/arm/delay.
* Arms:
  * 'carry' — the world write remains readable at t+k.
  * 'reset' — readable state is cleared immediately before t+k.
  * 'corrupt' — one stored bit is flipped immediately before t+k.
* The router is a deterministic state-conditioned equality control. It reads
  only the query's public state address and the value returned by persistent
  state.
* The hidden reference used to judge the task remains environment-side. It is
  never passed to the router or retrieval layer.
* Learning writes and repair are disabled for the measurement so no unregistered
  state transition can contaminate the delay.

## Primary endpoint

'decision_success' at the read boundary.

Secondary endpoints are 'read_available', 'router_candidates', and
'verification_passed'.

## Boundary gate

For every seed and delay, the benchmark executes all intervening boundaries and
asserts that the addressed value is still unreadable at t+k-1. A failure voids
the run.

This is a causal gate, not a capability result.

## Interpretation

The registered reading is:

    temporal boundary
        -> state availability
        -> decision capability
        -> verification

A successful carry arm at a given delay supports the causal contribution of the
persistent state under this synthetic control at that delay. It does not
establish long-horizon memory generally.

A failure under carry establishes only that the tested mechanism did not
preserve useful information at that delay. It does not establish that all
persistent-memory architectures fail.

Reset and corruption are intervention controls. They do not constitute
independent models.

## Leakage boundary

The router sees:

    Query
    persistent state read

The router does not see:

    acceptable_actions
    target_action
    hidden reference
    outcome
    verifier result

The environment/executor may access hidden truth after routing, because that is
the observation/evaluation boundary rather than a router input.

## Reproduction

The machine-readable contract is
'contracts/TACOSM-TEMPORAL-001.json'.

The instrument is:

    python scripts/measure_temporal_persistence.py

A '--smoke' invocation is explicitly non-result and records its contract
deviations.
## Amendment A1 — filler workload

The first registered run exposed excessive runtime because every intervening
filler boundary regenerated a full H=64 candidate set. No result artifact was
produced under that version.

Before the confirmatory result, A1 changes only the intervening filler
workload: filler decisions use one candidate, while the actual read boundary
continues to use H=64. The write/read delay, arms, seeds, primary endpoint,
router-visible information, and hidden-truth boundary are unchanged.

The purpose is to enforce the causal sequence without allowing non-measured
filler generation cost to dominate the experiment. A1 is recorded in the
machine-readable contract and applies before the result run.
