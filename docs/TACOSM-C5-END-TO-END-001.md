# TACOSM-C5-END-TO-END-001

Status: PREREGISTERED — result pending.

## Purpose

This experiment combines the two validated runtime boundaries into one
controlled system:

`persistent semantic state -> state address -> candidate retention -> execution`.

The state query is one-bit noisy. The target persistent item is hidden behind
an opaque address. After state retrieval, exactly four candidate programs are
relevant regardless of H.

The decisive difference from earlier selective tests is that the correct output
is the aggregate contribution of all four relevant programs. Executing only one
candidate is insufficient.

## Registered populations

- persistent state items M=64;
- candidate history H=64,128,256;
- relevant programs R=4 for every query;
- state-index shortlist = 1;
- candidate-index shortlist K=4;
- 100 repeated queries per seed/H cell;
- seeds 0..4.

## Arms

### Exhaustive

- scan all 64 persistent state items to recover the nearest state code;
- compare all H candidate descriptors with that state code;
- execute all H candidate programs;
- aggregate the H outputs.

### Selective

- use the Hamming-radius-2 state index to retain one state address;
- use the exact candidate index to retain the four matching programs;
- execute exactly four programs;
- aggregate the four outputs.

The same fixed-work executor is used in both arms.

## Registered execution cost

Each candidate program costs 12 fixed work units.

`W_exhaustive = 12H`.

`W_selective = 12R = 48`.

Expected execution work:

| H | Exhaustive work | Selective work | Reduction |
|---:|---:|---:|---:|
| 64 | 768 | 48 | 93.75% |
| 128 | 1,536 | 48 | 96.875% |
| 256 | 3,072 | 48 | 98.4375% |

The state-addressing and candidate-addressing terms are reported separately.
They are not folded into the execution-work units.

## Capability

The hidden evaluator computes the exact aggregate output from all relevant
programs. Both arms must match it for every registered query.

The selective arm must also retain exactly four relevant programs. Because the
candidate index is exact, this is a correctness test, not a recall estimate.

## Primary decision

The selective arm supports the bounded end-to-end mechanism only if capability
matches the exhaustive arm at every H and actual execution work remains 48 units
while H increases.

## Interpretation boundary

A passing result establishes only a synthetic end-to-end computation pattern:

`C_execute = O(R)` under a capability-matched exact selective runtime.

It does not establish learned semantic retrieval, natural-language reasoning,
long-horizon memory, hardware latency/FLOP superiority, or the broad L4 C5 claim.

C5 will remain ungraded as a broad program claim until the same decomposition is
tested with a learned/semantic addressing mechanism and a realistic workload.