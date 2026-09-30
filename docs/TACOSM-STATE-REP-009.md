# TACOSM-STATE-REP-009

Status: PREREGISTERED — result pending.

## Purpose

REP-008 showed that an exact semantic hash can make query-time state lookup
constant, but that task was favorable to exact hashing because query and index
keys were identical.

REP-009 introduces controlled one-bit query noise and a Hamming-radius-2
approximate index. The index must return a small shortlist before the same
learned state scorer acts.

## Task construction

Persistent state values are 10-bit binary codewords selected from a fixed
64-code Hamming-distance-3 codebook.

Each evaluation query is created by flipping exactly one bit of a target state
code. The target address remains opaque and is not present in the query.

M in {8,16,32,64} state items are stored in one fixed pool per seed/level.
Each evaluation block issues 256 target queries against that same pool, so the
one-time index build can be amortized across many queries.

## Index

The Hamming index probes every code within Hamming distance 2 of the query.
For 10-bit codes this is 1 + 10 + 45 = 56 bucket probes.

The retained shortlist is capped at K=4 and sorted by Hamming distance then
opaque address.

The target is guaranteed by task construction to be within the radius-2 query
neighborhood; whether it survives the K cap is measured rather than assumed.

## Capability test

The learned addressor is the same 10->8 bilinear matcher in both learned arms.

- learned_full scores all M state items;
- learned_indexed scores only the returned shortlist.

The weights are trained on a separate pool from the same code distribution using
512 verified-success episodes.

A fresh frozen addressor supplies a no-learning reference.

An identity encoder supplies the exact representability witness.

## Registered cost model

Full scorer:

`C_full = 80 + 88M MACs`

Indexed scorer:

`C_indexed_route = 80 + 88K MACs`, with K <= 4.

Index query cost:

`C_index_query = 56 bucket probes`.

Index construction:

`C_index_build = M state reads`.

At M=64, full scorer cost is 5,712 MACs/query. At K=4 the indexed scorer is
432 MACs/query before the fixed 56 bucket probes.

The index can therefore reduce learned-scorer arithmetic substantially, but the
experiment does not translate dictionary probes into FLOPs.

## Primary endpoint

Indexed learned Top-1 retrieval must remain within 0.05 absolute of full learned
Top-1 retrieval and the index must retain at least 95% of target states.

Mean shortlist size must remain strictly below M.

## Interpretation boundary

A positive result establishes only that a fixed approximate index preserves the
current learned scorer's capability on this synthetic noisy-code task while
reducing scorer population.

It does not establish learned semantic indexing, asymptotic sublinear end-to-end
computation, long-horizon memory, natural-language retrieval, or C5.

## Research value

REP-009 is the first experiment in this ladder that measures a real intervention
between state addressing and the scorer while holding the scorer itself fixed.
It therefore isolates whether the cost can move from:

`scan M -> score M`

to:

`cheap index -> retain K -> score K`.

C5 remains untested until this pattern is embedded in a matched capability-vs-
compute experiment with execution work and a full-history baseline.