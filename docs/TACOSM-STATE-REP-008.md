# TACOSM-STATE-REP-008

Status: PREREGISTERED — result pending.

## Purpose

REP-007 establishes that semantic state addressing works above the no-learning
control but pays O(M) because every state item is scanned.

REP-008 is a ceiling experiment. It asks what query-time addressing could cost
if a sufficient exact index already existed.

It deliberately does not learn an index.

## Design

For each registered M in {2,4,8,16,32}:

1. create one persistent state pool containing exactly one target semantic
   value;
2. build an exact inverted index from the stored semantic value to its opaque
   address once;
3. issue 256 repeated queries against that fixed pool;
4. compare indexed lookup with a full bilinear scan.

The query contains the semantic value and no address.

## Cost

Full scan:

`C_scan = 40 + 48M` MACs per query.

Exact index build:

`C_build = M` state reads.

Exact index query:

`C_query = 1` dictionary operation per query.

At M=32 and Q=256, the exact-index build cost amortizes to
`32/256 = 0.125` state reads/query, versus 1,576 MACs/query for the full
learned scan.

These terms are deliberately reported separately. The index is an upper-bound
ceiling, not a claim about a practical learned implementation.

## Primary endpoint

Both arms must preserve Top-1 state-address recall of 1.0.

## Interpretation boundary

Passing results would demonstrate only that the functional state-addressing
problem admits a constant-time exact lookup when its representation is directly
indexable.

It would not establish:

- learned semantic indexing;
- robustness to approximate queries;
- index maintenance under continuous writes;
- sublinear end-to-end computation;
- capability/computation parity for C5.

## Next architectural question

After the ceiling, replace the exact index with an approximate semantic index
whose shortlist K is smaller than M and whose recall is measured independently.
This is the mechanism required to connect semantic addressing to selective
computation rather than merely to exact hash lookup.