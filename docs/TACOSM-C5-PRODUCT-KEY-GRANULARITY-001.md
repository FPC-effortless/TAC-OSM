# TACOSM-C5-PRODUCT-KEY-GRANULARITY-001

Status: PRE-REGISTERED.

## Purpose

The fixed-K scaling experiment established that the factor-size-8 product-key
construction scores approximately 55%–56% of the persistent-state population
as M increases from 64 to 512. The final shortlist remains bounded at K=32,
but the over-fetched reranking set grows with M.

This experiment isolates the suspected cause: fixed cell granularity. It keeps
the teacher, query stream, factor beam, shortlist, state population and
downstream executor fixed while varying only factor size.

## Registered protocol

- M=128, 256, 512.
- H=256.
- Five seeds.
- K=32.
- Factor beam=6.
- Factor sizes=8,16,32.
- Eight cosine k-means iterations.
- Width-16 raw-trained cosine teacher.
- 512 teacher-training epochs with eight mean-gradient negatives.
- 48 training codes versus 16 held-out target codes.
- Training-only codebook fitting.
- Same exact downstream executor and query construction.

Each arm receives the same task stream. No target address or expected output is
used by codebook fitting.

## Mechanistic prediction

The existing factor-size-8 implementation creates 64 possible product cells.
The factor-beam-6 query considers at most 36 cells. As M increases, the average
number of states per cell grows, so the selected cell union becomes larger.

Larger factor sizes increase the number of possible cells:

- factor size 8 -> 64 cells;
- factor size 16 -> 256 cells;
- factor size 32 -> 1024 cells.

With the beam held at 6, this changes the candidate-cell resolution without
changing the query beam budget.

## Primary measurements

The primary retrieval endpoint is actual target-state Top-1 recall.

The scaling diagnostic is states scored / M. It is recorded independently from
the final K=32 shortlist size because the implementation may score an
over-fetched union before truncating to K.

## Interpretation boundary

A reduction in states scored / M would support the fixed-cell-occupancy
explanation for the previous scaling failure. It would not establish universal
sublinear retrieval, because the factor codebook cost itself changes with
factor size and the experiment remains on the registered synthetic workload.

A retrieval drop at larger factor size would be retained as evidence of the
coverage/granularity tradeoff rather than tuned away.

## Reproduction

Smoke: python scripts/run_c5_product_key_granularity_001.py --smoke

Full: python scripts/run_c5_product_key_granularity_001.py

Artifact: artifacts/TACOSM-C5-PRODUCT-KEY-GRANULARITY-001.json
