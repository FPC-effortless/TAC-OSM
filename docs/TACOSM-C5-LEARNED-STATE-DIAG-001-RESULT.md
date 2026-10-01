# TACOSM-C5-LEARNED-STATE-DIAG-001 RESULT

## Run provenance

- workflow: `36668376227`
- clean run: #2
- clean measurement head: `8fe72cfefea59303f8d321973b38dd74846ad568`
- precondition gate: passed
- full test suite: **894 passed**
- measurement step: passed
- artifact: `TACOSM-C5-LEARNED-STATE-DIAG-001`
- artifact id: `11076538524`
- artifact digest: `sha256:30bde43484f8a30dc6ea72364964efd6d10634e634b6dc1b0894736c49ce81f8`

## Registered comparison

The same 10 -> 8 dual linear encoder was evaluated three ways on M=64
persistent state items and 100 noisy held-out queries per seed:

1. random frozen continuous scoring;
2. trained continuous scoring;
3. the same trained encoder passed through 8-bit sign quantization, radius-1
   binary probing and K=1.

Training used 48 disjoint codes; evaluation used the remaining 16.

## Target-state recall

| Seed | No-learning continuous | Learned continuous | Learned binary |
|---:|---:|---:|---:|
| 0 | 0.00 | 0.20 | 0.26 |
| 1 | 0.00 | 0.25 | 0.14 |
| 2 | 0.01 | 0.21 | 0.25 |
| 3 | 0.01 | 0.23 | 0.14 |
| 4 | 0.00 | 0.18 | 0.06 |
| **Mean** | **0.004** | **0.214** | **0.170** |

The trained encoder improves held-out target recall by **0.210 absolute**
over the random continuous control.

The binary index is **0.044 absolute below** continuous learned retrieval
when averaged over seeds.

The registered binary-gap threshold was 0.20. The observed 0.044 gap is well
inside that threshold.

## Diagnostic decision

The registered diagnostic falls into the third branch:

- learned continuous retrieval is materially above the no-learning control;
- learned binary retrieval is within 0.20 absolute of learned continuous
  retrieval.

Therefore the C5-LEARNED-STATE-001 failure is **not primarily explained by
binary quantization/indexing** under this diagnostic.

The narrower finding is:

**The current semantic encoder learns a real held-out signal, but its
representation does not yet recover the target persistent state reliably
enough for selective C5 use.**

Binary quantization contributes some loss, but the dominant boundary is the
learned semantic representation/generalization itself.

## Retrieval rank

The learned continuous target rank means were:

| Seed | Mean target rank |
|---:|---:|
| 0 | 3.84 |
| 1 | 3.08 |
| 2 | 4.76 |
| 3 | 6.41 |
| 4 | 5.69 |

The correct item therefore tends to sit near the top of the 64-item pool, but
is not selected reliably enough for K=1 selective execution.

## Cost

Continuous learned scoring:

- query encoding = **80 MACs/query**;
- 64 cached state embeddings x 8-dimensional dot products = **512 MACs/query**;
- continuous total = **592 MACs/query**, excluding one-time state embedding.

Binary learned indexing:

- query encoding = **80 MACs/query**;
- radius-1 binary lookup = **9 probes/query**;
- state-index construction = **5,120 MACs/build**;
- amortized construction arithmetic over 100 queries = **51.2 MACs/query**.

The two arithmetic regimes are reported separately and are not claimed to be
hardware-equivalent.

## Scientific interpretation

This experiment changes the next engineering target.

The bottleneck is no longer best described as “the index cannot preserve the
semantic embedding.” The trained continuous representation itself is only
~0.214 Top-1 recall on the held-out noisy state codes.

That makes the next sensible intervention a representation-learning
experiment, not another indexing mechanism.

The current learner uses one positive and one deterministic negative per
update. The next measurement should therefore vary training budget and
negative coverage while keeping the held-out code split, query stream, state
pool, and continuous evaluation fixed.

## C5 status

Broad C5 remains **ungraded**.

Supported:

- a real learned semantic signal exists beyond the random continuous control;
- the binary index is not the dominant source of the current failure under the
  registered 0.20 diagnostic threshold.

Not established:

- reliable learned selective state retrieval;
- sublinear learned state addressing with capability parity;
- learned candidate retrieval;
- realistic end-to-end computation/cost parity;
- broad L4 C5.

## Reproduction

Measurement command:

`python scripts/run_c5_learned_state_diag_001.py`

Contract:

`contracts/TACOSM-C5-LEARNED-STATE-DIAG-001.json`
