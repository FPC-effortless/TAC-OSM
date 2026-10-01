# TACOSM-C5-FACTOR16-BEAM-7-8-FRONTIER-001 — Result

Status: MEASURED. Beam 7 is ELIGIBLE under the preregistered rule; beam 8 is NOT ELIGIBLE.

## Run and artifact integrity

- GitHub Actions run: 36810637731
- Full artifact: 11138988228
- Artifact SHA-256: 7888514cd45383c680c58e483a49f6adaccf14ec4bf66bcee120d614765e229a
- Artifact size: 5,085 bytes
- Dispatch head: 15c463b0ab10750bece76e0dd8bb95e8e6217cc0
- Protocol: H=256, M={128,256,512}, K=32, seeds=10-19, 100 tasks per seed/M/arm.

The artifact contains 60 cells and the pooled endpoint is count-conserving over 1,000
tasks at each M/beam combination.

## Primary result

| M | Beam 7 selective/exhaustive | Retention | Beam 8 selective/exhaustive | Retention |
|---|---:|---:|---:|---:|
| 128 | 348/386 | 0.901554 | 359/386 | 0.930052 |
| 256 | 151/166 | 0.909639 | 149/166 | 0.897590 |
| 512 | 48/53 | 0.905660 | 49/53 | 0.924528 |

Beam 7 has minimum retention 0.901554 and therefore clears the preregistered 0.90
floor at all three M levels. Beam 8 fails at M=256 with retention 0.897590.

The beam-7 margin at M=128 is only 348 - (0.90 × 386) = 0.6 task. This is a
boundary-level pass, not evidence of a large capability margin. Beam 8's
non-monotonic M=256 failure reinforces that the result should not be interpreted
as monotonic beam scaling.

## Computation result

Pooled states_scored_over_M:

- beam 7: 0.106148, 0.108602, 0.108545 for M=128,256,512; mean = 0.107765.
- beam 8: 0.147742, 0.149520, 0.149576; mean = 0.148946.

The beam-7 mean is 1.432x below the preregistered 0.1542864583 reference
(0.1542864583 / 0.107765). Beam 8 is also below the reference on mean reranking
fraction, but is ineligible because it misses the capability floor.

Absolute reranking counts for beam 7 rise approximately with M:
13.587, 27.802, and 55.575 states per task. Pair-generation work is 343
operations per task; beam 8 uses 512. This is an arithmetic/topology audit, not
an asymptotic claim.

## Admission versus post-admission selection

Beam 7 pooled proposal retention is 0.794, 0.762, and 0.786 at M=128,256,512.
Using the pooled proposal and selective counts, conditional selection given admission is
348/794 = 0.438, 151/762 = 0.198, and 48/786 = 0.061 respectively.

The registered result therefore does not isolate a single bottleneck: both
admission and post-admission selection contribute to end-to-end loss.

## Interpretation boundary

This is a narrow workload-specific frontier result under the preregistered rule.
It does not establish C5, sublinear scaling, universal beam optimality, semantic
retrieval guarantees, or hardware speedup.

The flat scored/M fractions show the current factorized mechanism remains
approximately linear in M with an improved constant. C5 therefore remains
UNTESTED.

## Reproduction

Full artifact:
TACOSM-C5-FACTOR16-BEAM-7-8-FRONTIER-001.json

Artifact ID: 11138988228.
