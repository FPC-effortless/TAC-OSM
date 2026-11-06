# DECISION-SYS1-002 — causal adapter, censored targets and matched baseline engineering

Woyengi Intelligence Labs · 6 November 2026

## Result

**85/85 local engineering tests passed: 32 inherited plus 53 new. Zero optimizer
updates, zero new real-data predictions, no GPU execution and no model promotion.**
The original DECISION-SYS1-001 source and all historical checkpoints are unchanged.
This is a continuation of TAC-OSM draft PR717 at
`74ffe64447fd61bda146a726904ea40712ec3368`, not a completed training experiment.

## Implemented

- A 212-feature causal adapter: 105 named publisher measurements, 105 missingness
  flags, log elapsed time and log observed age. Anonymized operating units are NOT
  days. Only declared FIT vehicles can fit streaming normalization; means impute
  missing values. No future interpolation, label inputs or retrospective feedback.
  Per-entity clocks reject duplicate/reversed timestamps; unrelated traffic leaves
  the same entity's feature stream unchanged.
- Separate event/censor target construction and a stable masked hazard loss. A
  truck followed for eight units without repair supplies survival evidence through
  six units, NOT negative labels at 12/24/48. Partial-bin censor information is
  deliberately coarsened away; no within-bin timing distribution is invented.
  Event observations exactly at 6/12/24/48 are ineligible until publisher boundary
  inclusivity is independently established. Readings at/after repair are rejected.
- A two-layer 64-hidden-unit GRU and two-layer 64-wide/four-head causal Transformer.
  Transformer feed-forward width is 128. The Transformer has a true incremental
  key/value cache using the SAME weights as its full-sequence path; no history
  truncation. Cached/full outputs AND parameter gradients agree on the registered
  fixtures. These are untrained baseline candidates, not tuned or fully qualified
  competitors. The original native model is unchanged.
- An offline TRAIN-only source auditor: exact input filenames, historical readout
  SHA256, strict schema and chronological checks, identity split/example digest
  persisted BEFORE opening labels, unique outcome joins, at-risk exclusions, and
  explicit non-admission of training. It is tested on generated CSV fixtures only.
- A runnable no-fit source/count/runtime receipt generator. No trainer or optimizer
  is added, and no absent prerequisite can silently pass.

## Actual instantiated counts

| Candidate | Trainable/deployed parameters | Carried model state |
|---|---:|---|
| NativeRiskStateModel | 99,963 | 128 FP32 scalars = 512 tensor bytes/entity |
| GRURiskModel | 99,977 | 128 FP32 scalars = 512 tensor bytes/entity |
| TransformerRiskModel | 99,816 | KV cache grows by 1,024 FP32 tensor bytes/token/entity |

Maximum gap is 0.1612968%, below the 1% count limit. Counts alone do NOT establish
resource parity, equivalent optimization, intelligence or training admission.
State/cache bytes exclude weights, metadata, framework allocations, preprocessing
clocks, lookup storage and transient allocations. The ~100K target is study-local,
not a global size ceiling. None of these numbers is an inference speed benchmark.

The required Mamba-2 baseline remains **NOT instantiated/qualified**: mamba_ssm is
absent and this runtime has CPU PyTorch 2.10.0, no CUDA. Official Mamba-2 module
source reference was located at state-spaces/mamba commit
`6b72c122713bb769cc82c6b8e6d019c53d27d6a1`; this is a file-history anchor, not a
completed install/kernel/dependency/geometry qualification. No SSM surrogate was
substituted. A complete competition cannot be claimed with that arm missing.

## Historical exposure recovered

Recovered original GitHub Actions run39334730465, artifact12225908024,
`me001b-original-scania-train-internal`. The downloaded archive SHA256 is
`4f7afcc3038d3e8688cc5fb1f3a15ccb2321dc14ed3fe5de7e8bf23de969e219`, matching GitHub's
artifact digest. Its unmodified source/results receipts are preserved in the packet.

The receipt records 1,122,452 TRAIN readouts from 23,550 vehicles, with 18,848
training and 4,702 internal heldout vehicles; 24 features were used in forecasting.
Original TRAIN CSV SHA256:
`e01cb0bd87dfab4c9dbad215d51d81282fc0d413be96d6f819ea872fb7a3c715`.
The original native forecasting promotion failed; PR246 preserves that negative.
New TRAIN-internal partitions must not be advertised as an untouched scientific
TEST. This archive does NOT establish absence of later exposure in other studies.

No raw TRAIN CSV was recovered from that checkpoint/report artifact. The current
runtime could not fetch the publisher CSV; no bulk ZIP fallback was requested.
Thus **the new adapter has not run over real TRAIN rows**, and the new outcome
join/censor profile is still unmeasured. Publisher VALIDATION and TEST files were
not opened. Historical receipts are not a substitute for the new actual audit.

## Source semantics checked

Publisher documentation establishes the 105 measurement fields, irregular
operating-time readouts, first-repair versus censor interpretation, and four risk
windows at 6/12/24/48 anonymized units. The challenge PDF pages 1–3 were inspected,
including its class-window description. Exact-edge inclusivity remains unresolved.

- https://researchdata.se/en/catalogue/dataset/2024-34/3
- https://www.nature.com/articles/s41597-025-04802-6
- https://api.researchdata.se/dataset/2024-34/3/file/documentation?filePath=2024_IDA_challenge_v2.pdf
- https://github.com/FPC-effortless/TAC-OSM/actions/runs/39334730465
- https://github.com/FPC-effortless/TAC-OSM/pull/246

## Executed verification

Python3.13.5 / PyTorch2.10.0+cpu / one thread:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_decision_sys1*.py' -v
PYTHONPATH=src python scripts/preflight_decision_sys1_002.py
```

The 53 new tests cover schema/time guards, forbidden future fields, missing-value
handling, FIT-only transforms, independent NumPy moments and likelihood checks,
censor loss gradcheck, unused-horizon zero gradients, class-mass equivalence,
native adapter integration, Transformer causal and gradient/cache equivalence,
parameter counts, checkpoint/batch isolation and generated-CSV source guards.
These checks are authored in this session, NOT an independent scientific review,
whole-repository CI, full formal proof, authentic trained capability or calibration.

## Open before any fit

Recover only permitted original TRAIN files and execute the hash/schema/at-risk
checks; finish source/exposure and endpoint qualification; choose a label-independent
readout sampling and vehicle-weighting rule; instantiate and qualify authentic
Mamba-2 kernels; register equally budgeted tuning/training and actual compute;
finish source/math/canonical registry correspondence and obtain the repository's
independent source/evaluator/parity admission and compute plan. Model fitting and
TEST access remain blocked. No paid provider or autonomous process was started.

The native transition, correction candidate, and their original partial PNDS PR117
remain unchanged. New baseline whole-graph/Lean correspondence is OPEN; local
cache-gradient tests are not a substitute for this requirement.
