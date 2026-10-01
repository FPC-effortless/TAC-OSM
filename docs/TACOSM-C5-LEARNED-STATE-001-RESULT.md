# TACOSM-C5-LEARNED-STATE-001 RESULT

## Run provenance

- workflow: `36668108313`
- confirmatory clean run: #4
- clean measurement head: `3dded5fa7c8fc219aac4ed3b335356dbb3715ccd`
- precondition gate: passed
- full test suite: **886 passed**
- measurement step: passed
- artifact: `TACOSM-C5-LEARNED-STATE-001`
- artifact id: `11076902563`
- artifact digest: `sha256:b00e56bf3570e31ec895be7a98bf5559d0a3d1228e2c7ec7590daf302fb992c4`

## Protocol amendment

The first registered run was quarantined after detecting that the query noise
bit depended on H. Amendment A1 was recorded before the confirmatory run.

The confirmatory runner now anchors query/state generation at H=64 and changes
only candidate population size with H. Evaluation target codes remain disjoint
from the 48 training codes.

## Direct retrieval result

The learned state index used:

- dual linear 10 -> 8 semantic encoder;
- 8-bit sign quantization;
- Hamming-radius-1 probing;
- shortlist K=1;
- 32 training epochs;
- M=64 persistent state items.

The direct endpoint is target-state retention. It is the least aliased
measurement of whether the learned index actually crosses the retrieval
boundary.

| Seed | H=64 target retention | H=128 target retention | H=256 target retention |
|---:|---:|---:|---:|
| 0 | 0.26 | 0.26 | 0.26 |
| 1 | 0.14 | 0.14 | 0.14 |
| 2 | 0.25 | 0.25 | 0.25 |
| 3 | 0.14 | 0.14 | 0.14 |
| 4 | 0.06 | 0.06 | 0.06 |

Across all cells, target-state retention is **0.1700**.

The learned index therefore fails to reliably identify the target persistent
state item from the held-out noisy query.

## Capability result

The exhaustive path is 1.0000 in every registered cell. Learned-selective
success is:

| H | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.38 | 0.33 | 0.44 | 0.29 | 0.15 | 0.318 |
| 128 | 0.56 | 0.55 | 0.64 | 0.52 | 0.29 | 0.512 |
| 256 | 0.94 | 0.99 | 0.95 | 0.93 | 0.93 | 0.948 |

The one-sided capability rule `learned_selective >= exhaustive - 0.05`
fails in 12 of 15 seed/H cells.

These capability numbers must not be read as state-addressing accuracy. The
downstream synthetic executor can produce the same aggregate output for some
wrong retrieved states. That aliasing is visible because target-state
retention remains fixed at 0.06--0.26 while capability rises sharply with H.
The interpretation order therefore makes direct state retrieval the load-
bearing endpoint.

## Cost

The learned index reports:

- query encoder arithmetic: **80 MACs/query**;
- state-index build arithmetic: **5,120 MACs/build**, or **51.2 MACs/query**
  amortized over each 100-query cell;
- binary lookup probes: **9/query**;
- mean bucket candidates: approximately **2.29--2.79**;
- mean shortlist size: **0.89--0.99**;
- exhaustive state scan: **640 bit-comparison operations/query**.

The 80-MAC query encoder and 51.2-MAC amortized build term are not converted
into the synthetic executor work units.

## Decision

**Negative for the registered learned state-index mechanism.**

The preregistered capability boundary is not preserved, and direct target-state
retention is only 17% pooled. The experiment therefore does not support
promotion of the learned coarse index as a valid replacement for the
hand-designed Hamming state index.

The result does **not** establish that learned semantic addressing is
impossible. It identifies a narrower failure boundary: the current dual
encoder plus binary sign quantization and radius-1 bucket probe does not
generalize reliably to the held-out noisy state codes.

## What remains valid

The following earlier findings remain unchanged:

- REP-006 supports bounded learned semantic addressing under a different,
  5-bit, exhaustive-pool protocol.
- REP-009 supports a hand-designed approximate Hamming index as a runtime
  mechanism.
- C5-END-TO-END-001 supports the bounded selective-execution mechanism with
  exact synthetic indexes.

This experiment does not retract any of those measurements.

## Next diagnostic

The next experiment should separate the learned representation from the
quantization/index boundary by evaluating the same trained encoder with a
continuous exhaustive scan. The key diagnostic is:

`continuous semantic top-1` vs `binary-index top-1`.

If continuous retrieval is high while binary retention is low, the failure is
primarily the coarse quantization/index. If both are low, the learned
representation itself is the limiting boundary.

## Reproduction

Measurement command:

`python scripts/run_c5_learned_state_001.py`

Contract:

`contracts/TACOSM-C5-LEARNED-STATE-001.json`
