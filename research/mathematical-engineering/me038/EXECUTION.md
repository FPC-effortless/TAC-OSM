# Executed model construction and verification

Core registration/source commit: 4ee2e56d, before numeric Tetouan parsing or fits.
Independent final-metric audit addendum: 2de9220d, before any test scoring.
Global selection/source freeze: 75d4dce0, before scoring all selected models.
Core model and trainer bytes remain identical to their registration; receipt in
execution-freeze.json. Reporting/export/replay helpers do not alter learned weights.

## Complete training, not a component study

All 192 exposure fits completed 500 updates; all 48 separate compute-budget fits
completed the registered deadline/selection procedure. Eight new Tetouan AR
parents completed 1,000 updates;40 legacy AR parents were verified and reused.
No failed/replacement fits. All initial predictions match their AR parent bit
exactly, with zero head correction; step0 is never selected. All optimizer/RNG
states, gradients/traces, endpoints and selected checkpoint hashes are retained.

The two lanes use identical initial parameters and sample-stream prefixes.
Their 48 matching traces replay 23,103 overlapping steps bit exactly. Compute-based
stopping changes the number of updates and is hardware/timing dependent:
**seeds alone do not promise identical weights on a new timed-training run**.
Original checkpoints are required for exact model restoration.

Charged training-CPU budget includes per-model construction, initial equality
check and optimizer updates. Checkpoint validation/serialization and worker
startup are separately recorded. Parent 1,000-step training and its validation are
attributed to every pipeline, while historical cached spend is deduplicated.
Pipeline cost fields are sums of measured components, not exhaustive end-to-end
CPU totals: shared CSV loading, final trace/report writing and some old parent
serialization are unmetered. No full FLOP/energy or end-to-end parity claim.
Five-second fits finish the deadline-crossing update; actual overshoot is recorded.
No sleeping or dummy computation was used to manufacture equal spend.

## What is verified

- All 1,008 parent/adaptation checkpoints replay 37,310,976 saved validation forecast
  values bit exactly. Initial AR equality covers 8,448,000 values. Optimizer clocks,
  original source hashes, sampler/RNG states and finite training traces are checked.
- The NumPy final-metric audit independently recomputes all 52 Tetouan selected
  models/persistence references: 1,479,493,632 test forecast values. Maximum metric
  discrepancy 3.8191672047105385e-14, within the preregistered float64 summation
  bound. This is aggregate verification, not saved bit-exact test-array replay.
  Legacy final test metrics are not independently recomputed in this audit.
- All 1,680 ME036 and 576 ME037 checkpoint hashes remain unchanged. Their earlier
  replay evidence is preserved; those entire older forecast matrices were not
  rerun. Historical ME035B precision failure remains unchanged.
- Local repository suite: 1,020 tests plus 12 subtests passed, 52 existing warnings.
  CI separately runs numerical/regression checks, not the full real-data fits.

Native state is 128 float32 values/channel (fast32,slow32,matrix64), compared with
ME037's 64. That is 512 recurrent-state bytes/channel before history, activation,
weight and output storage. No lifetime streaming or microsecond control claim.
All serving latency measurement happens after fitting/scoring, single thread,
with five warmups and 20 B1 samples. Eager arm comparisons are in resources.json;
TorchScript physical-model measurements are separate in runtime.json.
Supported profiler FLOPs omit unsupported operations and do not establish full
training/pipeline parity. All selected models use whole 96-step replay.

## Reproduce the original models without training

Verify archive SHA-256 from ARTIFACT.md before extraction. From the complete
archive's source root, with recorded Python 3.12.14/Torch2.14.1+cpu/NumPy2.3.5:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/me038_audit.py audit
```

The independent final-metric audit can also be rerun against the original CSVs
and checkpoint selections, without refitting or reselection:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/me038_metric_audit.py
```

Rerunning this audit changes CPU-time fields in its receipt; preserve the original
archived receipt separately. Neither command tunes models or opens new cohorts.
For the compact physical-input model archive, run `python verify.py`; no TAC
package import is required. TorchScript models expect raw Bx96xC input in data.json
channel order, return BxHxC in original units, and remove only training-anchor
buffers. Traced Python guards are not a dynamic shape validator; callers must
validate the 96-step input contract. B1 latency and B2 saved probes are exercised.
Float32 raw-unit roundtrip is compared against the same eager physical wrapper,
not declared equal to a separately standardized call. Preserve recorded warnings.
