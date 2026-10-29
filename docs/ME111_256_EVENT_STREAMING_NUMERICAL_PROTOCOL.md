# ME111 256-event streaming portability and mathematical consistency protocol

**Status:** No new training, no altered original checkpoints, no new sensor data,
no physical device qualification, and no useful-memory superiority claim. Tests
are entirely constructed numerical diagnostics; the original ME111 real-data
joint quality/calibration gate remains **FAIL / MIXED_TRADEOFF**.

## Frozen source and graph

- Original Git-owned trained checkpoint: ME111 seed 111001, 1,536 updates, exactly
  104,663 trained parameters, SHA256
  `337b882e9a10631dc85700846ef2b15ca8908699499668186ad3a248dc108449`.
- Immutable original model and export recipe commit:
  `3fc2fbe8e4974491349caecd8d6171e738810b54`.
- Exported one-step ONNX opset18 graph SHA256:
  `dd65519a9e4f3388966538af53119b56e9292b46a9209d63ad4927b87f0a176b`.
- Real trained architecture: original ME111 conditional-belief source model. This
  experiment changes no learned model parameter or training condition.

## One-step mathematical contract

For one event `t`, the export specializes the unchanged native parallel
prefix scan to the exactly equivalent **single-element** affine equations

```text
z_t = a_t ⊙ z_(t−1) + b_t
P_t = α_t [I − β_t k_t k_tᵀ]                 # writable event
Q_t = β_t v_t k_tᵀ                          # writable event
M_t = M_(t−1) P_t + Q_t
```

On an unwritable event the native operator uses `P_t = I, Q_t = 0`,
so `M_t` is preserved. Source admission is likewise masked; unusable
observations must not increment `count`. For a valid model event, `last_hour`
becomes its source-aligned hour even when all observation channels are masked.

These equations describe the *same* learned one-step TAC computation as the
original graph. A binary `T=1` ONNX representation **does not** establish
equivalence for full parallel `T>1` batched scans, or a finite-precision proof.
FP32 effects and generated ONNX `expm1` decomposition are tested numerically.
There is no Lean proof for this translation.

This original ME111 model checks that source hours increase but does **not**
infer learned continuous elapsed-time dynamics from an hour gap alone.
A numerically successful 3-hour jump test shows only source monotonicity and
state consistency, **not** a new time-aware neural capability.

## Independent CPU differential test

Run after regenerating the original frozen graph from the exact pinned source:

```sh
python scripts/verify_me111_256step_cpu.py \
  --source-root pinned-me111 \
  --onnx /tmp/frozen-me111/ME111_ONE_STEP_EXPLORATORY.onnx \
  --receipt /tmp/me111-256step-host-parity.json
```

Two source owners (0, 5) each receive 256 constructed sequential events.
Some are all-masked; some omit all target channels, others have partial masks;
the source-hour increment is 3 at registered positions rather than always 1.
No real sensor events or targets are accessed. Original PyTorch and ONNX
each carry **their own** previously computed state to the next step; no
teacher-forced substitution of original state into ONNX. On all 512 steps the
test compares every scalar in the forecast and returned source/evidence/memory
state at `rtol=2e-4, atol=2e-5`, and integer/bool chronology/count exactly.
It also performs ten lossless NumPy binary snapshots/reloads and asserts
source-owner stream counts. CI preserves both PASS and FAIL receipts.

**Falsifier:** a numerical drift, inconsistent shape, missing graph input/output,
wrong count/timestamp, nonfinite state, changed source/model checksum or broken
binary snapshot must fail the test. Do not relax tolerances after looking at
results without a prospective, dated version.

## Android JNI long-stream test

The source-pinned `scripts/create_me111_android_256step_fixture.py` generates
the **same** two constructed 256-event streams, preserving all event inputs and
24 original PyTorch forecast/state snapshots at registered positions. The file
is packaged in the **Android instrumentation test APK**, not shipped as customer
sensor data in the regular app APK.

`Me111LongStreamAndroidTest` calls the real Android ONNX Runtime CPU
session for all 512 events. It compares 24 PyTorch snapshots across model
forecast, fast/slow evidence, count, hour and associative 8×8 matrix, and
reuses actual app-private serialization after every 32 events (16 full
FP32/int64/matrix exact restarts). The original shorter three-test JNI suite
remains included. A separate physical-only benchmark class is excluded.

Android snapshot comparison uses fixed
`abs(got - original) <= 3e-4 + 2e-4*abs(original)`.
This is an emulator CPU test, not real phone measurement. Neither host nor
emulator parity proves the model's **actual 256-event retrieval accuracy**,
long-distance reasoning, new independent-site predictive performance,
calibration, action safety or customer return on investment.

## Preserved gate boundaries

| Gate | Status until the exact associated CI result is inspected |
| --- | --- |
| Frozen original checkpoint and operator-coverage export | Previously established |
| 512 host CPU source-PyTorch/ONNX steps | Exact CI receipt required |
| 512 Android emulator JNI steps, 24 original snapshots, 16 state roundtrips | Exact CI test report required |
| Interrupted-write and corrupt-state rejection | Existing Android instrumentation test |
| Device process kill/relaunch with state continuity | NOT RUN |
| Actual Android physical-airplane-mode, p50/p95, peak RAM, energy | NOT RUN |
| Authentic independent sensor site / source authority / calibration | NOT QUALIFIED |
| Real-world 256-event useful memory | NOT ESTABLISHED |

The strict separation between **numerical portability** and **learned
intelligence** is mandatory. Test output cannot be used as a new predictive
capability or architecture-superiority score. Customer inference stays fully
offline and advisory only.
