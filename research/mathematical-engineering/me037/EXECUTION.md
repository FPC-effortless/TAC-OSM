# ME037 execution and reproduction

Prediction source/registration: `0f0c0d9e`. All selections and source hashes frozen
at `3492ff2e` before test scoring. Delivery source snapshot is recorded in
archive.json. Parent AR models use validation selection; only positive adaptation
endpoints 100/250/500 can be selected. No retuning or fit replacement after test.

192 new adaptation fits, eight new AR parents, no failed trajectories. Adaptation
consumed 609.969 process-CPU seconds; new-house parents 7.781 seconds. Cached
legacy parent training consumed 58.464 seconds historically, deduplicated; each
pipeline's manifest attributes its parent's full pretraining cost. Reusing one
parent for four arms does not physically incur that cost four more times.

Mean process-CPU seconds for 500 adaptation updates (48 fits per arm):

| Arm | Mean CPU seconds |
|---|---:|
| Native + context | 5.4552 |
| Matched context-only | 2.1152 |
| Matched GRU + context | 4.3086 |
| AR-only adaptation | 0.8287 |

These are downstream training measurements, not streaming inference latency.
Single-thread workers, at most two concurrent, same AMD EPYC environment under
2-CPU quota. Parameters/data/update exposure match across the three neural arms;
CPU/FLOPs do not. No architectural-superiority claim. Runtime exports separately
record 20 B1 full-window latency samples after five warmups with no training or
scoring concurrent; those measurements are platform/implementation specific.
Window normalization and native encoding still replay all 96 observations.

Final scoring covers 20,611,906,560 neural scalar forecasts using all eligible
origins and complete target sequences. Those arrays are aggregated, not all
stored. Exact replay receipts refer to saved validation probes: 37,062,144 values
from 864 endpoints, plus 8,236,032 initial-parent equality values. Standard/raw
exports replay 4,118,016 values; compact standalone restoration replays 2,059,008
physical-unit values. All source/test/role boundaries are in PROTOCOL.md.

## Verify original weights without training

The complete archive includes the original 288 parent and 576 adaptation
checkpoints, optimizer/RNG states, learning traces, checkpoint probes, model
exports, final score receipts, raw source CSVs, source hashes and publisher
attribution/license records. It excludes unrelated old ME036 trajectories.
Verify the SHA-256 in ARTIFACT.md before extracting. Then from its source root:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/me037_audit.py audit
```

This restores original checkpoints and verifies probes; it does not fit models,
rescore the entire test matrix or revisit selection. Every indexed archive
payload is independently hash checked during the recorded restoration.
The compact archive requires only Torch and NumPy:

```sh
python verify.py
```

Use the recorded Python 3.12.14 / Torch 2.14.1+cpu / NumPy 2.3.5 environment for
the same-CPU bit-exact contract. Raw input order and training-only preprocessing
are in data.json. Exports remove only training-anchor buffers, retaining all
learned prediction weights. No TAC package import is needed for physical models.

The standalone models are traced for the 96-step contract; shape guards in Python
are not a dynamic input validator after tracing. Validate B x 96 x C with the
recorded channel count before calling. B1 serving and B2 replay were exercised.
TorchScript deprecation and trace warnings remain in export.log.
