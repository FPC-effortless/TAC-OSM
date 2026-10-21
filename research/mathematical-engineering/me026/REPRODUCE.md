# ME026 reproduction

Environment: Python 3.12, Torch 2.14.1+cpu, NumPy 2.3.5, pytest 9.1.1.
Install the repository with `pip install -e .`. Plotting used Matplotlib 3.10.8.
The runtime remains an optional Torch module; base imports do not require Torch.
Training/evaluation source for ME025 remains frozen at its original commit;
ME026 runtime, benchmark and gates were pushed at `ecd7d97` before any timing.

The delivered archive contains an untouched `me025-original-workflow.zip` and
`runtime/` with every derived export, probability file, timing repeat and audit.
After extraction to `/workspace/me026-restored`, extract the original ZIP into
`/workspace/me026-restored/original`. It contains `results/me025/` and the
original execution log. Before extracting, verify the published SHA-256 and
all indexed payload hashes. The bundle excludes raw observations.

Recorded-only reproduction additionally needs original authenticated ME022
checkpoints, manifest and `data-snapshot.pt` under `results/me022`. Those
fingerprints and hashes are checked; do not substitute a source cohort.

```bash
python -m scripts.me026_runtime_audit \
  --root /workspace/me026-restored/runtime \
  --original /workspace/me026-restored/original/results/me025 \
  --archive /workspace/me026-restored/me025-original-workflow.zip
python -m pytest -q
```

This replays all nine original neural models and six native exports, checks
every original checkpoint/probability and new export hash before loading any
response, reconstructs timing medians from retained samples, and tests all six
exports in fresh Torch-only subprocesses on genuine inputs. It performs no new
fits, timing selection or reserved outcome access. `me025-replay.json` and
`runtime-replay.json` are overwritten with the recorded replay results.

A deliberate retiming run requires a new output directory and the same verified
original archive. Timing can vary with machine/load and must not replace the
registered record:

```bash
python -m scripts.me026_runtime_research --out results/me026-separate-retiming
```

Runtime API for each exported `.jit.pt`:

```python
model = torch.jit.load(path).eval()
state = torch.zeros(batch, model.state_width)  # 64 RMS-only; 74 kernel-native
with torch.no_grad():
    state = model.stream_step(observed_item, observed_metadata, state)
    logit = model.readout(state, next_item, next_duration, causal_prior32)
```

Maintain the last 32 actual responses separately for the prior. A new entity
requires its own reset state; never feed the offered item's response before
forecasting it. Equivalence is measured after 128 genuine historical events
from zero state. Longer continuous streams and resetting sliding windows are
different forecasting contracts and have no accuracy qualification here.

`kernel_segment` and `compose_kernel_segments` expose the exact affine
sufficient-statistic algebra. Their associativity does not apply to the
nonlinear learned fast/slow recurrence. No custom scan accelerator is supplied.
TorchScript is the pinned, tested runtime; upstream deprecates this API and a
replacement exporter has not been verified by this study.
