# ME027 reproduction

Python 3.12, Torch 2.14.1+cpu, NumPy 2.3.5, pytest 9.1.1; Matplotlib 3.10.8 for
plots. Install with `pip install -e .`. The new Torch module is optional to the
base library. The fitting source, architecture and machine contract are frozen
before fitting; `results.json` records the exact commit and environment.

The delivered checkpoint archive excludes raw observations. Reproduction needs
the original ME022 twelve checkpoint files, manifest and authenticated snapshot
under `results/me022`. Source/labels cannot be silently replaced by another
cohort. All ME027 checkpoint/prediction/export digests are validated before
cached responses, then original ME022 hashes and canonical fingerprints.

After archive SHA-256 verification and extraction to `/workspace/me027-restored`:

```bash
python -m scripts.me027_audit --root /workspace/me027-restored --verify-only
python -m scripts.me027_runtime_probe --root /workspace/me027-restored
python -m scripts.me027_summarize --root /workspace/me027-restored --out /tmp/me027-summary
python -m pytest -q
```

The audit reconstructs optimizer steps, exposure and sampler draws; replays
every registered budget prediction and primary standalone export; recomputes
FIT feature means/interventions; and validates the retained latency medians.
`--verify-only` performs no new latency measurement or profiling and preserves
original `replay.json`. Its `verification.json` is an additional verification
record. The deterministic user bootstrap is recomputed. No new fit or reserved
outcome access occurs. Runtime probing uses fresh Torch-only processes and
four genuine inputs per model. It writes `runtime-probe.json` and makes no
architecture-package/research imports inside those subprocesses.

Original audit-only measurements are created with `--profile` and without
`--verify-only`. They contain nine batch128 inference repeats per checkpoint
and real FIT forward/backward profiler subtotals at step300/seed14001. No
optimizer step occurs in profiling; unsupported/opaque operations are retained
and full FLOP equivalence remains unestablished.

Repeating training requires a fresh output directory and is an additional set
of fits, not the original preregistered record:

```bash
python -m scripts.me027_state_research --out results/me027-separate-rerun
```

Every arm receives the same 50,000-step/180 charged CPU-second ceiling. The
registered run stops after both step300 and charged CPU20, independent of DEV.
First-crossing compute selection follows the frozen driver; no independent
per-step clock log exists, so it cannot be reconstructed as wall-clock truth
from checkpoint artifacts alone. Charged cumulative values and all budget
points are retained. Source authentication, shared preprocessing, setup,
optimizer, monitoring and whole-study costs have separate declared conventions.

The feature MLP reallocates capacity from sequence processing into a nonlinear
summary head; sequence models use linear additive state/common readouts.
Consequently its comparison is a practical state-allocation diagnostic, not
an isolated causal on/off intervention on the learned state. The same observable feature channels are present, but parameter placement/inductive bias differ.
No learned-state conclusion follows from a CPU score alone, particularly when
more rapid updates cause overfitting. The exposure and CPU lanes must both be
reported; no best-checkpoint selection is authorized.

Standalone exports are pinned TorchScript graphs. Upstream deprecates that
exporter; no replacement-exporter or cross-version guarantee is established.
Training checkpoints include all parameter/optimizer/RNG state. Persistent
state algebra, full-window inputs and training activations are distinct;
no new long-stream accuracy or KV-cache implementation is qualified.
