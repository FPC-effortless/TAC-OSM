# Original trained models and offline architecture-development evidence

Local delivery: `/workspace/me035-native-architecture-models-and-evidence.tar.gz`

- **25148524bytes**, SHA256
  **81bf3d7f00a62617a41d7c17c2a3eb9b6df55beb8d982ffeabd6f47299cf2c2a**.
- **853indexed payloads**, plus `ARCHIVE-INDEX.json`.
- Source snapshot `a23f996933d892e1265781a6af3de2c64a1d9244`.
- Original first-round freeze77e5ce19; registered refinement freezea91767b7.
- Local workspace archive, not a GitHub Actions training artifact.

The archive contains all51original endpoint checkpoints with Adam/sampler/model
RNG, original scripted models, forecasts, traces, ridge coefficients, original
unmodified source CSV/README/license and complete repository source for offline
replay. No original weights are recreated by retraining. It also includes the
3selected raw-input models, all development selection records and the preserved
strict initialization precision failure. No normalized/modified source dataset
is distributed: normalization occurs inside the model/reproduction code.

Publisher ETDataset license is CC BY-ND4.0, attributed to Zhou et al., Informer,
AAAI2021 and the ETDataset project, measurements collected with Beijing Guowang
Fuda. Original ETTh1 CSV checksum
f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066.
See the unmodified source LICENSE/README alongside the CSV in `results/me035/`.

## Load a selected trained model

Selected checkpoint: ME035B,25new updates after its250-update native parent.
All32154coefficients were trainable. The exported models contain their
training-only global normalization and output inverse scaling.

```
results/me035b/raw-ar-native-s35001.jit.pt
results/me035b/raw-ar-native-s35002.jit.pt
results/me035b/raw-ar-native-s35003.jit.pt
```

```python
import torch
model = torch.jit.load("results/me035b/raw-ar-native-s35001.jit.pt").eval()
# history: float32 [batch,96,7], preceding hourly readings in source units.
# Columns: HUFL,HULL,MUFL,MULL,LUFL,LULL,OT. No future measurements.
with torch.inference_mode():
    forecast = model(history)
# forecast: [batch,3,7], source units, horizon axis1/6/24hours.
```

No `tac_osm` Python imports are needed for these exported models. PyTorch is
required. Recorded environment: Python3.12,Torch2.14.1+cpu,NumPy2.3.5;
pytest9.1.1 for repository tests. These are individual seed models. The headline
meanMSE averages metrics across allthree, not one selected seed or a served
ensemble. Per-seed selected MSE:0.370058874,0.367738522,0.369521817.

## Offline replay

Check archive SHA256, extract to a fresh directory, then verify each file against
`ARCHIVE-INDEX.json` before loading model files. From the extracted root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/src" \
  python -m scripts.me035_audit --out "$PWD/results/me035" --quick
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/src" \
  python -m scripts.me035b_audit --out "$PWD/results/me035b" --quick
```

Both commands use only packaged source/data/model files and installed libraries;
no originating workspace results or network access is needed. They verify
frozen/final hashes before source values, reconstruct weights/optimizer clocks/
sampling RNG, rebuild input/target windows, reproduce509796scalar forecasts and
development selections, and verify source/parent/ridge provenance. Full audit
without `--quick` also checks fresh Torch-only processes and first-round resources.

The second audit intentionally retains
`REPLAY_PASS_INITIAL_EQUIVALENCE_TOLERANCE_FAIL`: the float32 initial linear path
exceeds registered1e-6agreement with float64ridge for3values/seed, max1.1409247e-6.
All trained forecast replay checks pass. Do not convert this preserved failure
into an all-checks-pass claim. See [results](RESULTS.md) and the initial failure log.

Independent extraction/replay is recorded in [restoration.json](restoration.json)
and [restoration.log](restoration.log). It is a separate source/filesystem copy
on the same CPU environment, not new hardware or statistical replication.
All853original indexed files are checked again for mutation after replay.
Circular archive metadata/restoration receipts are published outside the bundle.
