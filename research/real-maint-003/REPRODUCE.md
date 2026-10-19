# Restore the forecaster without refitting

Fitting source: `7fedb402ef1ea04c4ab2f471e79494ec9d0d27e2`; registration first
committed at 5dc9289. The six fitting/gate files remain unchanged. The original
protocol's Celsius label is erroneous; use UNIT-ERRATUM.md and the corrected
Fahrenheit adapter. No checkpoint or scientific gate changed with that correction.

Python3.12, CPU PyTorch2.14.1 and NumPy2.5.4; install TAC-OSM editable. Veritas is
not needed for this observational physical-data study. After packaging:

```sh
python scripts/restore_real_maintenance_003.py
python scripts/audit_real_maintenance_003.py --directory results/real003-main --output results/real003-replay.json --exports results/real003-replayed-forecasters
python scripts/analyze_real_maintenance_003.py --directory results/real003-main --output results/real003-replayed-statistics.json
```

The restore verifies every archive/derived SHA256. The audit verifies the run
manifest, frozen source, all 88 neural EMA checkpoints, all references, TRAIN-only
normalization, validation and held-out predictions. Raw weights stay preserved
and hashed. Neither neural nor closed-form models are refitted during CI. Cached
windows reproduce the recorded experiment, without downloading or opening any
new sensor period. Exact local replay is observed; cross-host tolerance is1e-8.

The standalone archive contains every arm at the first predeclared initialization:

```sh
mkdir -p /tmp/room-forecasters
tar -xzf research/real-maint-003/learned-forecaster.tar.gz -C /tmp/room-forecasters
cd /tmp/room-forecasters/real003-agents/mlp_revin
python runtime.py < demo-input.jsonl
```

Read metadata.json for input order/units: eleven temperaturesF, then powerW.
Output is `temperature_F`. Inference imports no research harness, simulator,
Veritas or NumPy. The copied model class and metadata are part of the release.

Refitting is separate: clean checkout of the fitting commit, acquire the published
OEDI909 zip matching the source SHA256, then run:

```sh
python scripts/run_real_maintenance_003.py --directory results/refitted-real003 --source /path/to/dataset.zip
```

That reproduces an opened benchmark, not a new confirmation. Do not discard bad
original seeds, retune on2019 or call another run a fresh holdout. Independent-site
or later-period claims need a new registration and untouched measurements, with
correct units from the start. MetroPT July B remains unopened.
