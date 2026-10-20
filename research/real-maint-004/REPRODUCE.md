# Reproduce the recorded negative result

Use Python3.12, NumPy2.5.4, Torch2.14.1CPU. No GPU, Veritas, network credential
or fitting is required for checkpoint replay. The public source CSV is separately
needed to independently reconstruct the grouped chronological input boundary.

```bash
python scripts/restore_real_maintenance_004.py
python scripts/audit_real_maintenance_004.py --directory results/real004-main --source /path/to/floodchlorinationsurvey.csv --output results/real004-replay.json --exports results/real004-replayed-forecasters
python scripts/analyze_real_maintenance_004.py --directory results/real004-main --output results/real004-replayed-statistics.json
```

Fetch the exact CSV at publisher commit
`a2755213c841fdfc7b65aaaff0b91016296f2985`, path
`inst/extdata/floodchlorinationsurvey.csv`, in repository
`https://github.com/openwashdata/floodchlorinationsurvey`. The loader requires
SHA256`6062eaa4c758d07820e7a13c4c31f783970dbf5598fa554afee2de5a36d58258`.
Do not substitute a moving main branch or inspect sealed target values.

The archive restores all24pilot reports,48raw/EMAweights, all curves, conventional
references and technical prepared train/validation data. `test-labels.json` and
`TEST-OPENED.json` must remain absent. Every standalone export must report seconds
per20L and match all five demo predictions per arm. Initial and metadata-complete
exports are separately preserved; model weights and predictions are unchanged.

Actual fitting source was frozen at4b047457b32412f99fcdf352a355e2050d968df6. Full
fitting is intentionally NOT part of CI replay. A new research recipe requires a
new registration and independent evaluation; this result is not repairable by
quietly modifying the protocol or screening the test cohort.
