# Recorded forecast qualification

Python3.12, CPU PyTorch2.14.1, NumPy2.5.4 and this repository. No training is needed
for committed evidence replay. In a fresh checkout:

```sh
python scripts/restore_real_maintenance_002.py
python scripts/audit_real_maintenance_002.py --directory results/real-maint-002-main --output results/replay-002.json --exports results/export-002
python scripts/analyze_real_maintenance_002.py --directory results/real-maint-002-main --output results/analysis-002.json
```

All archive hashes are verified before safe extraction. The reserved July B payload
is deliberately excluded; restore/audit never needs it. Source hashes/DOI and fixed
preprocessing reproduce B for a separately registered future experiment. Sealing
is a procedural guard, not independent third-party custody.

Extract `learned-agent.tar.gz`, then from an exported arm's directory:

```sh
python runtime.py < demo-requests.jsonl
```

The only non-standard runtime import is PyTorch. Inputs are24x8 raw analog contexts,
with the fixed source schema in metadata. Outputs are point sensor forecasts, not
maintenance actions, calibrated physical-fault judgments or independently verified
repairs. Source data: Veloso/Gama/Ribeiro/Pereira, MetroPT2 DOI10.5281/zenodo.7766691,
CC BY4.0. Forecasts remain unpromoted under the strict original registration.

The standalone belief bottleneck is a separate, UNTRAINED prototype. Its property
tests establish information paths/time symmetries, not operational performance.
No serialized belief-prototype weights are presented as a trained replacement.
