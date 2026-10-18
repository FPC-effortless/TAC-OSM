# Recorded learned-maintenance studies

Use Python3.12 and the pinned environment in
`.github/workflows/dev0615-062-robust-belief.yml`. Install this project and CPU
PyTorch2.14.1, NumPy2.5.4; native replay additionally requires the pinned public
Veritas runtime. No training is required to verify committed evidence.

For one study (`dev0615`, `dev062`, `dev062-r1`, `real-maint-001`):

```sh
python scripts/restore_research_evidence_dev062.py --study dev062
python scripts/audit_belief_dev062.py --directory results/dev062-main --output results/dev062-independent-audit.json --exports results/dev062-independent-export --veritas /path/to/veritas
```

Use the corresponding identity, R1 or real-maintenance auditor for the other
studies. The external sensor auditor takes no Veritas option. Each restoration
checks every archive SHA256 before safe extraction. Do not overwrite existing
research roots; use a fresh checkout for independent restoration.

Portable artifact example, without TAC-OSM/NumPy/Veritas imports:

```sh
mkdir -p /tmp/tac-belief-demo
cd /tmp/tac-belief-demo
tar -xzf /path/to/research/dev062/learned-agent.tar.gz
```

Each exported arm contains `runtime.py`, model weights, metadata and
`demo-requests.jsonl`. From that arm's directory run:

```sh
python runtime.py < demo-requests.jsonl
```

Inputs are the registered public numeric schema, not arbitrary natural language.
Scores and learned completion estimates do not independently verify a physical
repair. Exact runtime file manifests and checkpoint SHA256 are bundled per arm.
Use `research/dev062-batch/batched-agent.tar.gz` only for its separately measured
batching qualification; never silently replace the original recorded artifact.
Training reproduction commands and fixed budgets are in each PROTOCOL.md and
fitter CLI. Running new fits is a separate experiment, not audit replay.
