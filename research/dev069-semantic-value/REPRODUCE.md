# Recorded-only preservation replay

Install project CPU dependencies and pinned Veritasb21d829b62d3cab2811e746267ca305e110ed004.
Do not invoke the fitting runner: this registration was stopped by standing policy.
Restore parent DEV068 evidence and these archived files; no fitting/new test source
is needed. Run from the repository root using the Python environment with PyTorch:

```bash
python scripts/restore_public_outcome_dev068.py --destination results
python scripts/restore_semantic_value_dev069.py --destination results
PYTHONPATH=src python scripts/audit_semantic_value_dev069.py \
  --directory results/dev069-main --parent results/dev068-main \
  --output results/dev069-recorded-audit.json --veritas /path/to/veritas \
  --exports results/dev069-reexport
PYTHONPATH=src python scripts/analyze_semantic_value_dev069.py \
  --directory results/dev069-main --output results/dev069-recorded-statistics.json
```

Choose a nonexistent exports directory. The audit checks every archived file,
all24complete raw/EMA checkpoints, partial optimizer states/frozen bodies, original
validation trajectories and first-registered-group native/standalone behavior.
It never opens reserved tests. Frozen fitting source is71a7bb693b9cb4988cb1cdf570ac2a3242e88dfd;
audit/packaging additions were made afterward and do not change fitted tensors.

The evidence archive includes the administrative stop and partial work, not just
successful fits. Diagnostic-only portable exports use first declared completed
fresh policy seed96901, never a validation-selected successful seed.
