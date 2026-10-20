# Reproduce the recorded outcome-only study

Python3.12, NumPy2.5.4, Torch2.14.1CPU. Native replay requires Veritas commit
b21d829b62d3cab2811e746267ca305e110ed004. CI restores evidence; it does not refit.

```bash
python scripts/restore_public_outcome_dev068.py
python scripts/audit_public_outcome_dev068.py --directory results/dev068-main --output results/dev068-replay.json --exports results/dev068-replayed-agents --veritas /path/to/veritas --reconstruct-data
python scripts/analyze_public_outcome_dev068.py --directory results/dev068-main --output results/dev068-replayed-statistics.json
```

All88fit reports/checkpoints, prepared public training/validation tensors, paid
probe/main action traces, fees, separate teacher-only targets, curves, locked
selection and interrupted-fit records are archived. SHA256checks and the exact
fitting sourcea6d8cf30ddf69fe3c14980c54e27ba47059a8854are enforced. No TEST-OPENED
marker or reserved test report may exist. Both public datasets regenerate from
recorded procedural namespaces without reading private state as primary labels.

Exports are first fresh initialization96801, all four arms, diagnostic only.
Their JSON runtime accepts public events and remaining budget; no posterior,
hidden truth or analytical transition is accepted. Follow each export's README
and demo requests; output measurement/verifier probabilities are not certificates.

The recovery wrapper is hash-pinned and partial fits retained. Exact lost-process
CPU is unknown and conservatively charged. Raw optimizer state was NOT serialized
in this study, so source/data/seed rerun differs from mid-fit state continuation.
The two exposed-validation intervention scripts are diagnostics, not training,
sealed tests or deployable oracles. Their fixed first14seed scope is explicit.
