# DEV-060 foundations evidence

[Program](../../docs/WORLD-STATE-RESEARCH-PROGRAM.md) ·
[Results and limitations](../../docs/DEV-060-FOUNDATIONS.md)

`protocol-foundations.json` was committed before fitting. `evidence-index.json`
records archive digests, fitted-source integrity, raw benchmark measurements and
explicit **failed/unrun** claim gates. The full learned-policy protocol remains
[the draft](PARTIAL-OBSERVATION-DRAFT.md).

`evidence.tar.gz` preserves all twenty probe fits, curves, choices and summaries;
three reference arms' full uncertain-world trajectories; sparse compatibility
and benchmark records; and standalone existing-model exports.

`sparse-agent.tar.gz` is a convenient torch-only distribution of both existing
DEV-058 checkpoints with sparse execution and **temporal bias disabled**. It
contains no teacher, simulator or analytical controller. Extract and run:

```sh
cd 7581
python runtime.py < demo-requests.jsonl
```

Compare with `demo-expected.jsonl`. `7582` supplies the repeat checkpoint.
These are the original bounded agents, including their known reliability limits,
not newly calibrated models or agents trained on uncertain observations.

To verify research without fitting:

```sh
mkdir -p results
tar -xzf research/dev060/evidence.tar.gz -C results
tar -xzf research/dev058/evidence.tar.gz -C results
python scripts/run_memory_foundations_dev060.py --directory results/dev060-memory --replay
python scripts/evaluate_uncertainty_dev060.py --output results/dev060-ref-recheck.json
python scripts/audit_sparse_compatibility_dev060.py \
  --output results/dev060-sparse-recheck.json --veritas-root /path/to/pinned/veritas
python scripts/benchmark_sparse_dev060.py --output results/dev060-benchmark-recheck.json
```

Use the package versions and pinned Veritas revision from the DEV-060 workflow.
Replay keeps original probe evidence read-only and requires identical selected
indices. Timing is descriptive, never a passing threshold. No DRAM bytes were
measured. The exact latest-record reference remains perfect on the clean probe;
the temporal scorer still selects stale records at longer lengths. No new
architecture, partial-observation learner, JEPA, symbolic CASM or commercial claim
is accepted from this tranche.
