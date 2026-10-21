# Original ME030 weights and evidence

Archive: `/workspace/me030-checkpoints-and-evidence.tar.gz`, **12770103 bytes**.
SHA256: `ddb6b9f597310feffbf0cae3650bb1ac2a71a2acd57b769f8665032f9a8be7c3`.

Contains all9 original final checkpoints with optimizer/RNG,9 TorchScript exports,
original/mean-state probability arrays, dense curves,2700 per-update CPU/loss receipts,
source manifest, audit/profiler receipts, source files and protocol. See the
per-payload SHA256/size index in artifact-index.json. Raw source observations are
excluded. The authenticated parent ME022 cache is required for full-data replay.

Local independent extraction to `/workspace/me030-restored` verified every indexed
payload and replayed all18 original checkpoint/export paths plus9 mean-state paths.
All107136 primary forecasts matched bit exactly. This is a restoration from the
original archive, not refitting. Archive bytes are a local workspace artifact;
no GitHub artifact download is claimed for ME030.

From the source checkout with authenticated parent cache:

```bash
sha256sum /workspace/me030-checkpoints-and-evidence.tar.gz
python -m scripts.me030_joint_state_research --out /workspace/me030-restored/results/me030 --replay
python -m scripts.me030_artifact_audit /workspace/me030-restored/results/me030
python -m pytest -q
```

`--replay` verifies every payload digest before source loading. The artifact audit
also verifies Adam counters, exact seeded sample RNG, fresh Torch-only exports and
fail-closed corruption. Torch2.14.1+cpu, NumPy2.3.5, Python3.12; thread count1.

For a NEW reproduction only (this initiates9 new fits, not archive restoration):
`python -m scripts.me030_joint_state_research --out results/me030-new-reproduction`.
Never substitute new fit bytes for the originals or reopen reserved cohorts.
