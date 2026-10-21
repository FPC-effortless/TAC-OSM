# Reproduce or restore ME036

Python3.12, NumPy2.3.5, Torch2.14.1+cpu. Source package's ordinary installation
has no mandatory Torch dependency; the research models are optional Torch modules.
Install those versions and the repository before running the commands below.
Use one Torch/BLAS thread per fit, with the same variables for auditing:

```bash
export PYTHONPATH=src:.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## Restore original trained models without fitting

Archive location and SHA256 are in ARTIFACT.md/archive.json. Verify that archive
hash before extraction. On a new destination (must not already exist):

```bash
python scripts/me036_restore.py /workspace/me036-full-horizon-models-and-evidence.tar.gz /workspace/me036-restored
```

The archive preserves original source, protocols, source dataset bytes/card/license,
checkpoints with Adam/sampler/model RNG, learning traces, validation forecast probes,
selected TorchScript exports and scoring records. No refitting is needed. The
restorer hashes every indexed payload before loading weights and independently
replays initial and continued validation probes from the extracted source/data.
It does not re-open test metrics or claim independent-hardware replication.

## Retrain the registered trajectories

In a clean checkout/output directory, acquire the exact source bytes and run:

```bash
python scripts/me036_sources.py
python scripts/me036_research.py train
python scripts/me036b_continue.py train
```

The second command trains100updates and the third continues every model to1000.
Sequential execution gives the same model paths as the recorded independent
single-thread cohort/horizon workers. Hardware/CPU costs need not match.
No checkpoint is reconstructed by refitting in the original-model archive.
The protocols forbid retrospective retuning or replacement of failed trajectories.

Only after all selections/coefficient hashes freeze:

```bash
python scripts/me036_evaluate.py
python scripts/me036_research.py audit
python scripts/me036b_continue.py audit
python scripts/me036_runtime_audit.py
python scripts/me036_report.py
```

The evaluator uses two single-thread processes, all eligible test anchors (stride1),
and every target step/channel. Existing completed dataset receipts are reused;
do not delete receipts to repeatedly inspect test outcomes. Test means are stored
with scalar/window counts, selection step, seed and actual processCPU. Forecast
arrays are not all retained because the full matrix contains billions of values.
Checkpoint audits compare saved validation probe forecasts bit for bit.

`selected.jit.pt` consumes globally train-standardized values and returns the same
units. `physical-selected.jit.pt` also contains channel-wise train mean/scale and
accepts/returns raw publisher units. Both still replay the full96-step window.
Use the corresponding dataset's channel order in data.json. Float32 physical-unit
roundtrip changes some input values; the physical export is audited against its
own eager wrapper, not declared bit-identical to a separately standardized call.

Weather's81training sentinel codes, duplicate timestamp and100-minute gap remain
unchanged for benchmark reproducibility. The test period has no-9999codes. ETT
previous-exposure declarations and the historical ME035B precision failure remain
part of the evidence ledger. These are forecasting experiments, not action or
commercial world-model qualification.
