# ME091 original-model and input restoration contract

Five declared seeds91001–91005,104225active learned values each. All5 fixed
ME087 parents are index paired, with explicit paired SwiGLU reduction116→108;
94907 original values are retained and9318 visual/location values initialized.
Source original optimizer/RNG remain immutable. Each new model uses new AdamW,
4096steps,98304sampled windows,3145728observed prefix presentations, not that many
unique frames. Retained-base lineage is4096old+4096new=8192updates. New weights
receive4096updates. The prefit manifest is frozen at14cd30e5 before fitting.

Original fitting receipts and recoverable final checkpoints live at
`research/replay-bundles/ME091/seed-<seed>/`. Every `final.pt` contains original
weights, optimizer moments, Torch/NumPy/Python RNG, sampling digest, source stats,
registration digest and lineage. `live-budget.json` records measured100k policy
and guarded optimizer steps. Raw stdout uses a legacy`.jsonl`filename but may
include a plain completion marker; preserve it exactly, not as strict JSONL.
No fitted seed is restarted, selected away or ensembled.

All originals and receipts must be Git-frozen before any REPORT scoring.
`FIT_INDEX.json` is created only after all5 finish. Final parameter/step validation
then precedes original forecasts for8- and32-frame histories. Reporting retains
all27,600forecast records and the full fixed gate, including failures. No partial
seed score may determine training or inclusion of another seed.

## Inputs and independent restore

Pinned publisher source is BodenAI/RW-RL-HIL-Dataset revision
9290f5a96dc91ea58256e81d9ce2dd9621890b51, CC BY4.0, attributionBodenAI/RW-RL.
The original numeric160episode source digest is
4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776.
Only the same120FIT/40exposedREPORT episodes are used; CAL/TEST/tissue-box closed.
`research/source-bundles/ME091/` contains exact licensed gray32observed input
pixels and every source-video/pixel checksum, avoiding codec/version drift.
Those are source observations, not frozen learned encoder parameters.

From a clean full checkout with pinned repository environment:

```
python scripts/me091_restore_pixels.py
PYTHONPATH=src:. python scripts/me091_train_world.py finalize --study me091
PYTHONPATH=src:. python scripts/me091_train_world.py replay --study me091 --cache /tmp/numeric-source
PYTHONPATH=src:. python scripts/me091_cross_cpu_audit.py --study me091 --cache /tmp/numeric-source
```

These commands restore and score original weights; none fits a model. The
numeric loader independently checks the original pinned parquet source. Pixel
restoration checks raw pixel SHA256 against the frozen registration. Reference
inference is float64 promoted from exact original float32 weight values; original
rtol1e-5/atol1e-6 are unchanged. A replay failure cannot be relabeled success by
changing inputs, quantile iterations, weights or tolerances.

Derived fixed-profile TorchScript executables, when present, are not new fits.
They embed the same learned values, which remain counted as104225deployed learned
parameters even if freezing removes their registration from`.parameters()`.
`END_TO_END_GRAPH.json` and `COMPILED_GRAPH_AUDIT.json` concern first-original
real-FIT execution and all17outputs (3quantiles, factual risk,13state tensors),
not all-shape performance, predictive skill or full original REPORT replay.
Shared CPU/concurrent-training timing limits remain explicit.

The functional model cache is1331scalar state values per entity. The optional
versioned handler additionally stores numeric retry inputs, one32byte image
hash, cloned forecasts and container overhead;1331is not total application RAM.
No expanding raw camera history is retained by inference. This does not establish
arbitrary-history predictive accuracy or actual customer/robot-control value.

The original evaluation retains960registered H1origins (24per each of40recording
episodes),956H4and928H16 endpoints. It does not score all67797possible autonomous
REPORT windows. The inherited manifest's `all_eligible_origins` means all
registered selected origins/future endpoints retained, not every possible frame.
The separate32-frame denominator remains explicitly recorded; repeated seeds
and overlapping windows do not add independent episode clusters.
