# Original ME080 models and replay

All five original whole models are in
`research/replay-bundles/ME080/seed-8000{1,2,3,4,5}/final.pt`.
These are tracked Git bytes, not hash-only receipts or temporary CI artifacts.
Each includes weights, optimizer, Torch/NumPy/Python RNG, seed, frozen schedule,
FIT-only preprocessing and the prefit registration digest. Adjacent live-budget
receipts bind actual99,816parameter models to all2048successful optimizer steps.
FIT_INDEX and ORIGINAL_SCORE_INDEX record hashes and raw query denominators.

- Authoritative prefit/source/model code freeze:4c7309ff.
- All five original trained checkpoints frozen BEFORE scoring:e37a4b88.
- Original raw predictions/targets/scores:ea6cdd5e.
- Final metadata binding verified by`scripts/me080_finalize_originals.py`.
- Independent fresh remote clone atea6cdd5e restored all five originals and
  replayed all14,220raw forecasts within unchanged rtol1e-5/atol1e-6. Entire
  CAPABILITY_RESULTS summary independently recomputed and JSON-exact locally.
  See`research/replay-bundles/ME080/INDEPENDENT_REPLAY.json`.
- External[run40005926076](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40005926076)
  restored original real source and models; preflight, five tests and final
  live-budget/actual-model verification passed. **Strict replay FAILED at
  seed80002/lower quantile.** Not all-platform-green. The full no-fit census
  is recorded separately when complete; do not change originals or tolerance.

Old ME078/079strict cross-CPU failures remain unchanged. This study's float64
CDF inversion/scoring was specified before fitting; it does not guarantee
identical float32neural outputs on arbitrary hardware.

Reproduce from a clean checkout with a compatible CPU PyTorch installation:

```sh
PYTHONPATH=src:. python scripts/me080_finalize_originals.py
PYTHONPATH=src:. python scripts/me080_train_world.py replay --cache /tmp/me080-source
PYTHONPATH=src:. python scripts/me080_report_capability.py
```

Full source identification/licenses and original row/query hashes are frozen in
SOURCE_QUALIFICATION.json and FAIR_COMPARISON.json. Public original Parquets
must be retrieved via the pinned original loader; no unlicensed raw source in
Git. The protected tissue-box, sourceCALandTEST remain closed.

**No new fits under ME080.** No training mode is invoked by the replay workflow.
This is exposed-development physical forecasting, with a failed full capability
gate, no architecture-superiority or causal-control claim, and no measured
operator/customer value.

## Complete external census

[Run 40006593903](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40006593903)
restored all five original models and recomputed all 14,220 forecast records.
The read-only census completed successfully, but **strict original replay is
FAIL**: three lower-quantile records and one derived interval score exceed the
original tolerance. Maximum absolute lower-quantile drift is 8.3226e-6;
maximum interval-score drift is 1.09736e-4. These are field counts and may
refer to overlapping records. Every other recorded field, including actual
point predictions, targets, physical MAE, NLL, probabilities, Brier, upper
quantiles, widths and coverage, meets the frozen contract. Coverage and target
arrays are exact. This does not authorize a whole-replay PASS.

Full per-model/per-field counts, maxima, examples and actual original checkpoint
hashes are committed at `research/replay-bundles/ME080/GITHUB_CROSS_CPU_AUDIT.json`.
No weight, original output, registration or tolerance was changed.
