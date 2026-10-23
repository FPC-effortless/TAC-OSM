# ME081 original model archive and replay

Each `research/replay-bundles/ME081/seed-8100{1,2,3,4,5}/final.pt` is actual
tracked original Git bytes. It contains whole-model weights, optimizer, Torch/
NumPy/Python RNG, source-derived preprocessing, deterministic sampler seed/digests
and frozen registration hash. Adjacent live-budget receipts bind actual
100,165-parameter whole models to all 2,048 successful optimizer steps.

Prefit source/registration: `b5698c86`; all five originals frozen before scoring:
`dd71fee2`; original raw forecasts/targets/scores: `bf23f52b`.

Primary files `ORIGINAL_PREDICTIONS.json` use the original eight-frame task.
Secondary `ORIGINAL_PREDICTIONS_H32.json` use the prospectively registered actual
32-frame cohort. Both are original forecasts; they must not be overwritten.
`ORIGINAL_SCORE_INDEX.json` records their checkpoint/output hashes and denominators.

```sh
PYTHONPATH=src:. python scripts/me081_train_world.py finalize
PYTHONPATH=src:. python scripts/me081_train_world.py replay --cache /tmp/me081-source
PYTHONPATH=src:. python scripts/me081_report_capability.py
```

These commands perform no optimizer update. Source original Parquets must be
fetched through the pinned, licensed, checksum-verifying loader; no unlicensed
raw data is in Git. Independent local fresh-clone and external status will be
recorded separately here. The original replay tolerance remains rtol1e-5/atol1e-6.
No all-platform claim until actually verified; historical exceptions unchanged.

Five fits are complete; full capability gate FAIL / MIXED_TRADEOFF. No further
ME081 fitting, sealed target access, architectural superiority or customer claim.

## Completed independent replay status

Fresh remote checkout at `bf23f52b` restored all five originals and both
information conditions, **27,600 original forecast records**, within the unchanged
tolerance. The entire capability summary recomputed exactly locally. The first
full checkout exhausted workspace disk; only that disposable failed clone was
removed. A fresh sparse remote checkout included all code, configs and required
original model archives. Original research worktrees/checkpoints were untouched.
See `research/replay-bundles/ME081/INDEPENDENT_REPLAY.json`.

[External run40021655456](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40021655456)
restored original source and all five models. Actual final-model/2048-step
receipt verification passed. **Strict external replay FAILS**, first at
seed81001/eight-frame lower quantile. The full read-only audit completed over
both information conditions and all27,600records. Out-of-tolerance field counts:
point prediction8, physicalMAE5, lower quantile8, upper quantile6, width12,
interval score11. Counts can overlap and are NOT distinct failed cases.
Max absolute point-array drift1.50443e-5, physicalMAE drift4.18854e-5;
max interval-score drift6.72926e-5. Actual targets, coverage, motion probabilities/
Brier and physicalNLL meet the frozen tolerance. No all-platform replay PASS.

Full per-model/history/field evidence is at
`research/replay-bundles/ME081/GITHUB_CROSS_CPU_AUDIT.json`. No original weight,
score, registration, old study or tolerance changed. The audit-process success
does not replace the strict failed replay. Research remains draft/unpromoted.
