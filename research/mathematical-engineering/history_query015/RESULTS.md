# HISTORY-QUERY015 — measured native historical query specificity

**Internally measured, not independently qualified. Registered retrieval gate: FAIL for both original seeds.**

[GitHub CPU source run and byte-identical fresh-process replay](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40483178575), code `b0c09f28c4411ee11f6f03a26767f3eca6fa25cd`. Four original LATFUSE018 checkpoints at pinned code `fd201207cb76b30470f64e1fc1baecf9de44560d`; 117 original exposed FIT windows, eight real timestamps each. There are **915 observed targets and21 missing**, not936 imputed values. The same targets are reused across the four models; these are not3,660 independent observations.

## Measured source scores

All errors below use the original pre-March2014 normalized log1p(PM2.5) scale; lower error is better. Centered skill is dimensionless; positive means better within-window discrimination than a constant.

| Associative model measure | Seed18018 | Seed18019 |
| --- | ---: | ---: |
| Correct trained-H64 clock MAE | 0.390067 | 0.329604 |
| Seven wrong-clock mean target error | 0.394540 | 0.357811 |
| Timestamp-average predictor H64 error | 0.389277 | 0.333051 |
| Same-checkpoint zero matrix H64 error | 0.352914 | 0.352853 |
| Original training-mean prior H64 error | 0.361321 | 0.361321 |
| Within-window centered skill | -0.128303 | -0.063212 |
| All observed requested-time MAE | 0.748002 | 0.652617 |
| Circularly wrong query-target MAE | 0.739971 | 0.647260 |
| Timestamp-average all-target MAE | 0.732130 | 0.645463 |
| Mean within-window prediction range | 0.248295 | 0.431752 |
| Mean observed within-window target range | 1.777121 | 1.777121 |

Correct H64 query gains only **−0.20% / +1.03%** relative to the target-free timestamp-average predictor; both station-bootstrap95% error-gain intervals cross zero. Neither meets the frozen >=5% gain +positive lower interval condition.

The result is not complete query insensitivity: seed18019 improves H64 error over wrong clocks by **7.88%**, with positive grouped error-gain interval [0.01166,0.04226]. Seed18018 gains1.13% and its interval crosses zero. Preserve this partial positive contrast instead of treating every test as a failure.

Native within-window centered skill is **−0.1283 / −0.0632**. Centering removes window-level predicted/true means only in scoring; no target mean is supplied to either predictor. The fast/slow control has skill **+0.0433 / −0.1654**. Native is worse than that control in seed18018 and less bad in seed18019, but it is not positively skillful in either seed. Both raw native all-target MAEs are also worse than the target-free collapsed predictor.

## Meaning and next experiment

These frozen content-trained checkpoints do not establish robust requested-time factual access under the preregistered controls. Their source-level H64 content scores must not be presented as proof that eight distinct older observations can be recalled. Predictions do change with the legal clock, and one seed shows a trained-lag contrast. The evidence therefore supports a specific generalization/readout gap, not a proof of information-theoretic absence or architectural incapacity.

Only H64 was trained. The other seven historical lags and deliberate clock/target mismatch controls are distribution changes; the negative within-window result cannot establish that variable-lag supervision would fail. The next justified training hypothesis is explicit query-diverse historical-content supervision, with the same authentic histories/targets and matched budgets for both arms, preregistered before outcomes. Do not repeat the old fixed-lag fits or add an architecture solely to rescue this exposed cohort.

## Original-source and replay checks

- Original source, original plan/results lineage, all four trained checkpoint hashes, source/seed/training-order metadata and state_dict immutability passed. No optimizer, calibration, graph or weights changed.
- Actual normalization and source last/age prefixes passed. Each predictor received only its own88-step observed feature window, legal source ID and eight known past calendars; raw targets and future feature rows were not inputs. Targets/masks were consulted by the scorer after both predictors.
- H64 reproduces the original saved content error within2e-6 in every arm/seed (actual max difference below2e-8). First N=8 output agrees with N=1 within2e-6 (observed maxima1.19e-7 /2.38e-7).
- Fresh-process replay is byte-identical. Original full output SHA256: `dd7a482814dab5ed61bebe9150c6b121871fcd7a298fe5ad1209e6f008f516b4`. `RESULTS.json` retains every case, mask, target, prediction and input digest in compact serialization; `SUMMARY.json` explains exact original-byte reconstruction.
- Full source-file-reloaded future poisoning remains **INCONCLUSIVE_NOT_IMPLEMENTED**; whole-model PNDS Lean/Python/IEEE/AD equivalence remains **OPEN**. No fixtures or local checks are substituted for those obligations.

The source is repeatedly exposed May–June2014 FIT, not an independent holdout. Station-grouped intervals preserve windows/query targets but do not eliminate regional correlation. There is no independent qualification, safety/reward certification, causal decision utility or checkpoint promotion. Previously exposed2015 and protected2016–17 numerical rows were not opened by this FIT-only loader; no GPU quota or new training was used.

Attribution: Chen, S. (2017), *Beijing Multi-Site Air Quality*, UCI Machine Learning Repository, [DOI10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY4.0.
