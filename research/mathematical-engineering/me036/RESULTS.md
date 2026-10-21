# ME036 — full-horizon patched-native forecasting results

Architecture-first real-data development on five forecasting benchmarks. The original 100-update stage was extended prospectively to 1,000 updates before any final test score was opened. All arms received the same continuation opportunity.

280 neural trajectories (two seeds), 20 fitted ridge references. Failed trajectories: 0. No model promoted or merged.

## Full-test results

Train-only standardized MSE; all output steps/channels; every eligible test anchor (stride1). Separate models for horizons 96/192/336/720. Macro mean gives each of the 20 dataset/horizon cells equal weight. Two-seed means are descriptive; no independent retraining interval or superiority test is claimed.

| Model | Macro MSE | Native parameter parity |
|---|---:|---|
| Patched native TAC | 0.346956 | within1% |
| AR only | 0.348944 | unmatched |
| AR + context correction | 0.343989 | within1% |
| AR + GRU correction | 0.348461 | within1% |
| Compact shared DLinear-style | 0.348481 | unmatched |
| Compact PatchTST-style | 0.345084 | unmatched |
| Compact iTransformer-style | 0.355949 | unmatched |
| Locally normalized ridge (original objective) | 11.090221 | unmatched |

TAC improvement over AR: **0.57%**. Gap to the best macro baseline: **0.86%**. Registered practical gate: **FAIL**.

## Dataset/horizon detail

| Dataset | H | TAC | AR | Context | GRU | DLinear | PatchTST | iTransformer | Ridge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.384423 | 0.391514 | 0.387186 | 0.385528 | 0.389948 | 0.383947 | 0.398963 | 0.486302 |
| ETTh1 | 192 | 0.441341 | 0.441269 | 0.443272 | 0.442729 | 0.440833 | 0.438898 | 0.452652 | 0.565901 |
| ETTh1 | 336 | 0.501367 | 0.482807 | 0.495388 | 0.498592 | 0.493054 | 0.498731 | 0.500322 | 0.683129 |
| ETTh1 | 720 | 0.562331 | 0.479805 | 0.530895 | 0.560380 | 0.482617 | 0.519841 | 0.504015 | 0.960629 |
| ETTh2 | 96 | 0.290239 | 0.291732 | 0.290367 | 0.299600 | 0.291587 | 0.291078 | 0.295667 | 2.184588 |
| ETTh2 | 192 | 0.377168 | 0.379625 | 0.372623 | 0.380924 | 0.378449 | 0.373309 | 0.381854 | 11.241882 |
| ETTh2 | 336 | 0.437208 | 0.420355 | 0.417235 | 0.424349 | 0.415449 | 0.424566 | 0.406213 | 41.951015 |
| ETTh2 | 720 | 0.465710 | 0.419468 | 0.467305 | 0.451892 | 0.420058 | 0.454946 | 0.424502 | 142.032695 |
| ETTm1 | 96 | 0.333330 | 0.359910 | 0.332631 | 0.357493 | 0.355420 | 0.333208 | 0.359550 | 0.808070 |
| ETTm1 | 192 | 0.377360 | 0.395281 | 0.375349 | 0.382402 | 0.391811 | 0.375461 | 0.401699 | 2.311494 |
| ETTm1 | 336 | 0.410846 | 0.428910 | 0.406970 | 0.415070 | 0.426536 | 0.407472 | 0.432521 | 3.655022 |
| ETTm1 | 720 | 0.478871 | 0.493625 | 0.481546 | 0.492136 | 0.492462 | 0.474030 | 0.500629 | 3.140693 |
| weather | 96 | 0.182918 | 0.199893 | 0.184503 | 0.185493 | 0.198717 | 0.186275 | 0.192249 | 0.972304 |
| weather | 192 | 0.226968 | 0.245109 | 0.227510 | 0.227261 | 0.248254 | 0.233645 | 0.238693 | 1.317767 |
| weather | 336 | 0.277107 | 0.299573 | 0.278112 | 0.277196 | 0.297917 | 0.286067 | 0.291009 | 1.521008 |
| weather | 720 | 0.348865 | 0.365724 | 0.345049 | 0.345033 | 0.367411 | 0.359174 | 0.362253 | 1.710481 |
| electricity | 96 | 0.193090 | 0.203855 | 0.192454 | 0.192920 | 0.202286 | 0.196693 | 0.216370 | 1.032673 |
| electricity | 192 | 0.195520 | 0.203378 | 0.195504 | 0.195638 | 0.201996 | 0.197422 | 0.224989 | 1.333417 |
| electricity | 336 | 0.210133 | 0.217798 | 0.210122 | 0.209786 | 0.216693 | 0.212813 | 0.244766 | 1.652275 |
| electricity | 720 | 0.244323 | 0.259256 | 0.245764 | 0.244801 | 0.258126 | 0.254094 | 0.290060 | 2.243070 |

## Engineering and comparison limits

TAC uses twelve ordered eight-step patches, learned 32D fast/32D slow native dynamics, residual slow updates, trainable multiscale rates, a SwiGLU/32-basis correction and joint channel-shared linear forecast path. All models start from scratch; ridge is a separately fitted reference, not a TAC initializer. Models are trained separately per dataset/horizon; this is not zero-shot transfer.

TAC/context/GRU parameter gaps are below 1%; all other model sizes differ. Examples, samplers, 1,000-update ceilings and selection privileges match for the neural models. Actual CPU spend differs: the strict compute-parity gate **fails**, so there is **no architectural-superiority claim**.

Actual neural fitting CPU: 2248.59s; validation: 163.32s; neural final scoring: 1307.08s. Ridge fitting: 3.40s. Initialization/optimizer/update costs are charged; checkpoint I/O and export are separate. RSS in fit records is process high-water RSS, not isolated per-model activation memory. CPU timing is process time, single-thread.

Standard split/horizon/stride definitions are now used, but compact independent baseline implementations, 96-step lookback, 1,000-update training and stride 24 validation differ from published recipes. These scores do not reproduce the published DLinear/PatchTST/iTransformer tables. Weather and Electricity are authenticated THUML benchmark copies, not raw instrument files. Earlier ETT period exposure remains declared; no pristine-population claim. ETTh1 and ETTm1 are resolutions of the same transformer records, not independent sites. Weather contains 81 training-only `-9999` codes, one duplicate timestamp and one 100-minute gap; original rows remain unchanged under the registered benchmark convention. Its test targets contain no `-9999` codes, but training normalization includes them. Weather results are sensitive to this source-quality limitation.

Serving still re-encodes the entire normalized96-step window. Twelve patch updates do not prove an eightfold runtime speedup or constant-time streaming. Runtime JSON records actual B1 latency and weight/input/output/state bytes. Standardized exports and physical-input exports are both supplied; physical exports include training mean/scale buffers and inversion. data.json records those statistics. The old ME035B `1e-6` equivalence failure remains recorded and unchanged; new linear numerical tests use a prospective gamma_n bound.

See summary.json, results.json (per seed), comparison-manifest.json, frozen.json, replay.json and runtime.json for machine-readable evidence. Final metrics stream without retaining billions of forecast values; checkpoint audits replay the saved validation probes, not the entire test metric array.

## Post hoc ridge objective correction

The registered ridge fit minimized locally normalized residual errors without the variance weights needed to match the scored forecasting MSE. Its extreme ETTh2 results remain in the original table. ME036C fixes that algebra with the same alpha 100 and training anchors; no TAC changes or validation tuning. It was registered after initial test scores, so it is descriptive corrected reference evidence, not prospective confirmation. See RIDGE-CORRECTION.md.

Corrected ridge macro MSE: **0.389767**; TAC relative gap: **-10.98%**. The registered gate above is unchanged and excludes this later correction.

| Dataset | H | Corrected ridge MSE |
|---|---:|---:|
| ETTh1 | 96 | 0.470688 |
| ETTh1 | 192 | 0.523650 |
| ETTh1 | 336 | 0.579395 |
| ETTh1 | 720 | 0.619212 |
| ETTh2 | 96 | 0.343069 |
| ETTh2 | 192 | 0.448982 |
| ETTh2 | 336 | 0.485558 |
| ETTh2 | 720 | 0.483975 |
| ETTm1 | 96 | 0.382161 |
| ETTm1 | 192 | 0.415061 |
| ETTm1 | 336 | 0.447215 |
| ETTm1 | 720 | 0.507152 |
| weather | 96 | 0.199261 |
| weather | 192 | 0.244190 |
| weather | 336 | 0.295727 |
| weather | 720 | 0.367920 |
| electricity | 96 | 0.229663 |
| electricity | 192 | 0.223458 |
| electricity | 336 | 0.239344 |
| electricity | 720 | 0.289651 |

## Material long-horizon regressions

The macro mean masks regressions at 720 steps: TAC is 17.20% worse than AR-only on ETTh1 and 11.02% worse on ETTh2. The complete model is therefore not consistently competitive across horizons. Future architecture development should strengthen its horizon-dependent predictive path; no mechanistic-ablation phase is required before building that next model.

Scoring resumed at model granularity to use both CPUs after four datasets finished. All 248 completed neural scores were preserved unchanged; one incomplete arithmetic score may have restarted. No checkpoint/selection/metric changes. See EVALUATION-EXECUTION.md and evaluation-reschedule.json.
