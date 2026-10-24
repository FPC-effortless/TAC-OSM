# ME082–ME084 original complete-model results

All three prospectively frozen recipes completed five cold starts each on actual licensed robot recordings. All original weights/optimizer/RNG were committed before REPORT scoring. These are exposed-development capability iterations, not a matched architecture comparison or independent qualification. No sealed tissue-box/CAL/TEST was opened.

| Study | NLL loss weight | H1 MAE | Moving H1 | Stationary H1 | H4 MAE | H16 MAE | Full predictive gate |
|---|---:|---:|---:|---:|---:|---:|---|
| ME082 | 0.2 | 0.05302660 | 0.40823445 | 0.01896557 | 0.18896452 | 0.85704385 | FAIL |
| ME083 | 1.0 | 0.05273028 | 0.39716496 | 0.01970230 | 0.19087094 | 0.88596384 | FAIL |
| ME084 | 0.02 | 0.05445141 | 0.39042933 | 0.02223435 | 0.18210199 | 0.86214348 | FAIL |

The first two candidates failed stationary precision, the strict four-frame target and both original95% and additional98.333% cluster-CI gates. The third also missed the stronger ME081 one-step target. Every failed condition remains in its original JSON. All five seeds, all tasks and both history lengths are retained; no model ensemble is presented as one model.

The first two originals independently replayed27,600 forecast records each in GitHub Actions under the unchanged strict tolerance. ME084 external receipt is recorded separately when available. Numerical replay is independent of predictive success.

## Subsequent architecture, separately registered

ME085/086 change the native integration substantively: actual observed capture time, time-rate native state dynamics, time-decayed vectorized filters and direct convex retrieval of recent actual values. Their two-recipe4096-update cold-start search was frozen separately after these failures. It cannot be counted as an untouched confirmatory test. No old checkpoint, score or gate was changed.
