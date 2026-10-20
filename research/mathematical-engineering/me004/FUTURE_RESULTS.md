# ME-004F — Original frozen out-of-time real forecasting results (2026-10-20)

**Result:** Limited one-source out-of-time corroboration of the native TAC two-timescale **forecasting submodel**, **not a full TAC-OSM agent qualification**.

- Frozen fitted weights from ME004 original successful [run 39357593422](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39357593422); artifact `12235023980`, source commit `954baafe63d1450e4968cc1b49dac02a5e8f6f1d`.
- Independent **original-source/score replay** before test: [run 39358264875](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39358264875), all six frozen weights and forecasts verified with **zero refits**, portable model inference smoke test passed.
- Preregistered one-time future-week [run 39358794541](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39358794541), artifact `12236311978`. Exact publisher ZIP SHA256 `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`.
- **Source period:** March 2 inclusive to March 9 exclusive, 2020 (outside both original fitting and validation periods and beyond the earlier ME003 first-thirty-day source audit). 51,497 authentic observed rows, **4,200** correctly ordered forecast windows, two forecast horizons (12 and 60 later observations), exact original TRAIN-only normalization and change thresholds. No source rows beyond the test cutoff opened in this run. **Zero retraining, zero tuning, no selected checkpoint replacement.**

## One-time future-week macro normalized absolute error (lower better)

| Frozen model | Seed 240401 | Seed 240402 | Mean |
|---|---:|---:|---:|
| Persistence | 0.899034 | 0.899034 | 0.899034 |
| GRU | 0.672873 | 0.679193 | **0.676033** |
| TAC identity-first | 0.668656 | 0.681937 | **0.675297** |
| **TAC two-timescale** | **0.656969** | **0.658950** | **0.657959** |

Two-timescale TAC beat both persistence and GRU in both frozen seeds, with **26.81% lower error than persistence**, **2.67% lower error than GRU**, on this one site/week.

## Required negative and qualifying findings

- This later week has **13,486 genuine large measured changes**, **6,734 positive and 6,752 negative**, avoiding ME001B's degenerate all-positive sign task.
- TAC two-timescale does **worse on changed-value DIRECTION** than GRU: mean TAC 80.26% vs GRU 83.38%. Its changed-value MAGNITUDE error is better: TAC approximately **2.025** vs GRU approximately **2.071**, compared with persistence 3.051. Therefore do not claim it is the better direction classifier.
- Paired per-channel/horizon audit: two-timescale wins **17/24** channel×horizon cells in seed 240401, **15/24** in seed 240402; **8/12** physical sensor means across both seeds. Several compressor status/valve channels are slightly worse than GRU. Large improvement on TP3 and Reservoirs contributes heavily to aggregate gain.
- Model complexity is unequal: TAC two-timescale **9,368 params**, GRU **6,360**, with TAC model ~2.18× training CPU time in the original matched-step experiment. **Do not attribute gains specifically to memory mathematics until a capacity/compute-matched GRU or other strong control is evaluated.**
- Single compressor and one later week; two optimizer seeds; no bootstrap CIs, new independent installation, full calibrated uncertainty, action-effect verification, entity discovery or commercial economic outcomes.
- New test was exposed exactly once; future CI runs must skip the original frozen evaluation. Original weights and test results must not be overwritten or retuned.

## Next decision (model-first)

Keep the frozen TAC two-timescale checkpoint as a **working experimental real-data forecasting component**. Do not replace the existing DEV067 world-state/action controller or merge it into production yet. Next direct engineering work: tune/capacity-match GRU and boosted baseline on TRAIN-only, frozen evaluation on a truly new time period or independent site; measure uncertainty/calibration and robust direction/rare-change performance. Integrate as a *predictive head* into the established TAC agent only if it continues to improve actionable risk/cost signals.

TAC-OSM remains its own mathematical predictive-executable world model, not an SSM conversion. Other master-plan ideas are parked, not rejected, until demonstrated deficiencies justify them.
