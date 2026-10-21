# BENCH-FM-001 — Original TAC vs official TimesFM 3.0 and TabFM on genuine MetroPT-3

**Date:** October 21, 2026. **Scope:** 24 previously exposed genuine March 2–9, 2020 MetroPT-3 windows, 12 physical channels, 32 observations of context, TP3 pressure target 12 and 60 later *recorded observations* in original TRAIN-standardized channel units. The study is exploratory; it is **not** a new untouched test or a matched-pretraining/computation architecture comparison.

## Original frozen TAC/GRU and TimesFM3 successful execution

Successful source/weights verification: [GitHub Actions run 39596526571](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39596526571), with original trained checkpoints restored byte-for-byte from ME004 fitting [run 39357593422](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39357593422), artifact 12235023980. Real publisher archive SHA256 aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a. Complete-source candidate future windows 4200; 24 fixed evenly spaced windows (all 32×12 real context measurements present) have been scored; none selected by looking at new targets. TP3 index1 in declared source order.

| Actual deployed model | TP3 +12 absolute error | TP3 +60 absolute error | Mean (lower better) | Result source |
|---|---:|---:|---:|---|
| TAC 2-timescale frozen seed240402, 9,368 weights |0.311184|0.908590|**0.609887**| original weights |
| TAC 2-timescale frozen seed240401, 9,368 weights |0.344730|0.915040|0.629885| original weights |
| GRU frozen seed240402, 6,360 weights |0.380030|1.006550|0.693290| original weights |
| GRU frozen seed240401, 6,360 weights |0.396250|0.991320|0.693785| original weights |
| Persistence, 0 weights |0.639700|1.751180|1.195440| original physical readings |
| **Official TimesFM3.0** `google/timesfm-3.0-pytorch` zero shot|**1.239519**|**5.172394**|**3.205957**| official loaded pretrained inference |

Numbers above were rounded for display; exact per-model data remain in GitHub artifact [12343014765](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39596526571/artifacts/12343014765) and official TimesFM artifact [12343399436](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39596526571/artifacts/12343399436). TimesFM output is the official multi-variate zero-shot point forecast returned by `TimesFM3Evaluator.predict_batch` for full 12×32 context, horizon60, evaluated at forecast indexes 11 and 59 and differenced from last observed TP3. Official forecast produces exactly 24 valid numerical samples; inference stage records 4.817s model run time (excludes loading/downloading). No real-target tuning or synthetic model training performed.

TimesFM loses to persistence on this specialized **32-readout, high-frequency compressor** pilot, but this cannot be generalized to other context lengths, data domains, or model rankings. TAC was supervised-trained on this exact type of compressor signal; TimesFM3 is general pretrained and has received no MetroPT task-specific adaptation. TAC has larger capacity/training time than original GRU and only 24 observations were scored; per-origin bootstrap or independent host/site inference is absent.

## TabFM official model execution boundary

- Initial attempt to use HF model-card `google/tabfm-1.1.0-pytorch`: **NO numeric result**. Official installed Python package lacks `tabfm_v1_1_0_pytorch`. Original unchanged failure log GitHub artifact [12343219358](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39596526571/artifacts/12343219358).
- Compatibility fallback preregistered after this import failure uses **official TabFM v1.0.0** `google/tabfm-1.0.0-pytorch`, NOT 1.1. On [GitHub run 39597032418](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39597032418) weights downloaded but loader threw `ModuleNotFoundError: No module named safetensors`; original error log retained artifact 12344243765.
- Separate strictly environment-only load correction: [run 39597312619](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39597312619) installs `safetensors`, uses original immutable 24 samples, original train-only 64 example in-context data, flattened 32×12=384 features; the same two TP3 horizons are independent regression contexts. **Do not report a TabFM score unless a new `tabfm10-results.json` is actually uploaded and verified.** Do not claim v1.0 is v1.1.

## Interpretation / limits

This is a real-input, fixed-output *inference-product* comparison, **NOT fair parameter-, pretraining-data-, or compute-matched architecture research**. Google models have vastly larger model and source pretraining. TabFM uses 64 data-specific train example labels as in-context examples whereas TimesFM uses zero training examples at inference and TAC's weights were fully task-trained; conditions are intentionally disclosed, not called matched training. No TabFM result is estimated from its papers, no failed model is assigned an imputed score. Original full 4200-window ME004F results cannot be directly compared numerically to these 24-window TP3-only results.

The official TimesFM3 and TabFM1.0 weights are non-commercial. Do not redistribute their weights, train any model on sealed data, or claim commercial action/control quality, System 2 reasoning or foundation-model competitiveness from this study. All research is draft and unmerged.
