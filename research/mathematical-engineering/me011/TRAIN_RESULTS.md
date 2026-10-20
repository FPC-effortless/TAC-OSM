# ME-011 — Original real transformer training and historical calibration evidence

**Original run:** [GitHub Actions 39388044790](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39388044790), original model source commit `957d45e939e66d8f0bbce8d1f7537bed6e2f6d69`, artifact `12248760583` (`me011-original-etth1-quantiles`). All six exact PyTorch checkpoint SHA-256 values independently verified against original source ledger. Mathematically required q10 <= q50 <= q90 held with zero crossings.

## Authentic data and causal contract
Official ETDataset hourly ETTh1 recorded electrical load and oil-temperature sensor readings, publisher raw source MD5 `8381763947c85f4be6ac456c508460d6`, source SHA-256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`. Original trainer parsed rows **0–11639 only**. **TRAIN** rows 0–8759 (1,417 causal subsampled prediction windows); **historical calibration/dev** rows 8760–11639 (452 evaluation windows), using TRAIN-only normalization. Original later ETTh1 test rows >=11640 and separate physical ETTh2 site were never parsed/opened by original trainer. No trained MetroPT weights reused: second physical task retrained the same native two-timescale recurrence type from scratch.

Real inputs: 96 hourly previous readings of 7 actual sensor channels + availability masks, oil-temperature signed deltas 24/168 hours into future, 0.1/0.5/0.9 quantiles. Three seeds 11001/11002/11003, 450 fixed AdamW updates each, pinball objective, gradient clip1, identical training batches. Native TAC two-timescale **7,558 trainable parameters**, GRU40 **6,966** (8.5% smaller); optimizer-step matched not parameter/FLOP equal. Original CPU training native **9.78/9.59/9.87 seconds** vs GRU **5.29/5.47/5.62 seconds**, so native ~1.78× slower in these recorded fits.

## Historical development results — unpromoted

| Model | Median error (°C), 3-seed average | Nominal 80% interval coverage, 3-seed average | Average pinball, TRAIN-normalized |
|---|---:|---:|---:|
| Native TAC | **2.883** | 66.48% | 0.1081 |
| GRU | **2.795** | 72.97% | 0.1030 |

Original per-seed native MAE **2.945802 / 2.775937 / 2.927451**; GRU **2.905151 / 2.786039 / 2.692983**. Native coverage **0.674779 / 0.670354 / 0.649336**; GRU coverage **0.721239 / 0.703540 / 0.764381**. Both models UNDERCOVER nominal 80%, GRU generally closer in this already inspected development period.

**No model qualified** from historically exposed dev data; later time and cross-site tests are registered separately in ME011F. Zero new executive world-model claims; persistent entity memory, learned CASM, executable operations, CDL-PHR, PNDS Lean and neuroplasticity master-plan sections remain preserved.
