# Initial remote strict-replay exception

CI39654740464 passed1041 tests, source/archive hashes and all584 indexed payload checks, but failed strict original forecast equality on the first ETTh1 native model. Local original and independently restored80-model and16compiled-export replays passed exactly. The initial remote failure log is preserved unchanged.

A post-fit numerical portability check compares original input windows and intermediate arrays for the frozen first model, using public historical DEV only. Explicit CPU dispatch profiles are tested solely to reproduce the same frozen forecast bits, not to select better predictive scores. No training, checkpoint replacement, metric selection or forecast-tolerance change is authorized by this check. Any profile claimed compatible must subsequently pass all80 original forecasts, labels, optimizer/cheap checks and16 saved compiled exports exactly. Failure remains visible until that full verification succeeds.

## Explicit AVX2 serving variant (registered before its references)

The remote CPU uses AVX2; the training workspace uses AVX512. Frozen inputs are identical. Restricting only local PyTorch dispatch to AVX2 reproduces the remote first-model discrepancy exactly:49 changed native-state values,19 coefficients and15 final forecasts, maximum forecast absolute difference1.1920928955078125e-7. None of the tested runner profiles reproduces original AVX512 forecast bits. The original cross-ISA strict check remains FAILED.

Define a separate serving numerical variant before generating its references: ATEN_CPU_CAPABILITY=avx2, MKL_ENABLE_INSTRUCTIONS=AVX2, ONEDNN_MAX_CPU_ISA=AVX2, single thread, same pinned libraries, same original80 checkpoint weights/optimizer records. No retraining, calibration, target change or new test data. Record separate variant forecast/truth/gate/cheap arrays and metrics, plus exact bit differences/max absolute drift against originals. Archive references separately; never overwrite original forecasts or mark the original cross-ISA failure resolved. Existing original timing/1.414x result remains specific to the original environment, not automatically transferred.

CI must replay the new AVX2 references exactly, using original weight/optimizer hashes and all original cheap retention checks. Optional compiled exports must also match the separate AVX2 references exactly. This is an explicitly different execution profile, not a relaxed tolerance for old forecasts. Both original failure records and the AVX2 serving record remain public.
