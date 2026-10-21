# Initial remote strict-replay exception

CI39654740464 passed1041 tests, source/archive hashes and all584 indexed payload checks, but failed strict original forecast equality on the first ETTh1 native model. Local original and independently restored80-model and16compiled-export replays passed exactly. The initial remote failure log is preserved unchanged.

A post-fit numerical portability check compares original input windows and intermediate arrays for the frozen first model, using public historical DEV only. Explicit CPU dispatch profiles are tested solely to reproduce the same frozen forecast bits, not to select better predictive scores. No training, checkpoint replacement, metric selection or forecast-tolerance change is authorized by this check. Any profile claimed compatible must subsequently pass all80 original forecasts, labels, optimizer/cheap checks and16 saved compiled exports exactly. Failure remains visible until that full verification succeeds.
