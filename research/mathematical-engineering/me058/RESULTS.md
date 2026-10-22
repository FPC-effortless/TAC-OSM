# ME-058 authentic original FIT resource qualification

**Status: CODE SUBMITTED; NO VERIFIED ME-058 PROFILE RECEIPT YET; NO TRAINED MODELS.**

Pre-registered diagnostic input and provenance:
- Frozen original FIT source SHA manifest: `c4064651e91cced7aafc11337c32a27266f470ea10388bb0dd2562ae955f88fd`, recomputed from all 228 original FIT publisher parquet files.
- Real robot 14D physical/action interface, 256 observed burn-in frames, 16-step factual rollout.
- Registered model active weights: native anchored 38,266; GRU world 38,295; five distinct seeds (58001 through 58005), 600 updates / paired arm and seed in eventual training.
- No new training or accuracy claims. The source-only prefit will explicitly record discarded one-update profiling per architecture, actual source byte digests, fixed FIT window schedules and FLOP parity.

**Prior evidence**: archived PR #334 reported 0.680% profiled FLOP gap on a single genuine FIT episode with a post-feedback dense-target leakage flaw. No expectation that the corrected preflight will pass is presumed. CPU historical gate failure remains archived at 8.007%, not a scientific fairness blocker. A new verified source-backed profile is mandatory.

**Next**: Read GitHub Actions receipts from the ME-058 run. If the test, original source or parameter/profile gates fail, preserve the failure and do not train. If the source and resource gates pass, freeze a separate complete parity-v3 `architecture_comparison` manifest **before** fitting any scored model. Precomputed priors and sample orders must be identical across the arms. Complete-operator FLOP uncertainty must be reported.
