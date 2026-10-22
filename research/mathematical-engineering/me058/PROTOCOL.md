# ME-058 — corrected causal original-FIT prefit on independent robot task

**Registration date:** 2026-10-22, before the first ME-058 source profiler update. Source-only noncomparative diagnostic; `training_authorized=false`.

## Known evidence and source
- Direct parent draft [PR #335](https://github.com/FPC-effortless/TAC-OSM/pull/335): prospective CPU-independent parity-v3, original all-FIT source qualification, pre-feedback dense-state fix.
- Historical anchored native [PR #334](https://github.com/FPC-effortless/TAC-OSM/pull/334): 38,266 TAC / 38,295 GRU active weights; historical CPU overhead 8.007% remains a **failed 5% engineering qualification**. History must not be retrospectively rewritten.
- Original independent publisher source: BodenAI/RW-RL-HIL-Dataset `clean_stains_off_tabletop`, pinned revision `9290f5a96dc91ea58256e81d9ce2dd9621890b51`; FIT/DEV/TEST = 228/76/80 episodes. Previous independent artifact 12451242831 from GitHub Action 39836744099 established original FIT receipt canonical hash `c4064651e91cced7aafc11337c32a27266f470ea10388bb0dd2562ae955f88fd`.
- This experiment will reverify the complete 228-episode FIT receipt, not trust filenames or metadata alone. No DEV/TEST trajectory is to be read during profiling or selection.

## Fixed genuine two-arm profile
- Both physical models process the same original FIT physical observations (14D), executed actions (14D), known control authority, source-derived physical servo prior and 256-frame causal history, with a 16-step action-conditioned rollout.
- The native ME-056 128-frame consolidation is preserved. Its dense transition auxiliary now snapshots the state **before** feeding the future physical target back into verification. The historical post-feedback target information leak must not be reintroduced.
- The native objective includes a 128-transition dense supervision auxiliary from the common FIT window; GRU uses a 16-step rollout without a dense auxiliary. The difference is explicit **architecture plus training objective**, not evidence that memory alone accounts for any future effect. The same original observation/label stream is available to each arm.
- All physical normalization and servo prior statistics are derived from original FIT only, within-episode transitions only. No cross-episode synthetic physical transition is permitted.
- Deterministic outcome-blind window sampler for seeds 58001–58005; 600 planned updates, batch 2, history 256, rollout 16, zero validation hyperparameter tuning. Eligibility uses only current FIT authority at the rollout origin.
- Source-backed pilot performs **one discarded prefit optimizer update on the same first real FIT pair per arm** with PyTorch CPU operator FLOP profiler. This is counted as a diagnostic update, not silently excluded optimizer work.
- Registered parity threshold on active trainable weights: relative gap ≤1%; on prefit profiler-reported FLOPs/update ≤1%. CPU elapsed/process time is logged for deployment efficiency only and is never matched or capped.
- No imaginary FLOP padding or dummy weights. PyTorch profiler does not count every possible operator: matching profiled operations does **not** certify full end-to-end true FLOP equality.

## Existing literature reuse
- R2D2, https://openreview.net/forum?id=r1lyTjAqYX — recurrent burn-in and stale state.
- DreamerV3, https://doi.org/10.1038/s41586-025-08744-2 — jointly learned action-conditioned world dynamics.
- TD-MPC2, https://arxiv.org/abs/2310.16828 — multi-step predictive latent dynamics.

The unresolved TAC question is whether anchored relevance state can predict authentic future physical transitions better than last-state persistence **and** a budget-matched GRU, not whether recurrence is useful at all.

## Next phase (requires a separate predeclared genuine architecture_comparison registration)
Once this source-backed prefit passes, freeze its reported real original source SHA, original sampling-order SHA per seed, FIT physical prior, model counts, profile receipts, the actual optimizer and evaluation contracts, all five paired seeds, and identical training budget into a new real `architecture_comparison` parity-v3 manifest. Only then allow full model fitting on original FIT, followed by all checkpoints frozen before DEV and TEST outcome access. Both sets of trained model results, losses, validation and independent TEST metrics must be retained with raw SHA receipts. If compute cannot be certified, preserve developmental results but withhold fully matched superiority claims.

**Primary practical baseline:** raw persistence `x_(t+1)=x_t`, scored with the exact same real frames, horizons, masks and original physical units. An untrained native or GRU model is not a capability result.

**Policy limit:** future human actions and post-takeover trajectories do not define ex-ante robot-policy outcomes; report physical factual predictions only until first observed takeover. The publisher's four-adjacent-episode split cannot verify operator/session independence.
