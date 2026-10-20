# ME-008 — Optimize the unchanged native TAC before changing its mathematics

**Preregistered 2026-10-20 before any ME008 fit or new later-week test.** This is a *direct falsification of optimization-limited performance*, not a search for new architecture. TAC-OSM is not converted into an SSM.

## Known limitation from original verified results
ME007 GRU41 (9,249 parameters) at 1,000 steps beat the existing native TAC+direct adapter on real March16–23 data (macro MAE 0.331029 vs 0.341877), after weaker 450-step GRU lost (0.362179). The original native TAC two-timescale recurrence has 9,368 parameters and trained for only 450 optimizer steps; its ME006 direct future-trained residual added 160 more optimizer steps on a frozen backbone. Therefore raw TAC-vs-GRU evidence is confounded by training allocation.

## Exact unchanged native TAC test
Model must be **exact** `NativeTAC("tac_two_timescale")` from `scripts/me004_native_real_forecast.py`, with no modified recurrence, learned module, output head, features, optimizer or objective. Train using authentic UCI MetroPT-3 official publisher TRAIN-only first 20 days, source/hash, same 12 named authentic physical channels, missingness masks, 32 past readouts, prediction horizons 12 and 60 samples, Huber beta=.2, AdamW lr=.001 weight_decay=.0001, batch128, gradient clipping maxnorm 1. Fixed three initializations **240401, 240402, 240403** and exactly **1,000** iterations each; archive snapshots and scores at 450 and 1000 updates. TRAIN-derived normalization and first 20–25 day historically reused validation diagnostic are unchanged. Do not tune against historical validation.

At step450, compare against historic ME004 original native TAC checkpoint for seeds 240401/240402 (parameter equality when torch deterministic); difference is a reproducibility diagnostic. At step1000, compare to ME007 GRU41 weights trained 1000 updates; same optimizer step count but recurrence compute wall clock and parameters are not mathematically identical. Record CPU time.

## New genuinely time-shifted cohort and scientific gate
Before opening future data, record an independent one-time test for **2020-03-23 inclusive to 2020-03-30 exclusive**, subsequent to ME007F March16–23 and after all previous source audits. Restore all frozen GRU41-1000 (3 seeds) and ME008 TAC-1000 (3 seeds), unchanged original reference/persistence; measure mean normalized MAE, two horizons, 12 per-channel accuracy, real significant changes and direction, CPU inference. Never optimize on this future week. Preserve original weights and failures.

If native TAC-1000 beats GRU41-1000 in all three matched seeds, consider the optimizer-budget hypothesis supported (limited single-site, not architecture promotion); if GRU remains competitive or superior, focus next on training objective and where representation encodes critical changes, before adding CDL-PHR/BDH/oscillations/graph components. All unified master-plan ideas remain recorded, with separate relevant evidence gates. No production integration/merge, no repair/counterfactual claims.
