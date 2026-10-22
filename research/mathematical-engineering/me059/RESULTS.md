# ME-059 independent real robot model results

**Status before new GitHub Actions run: REGISTERED, NO VERIFIED FULL MODEL FITS.**

Source and training are frozen in `FAIR_COMPARISON.json` with original manifest and per-seed training schedule hashes. Do not infer predictive improvement from the preceding ME-058 profiled update. The registered compute comparisons are based on incomplete CPU operator counts and can establish prospective reported-operator parity, **not full actual true training FLOP certification**.

### Acceptance table (to fill only from original measured run artifacts)

| Gate | Source/evidence | Status |
|---|---|---|
| Original 228 FIT files hash | ME058 FIT source receipt, original bytes | PASS in ME058 |
| Pre-feedback dense state invariance | ME057/ME058 numerical tests | PASS |
| Parameter match ≤1% | ME058 actual FIT update receipt | PASS 0.076% |
| Partial profiler reported FLOPs ≤1% | ME058 full shaped authentic FIT update | PASS 0.680%; exact full FLOPs unverified |
| Original independent FIT full model fits | ME059 training job | NOT YET RUN |
| Ten original checkpoints and exact per-seed window order | ME059 training artifact | NOT YET VERIFIED |
| Postfit resource accounting | ME059 full training receipt | NOT YET VERIFIED |
| Sealed DEV/TEST conditional physical MAE | Original ME059 recorded robot outcomes | NOT YET MEASURED |
| Raw persistence and physical servo baselines | Same real physical evaluation frames | NOT YET MEASURED |
| Scientific full-FLOP architecture superiority qualification | Independently measured full training FLOPs | **NOT CERTIFIED** |

Never silently replace these with forecast guesses or synthetic successes. Preserve original errors, including training stops, missing checkpoints and worse-than-persistence outcomes.
