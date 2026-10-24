# ME082 — completed native capability attempt

Full predictive gate **FAIL / MIXED_TRADEOFF**. No further fits are authorized under this study.

[Original results](CAPABILITY_RESULTS.json), [actual optimizer/parameter audit](FINAL_COMPARISON.json), [source](SOURCE_QUALIFICATION.json), [complete attempt ledger](../../equilibrium-search/RESULTS.md).

Original five model checkpoints, optimizer/RNG, actual learning traces and both8/32-frame histories are tracked under `research/replay-bundles/ME082`. All were Git-frozen before REPORT. Exposed40episode clusters, not new heldout evidence.

The qualified reference promotes original trained float32 weight values to float64 for the entire forward pass, not merely the quantile inversion. Original rtol1e-5/atol1e-6 remains fixed. Prior failed external numeric exceptions are unchanged.
