# ME101 original delivery and replay

Original models, optimizer states, RNGs, exact sampled histories, FIT normalization,
live parameter receipts, original forecasts and complete scores are stored directly
in Git under `research/replay-bundles/ME101`. The licensed publisher archive remains
at `research/source-bundles/ME100/PRSA2017_original.zip`; it is reused unchanged,
not repackaged as a new data source. ME100 originals remain directly tracked.

The training source/registration was frozen at
`493032ff7210604d6123a923d0d1a786607552ea`. FIT_INDEX pins full source-file and
checkpoint hashes. All five originals must be Git-frozen before scoring2015.
There is no private local-only archive dependency or posthoc calibrated model.

On a complete checkout of this branch, with NumPy and workflow-pinned PyTorch:

```sh
sha256sum -c research/replay-bundles/ME101/SHA256SUMS
python scripts/validate_research_parity.py --manifest research/mathematical-engineering/me101/FAIR_COMPARISON.json --phase preflight
python scripts/validate_research_parity.py --manifest research/mathematical-engineering/me101/POSTFIT.json --phase final
PYTHONPATH=src:. python scripts/me101_belief_sensor.py replay
PYTHONPATH=src:. python scripts/me101_fitted_graph_audit.py
PYTHONPATH=src:. python scripts/me101_uncertainty_report.py
```

Restore both ME100 and ME101 replay/source paths if using sparse checkout. Replay
never trains or opens protected2016–2017numerical outcomes. Comparison tolerance
is rtol1e-5/atol1e-6, not bit-exact. The training entry point rejects overwrites;
repeat verification reuses original checkpoints and cannot count as a new fit.
