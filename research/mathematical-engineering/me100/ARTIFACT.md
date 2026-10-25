# Original repository-owned delivery

Every original fitted model and optimizer is directly tracked under `research/replay-bundles/ME100/seed-100001` through `seed-100005`. The canonical publisher source archive and CC-BY attribution are under `research/source-bundles/ME100`. This delivery does not depend on a private workspace archive or expired Actions artifact.

`FIT_INDEX.json` pins training source code and all five checkpoints. `SHA256SUMS` pins model, receipt, original forecast and result payloads. `FITTED_GRAPH_AUDIT.json` contains actual original-weight source-refinement tests. `RESTORATION_RECEIPT.json` records the separate clean-checkout restoration and the source-sparse-checkout setup correction.

From a complete checkout of the ME100 branch:

```sh
sha256sum -c research/replay-bundles/ME100/SHA256SUMS
python scripts/validate_research_parity.py --manifest research/mathematical-engineering/me100/FAIR_COMPARISON.json --phase preflight
python scripts/validate_research_parity.py --manifest research/mathematical-engineering/me100/POSTFIT.json --phase final
PYTHONPATH=src:. python scripts/me100_entity_sensor.py replay
PYTHONPATH=src:. python scripts/me100_fitted_graph_audit.py
```

Install NumPy and PyTorch as pinned in the ME100 workflow. Replay restores original models, independently recomputes FIT-only preprocessing/ridge references and2015scores, and compares original arrays/results within rtol1e-5/atol1e-6. It never trains, tunes intervals or opens2016–2017numeric outcomes. Under sparse checkout, restore the ME100 replay/source paths from Git first. The frozen training script rejects attempts to overwrite original fits.

No original failed seed is omitted. Parameter-policy PASS, replay PASS and numerical math PASS are distinct from the failed predictive full gate.
