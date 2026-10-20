# ME-005C — post-training mechanism influence on real validation

**Independent recorded-only GitHub Actions run:** [39378702013](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39378702013). Artifact \`12244113709\` (\`me005-original-recorded-only-ablation\`). Both full original trained checkpoint weights and the original authentic UCI MetroPT 20–25 day validation data were independently restored and replayed: original scores reproduced to within the audit tolerance, **zero model refits, zero future test exposure**.

The test disables one pathway *after training* and reports the change in normalized macro MAE relative to the same model with that pathway active. This is a sensitivity/necessity probe, **not** isolation of why the pathway was learned, nor evidence of task-specific reasoning.

| Pathway disabled | Seed 505001 ΔMAE | Seed 505002 ΔMAE | Interpretation |
|---|---:|---:|---|
| Sparse assembly graph | **-0.000500636** (improves) | +0.000652796 (worsens) | Unstable/inconsistent; not a reliable entity-organization gain |
| Executable-operator mixture | +0.000096585 | +0.001071462 | Measurable beneficial influence in this fixed model, *not* verified compositional generalization |
| Hebbian predictive fast association | **-0.000018912** (improves) | +0.000178499 (worsens) | Tiny and inconsistent effect |
| Adaptive attractor refinement | +0.000055223 | +0.000047910 | Tiny positive effect; no causal multi-step reasoning demonstration |
| Oscillatory gates | **-0.000002980** (improves) | **-0.000001903** (improves) | Essentially zero; rhythmic modulation adds no measured value on this task |

The two full checkpoint predictions were not identical to all ablations. Observed mean absolute output changes: graph ~0.00475/0.00505; operators ~0.00229/0.00312; plastic ~0.000323/0.000505; refinement ~0.00055/0.000467; oscillatory gates ~0.0000288/0.0000247. No gradient/train fits or future datasets were accessed in this comparison.

**Scientific decision:** The native TAC-OSM trained ME004 forecasting backbone remains the working research default. Simply stacking all mathematical ideas is *computationally feasible but scientifically unqualified*; the full ME005 model improves against a residual-only adapter only negligibly, and its graph/plastic/rhythm modules do not earn promotion. Preserve all ideas as registered alternatives and integrate them only after their real-data-specific objectives and strong controls pass.

**Next action to advance the architecture rather than rediscover it:** train entity-correctness on genuine multi-entity logs, CDL-PHR on predictive conditional likelihood with fixed noncollapsed targets, operator binding on real observed compositions, adaptive refinement on genuine long-horizon decisions, critics on verified failure outcomes, and PNDS on trusted independent receipts. Sensor-only telemetry cannot establish human-language logic, causal decisions or safe tool calls. Spend compute on the existing working world-state/control agent and explicitly targeted bottlenecks, not a continually expanding assembly of unproved modules.
