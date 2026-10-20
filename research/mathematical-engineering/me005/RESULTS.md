# ME005 original authentic-data integration results — complete exploratory evaluation

**Source/provenance:** [Original completed GitHub Actions 39377990208](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39377990208); archive \`12244152736\` (\`me005-original-four-real-adapters\`). Original runner/code SHA \`6d9788cf78d9c5540072388dad055e32b4412264\`. Four complete real model adapter fits, 240 optimizer steps each, exactly **zero original pretrained backbone refits**. All four original checkpoint SHA256 digests independently checked against the original results JSON.

**Real data:** Same official UCI MetroPT source bytes and previously explored 12-channel TRAIN/validation split used in ME004, 190,443 source rows; 12,412 real training windows and 3,235 real exploratory validation windows; 12/60 future-observation horizons. No records beyond ME004's first 25 days opened. **This validation split has already been inspected before ME005; it is not independent scientific test evidence.**

## Mean absolute error in TRAIN-normalized sensor units

| Arm | Seed 505001 using frozen base 240401 | Seed 505002 using frozen base 240402 |
|---|---:|---:|
| Original trained two-timescale TAC (never refitted) | 0.2687310955 | **0.2714250019** |
| Residual-only adapter, 240 updates | 0.2685136241 | 0.2722446860 |
| Full unified TAC adapter, 240 updates | **0.2685131161** | 0.2721002301 |

Both full and residual-only adapters used the exact same original inputs, batch, optimizer/loss schedule and 17,546 parameters eligible for gradients at the model-definition level, but the residual-only gate disables many paths, so the comparison **is not matched active capacity**. Each full adapter's improvement relative to residual-only is **negligible** (about 0.000000508 in seed 505001 and 0.000144456 in seed 505002). The full adapter fails to consistently improve over the original pretrained TAC model (one improves, one gets worse). All improved substantially over persistence only because they retained the already-working pretrained base. Thus **do not attribute original ME004 performance to the novel ME005 additions**.

## Operational limits
- Full adapter train wall time ~32.89/33.34 seconds per 240 updates, residual-only ~21.04/20.96 seconds — roughly **1.57× extra CPU training time** with no meaningful incremental predictive quality.
- Large future change MAE / sign accuracy:
  - Seed 505001: residual 1.77857555 / 0.9121989; full 1.77940428 / 0.9111629. Full **slightly worsens** both.
  - Seed 505002: residual 1.77042186 / 0.942761; full 1.77017940 / 0.943279. Full improves only slightly.
- The backbone remains frozen, residual initially zero, and source preprocessing/normalization must reproduce original; both baseline-preservation checks passed.
- No independently observed maintenance action effect, cost, causal intervention, learned operator composition, semantic entity identification, System 2 reasoning, physical safety, lifelong consolidation, or multienvironment validation demonstrated.
- Existing ME004F one-source future-week result remains valid for the original frozen model; NO future data were opened by ME005 and no ME005 gain on any new heldout cohort can be claimed.

**Result/decision:** **ENGINEERING INTEGRATION PASSED; INCREMENTAL ARCHITECTURE VALUE UNPROVEN/FAILED PROMOTION GATE**. Keep original frozen ME004 forecasting checkpoint as default. Keep all unified mechanisms as explicit, independently ablatable research components; do NOT merge the unified model or promote its trained adapter for deployment. Preserve all negative outcomes.

## Efficient next work

1. Do **not** keep modifying the repeatedly used validation data until a desired metric appears. Check whether the learned modules get useful gradient and genuinely affect outputs via targeted **recorded-only** ablations on frozen checkpoints and train-only diagnostics; no refits.
2. Define supervised/contrastive *real* objectives for entity correspondence, CDL utility, compositional operators and reasoning only where corresponding ground truth exists, rather than assuming future sensor MAE will train every capability.
3. Strengthen E/P verifier's evidence authenticity interface to be fail-closed unless an independent receipt validator is provided; mock receipt IDs alone do not prove real verification.
4. Compare against compute- and capacity-matched GRU and boosted forecasting controls on a newly registered real cohort **before** claiming the integrated model outperforms simple alternatives.
5. Use logged actions/outcomes separately to train CASM, critic reliability, counterfactual policy value and real decision costs; never manufacture them from sensor trajectories.

Original source hash, checkpoint digests, per-fit loss curves, parameter counts, baseline comparisons and complete scores remain available in the immutable Actions archive.
