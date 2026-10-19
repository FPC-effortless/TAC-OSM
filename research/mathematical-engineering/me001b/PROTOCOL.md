# ME-001B — Native TAC assembly recurrence on authentic SCANIA readouts
**Registered:** 2026-10-19, **before any ME-001B model fit or outcome inspection.**
**Evidence class:** exploratory original TRAIN-internal real operational forecast.
**Scientific qualification:** Not a final independent TEST; no model promotion on this arm.

## Hypothesis
A native TAC multi-assembly gated recurrence, with masked real sensor-to-assembly routing, can learn real next-observation changes and preserve useful temporal signal. This is NOT converting TAC-OSM into an SSM; the SSM inspiration is mathematical discipline, not architectural inheritance.

## Authentic data and exclusions
Inherit the fixed public SCANIA Component X v3 extraction pipeline from SEQ010 (scripts/seq010_real_recurrent.py) in stacked PR #242; DOI 10.5878/bnh5-ka77. Extract ONLY train_operational_readouts.csv, hashing the archive and uncompressed member separately. Publisher VALIDATION and TEST and all repair labels MUST remain unopened. Fail closed on schema, chronology or source integrity errors. ZIP metadata hashes may differ across publisher downloads; no source equivalence claim from archive hashes alone. Record actual hashes.

Only 24 first genuine sensors, at most eight past readouts per truck, exact missingness mask, signed-log transform and residual-to-last-observation target. Model sees chronological index order, NOT actual elapsed-time gaps. This does not test entity identification across trucks or multi-entity action selection.

## Frozen split and train budget
New deterministic vehicle-disjoint SHA-256 partition with salt: me001b-vehicle-disjoint-train-internal-20261019-v1 (distinct from SEQ010), ~20% source internal holdout. Previous work on the same publisher TRAIN cohort means this is a NEW EXPLORATORY SPLIT, not an independent external test. No fitting or threshold tuning on heldout trucks.

Seeds: 119001, 119002, 119003.
Fixed 250 AdamW updates, LR=0.001, weight_decay=0.0001, batch 64, smooth-L1 beta=0.1, max grad norm=1. Architecture and hyperparameters frozen before first original run. No seed selection or score-driven reruns.

## Primary arms (identical observational inputs and steps)
1. Persistence, no training.
2. Flattened linear history residual predictor, fixed eight readouts.
3. Global GRU residual, hidden 32.
4. Native TAC assembly recurrence, four assemblies/width 24, masked sensor-feature routing, null association, learned anchors and sensor-conditioned readout.

Native assemblies are NOT discovered truck entities. Architectures are NOT FLOP/parameter matched: report parameters and runtime. Tuned LightGBM and known-ID routed GRU are NOT included in this pilot; register them as subsequent controls, not as tested baselines. Public truck identifiers are used only to partition histories, not fed to models.

## Endpoints and software integrity
Primary: heldout sensor-macro signed-log next-readout MAE, matched observed sensor pairs.
Secondary: micro-MAE, TRAIN-only 90th-percentile absolute-change thresholds for each feature (minimum 1e-4), change-MAE, change sign accuracy, state-midpoint reset control, gradient and boundedness checks, seed variability, parameter count and CPU wall time. Preserve counts and undefined results.

Software-only random-input checks: masked-value isolation, bounded state, finite nonzero gradients, trainable router and sensory zero-input isolation. Software correctness is NOT real-world evidence.

## Scientific decision
No architecture promotion even if this exploratory study wins. Future qualification requires 3/3 seeds to beat persistence, fair compute/capacity controls, robust rare-change metrics and independent reserved real evaluation. With 250 steps, limited sensor channels, previously studied original TRAIN cohort, and no intervention outcomes, this protocol cannot establish commercial or causal control.

Archive every original run, checkpoints, source hashes, scores and failing tests. No overwrites of historical DEV/SEQ results. Do not open official publisher VALIDATION/TEST or reserved SEQ006C/SEQ008 synthetic tests.
