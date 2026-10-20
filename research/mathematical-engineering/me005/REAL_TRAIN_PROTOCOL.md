# ME-005B authentic-data integrated adapter experiment — frozen protocol

**Registered 2026-10-20 BEFORE any ME005B original model fitting.** 
**Scientific evidence classification:** exploratory real TRAIN history on an **already-used** ME004 calibration cohort. This is NOT a new blinded holdout or promotion test. All previous ME004 trained checkpoints, audits and source/provenance history remain unchanged.

## Objective
Measure whether the complete, unified native TAC mathematical engine is trainable *without destroying the already-working ME004 predictor*, and whether the complete integrated adapter materially outperforms a lightweight residual-control under identical authentic observations.

## Real source and no-leak contract
UCI MetroPT-3, exact official archive SHA256 aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a. Same 12 physical signals, 32 historical observations, next-readout deltas at 12 and 60 future observation steps, chronological real TRAIN-only days 0–20 and previously inspected validation days 20–25 as the original ME004 protocol. Days after 25 are not opened in this training study. Original ME004 evidence artifact 12235023980, fit SHA 954baafe63d1450e4968cc1b49dac02a5e8f6f1d, read and checksum each base checkpoint before use. Frozen original two-timescale TAC base weights **never updated**.

## Strict fixed arms
A. **Residual-only control:** Same pretrained real-data backbone and residual-head math with graph, local plasticity, executable operators, oscillatory phase and attractor refinement all deactivated. Difference from baseline is a new trainable residual.
B. **Unified-all research architecture:** Same pretrained base plus sparse entity-like assemblies/null route, sparse graph, bounded CDL-PHR-style predictive Hebbian fast associations, VSA signed role bindings and learned operator mixture/no-op, four independent oscillatory gates, attractor-energy refinement with adaptive halting, forecast uncertainty head and continuous critic. The action-conditioned outcome head stays untrained/inactive because these observations contain no logged actions and verified outcomes. External Veritas authorization and E/P receipt requirements are not relaxed.

Two adapter seeds 505001 and 505002 paired to frozen original base seeds 240401 and 240402; two arms per pair (4 adapters). Exactly 240 AdamW updates each, batch 64, LR 0.0005, decay 0.0001, Huber beta .2, norm clip 1. Proper Gaussian uncertainty auxiliary weight .008, compute auxiliary penalty .0005, no post-hoc loss adjustment; init residual exactly zero. Record parameter/CPU budgets, unmodified pretrained score, multi-horizon MAE, selected large-change MAE and direction, and every result even if worse.

## Falsification and next allocation
Pass engineering if baseline output is preserved **before fitting**, gradients are finite, all four adapters finish, checkpoints/reproducibility evidence are hashed, and no future TEST is opened. A useful exploratory architectural increment must beat original base and residual-only on BOTH pairs at comparable compute, without making large-change direction worse. **No scientific promotion regardless**, because the validation period is reused, module effects are not individually attributed, and parameter counts/time are not matched. If unified-all fails, retain its negative result and revert to compact base or single justified additions instead of discovering new modules. If it succeeds, isolate graph, plasticity, operator and refinement ablations using TRAIN-internal validation only before any one-shot future test.

## Mathematical invariants
Convex native base states/slot updates stay in [-1,1] when initialized there. Fast associative state is projected onto Frobenius norm <=2. Sparsemax graph probabilities lie on a simplex. VSA bipolar roles have exact unbinding. Frozen base and zero residual guarantee identical original forecasts at initialization. Contractive refinement holds only for its **fixed chosen attractor target**; changing targets/critics do not guarantee physical safety or logical deduction. Receipt-only persistent-memory promotion and action authorization are external software requirements. All mechanisms are research hypotheses and must be measured, never announced as spontaneous intelligence.

No synthetic task can promote a model. Keep all historic DEV/SEQ/ME weights, negative findings and hypotheses, and do not merge without independent verification.
