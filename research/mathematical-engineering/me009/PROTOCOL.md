# ME-009 — Native TAC reliability gate (frozen preregistration)

Registered on 2026-10-20 before gate fitting or new future label access.
TAC remains its own world model, not a State Space Model.

## Known failure and proposed correction
ME008F untouched March23–30 physical compressor data: native TAC1000 MAE 0.70078, GRU41-1000 0.755996; both lost to persistence 0.59548. Native TAC nevertheless beat GRU and persistence on significant actual-change conditional magnitude and direction. Hypothesis: overpredicted changes on quiet episodes contribute to the global failure. This is not proven from the existing aggregate data.

Keep the native recurrence entirely frozen. Use a small reliability gate with 161 learned parameters:

    predicted_future_delta = alpha(history, original_delta, masks, horizon) * original_delta
    alpha = sigmoid(MLP(original_delta, abs(delta), prior_volatility,
                        prior_trend, abs(trend), current_mask,
                        observed_fraction, horizon_id))

For every coordinate and horizon, 0 <= alpha <= 1, so the forecast sits between zero-change persistence and original native forecast. This mathematical containment does **not** guarantee forecasting error is smaller than both endpoints.

## Genuine source and original historical calibration
Restore all three original unchanged native TAC1000 SHA-256 verified weights from ME008 original run 39382850579. Official UCI MetroPT-3 publisher ZIP hash frozen. TAC base was trained on first20days. Train only the gate on authentic historical days20–25 records **previously used as validation**; this is calibration/training data and is not independent scientific evaluation. Exact same 12 physical channels, masks, 32-observation history, horizons12/60, original TRAIN-only normalized observations. Gate initial alpha=0.5, training seeds 909001/909002/909003 paired respectively with TAC seed 240401/240402/240403. Fixed 220 AdamW updates, batch64, LR0.002, Huber beta0.2, plus 0.2 weighted change-only Huber on original TRAIN-only quantile-defined real changes; no post-hoc hyperparameter search.

Software preflight: convex gated residual, finite gradients, masked observation isolation, no accidental base gradient updates, bounded alpha.

## Preregistered ONE new future week
After original trained gate checkpoints are hash-verified, score 2020-03-30 inclusive to 2020-04-06 exclusive with zero optimizer steps and source cutoff. Compare persistence, original native TAC1000, fixed half-gated native, learned dynamic gate and GRU41-1000 on three matched seeds. Original direct adapter can be secondary context, but no fresh model selection using later labels. Primary macro MAE, also per-horizon and per-sensor, actual significant-change magnitude/direction, positive/negative class counts and quiet-state false-movement rate.

Source/coding integrity: no data at or after April6 consumed; hash original checkpoints before opening future rows; archive failure traces and unpromoted outputs. If learned gate fails to beat persistence and the fixed half-gate, reject current gate, do not tune on the opened future period. If it wins, only limited one-equipment, one-week evidence; require independent source and action-relevant evaluation before promotion.

## Unified research continuity
All previous TAC-OSM and PNDS mechanisms (predictive entities, CDL-PHR, executable memory, neural CASM, BDH, oscillatory gates, latent attractors, role–filler VSA, verifier, counterfactual costs) remain registered, but none is reintroduced without relevant real-world causal/forecast evidence. No SSM conversion. No automatic merge.
