# GP-UNC-001 — six TAC-OSM uncertainty research contracts

**State:** mathematical / anti-leakage NO-FIT tests only; **not** evidence of real-world GP efficacy, calibrated forecasts, zero-latency inference or formally verified TAC-OSM. Nothing in this branch trains, opens protected TEST, modifies historical weights or uses Kaggle quota.

## Study question and predeclared five arms

Across six distinct families (**System 1, Enterprise, Temporal, Physical, Edge-Lite, Counterfactual**), compare the *eventual* original-source forecasting evaluation of:

1. Original TAC point forecast (no probabilistic claim).
2. TAC + Gaussian parametric variance head.
3. TAC + bounded inducing-point GP **on residuals**, using only source-time-legal training/CAL features.
4. TAC + split/online-appropriate conformal interval method, with exchangeability and temporal drift limitations explicitly stated.
5. A matched GRU with a probabilistic head, and mandatory persistence/classical controls on the same original task.

The code registers **30 proposed family × arm cells**, all **NOT_FITTED_OR_QUALIFIED**. Their names are not executed models. The `posterior` mathematical kernel is an **exact dense small-n reference** (maximum 128 observations), not the proposed sparse GP residual model. Neither a sparse GP, neural Gaussian head, GRU, nor full conformal forecaster has been fitted by this study.

## Research constraints / available genuine-data paths

- Temporal: original recorded multi-site pollution work, e.g. LATFUSE018–019; audit current source and exposed DEV limitations before freeze. LATFUSE019's rainfall extreme NLL and undercoverage are a *motivation*, not a validated GP correction.
- Enterprise: BPI2012 source-time event/waiting-time program; qualify actual zero-time event law and site/case split before choosing a likelihood.
- Physical: original robot logged episodes and action source; distinguish recorded actions from counterfactual commands.
- System 1: choose an original source with real decision-time cutoffs and response-budget limits; no unsupported keyed-recall generalization.
- Edge-Lite: same valid original forecasting cohorts as a qualified parent, with fixed deployed memory/latency ceilings.
- Counterfactual: BLOCK learned intervention-value claims until logged propensities/support, observed actual actions/outcomes and independent evaluator are present; ordinary predictive intervals are not causal uncertainty.

These are **candidate sources**; no new source hashes, licences, entity splits, source authorization or signer receipts are claimed in this registration. Separate per-family preregistrations must be frozen before fitting. Do not pool evaluation results across incomparable tasks to claim architecture superiority.

## Prospective head-to-head gate

Pre-register exactly matching publisher source, entity/time cohorts, FIT/CAL/DEV/sealed TEST, causal features and labels, train-only preprocessing, horizon, mask/censoring policy, seed sets and source SHA256. Actual neural models must individually be 95k–105k trainable/deployed learned parameters and each matched arm within 1%, including GP feature projections and likelihood/inducing learned parameters when trainable. Report nonlearned inducing state bytes separately. Match training data/exposure/tuning budget and FLOPs; CPU wall time is an efficiency metric, not a parity gate. Require exact checkpoint hashes/replay, actual anti-cheat adapter, PNDS Lean+Python exact source mapping, original independently signed source/outcome/reward/ACL receipts before optimizer step and before model qualification.

Metrics per eligible family: task-appropriate MAE/RMSE/log-loss, NLL or CRPS, coverage+interval width per horizon and entity/site/channel, sharpness versus calibration, risk–coverage under abstention, extreme-event tails, FIT/CAL/DEV/TEST identity, latency CPU/accelerated and peak RAM. Never calibrate on repeatedly exposed DEV then call a rescore independent generalization. Predeclare numerical success thresholds in each real-data registration; no new winner is asserted here.

## Executable no-fit checks

```bash
python -m unittest discover -s tests -p 'test_gp_uncertainty_nofit.py' -v
python scripts/validate_research_parity.py --manifest research/mathematical-engineering/gp-unc-001/REGISTERED_NOFIT.json --phase preflight
```

The reference checks Gaussian algebra and constructed source-time-contract falsifiers. **OPEN:** source-equivalent full TAC head, sparse inducing GP, calibrated decisions, PNDS Lean derivation/FP/AD correspondence, independently qualified real FIT/CAL/DEV data per family, actual parameter/compute receipts and signed evaluator gates. Preserve original negative findings and sealed TEST untouched.
