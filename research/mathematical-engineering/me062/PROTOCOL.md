# ME-062 — selective physical residual forecast: prospective registration

Status: **UNTRAINED CANDIDATE**. The source and software tests in this branch are not evidence of MAE, superior memory, calibrated uncertainty, decision utility, or Woyengi integration.

## What the evidence actually supports

ME-061 source DEV (76 episodes): TAC 0.603894, GRU 0.602600, persistence 0.702952, servo 1.105522. Frozen tea-pouring (80 episodes): TAC 0.493206, GRU 0.509535, persistence 0.455992, servo 1.421244. Source results are historically exposed DEV, not independent TEST. Transfer supports an advantage over this GRU comparator on this task, not over strongest persistence, not independent operators/sites and not native memory causal attribution. Keep ME-057, ME-060 and ME-061 negative superiority gates unchanged.

## ME-062 hypothesis (one integrated candidate)

Preserve ME-061 bounded native fast/slow/8x8 associative state, factual action-prefix conditioning, direct horizon prediction and physical-unit loss. Add the SAME causal reliability and heteroscedastic Gaussian design to native TAC and trained GRU. For time t and horizon h:

    d_h = f_ME061(s_t, a_{t:t+h}) - x_t
    alpha_h = sigmoid(g_theta(x_t, servo_delta_h, v_t, e_t, log(1+sigma_h), h))
    mean_h = x_t + alpha_h d_h
    sigma'_h = sqrt([sigma_h exp(tanh(q_theta(features_h)))]^2 +
                     [(1-alpha_h)d_h]^2)

All features are available when a forecast is requested; no future *observations*, losses, hidden oracle outcomes, or unrevealed human interventions enter the input. Observed historical executed actions and explicitly proposed future actions are separate. Alpha is **trust in a learned correction**, NOT confidence in factual causality, an epistemic uncertainty guarantee, a policy score, or authorization to execute.

Initialize the inherited correction output to zero, so both arms exactly predict persistence before fitting. The gate's initial value 0.5 does not block learning through the residual decoder. Scales are positive Gaussian standard deviations and require fresh calibration; do not reuse ME-061's Laplace conformal quantiles.

Loss for both architectures: physical-coordinate MAE plus 0.01 times TRAIN-only physical-step mean times heteroscedastic Gaussian negative log score. Include the identical 16-horizon objective and equally weighted dense one-step pre-feedback auxiliary (coefficient 0.25) as ME-061. Changes relative to ME-061 are joint: do not attribute any performance change solely to the gate without a matched ablation.

## Mandatory source and fairness registration BEFORE fits

- Authentic publisher source: BodenAI/RW-RL-HIL-Dataset revision `9290f5a96dc91ea58256e81d9ce2dd9621890b51`; use the pinned `clean_stains_off_tabletop` FIT-only source partition from ME-061 for training. The old TEST remains exposed and MUST NOT become the independent holdout.
- Keep independent whole-episode/time-group calibration and historical DEV partitions. Source normalization/servo identification is fitted on FIT only. No CAL/DEV/target data may be opened before **all** arms/seeds' selected checkpoint hashes freeze.
- Compare five paired seeds, identical 64-transition input histories, 16-step factual targets, actions, takeover censoring, sample order, loss, batch eight and nominal 600 AdamW updates. Selection limited to resource-match width adjustment, not outcome-based tuning. Exact seed set, hashes and schedules must be frozen in an immutable `FAIR_COMPARISON.json` **before** training; ME-061's seeds may not be reused to pretend new independence.
- Compulsory approximate 1% equality for trainable parameters and supported-operator training FLOP accounting, matched actual training examples/tokens/updates/data, and clean final parity receipts. True all-operator/hardware FLOPs reported UNKNOWN unless independently measured. Record CPU overhead separately; no forced CPU runtime parity.
- Baselines: trained equal-budget GRU with same head/features/loss; last-state persistence; FIT-derived servo; and a same-budget ME-062 no-memory control for hypothesis-specific attribution. Any no-memory control must be declared and trained in the same fair comparison, never cherry-picked after labels open.
- Score original-unit MAE by episode/horizon/channel and movement magnitude; report counts of motion/near-stationary origins from FIT-defined thresholds (predeclared, not TEST tuned). Report gate alpha distribution, gate versus *future factual* improvement on heldout labels (diagnostic only), movement error, and absolute uncertainty calibration. NEVER select a gate threshold from exposed TEST.
- CAL-only whole-episode split conformal, with documented finite-sample quantile, and 90% episode coverage, widths, reliability curves and risk-coverage/abstention on DEV and prospective transfer. Do not claim guarantees where trajectory independence/exchangeability is unsupported.
- Conduct (k=1,4,16,64) history-length tests in **two separate lanes**: (1) fixed trained-model truncation with reset memory (diagnostic); (2) full independently matched training at each k (memory necessity test). Do not confuse either with long-term recall or with richer source features. If newly trained controls add arms/seeds, satisfy the repository-replay storage policy for all checkpoints.
- Register a new untouched real physical task **before** accessing its trajectory labels, distinct from already exposed source DEV, old TEST, clothes-hanging and tea-pouring. Frozen source checkpoints only; no target normalization, calibration, thresholds, checkpoint selection or updates. Report domain/hardware/site/operator shifts explicitly, without falsely claiming site independence.
- Primary prospective transfer gate: TAC at least 2% lower physical MAE than **each** of GRU, persistence, and servo, with episode-clustered TAC-minus-strongest-comparator 95% interval below zero; no unacceptable risk-coverage degradation. Source DEV is diagnostic even if it passes. Gate is FAIL if persistence wins, even if GRU loses.
- Policy/causal claims remain BLOCKED absent authentic candidate-action support, logged propensities/intervention costs and verified outcome/authority records; Veritas generated rollouts are engineering and synthetic RL evaluation, **not real-world causal proof**.

## Repository checkpoint gate — must be satisfied before training is declared complete

The existing binding [repository replay policy](../REPOSITORY-REPLAY-AND-CHECKPOINT-RULE.md) controls this study.

Before an ME-062 full fit, reserve `research/replay-bundles/ME062/` in the TAC repo; verify commit/upload rights and capacity, source licensing, expected number of model/seed checkpoint files, restore loader, full optimizer and RNG states if exact training replay is claimed, replay commands, manifest hashes and independent clean-checkout verification. Include all compared arms and rejected/failed recoverable checkpoints. Commit original checkpoint **bytes**, not only their hashes, Actions artifacts, or a local-only archive. Pinned legal source-acquisition and bitwise or tolerance-registered replay must be documented; do not commit publisher trajectories without a redistribution right.

Record separately:
- `TRAINING=NOT_RUN` until authentic fits occurred.
- `REPO_REPLAY=BLOCKED` until checkpoints and independently verified repo-only restore exist.
- `FULL_EVALUATION_REPLAY=BLOCKED` if originals are inaccessible legally.
- `SUPERIORITY=UNKNOWN` before real heldout scoring, otherwise exact PASS/FAIL.

## Woyengi and Veritas boundary

An authorized Woyengi state reconstruction supplies a time-cut, identity-scoped, feature-schema-versioned observation history and candidate action sequences. The ME-062 prototype envelope in `woyengi_forecast_contract.py` checks that supplied time and shape constraints, **not the validity of Woyengi permission receipts**. Only Woyengi can authenticate, authorize, sanitize, choose an allowed action set, verify fresh state revision, and append governed action/outcome records. TAC responds with forecast proposals, risk/uncertainty and checkpoint/replay identifiers; it has no write or execution rights. The model may not consume security tokens.

Veritas may replay observed traces and conduct sandbox action-proposal comparisons and synthetic RL stress tests. Counterfactual predictions are model hypotheses until supported by real data/identification; Veritas simulator success does not qualify real intervention value. Candidate comparisons are safe read-only until Woyengi separately authorizes any real action.

**No automatic deployment, merge, or real-world action authorized by this candidate.**
