# ME063 original robot physical-regime mixture — prospective research protocol

## Scientific reason for this experiment
ME062 completed 5x2 authentic FIT-trained checkpoints and independent score replay but **FAILED** both superiority gates: source DEV TAC 0.609373 / GRU 0.605164 / persistence 0.702952; frozen task trash-bag TAC 1.256390 / GRU 1.223530 / persistence 1.295032 / **servo 1.031359**. Source CAL-based nominal 90% TAC episode coverage collapsed to 30% on new task.

The **causal prior selection** problem changes across physical tasks: persistence can dominate (ME061 tea pouring) but a FIT-identifed physical servo prior dominated ME062 trash-bag transfer. The previous gate only interpolated neural predictions and persistence and could not fully choose the servo baseline. This does not prove feature routing or TAC memory superiority.

## Hypothesis (integrated mathematical change, not rediscovery)
Retain native recurrent fast/slow/associative memory, 64 observed physical transitions, innovation/velocity observer, action-conditioned 16-step model forecast, and a trained matched GRU. For each factual proposed action prefix a[0:h], compute three regime means:
- P: x_t (persistence)
- S: FIT-calibrated servo rollout x^servo_{t+h}, using only current state and known action prefix
- N: native TAC/GRU action-conditioned recurrent predicted x^neural_{t+h}

A shared **observational, horizon-conditioned** router g produces weights w=softmax(g(x_t, x^servo-x_t, historical velocity, historical innovation, forecast scale, h)); prediction is w_P P + w_S S + w_N N. All arms see identical physical and historical signals. Positive variance is a moment-matched mixture of predictive scales plus mean-expert disagreement; training objective remains shared physical MAE plus a FIT-scaled Gaussian score, with dense physical supervision identical across arms. The output is not a causal guarantee or authorization. No physical target labels or future observed outcomes enter the router. No post-hoc comparison to the already exposed ME062 task will qualify independent transfer.

Established methods: model averaging, classical observer gains, mixtures of experts, and probabilistic predictive scoring; no component novelty claimed. See Jacobs et al. Adaptive Mixtures of Local Experts (https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf), Lakshminarayanan et al. Deep Ensembles (https://arxiv.org/abs/1612.01474), R2D2 recurrent replay (https://openreview.net/forum?id=r1lyTjAqYX). Novelty question is whether the **native TAC recurrent state**, under compulsory matched training and resource budget, is actually more useful than GRU after sharing the same adaptive physical regime-selection design.

## Mandatory prefit checks
- SOURCE: authentic BodenAI/RW-RL-HIL-Dataset pinned revision 9290f5a96dc91ea58256e81d9ce2dd9621890b51; original source `clean_stains_off_tabletop`, FIT hash b55d0e1f894072c5738ef8e45aa283a3b464c1a1dc8f60c63594666d16b30a49; 208 FIT, 20 CAL, historical 76 DEV, previously exposed old TEST prohibited.
- **Before any fit**, measure exact parameter counts, supported-operator forward+backward FLOPs, gradient activity, and original source sample-order hashes for seeds 63001..63005. Fail closed if any >1% gap in parameters or supported-operator FLOPs, or unequal input data/steps. CPU runtime is efficiency data, NOT a parity gate. Do not falsely label full-hardware FLOPs certified.
- Persist full comparable `FAIR_COMPARISON.json` **from original numeric receipts before first optimizer step**, then training workflow must call scientific validator before fitting and after actual fit.
- Five paired seeds, same 600 optimizer updates and batch 8, 64 historical transitions + 16 target horizons, same model feature/target masks, same zero tuning allowance, same FIT sample order per seed.
- The no-memory control and history k=1,4,16,64 must be a later separately matched and resource-registered experiment; do not infer unique memory mechanism from this two-arm experiment.

## Checkpoint and source retention
- Before training verify ability to Git-commit real model/optimizer/RNG bytes under `research/replay-bundles/ME063/`; expected first arm count 2 x 5 = 10, approximately 0.5 MiB each based on ME062 but actual byte size unknown until measured; do not treat historical Actions artifacts or hash-only receipts as sufficient. No publisher parquet published unless lawful.
- Freeze all fits, save last/selected weights and optimizer/RNG states with original source and schedule receipts, commit checkpoint **bytes** into TAC repo before opening CAL/DEV/transfer labels. A *separate clean GitHub checkout* must load all and verify hashes before scores are opened.
- Register a new unused real robot task via publisher **metadata only** before opening a single future target trajectory. Exclude historical source, clothes hanging, tea pouring, trash bag and any formerly selected target (verify research registry). Frozen source checkpoints only; no target normalization, fitted thresholds or updates.
- Score original physical MAE by seed/episode, persistence, servo, matched GRU, and uncertainty/risk coverage. Predeclared main gate: TAC >=2% lower MAE than **every** comparator including strongest practical baseline and paired episode-cluster CI upper bound <0 on genuinely new task, with acceptable source-calibrated uncertainty. Source DEV is exposed development and cannot establish independent model superiority.
- Real recorded logged future actions are **observational factual covariates**, not counterfactual policy-effect ground truth. No Woyengi auto-execution or Veritas simulated policy win as proof of real-world action utility.

## Status
`TRAINING=NOT_RUN`, `TEST=UNOPENED`, `REPO_REPLAY=NOT_PROVEN`, `SUPERIORITY=UNKNOWN` until actual measurements and replay change these labels. The branch is draft and unmerged.
