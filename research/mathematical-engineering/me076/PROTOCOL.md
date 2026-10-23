# ME076 — Native System 1 on authentic original robot one-step outcomes

**Preregistered before first ME076 fit · 2026-10-23.** [Source](https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset) CC BY 4.0. Cite BodenAI *and* the RW-RL parent. All forecasts are retrospective development on previously exposed source FIT; *no fresh independent model performance, no production safety or sub-ms latency*.

## Hypothesis and intended capability

Can a native **100,476-trainable/deployed parameter**, *fast-only* TAC-OSM System1 recurrent model predict **one actual physical next-state frame** using eight actually observed state frames and prior known issued action qpos better than persistence, especially on genuine movement-rich cases? This is not a policy-controller validation. The genuine original robot dataset does not supply verified task completion, self-supervised risk utility, real torque or autonomy permission.

This experiment specifically addresses the original ME068 problem: 91.23% near-stationary original autonomous FIT windows and physical one-step persistence ~0.0702 MAE compared with lag1 servo ~0.1191 MAE. Those FIT-only prior figures were calculated differently from ME076's balanced, episode-disjoint REPORT; they are *not numerically comparable* without rescoring identical cases.

## Registered source and exclusions

- Publisher original immutable five-task `BodenAI/RW-RL-HIL-Dataset` commit `9290f5a96dc91ea58256e81d9ce2dd9621890b51`; ME068 canonical original FIT 160-episode SHA256 `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`.
- No video/person records downloaded. Only already previously exposed original registered *FIT* 160 original parquet episodes with verified source publisher episode metadata and actual intervention authority flags, 32 episodes per task.
- Episode-level prospective deterministic partition per task: sort `sha256("ME076/episode-split/{task}/{episode_index}")`, first 24 original FIT for optimizer `FIT`, remaining 8 for `REPORT`. **120 FIT, 40 REPORT**, original episode identities disjoint. Source original historical FIT was exposed by prior ME studies, so REPORT is explicitly **DEV** not an independent final holdout. No previously unexposed tissue-box target, original CAL, old TEST, or other task opened.
- FIT-only source mean/std, available issued action normalization, servo gain and physical scales; REPORT does not influence those estimates. Original publisher content/source hash verified before any fitting. Training outcome labels from FIT only.
- At `t`, measured current and seven previous states `x[t-7:t]` and eight prior *issued/logged* actions `a[t-8:t-1]`; lag1 is definitely not a future logged action, but does not establish the actuator's actual causal delay. A constant `1/15s` tick uses publisher's **nominal retained timestamp grid**, not clock precision or device latency. True chronological observations and next `x[t+1]` are factual.
- Authority policy: diagnostic requires all original frames `[t-8,t+1]` to have `intervention == false`. The future authority condition is used to define **observed autonomous-motion evaluation eligibility**, *not* a predictive feature. It introduces selection and must be named with eligibility denominators; no general transition/policy claim across takeovers.
- Sampling: `HISTORY=8`; real FIT 5-task round-robin minibatches of 12 query histories, half from authentically moving next-state rows (physical mean 14D absolute movement >= 0.025) and half uniform if available, pre-registered. Target-based oversampling applies **training only**, not evaluation.
- REPORT queries: up to 24 deterministic SHA-ranked genuinely eligible observed transition origins per original REPORT episode, no outcome-based selection; raw scored predictions retained.
- Model: `System1FastModel(dimension=14)` 100,476 active live/deployed learned parameters. Its 96D fast recurrent memory is carried over eight factual observations, with a bounded one-step delta mean and positive Laplace scale. No slow/8x8 association, risk/action head, future action conditioning or plan rollout.
- Five seeds **76001–76005**, 160 optimizer updates each, batch12, 1,920 physical sequence exposures/seed of eight observed frames, AdamW lr0.0007, weight decay0.0001, gradient norm clip1.0. Selected final-step checkpoint for every seed, **no REPORT-driven training/hyperparameter tuning**, exact honest size verified on every optimizer update and checkpoint.
- Hard requirement: publish 5 original real trained `.pt` weights+optimizer/RNG, exact original data split/sample-order digest, raw source forecast vectors/targets and per-case scores, original metric summaries and receipts in Git, plus separately clean-checkout **independent replay** of all five models without refits. Fail if source unavailable, source hashes differ, permissions missing, nonfinite gradients or checkpoint cannot be Git-owned.

## Registered metrics and failure gates

**Primary developmental utility:** mean **physical-unit 14-joint next-state MAE** on all authentic eligible original 40 REPORT episodes, pair-reported against same-case persistence `x[t]` and FIT-only `x[t] + clip(gain*(a[t-1]-x[t]), servo_step_limit)`. Independently state how many source original episodes and original case origins were eligible. No source log END/success labels.

**Required separate outcomes:** actual movement-rich `mean(|x[t+1]-x[t]|*FIT_state_scale)>=0.025`, near-stationary, task-level and seed-level MAE; normalized Laplace NLL, prediction scale distribution and interval calibration on real observations; original observed next-state vectors and unconditional population coverage limitations. An apparent all-case gain with movement-rich regression is **MIXED_TRADEOFF** (if supported), not improved reflex intelligence. Neither target-defined stratification nor validity mask may be an inference-time model input.

**Provisional success criterion (DEV only):** raw all-case MAE <=0.95×same-case persistence MAE **and** raw motion-rich MAE <=0.95×movement-rich persistence; no severe task regression >10% relative on a qualified task. If five fitted seeds yield inconsistent signs, call **INCONCLUSIVE** rather than a confirmed improvement. Existing relevant episode-cluster uncertainty should be reported where sample size supports; repeat model seeds are not independent physical cases. This gate was registered before data opening and cannot be weakened later.

**Scientific tier:** even with success, `DEV_GAIN` at most for a same-task model vs prior frozen native full model; practical superiority to persistence can be reported as **developmental practical gain**. `INDEPENDENT_GAIN` requires a *new previously untouched authentic* robot task/source and a fully preregistered confidence and seed gate. Any GRU/Transformer architecture winner additionally needs <=1% exact whole-model parameter parity, equal source/data exposure/training+selection compute, and identical heldout cases; CPU execution time itself not required equal.

**No System1 policy claims:** model predicts state only; real risk/ALLOW/ABORT requires explicit observed risk targets and safe independent authorization, calibration and application-specific cost. No submillisecond inference or 25k fit claims. Original 95k–105k new fitted-model budget persists. Historical DIAGKEY synthetic keyed recall failure is separate and does not prove or disprove this original physical forecasting ability.

## Verification commands

`python scripts/validate_research_parity.py --manifest research/mathematical-engineering/me076/FAIR_COMPARISON.json --phase preflight`

`PYTHONPATH=src:. python scripts/me076_train_system1_real.py fit --cache <original-publisher-cache>`

`PYTHONPATH=src:. python scripts/me076_train_system1_real.py replay --cache <same-immutable-verified-source-cache>`

The training stage is **not** itself a replay. The workflow must commit original five checkpoint bytes and source/score ledgers first, then a distinct new checkout restores and recomputes scores before labeling scientific reproducibility PASS. **No original checkpoint overwrites on rerun.**
