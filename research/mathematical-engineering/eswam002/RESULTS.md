# ESWAM002 — trained original authentic robot outcomes and independent replay

**Date: 2026-10-23. Scientific verdict: trained/replayed authentic exposed DEV, practical prediction FAIL, action imitation FAIL, no architecture-superiority claim.**

Source of truth: original per-seed files under research/replay-bundles/ESWAM002 and registered protocol/manifest in research/mathematical-engineering/eswam002. No prior study, checkpoint, source normalization, baseline score or held-out TEST bytes were changed.

## Actual original training

- Five whole-model ES-WAM fast/slow exchange-structured candidates, **100,343 trainable/deployed parameters each**, seeds 76001–76005, exactly 160 guarded AdamW updates per seed, batch12, identical original ME076 real FIT sampled cases/order and the same previously exposed episode-disjoint DEV queries.
- 120 original FIT episodes, 40 already exposed REPORT episodes from five BodenAI real robot tasks. Original frozen original source digest 4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776. Publisher-provided nominal 15Hz grid, lag1 previously recorded issued qpos; causal actuator timing not independently established.
- Source-original actual real next-state targets x[t+1], 960 prediction-time available-case queries per seed (40 distinct episodes, 84 observed movement-rich, 876 near-stationary). The same physical cases are repeated for five model seeds, **not 4,800 independent trajectories**.
- Auxiliary action-head target: logged a[t] after the last known input a[t-1]. This is observational next-recorded-command imitation, **not actual decision utility, reward, policy, safe authority, or causal action-conditioned intervention evidence**.
- First original attempt was rejected before first optimizer update due a code-only action-shape check (a.shape compared with Tensor instead of x.shape); fixed before starting the only actual registered training. The first attempt yielded NO original learned checkpoints. The full second attempt completed; original weights, optimizer/RNG/source fingerprints, full per-case prediction/target vectors and five SHA-hashed original checkpoints committed to Git. No REPORT-driven tuning.

## Physical one-step real forecast results

Lower MAE is better. Each seed evaluates the SAME 960 real original cases as the corresponding frozen ME076 checkpoint.

| Seed | ES-WAM mean physical MAE | Original ME076 mean physical MAE | ES-WAM movement-rich physical MAE | ES-WAM near-stationary physical MAE |
| --- | ---: | ---: | ---: | ---: |
| 76001 | 0.072932 | 0.083078 | 0.745539 | 0.008435 |
| 76002 | 0.079334 | 0.076604 | 0.749324 | 0.015089 |
| 76003 | 0.074629 | 0.080977 | 0.746511 | 0.010202 |
| 76004 | 0.074743 | 0.080320 | 0.746234 | 0.010353 |
| 76005 | 0.081488 | 0.079813 | 0.749311 | 0.017450 |
| **Mean** | **0.076625** | **0.080158** | **0.747384** | **0.012306** |

Same-case genuine physical baselines (identical for every seed):
- All-case persistence: **0.068668**; ME076 servo prior: **0.097457**.
- Motion-rich persistence: **0.743241**; motion-rich FIT servo: **0.704517**.
- Near-stationary persistence: **0.003983**.

Interpretation:
- The new model's all-case MAE is approximately **4.4% lower than the original ME076 neural checkpoint mean**, with mixed seed direction (**3 of 5 seeds** improve). This is descriptive across training architectures/objectives; not an architecture superiority claim. Nominal parameter gap is about 0.13%, but full training compute and objectives are not matched.
- ES-WAM all-case MAE is **11.6% worse than persistence**. ALL five trained seeds lose to persistence on the all-case original observations.
- Movement-rich ES-WAM **0.747384** is 0.56% worse than movement-rich persistence, and about 6.1% worse than motion-rich servo. ALL five seeds lose to persistence on movement-rich cases.
- Near-stationary ES-WAM 0.012306 remains about 3.09 times persistence error 0.003983, although lower than ME076's original near-stationary mean.
- The preregistered >=5% practical gain over persistence simultaneously on all cases and genuine movement-rich cases **FAILS** for every seed. No seed choice can change that conclusion.

## Recorded-action behavioral imitation (not policy intelligence)

Mean next-logged-command normalized L1:
- ES-WAM trained action proposal: **0.804647**.
- Repeat the last source-recorded command: **0.009537**.

The new action head is dramatically worse than a trivially available observational baseline. It cannot be qualified as a useful policy. This is NOT evidence that any counterfactual action is good or bad; the original data and training do not establish candidate-action causal effects or verified external control benefits.

## Architecture capability status

| Capability | Present engineering mechanism | Trained real skill demonstrated? |
| --- | --- | --- |
| Native fast/slow recurrence | YES | Trained, but all-case forecast fails persistence |
| Evidence roles observed/predicted/verified | YES; protected separate occupancy | No entity-role intelligence qualification; no verified labels in this study |
| Exchange symmetry/antisymmetry | YES; software-tested | No empirically isolated real benefit |
| Content retrieval | YES; query masks and memory software checks | No realistic multi-entity keyed retrieval evidence |
| Long event retention at 16/64/256 | NO qualified mechanism test on authentic histories | UNKNOWN |
| Action-conditioned dynamics | Previous logged action context used | Actual current-candidate intervention effects NOT identified |
| Multi-horizon state futures | Not trained in ESWAM002 | UNKNOWN |
| Uncertainty calibration | Laplace scale head is fitted | No calibrated deployment reliability claim |
| Direct action proposals | YES; trained logged-action imitation | FAIL vs repeat-last-action |
| CASM policy-guided counterfactual | Not integrated or trained here | UNKNOWN |
| Real-world verified state correction | Predicted/observed-role innovation is active | No independently grounded corrected-decision benefit |
| Operational value/customer outcomes | No authorized pilot | UNKNOWN |

The prior DIAGKEY004 immediate synthetic recall pass does **not** transfer to this model; that is a distinct original memory checkpoint and does not prove this candidate can perform real long-delay recall.

## Original checksum and replay result

Five original model/optimizer/RNG checkpoints, complete normalized actual source/forecast arrays, original per-case scores and source IDs are Git-owned under research/replay-bundles/ESWAM002. An independent separate GitHub Actions checkout restored all five original model weights and rescored **each of the same 960 original real query cases per seed without optimizer updates**. Strict registered physical MAE, servo/persistence and Laplace NLL within abs/rel 1e-6: **PASS**. Original source fingerprint and per-case identity matched.

Independent original trained checkpoint SHA256:
- 76001: a5cfdc56f61de4369922f1a2337e853ccf96aa24f6da5ee5f2fe4067c5ec6584
- 76002: 0e1926f06cf4a7ef001544e9a0f1f876c1ad4b46f3a3bc18d45141b70e604b95
- 76003: 28f8e3e507de82953777a02b76f811a1b5b6381f888fe9645ea1059a145726a6
- 76004: b9964c572abc5ac839b9a3ca0ce44e0834f298b06f12d705f0fafe29ea8e998c
- 76005: 8d5a839c749046837a63a60e20df102e79f450a0f5f1c803d2db8c2334641fe6

Software CI full suite 1,138 passed. Original prefit checks: eight exchange tests PASS, 100343 live parameters PASS, whole-run parameter audit PASS, source license/fingerprint PASS, 160 guarded updates per model saved; training workflow and clean replay workflow both SUCCESS. No private customer or sealed source TEST was accessed.

**Scientific tier:** authentic exposed-development trained candidate and fully replayed failure. No qualified predictive, policy, real long-memory, architectural superiority or commercial claim. Remains draft/unmerged, no device action authority.

## Next nonduplicative capability targets

1. Rework the integrated training objective around **actual meaningful physical motion and stable no-change predictions**, while explicitly retaining strong persistence/servo as true prediction-time alternatives; evaluate on a new genuine held-out source rather than retuning this exposed REPORT.
2. Qualify authentic longitudinal histories with multiple independently observed entities and true retrieval questions; only then train exchange exclusion/retention against a matched content-addressable control and measure long-history delayed decisions. Eight-step single-entity physical windows cannot test Pauli-inspired occupancy intelligence.
3. Choose real datasets with credible known-at-action-time candidate actions and observed task consequences before testing counterfactual policy improvements. Logged previous qpos and imitation alone do not support causal action-value evaluation.
4. Do not promote or integrate into production until an original preregistered unseen source and appropriate decision-cost/authority evidence pass.

## Capability-to-value

Verified research skill gain: **NONE yet**. Technical system trained and independently replayed: **YES**. Actual physical state prediction better than a simple baseline: **NO**. Independently qualified long-memory/action intelligence: **UNKNOWN**. Customer pilot, measured value/revenue, autonomous authorization: **NONE**.
