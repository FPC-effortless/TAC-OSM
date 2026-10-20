# ME020 — ORIGINAL genuine FIT frozen long-history mathematical diagnosis

**COMPLETE, all checks passed.** Original GitHub Actions [run 39451888197](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39451888197), [archived evidence artifact 12278972670](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39451888197/artifacts/12278972670), artifact zip SHA256 \`b4211d39cc837672a163e13566fc6ce3368f3bc8d2c7199172094e73bd560f1c\`. Original raw inner \`results/me020/frozen_genuine_fit_128_dynamics.json\` SHA256 **ec8bc131ba79d63e89d178fd00a4408861c4866151c71656e6fa246f76dc3ca2**.

All 18 **unchanged original** ME019 135050-trainable checkpoints from successful original run 39443325909 individually SHA256-verified BEFORE opening the authentic publisher FIT source. Original ME019 manifest raw SHA256 \`2a10ae3205e866ed792f5ad145e3c3c775e95c39de675f421184de44e56455b6\`; KuaiRec authentic publisher ZIP SHA256 \`c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c\`. **Only user_id%16==0 original early FIT before Aug 1 2020** parsed; not previously opened DEV nor any unseen new-user outcome cohorts. Original training models not updated. Preflight analytically computes the FULL coupled 64×64 original fast/slow Jacobian and compares **all matrix entries** against PyTorch autograd. A revised numerical probe control was fixed BEFORE the successful run to avoid using observations after intermediate snapshot depth t; canceled prior run 39451160586 is NOT an evidence source. 9 original model probes: original TAC32, TAC128, original GRU128, each seeds 14001/14002/14003.

Study sizes: 72 identical actual real FIT histories/model for neural-logit input gradient and TAC fast/slow gates, 12 genuine histories/model for full 64×64 Jacobian at depths 8,32,64,96,128, 1200 fixed genuine old FIT histories for user-disjoint separate ridge probes (975 train, 225 probe-test from 68 nonoverlapping probe-test real user identities). These are **exploratory in-sample original FIT diagnostics**, not new independent generalization.

## A. Exact joint recurrent Jacobian and neural gradient propagation

### Oldest eight / newest eight input neural-logit gradient norm ratio

| Model | T | Seed 14001 | Seed 14002 | Seed 14003 | Mean ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| unchanged native chrono+prior TAC | 32 | 3.328164 | 3.165814 | 2.171218 | **2.888399** |
| unchanged native chrono+prior TAC | 128 | 0.108540 | 0.050562 | 0.012477 | **0.057193** |
| original GRU (unanchored) | 128 | 0.002393 | 0.003132 | 0.001782 | **0.002436** |

The original analytic prior32 bypass is **excluded** from these neural-history logit derivatives (same ME017 convention). These are not end-to-end gradients to the original discrete watch event or causal utility. Nonetheless H128 native oldest-neural-input sensitivity decays sharply, relative to H32 old/new ratio.

### FULL exact fast/slow coupled recurrent Jacobian

Let \`A=∂z'/∂z\`, \`B=∂s'/∂z'\`, \`C=∂s'/∂s\`. **J = [[A,0],[B@A,C]]**, not merely C. Analytical full 64×64 J passed all-entry numerical autograd equality. Measurements are computed with original frozen state equations, float64 copies of original weights to prevent numerical underflow. Spectral norm below is the spectral norm of **the genuine chronological product** \`P_t=J_t @ ... @ J_1\`, NOT the product of individual operator norms. Reported values average across the 12 actually observed early FIT source windows for each frozen seed.

| Native T128 seed | ||P8||2 | ||P32||2 | ||P64||2 | ||P96||2 | ||P128||2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14001 | 1.521351 | 1.787739 | 1.310042 | 0.462854 | **0.179983** |
| 14002 | 1.359624 | 1.368406 | 0.656736 | 0.283608 | **0.107875** |
| 14003 | 1.224960 | 0.796133 | 0.421155 | 0.107398 | **0.037161** |

All three original T128 nonlinear recurrent states exhibit **net local contraction after all 128 events**, even though local J spectral norms at the measured individual depths are above one (~1.04–1.09). Minimum cumulative singular values at T128 are extremely small (mean log10 roughly −15 to −17); directional information transmission is strongly anisotropic. That alone **does not prove** that all earlier predictive information was lost or that Jacobian contraction *causes* heldout Brier regression: the states can still encode past facts and predictions also receive an analytic last32 logit.

Critically, GRU128's oldest/newest **neural input gradients are even smaller** than TAC128, yet original GRU128 achieved the **better unseen-user Brier** in ME019F. Therefore *vanishing early input gradients are neither sufficient nor uniquely diagnostic of the performance failure*. Do NOT automatically impose gradient norm constraints to boost every earlier observation.

## B. Actual native fast/slow gates DO NOT saturate at 0 or 1

| Native model | Seed | Mean fast gate (H128) | Mean slow gate (H128) | Mean slow gate (H32) |
| --- | ---: | ---: | ---: | ---: |
| native chrono+prior | 14001 | 0.1487 | 0.06034 | 0.04510 |
| native chrono+prior | 14002 | 0.1420 | 0.05772 | 0.04635 |
| native chrono+prior | 14003 | 0.1536 | 0.05512 | 0.04444 |

Across all 72 genuine T128 history windows per seed, fraction of either fast or slow gate <0.01 or >0.99 was 0 in these sampled data. Gate means stabilize across blocks and do **not** collapse toward exact zero/full overwrite. Original fast and slow state L2 norms increase through early events and settle to a plateau over middle/late contexts (magnitudes are norms across 32 bounded coordinates, not evidence of individual tanh saturation). No measured discrete stale-record deletion or "pruning" is proven in native or GRU.

Direct carry-only gate retention over 128 updates has strongly negative log survival (original per-model fast about −20 to −22, slow −7 to −8), but these **are only the isolated product of (1−gate)**. The full Jacobian includes state proposals and gate-derivative terms. Treating direct gate products as full forgetting is mathematically incorrect.

## C. Early-factual observation information survives better in TAC than GRU linear probes

Separate frozen **ridge decoders** (λ=1, train-only feature standardization, no network updates) attempt to reconstruct the genuine mean binary watch outcome of the **oldest eight actual events** from snapshots at chronological depth 8,32,64,96,128. Probe *training/testing user IDs* do not overlap, all inside the earlier training-time FIT population. This measures linear decodability, NOT mutual information or causal information value.

| History snapshot depth | Native T128 mean 3-seed MSE | Original GRU T128 mean 3-seed MSE |
| --- | ---: | ---: |
| 8 | 0.001543 | 0.002543 |
| 32 | 0.017335 | 0.034062 |
| 64 | 0.026736 | 0.036110 |
| 96 | 0.030893 | 0.040489 |
| 128 | **0.034584** | **0.041422** |

Common constant FIT-trained label-mean baseline MSE **0.046174**. A time-causal observable rolling response-rate control was also evaluated at each snapshot and is in the archived raw curves; at t=8 it is trivially perfect because the target is exactly the visible first8 rate. Native T128 retains **more linearly accessible early factual information** than GRU128 in this particular probe, despite worse real new-user full-watch forecasting under the models' actual trained prediction heads. Thus a blanket conclusion "TAC memory simply erased early observations" is **not supported**.

### Are older observations even relevant?

A distinct, exploratory real FIT **225-response user-separated probe test** measures predicting a genuine current full-watch label from recent32 response-rate vs recent32 plus older96 response-rate, both FIT-fitted ridge with equal fixed λ=1:

- Recent32 only real-probe Brier: **0.208194742**
- Recent32 + genuine older96 real-probe Brier: **0.205701236**
- Paired difference: **−0.002493507**, favorable for older-history summary on these real FIT users only.
- 975 training rows, 225 true observed probe-test rows, 68 actual probe-test user identities distinct from probe-train IDs.

This supports the narrow observation that an actual older-history statistic has additional **linear predictive association** in this FIT probe. It cannot establish a causal action advantage, independent external generalization, or that the native state retains all the specific relevant older events.

## Preregistered diagnostic decisions

1. **Joint dynamic contraction / long-history neural gradient attenuation:** **OBSERVED.** Full J128 product contracts and oldest/newest input gradients shrink vs H32 across all 3 native seeds. Not by itself a unique explanatory cause because original GRU has even smaller oldest/latest input-grad ratios.
2. **Gate saturation as dominant mechanism:** **NOT OBSERVED** on measured data. Neither the slow nor fast gate is mostly near 0 or 1. Do not indiscriminately add another reset/forget gate or clamp gate derivatives.
3. **Complete early-information erasure:** **CONTRADICTED** by real FIT user-separated decoding. Both architectures degrade old8 decoding with time, yet native T128 linearly preserves early8 mean MORE accurately than original GRU128 on this particular old8 target.
4. **Older real history lacks useful target-relevant information:** **NOT SUPPORTED** by recent32+old96 FIT regression test, though current probes have limited sample and no fresh external validation.
5. **Highest-priority *working hypothesis*, not proven causal explanation:** learned **predictive utility alignment / temporal credit/readout mismatch**. Useful old facts are partly represented, the long-horizon Jacobian transmits their gradients weakly, and existing native final readout fails to convert additional history into better real heldout full-watch prediction. Compare representation/readout vs temporal optimization without altering native memory merely to maximize early gradients. No inference that GRU "better prunes" records follows from these measurements.

## Minimum safe ME021 direction — no blind architecture replacement

Preserve original 32-step native and ME018 mathematically equivalent scripted speed reference. **ME021 should preregister an equal-parameter, equal-data, measured-compute/head-to-head real FIT study of readout/temporal credit changes**, initially without any new memory module. Use actual observed response losses and decision-relevant credit, not universal early-token gradient inflation. Candidate family: original learned head and fully shared native gated dynamics with a **parameter-free, depth-stratified causal auxiliary prediction objective** (e.g. same trained prediction head at actual 32/64/96/128 genuine real prefix depths, fixed weights and no new trainable parameters). This is a **HYPOTHESIS ONLY**: training budgets measured in CPU/FLOPs, and comparisons must be compute matched; no claim the proposed objective will help until new evidence. The original future/new-user user4 set has been opened already and MUST NOT be called untouched for ME021.

Do not add attention, Titans/MIRAS, SSM or extra gates until this identified predictive-utility bottleneck receives discriminating genuine data evidence. All original PR #276/277 negatives, original native TAC candidate and independent test records stay intact. The new experiment must reserve an actually unexposed population/source before any unseen test after training/checkpoint hashes seal.

**Commercial boundary:** observational video watch responses and in-sample mathematical probes do not establish action intervention value, predictive world-state planning or real cost savings. Active architectural comparisons still have unequal original training CPU budgets even with identical 135050 parameters, real target rows and optimizer updates.
