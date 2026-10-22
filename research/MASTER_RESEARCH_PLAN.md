# TAC-OSM — Unified Master Research Plan
Version: 2026-10-22 | Status: research proposal and evidence register

## Mission and constraints
Build a persistent action-conditioned predictive world model that learns what to remember, where to store it, how to retrieve it, how dynamics evolve, when to refine predictions, and when to correct beliefs. Retain ME-054 native and ME-056 Concept-Swarm as independently testable branches. Do not replace TAC-OSM with an LLM, Transformer, MTSK, or SSM.

**Compulsory:** real-world longitudinal data and authentic observed outcomes for performance claims; synthetic data only for labeled diagnostics. Architecture comparisons match trainable parameters, data/splits, training budget/compute and evaluation. Report CPU wall time, do not force equality. Include persistence, GRU and relevant stronger baselines. Preserve untouched holdouts, seeds, checkpoints, code, provenance, logs, and results. Do not call untrained designs improvements.

## Existing baseline (source review, verify at pinned commits)
ME-054 persistent fast32 + slow32 + associative 8x8 = 128 state elements. ME-056 adds 6x15 concept states = 218 elements; 4 shared-weight CED passes. State elements are not parameter counts. Both use gated recurrence, associative delta updates, bounded feedback. Neither explicitly models elapsed time. The 2026-10-22 reviews report real developmental forecasting losses to persistence; these are not independent confirmations of fused ideas.

Native associative update: M_next = rho*M + beta*(v-rho*M*k)*k^T, rho=0.95+0.0499*sigmoid(r), initial ~0.99. Unrefreshed contribution at 512 steps: 0.99^512 ~ 0.0058. Write interference into another key j: delta(M*k_j)=beta*e*(k_i^T*k_j), excluding retention. Native observation updates additionally tanh-compress M. The physical prediction head x_next=x+D*f(fast,slow,action) does not directly consume an action-conditioned associative read.

ME-056 concepts communicate by softmax((q_i dot k_j)/sqrt(d)+bias_ij), weighted value sums. Uniform routing and concept collapse are risks, not established failures. CED pass: q_next=q+0.1875*g*(candidate-q). Bounded interpolation does not imply contractive convergence or improving forecast at every pass. Initial preflight: CED 45,822 parameters and 143.60M profiled training-update FLOPs; initial GRU 45,995 parameters and 55.54M FLOPs (~2.59x). This is before registered GRU refinement and not ME-054 versus ME-056.

## Integrated architectural hypotheses (not implemented)
1. **Memory preservation:** learned multi-timescale retention within compact memory, responsibility-weighted delta writes, entity-local updates only where persistent identities are evidenced. Avoid MTSK unrestricted correction (DEV-035 reported state-difference amplification).
2. **Memory retrieval:** key collision diagnostics, direct action-conditioned associative read into physical predictor (r=M*normalize(a), fast_bar=fast+W*r), compare with indirect read. Candidate norm projection M_next=M*/max(1,||M*||_F/C) versus repeated tanh compression.
3. **Working versus persistent state:** fast transient dynamics; slow evolving context; associative retrievable knowledge; concept working representations; bounded verifier. These are learned roles, not imposed semantic labels.
4. **Concept specialization:** measure effective rank, diversity, routing entropy and causal forecast contribution before trying sparse/competitive routing or physical/action/memory-conditioned cross-attention. Softmax is not inherently incorrect; identical scores across nine entries give each 1/9 weight, not evidence of the same TAC failure.
5. **Useful iterative depth:** inspect factual physical errors at each of four CED passes; optionally supervise intermediate forecasts and learn bounded adaptive stopping against measured predictive benefit, not latent movement alone.
6. **Hierarchical latent-first world prediction:** global environment state, persistent relational/entity state and fine event residuals; predict future latents conditioned on *executed* actions then decode physical measurements. Prevent latent collapse and target leakage.
7. **Elapsed-time dynamics:** if real timestamp irregularity matters, compare decay exp(-lambda_t*delta_t) and input-dependent liquid-inspired dynamics to discrete recurrence. Borrow mathematics, not wholesale architecture replacement.
8. **Verification and decision intelligence:** action-conditioned multihorizon forecasting, grounded bounded correction, counterfactual decision-preserving memory retention when data supports identification; avoid claiming unobserved action effects are measured.
9. **Scalability:** batched operators, valid chunked recurrence, useful computation per CED pass; concept exchange approximately O(N²d+Nd²) per pass. Parameter sharing does not eliminate FLOPs.

## Candidate unified objective (research proposal, not current loss)
L = sum_h w_h[lambda_x*L_physical(h)+lambda_z*L_latent(h)] + lambda_o*L_outcome + lambda_v*L_verification + lambda_r*L_retrieval + lambda_c*L_concept + lambda_d*L_decoder + lambda_b*L_budget.
Introduce only justified terms in controlled ablations; prevent future-information leakage.

## Prior evidence and known limitations
- DEV-016 joint training reported 73.64% versus 59.32% frozen writer, but reliability gate failed.
- DEV-017 structured 41-value memory reported 91.84% versus TAC 57.90% future prediction and 38.3% unnecessary interventions; historical protocol details must be checked before cross-experiment comparison.
- DEV-020 audited structured state/read probes at 100%, TAC near chance; do not infer a unique causal defect.
- DEV-043 preserved pairwise ranking did not translate into hard cost-sensitive action accuracy.
- DEV-058 learned controller had strong synthetic completion but imperfect repeat seed; synthetic results do not establish real-world success.
- DEV-061/R1 reported strong synthetic completion; real-world forecasting still lost to persistence.
- ME-054 recovered developmental MAE 0.07881 versus matching GRU 0.09232; ME-056 0.08037 versus matching GRU 0.09082; persistence 0.06695. Check provenance, data splits, parameter and compute parity before claiming fair superiority. Neither beats persistence.
- ME-056 full fair GRU comparison not established by the initial preflight. Do not confuse early preflight with later experiments.

## Ordered program
0. Audit branch heads, experiment evidence, current training and test protocols; finish fair ME-056 GRU comparison.
1. Measure real-data retention, key collision, physical retrieval, concept rank/entropy, CED per-pass error, horizon-conditioned errors.
2. Ablate direct physical memory retrieval, projected versus tanh associative memory, multiscale retention, responsibility writes.
3. Test entity-local routing only with reliable identity evidence.
4. Compare concept cross-context routing and specialization; only adopt if it contributes factual prediction.
5. Compare latent-first hierarchical physical forecasting.
6. Compare fixed four-pass CED, deep supervision and adaptive depth.
7. Test liquid elapsed-time recurrence if timestamps support it.
8. Combine independently validated components in both branches, with parameter/data/compute parity.
9. Assess long-horizon real-world forecasting, verification, action selection, unnecessary interventions, decision cost and generalization on untouched data.

## Provenance and research hygiene
Every idea needs ID, origin, repository paths, mathematical specification, claimed benefit, counter-hypotheses, implementation commit/PR, datasets and splits, controls, budgets, seeds, artifact links, quantitative outcomes, limitations, and status. Do not overwrite negative results. Keep exploratory tests separate from confirmatory holdouts.

Source entry points:
- ME-054: https://github.com/FPC-effortless/TAC-OSM/blob/research/me054-real-physical-residual-stable-feedback-2026-10-22/src/tac_osm/predictive_state_agent.py
- ME-056: https://github.com/FPC-effortless/TAC-OSM/pull/332
- Historical DEV PRs: #172, #173, #176, #217, #220, #221. Revalidate precise linkage and metrics before treating summaries as independently audited.


## 2026-10-22: Anchored-native ME-056 and successor ME-057 evidence correction

Two separate ME-056 study lineages exist: concept-CED ([PR #332](https://github.com/FPC-effortless/TAC-OSM/pull/332)) and 128-frame anchored-native ([PR #334](https://github.com/FPC-effortless/TAC-OSM/pull/334)). Never aggregate their parameter counts, source splits or outcomes.

Anchored-native PR #334 reports 38,266 TAC vs 38,295 GRU trainable parameters (0.076% difference), 42,257,505 vs 42,546,619 profiled FLOPs/update (0.680% difference), and an 8.007% paired CPU overhead (95% CI 6.353–9.999%) against the historical 5% CPU gate. This is an engineering result, **not a trained model improvement**.

The owner-confirmed CPU rule is diagnostic-only. Successor [draft PR #335](https://github.com/FPC-effortless/TAC-OSM/pull/335) implements prospective scientific parity-v3, preserving historical v2 failures. The successor audits all 228 authentic FIT robot episodes before new training registration. Its separate source-only diagnostic manifest explicitly forbids model fitting until original source hashes and a complete comparison plan have been verified.

**Critical discovered information-boundary issue:** PR #334's dense auxiliary snapshots were saved after the same transition's actual next-state feedback. This can leak the supervised target into the state representation. The ME-057 successor saves pre-feedback snapshots and tests that mutating an unseen future target cannot change its own input. Historical resource receipts remain valid as descriptions of the old execution, not validated measures of corrected candidate training. Reprofile and train only after source qualification and complete preregistration. No new predictive performance result follows automatically.

Primary capability gate remains authentic future-state MAE versus persistence and a genuinely equal-resource GRU under complete audited real data/compute exposure. After-takeover outcomes and uncontrolled action effect claims remain outside the supported causal estimand.


### ME-057 FIT source qualification completed — audited real data, not training

Original GitHub Actions qualification [run 39836744099](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39836744099) passed. Its downloaded artifact 12451242831 was parsed, and the canonical per-episode receipt hash independently recomputed.

Verified original FIT source: 228/228 parquet episodes, 329,306 measured frames, 210,192 control-eligible candidate origins, and 375 actual takeover onsets. The canonical original source FIT evidence SHA256 is `c4064651e91cced7aafc11337c32a27266f470ea10388bb0dd2562ae955f88fd`. DEV and TEST trajectory data stayed unopened; no model trained in this phase. Detailed evidence is preserved at [research/ME057_AUTHENTIC_FIT_SOURCE_AUDIT.md](https://github.com/FPC-effortless/TAC-OSM/blob/research/ideas/research/ME057_AUTHENTIC_FIT_SOURCE_AUDIT.md).

Also recovered and byte-verified all ten archived ME-054 original robot checkpoints from workflow 39781336855. Authentic developmental original TEST mean physical MAE: TAC 0.07880869, GRU 0.09231612, persistence 0.06694929; original TEST previously exposed, full training FLOP parity unverified, and takeover labels very sparse. See [ME054 artifact audit](https://github.com/FPC-effortless/TAC-OSM/blob/research/ideas/research/ME054_RECOVERED_ARTIFACT_AUDIT.md).

**Next genuine research gate:** reprofile the corrected pre-feedback dense objective on verified FIT; freeze a prospective matched comparison with identical real trajectories/target access, active parameters, profiled and budgeted training FLOPs, paired schedules, seeds and final evaluation. Do not promote model performance until actual original new-task FIT training and heldout evidence exist.

### ME-058 original FIT parity preflight passed; ME-059 real fit preregistered

Source-backed original FIT comparison [GitHub Actions run 39840368795](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39840368795) passed with 228 original physically measured FIT episodes, original per-episode source SHA `c4064651e91cced7aafc11337c32a27266f470ea10388bb0dd2562ae955f88fd`. It reprofiled the **corrected pre-feedback causal dense objective**; do not substitute the leaked PR #334 preflight.

| Original-source corrected one-update profile | Anchored TAC | GRU |
|---|---:|---:|
| Active trainable parameters | 38,266 | 38,295 |
| Reported operator FLOPs | 42,257,730 | 42,546,844 |
| Instrumented CPU time, efficiency only | 0.87256 s | 0.64799 s |

Gaps: parameters 0.076%, reported FLOPs 0.680%; the prospective numeric preflight passed both 1% thresholds. Whole true training FLOPs are not independently certified. The original FIT-only normalization/servo prior SHA is `3d66d00ba01f345d7fcf3c23149f808986da795b1dea07ead4520b8bafd8af96`. Five outcome-independent paired sampling schedules for seeds 58001–58005 are frozen in the preflight receipt.

A new genuine original task training [draft PR #337](https://github.com/FPC-effortless/TAC-OSM/pull/337) prospectively commits the source/feature/target/optimizer/evaluation contracts, identical real source windows and 600 updates per arm per seed, and a genuine ten-model FIT training workflow. Strong physical persistence and servo-only priors will be scored on factual new-task DEV/TEST **only after all ten checkpoints freeze**. The anchored TAC has a disclosed extra 128-FIT-transition auxiliary while the GRU uses the same observed original physical frames without that gradient loss; improvements cannot be attributed solely to native memory. This protocol is a machine-qualified developmental experiment, **not a certified architecture-superiority result** until complete true FLOP parity is independently measured.

As of this documentation commit, no ME-059 checkpoint result or new source TEST predictive MAE is yet available. Keep that distinction explicit and update only with original run artifacts.
