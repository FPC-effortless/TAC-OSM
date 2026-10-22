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
