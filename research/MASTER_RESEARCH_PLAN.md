# TAC-OSM — Unified Master Research Plan
Version: 2026-10-23 (ME069 authentic enterprise event-model update) | Status: research proposal and evidence register; not a commercially qualified system

## Mission and constraints
Build a persistent action-conditioned predictive world model that learns what to remember, where to store it, how to retrieve it, how dynamics evolve, when to refine predictions, and when to correct beliefs. Retain ME-054 native and ME-056 Concept-Swarm as independently testable branches. Do not replace TAC-OSM with an LLM, Transformer, MTSK, or SSM.

**Compulsory:** real-world longitudinal data and authentic observed outcomes for performance claims; synthetic data only for labeled diagnostics. Architecture comparisons match trainable parameters, data/splits, training budget/compute and evaluation. Report CPU wall time, do not force equality. Include persistence, GRU and relevant stronger baselines. Preserve untouched holdouts, seeds, checkpoints, code, provenance, logs, and results. Do not call untrained designs improvements.

## Mandatory repository-backed model preservation and replay (2026-10-23)

**Hard rule, all local and remote/GitHub training:** before any new or continued ME/DEV/model fit is treated as complete, every applicable arm/seed's original trained checkpoint and the complete lawful, auditable model-loading and experiment-scoring replay bundle **must be preserved in this TAC-OSM GitHub repository at pinned Git commit paths**. This includes failed/rejected runs with valid recoverable weights, all dependencies and data manifests, optimizer/RNG state when resumption is claimed, and independent restoration/scoring verification from a **clean Git checkout**. Temporary Actions artifacts, caches, local `results/`, remote notebooks and hash-only receipts **never** replace actual retrievable repository checkpoint bytes. Use tracked checksum-indexed shards or fully retrievable repo Git LFS if needed. If source data cannot legally be redistributed, store pinned legal source-acquisition instructions/hashes and label any inaccessible full-evaluation replay **BLOCKED**, not PASS. No secrets, unlicensed data or private user records may be uploaded.

**Final research gate: REPO_REPLAY=PASS**, separate from real-world validity and matched-comparison qualification. If actual checkpoint bytes, loader, exact evaluation dependencies or independent clean-checkout replay are missing, status is **NOT_PRESERVED/BLOCKED; no new completed-fit, verified-experiment, model-promotion or superiority claim**. Preserve historical studies as historical with disclosed incomplete archives, never retrofit claims. This rule covers local, GitHub Actions, Kaggle, GPUs and other external hosts equally. [Full mandatory policy and file checklist](REPOSITORY-REPLAY-AND-CHECKPOINT-RULE.md).

## Company-aligned north star and scope (2026-10-22 revision)

### Current architecture-first execution: authentic enterprise workflows (ME069)

The user's latest direction prioritizes making the integrated native model more
capable for enterprise operations. ME069 therefore adds an authentic enterprise
observation lane rather than another robot-memory diagnostic. Preserve all
physical negatives and the untouched robot tissue-box holdout. Synthetic
Woyengi/Veritas logs may test interfaces; they cannot qualify predictive or
commercial ability. Claims about current revenue, valuation, fundability or
being one pivot from commercial readiness remain unverified.

[ME069 draft PR #348](https://github.com/FPC-effortless/TAC-OSM/pull/348)
implements categorical activity/resource encoders, elapsed-time decay of native
64D fast/64D slow/8x8 associative state, a learned repeat/empirical-transition/
neural event mixture and a hurdle-lognormal waiting-time distribution. It
retains TAC's native dynamics. Standard predictive-process monitoring,
time-aware recurrence and probabilistic heads are established prior art, not
claimed inventions. State dimensions are configurable resource choices, not
proof that earlier 32D state was inherently inadequate.

The original BPI Challenge 2012 loan-process log is publisher-hash verified:
13,087 cases, 262,200 events; a chronological, outcome-availability split yields
7,096 FIT, 1,850 DEV and 2,618 TEST cases. FIT-only vocabulary and priors, no case
ID or future event features. Five paired seeds receive identical schedules,
800 updates and dense supervision over at most 32 observed events. Native has
80,296 parameters versus GRU 80,966; common supported-operator training FLOPs
differ by approximately 0.284%. Final registered parity PASS under that
profiler convention; full hardware FLOPs remain uncertified. Native mean
optimizer CPU is 35.251s versus GRU 13.777s; CPU is an efficiency measurement.

| Frozen temporal TEST | Native TAC | Matched GRU | Empirical transition/time reference |
| --- | ---: | ---: | ---: |
| Next-event log loss | 0.122626 | **0.119182** | 0.348946 |
| Next-event accuracy | 94.77% | 94.77% | 84.76% |
| Waiting-time MAE, hours | 6.3561 | 6.3618 | 6.4953 |
| FIT-threshold long-wait proxy Brier | 0.070188 | 0.070767 | 0.077039 |

The whole TAC predictor demonstrates useful next-event forecasting relative
to the empirical reference (64.86% lower log loss, 5/5 seed wins). It does
**not** establish native superiority: TAC log loss is 2.89% worse than GRU,
1/5 seed wins; paired case-bootstrap TAC-minus-GRU 95% interval
[+0.001552, +0.005243], conditional on fitted seeds. The registered joint
superiority gate **FAILS**. Nominal 90% interval coverage is 92.28%, with a
wide 230.12-hour mean width; useful operational uncertainty is not qualified.

**Original-environment REPO_REPLAY=PASS; cross-CPU numerical exception retained:** all ten original model/optimizer/RNG checkpoints were
Git-owned at `491f085b` before scoring, restored from a separate clean checkout,
and all 44,680 DEV/TEST case-score records were independently recomputed
bit-identically. Original score commit `7b458208`; no retraining replacement.
GitHub's different CPU runner subsequently failed the fixed waiting-time
per-record tolerance at DEV / GRU seed 69005 / row 520. Preserve that numerical
portability exception and the original tolerance; no all-environment replay
pass or model promotion is claimed.
Subsequent GitHub run `39905632298` passed the identical fixed tolerance without
a numerical/model change; this does not erase the original failing run
`39905268211` or establish uniform CPU portability.
Latest code-validation run `39905869711` printed a bit-identical DEV/TEST
replay PASS receipt. The second failing run `39905753892` reported a
1.221ms waiting-error discrepancy against a 0.226ms permitted difference.
Both failures and successful receipts are repository-owned; no tolerance was
relaxed and no original score was replaced.
The full repository regression passed 1,005 tests. See the
[results and limits](https://github.com/FPC-effortless/TAC-OSM/blob/research/me069-authentic-workflow-world-model-2026-10-23/research/mathematical-engineering/me069/RESULTS.md)
and [original checkpoints](https://github.com/FPC-effortless/TAC-OSM/tree/research/me069-authentic-workflow-world-model-2026-10-23/research/replay-bundles/ME069).

**Next architecture objective:** improve the integrated event model's history
representation and waiting-time sharpness using FIT/DEV only, then preregister
qualification on an untouched authentic cohort/source. This midpoint TEST is
now exposed and cannot supply another prospective claim. Keep a strong
empirical process prior and matched GRU in the results table; improve the
complete system rather than inserting a diagnostic-only stage after every
failure. Do not assume adding width, hierarchy or reasoning loops will help.

**Unchanged capability boundaries:** this bounded observational forecast does
not establish lifetime state, learned actions, actual SLA failure, verified
process completion, monetary benefit, causal effects, System 2 reasoning or
Woyengi deployment. Woyengi owns authoritative state and execution permission;
TAC emits advisory predictions. Actual operational actions/costs/outcome
verification remain required for subsequent decision-intelligence stages.

**North star:** build TAC-OSM into a reusable *learned predictive-state and decision-intelligence engine* for real operational systems and authorized autonomous software. It must maintain useful beliefs across change, forecast *observed* futures under available action contexts, recognize uncertainty and information gaps, choose or recommend cost-aware interventions, and improve after independently verified outcomes. This is a scientific target, **not a claim of current intelligence or product readiness**. The primary commercial proof is measured improvement over the best appropriate simpler solution in an independently held-out, authentic operational setting.

**Portfolio review scope:** the connected GitHub owner `FPC-effortless` was reviewed 2026-10-22. GitHub-confirmed forks and the owner's online-banking, crypto and Paws & Claws projects are excluded from this strategy. Repositories are possible components or test beds, not evidence that integrations or deployments exist. An app, codebase, mock, synthetic world or README cannot stand in for independently collected real outcomes.

### System boundary — do not duplicate the rest of the company

| Project / research family | Appropriate responsibility | TAC-OSM interface / restriction |
| --- | --- | --- |
| [Woyengi](https://github.com/FPC-effortless/Woyengi) | Identity-aware, authoritative, reconstructable organizational and agent records, provenance and permissions | Read *authorized time-cut snapshots/events*; write **proposals/predictions**, never silently overwrite canonical truth or bypass authority. Learned belief is distinct from authoritative state. |
| [KoreFrame](https://github.com/FPC-effortless/KoreFrame) | Agent runtime, tool scheduling and execution | TAC predicts action outcomes / recommends a bounded next action; runtime executes only if separately allowed. |
| [Veritas](https://github.com/FPC-effortless/veritas) | Executable operational worlds, independent agent evaluation, rollout and evidence packaging | Use for **reproducible engineering diagnostics and replay**. Generated worlds are synthetic; only separately collected real-world evaluations can promote TAC capabilities. |
| SafeEdge and AegisFlow | Permission-bound action enforcement, security, policy and audit trails | Safety/governance remains **external hard constraint**: confidence scores never grant authority. Log proposal, authorization, action and result. |
| BusinessData / knowledge-refinery | Versioned, policy-bound ingestion and retrieval | Supply legally available observation/context with timestamp and provenance; TAC learns relevance and state, does not certify source truth. |
| [Nsat](https://github.com/FPC-effortless/Nsat) | Nigerian market, economic, weather and Earth-observation data engineering | Candidate *authentic forecast/transfer* lane (time/site/market split); **not** action-effect or policy-value evidence without actual interventions. |
| Shipping / Scrappy-HCSE / other operational apps | Potential logistics and workforce application surfaces | Candidate measurement/pilot paths **only if** actual entity-linked temporal events, action logs, costs and verified results exist. No deployment assumption. |
| [TAC-Transformer](https://github.com/FPC-effortless/TAC-transformer), [TAC-Prime](https://github.com/FPC-effortless/TAC-Prime), [ATLAS-CASSA](https://github.com/FPC-effortless/ATLAS-CASSA), PSN2, Neural-Field-Dynamics, USEF-X, CDL, PNDS-Lean | Comparative architectural hypotheses: reusable structures, hierarchical latent reasoning, concept dynamics, emergent specialization, mathematical proofs | Import **specific falsifiable insights only**. No automatic fusion, architecture replacement or component novelty claim; keep every imported idea's provenance and matched control. |
| Product Bible / education / sports / consumer prototypes | Potential later product experiences | Not first-line evidence until legal data, entity continuity, measured outcomes and consent/permissions are independently established. |

**Proposed integration contract** (design only):
`authorized {entity_id, event_time, observed_at, provenance, observation, executed_action?, observed_outcome?, measured_cost?, available_actions, constraints}`
→ TAC `{belief_version, horizon, factual_forecasts, uncertainty, proposed_action_or_inspection, predicted_value?, abstain_reason?, evidence_refs}`
→ governed executor / human approval → signed actual execution receipt → independent outcome verification → delayed TAC update.
Explicitly model missing actions, delayed/censored feedback, abstention and unknown costs. No permission transfer via model output, and no model-invented verified result. For causal claims add randomized assignment/propensity and support/identification fields.

### Portfolio-specific first product hypothesis

1. **First-choice integration:** Woyengi + KoreFrame + SafeEdge/Aegis + Veritas for long-running *authorized workflow decision support*. Target actual delayed completions, failure/intervention cost and evidence-backed stopping; demand real operational logs or authorized prospective collection before claiming improvement.
2. **Second candidate:** logistics/shipment risk and next-action recommendations only where actual time-ordered shipments, interventions and verified delivery outcomes are available.
3. **Third candidate:** Nsat regional/market forecasting and across-site transfer as a separate **prediction-only** lane, not a surrogate for decision intelligence.
4. Workforce/other applications follow domain-data and legal qualification. Do **not** build bespoke product logic into the TAC core; use explicit domain adapters. Avoid implying a dataset exists just because a related app or data pipeline exists.

### Capability scorecard (evidence status, not a product-completion percentage)

`ME` studies and DEV variants often sit in **separate open draft PRs**, not a merged deployable checkpoint. Classifications below are evidence across research lineages, not an assertion that one model possesses the whole set.

| Priority | Target capability | Evidence as of branch review | Promotion gate |
| --- | --- | --- | --- |
| P0 | Accurate persistent world-state belief | **Partial:** recurrent/entity-aware learned state and long-history tests exist; identity transfer and robust retention fail some gates | heldout real entity/time continuity and better calibrated future/error performance versus matched recurrent and structured baselines |
| P0 | Real action-conditioned factual forecasting | **Partial, trained ME061:** source DEV TAC MAE .603894 vs persistence .702952 and GRU .602600; different-task TAC .493206 vs GRU .509535 but persistence .455992. Both superiority gates fail | sealed real factual multi-horizon outcomes, persistence/servo priors and fully resource-audited matched GRU |
| P0 | Cost-sensitive decision / stop / inspect / abstain | **Partial synthetic only** for learned multi-step action/stop; randomized ME049 chose the same fixed action | verified real intervention/action support, uncertainty-aware decisions, net cost or reward improvement versus best fixed/no-action policies |
| P0 | Uncertainty and value of information | **Partial, measured ME061:** positive forecast scales and source split-conformal intervals implemented; nominal 90% whole-episode coverage reaches 85.79% on DEV with broad intervals. Useful uncertainty/inspection value unresolved | out-of-time calibration, selective-risk/coverage, paid-inspection utility and safe abstention on genuine records |
| P0 | Verified closed-loop belief correction | **Partial engineering:** masked bounded corrections exist, and synthetic learned controllers replay; no qualified real improvement | independent real outcome receipt, strictly delayed pre-feedback update, error reduction without overwriting verified facts |
| P1 | Entity and relationship representation | **Partial synthetic:** entity-local DEV061 98.5% completion, below structured GRU 99% and public MLP 100%; unfamiliar IDs fragile | unseen-real-entity generalization, interference reduction, true identity continuity |
| P1 | Decision-preserving selective long-term memory | **Partial:** routing/consolidation mechanisms exist; ME051 old-event gains inconsistent, GRU stronger | controlled real long-history decision/forecast gain and verified forgetting/interference trade-offs |
| P1 | Multi-horizon latent and hierarchical abstraction | **Partial architectures:** fast/slow/associative and concept-CED prototypes; added depth does not imply thought | consistent real future/policy improvement across horizons over simpler matched controls |
| P1 | Continual adaptation under shift | **Not established:** recurrent *state* updates are not online *weight learning* or useful adaptation | authorized sequential real site/time adaptation without leakage, forgetting or unauthorized side effects |
| P1 | Transferable state intelligence | **Not established:** ME047 new-site inference executes but usefulness/persistence gates fail | sealed different site/task transfer with advantage over strong target-independent references |
| P2 | Multimodal grounding | **Not established as native understanding:** structured events, actions and sensors exist | properly time-aligned independent audio/image/document observations and measured downstream lift |
| P2 | Selective computation at scale | **Partial mechanics only:** actual routing/dispatch and timing exist; cost-law C5 untested | end-to-end relevant-subset scaling including index/addressing without loss of accuracy; real latency and resource receipts |

**No capability is yet qualified as a cross-domain, commercial TAC-OSM advantage.** DEV058/DEV061 synthetic success remains diagnostics. The honest real-data negatives remain prerequisites, not inconveniences to optimize away.

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
- ME049 genuine three-arm randomized dataset: trained TAC's heldout action selection collapsed to the training-best fixed Mens email action, yielding no personalized policy benefit; context achieved better Brier (PR #314). Do not treat action-head existence as real decision intelligence.
- ME051 genuine MovieLens H256: oldest-history TAC MAE 0.787059/0.780177 across two seeds, recent TAC 0.791441/0.779113; GRU controls 0.775609/0.776760 best recent (PR #317). Old-history benefit inconsistent; cross-model capacity not parity-qualified.
- ME047 executed frozen cross-site ETTm2 inference but failed useful forecast-transfer gates (PR #312). 
- ME057 fused completed a five-seed real-robot comparison and failed its registered superiority gate (PR #338); ME059 remains a separately registered lineage whose result must be checked independently (PR #337).
- DEV-016 joint training reported 73.64% versus 59.32% frozen writer, but reliability gate failed.
- DEV-017 structured 41-value memory reported 91.84% versus TAC 57.90% future prediction and 38.3% unnecessary interventions; historical protocol details must be checked before cross-experiment comparison.
- DEV-020 audited structured state/read probes at 100%, TAC near chance; do not infer a unique causal defect.
- DEV-043 preserved pairwise ranking did not translate into hard cost-sensitive action accuracy.
- DEV-058 learned controller had strong synthetic completion but imperfect repeat seed; synthetic results do not establish real-world success.
- DEV-061/R1 reported strong synthetic completion; real-world forecasting still lost to persistence.
- ME-054 recovered developmental MAE 0.07881 versus matching GRU 0.09232; ME-056 0.08037 versus matching GRU 0.09082; persistence 0.06695. Check provenance, data splits, parameter and compute parity before claiming fair superiority. Neither beats persistence.
- ME-056 full fair GRU comparison not established by the initial preflight. Do not confuse early preflight with later experiments.

## Ordered integrated capability program — supersedes the previous component-first sequence

Run **a small number of complete preregistered capability experiments**, not an endless queue of memory-gate/decoder microvariants. Each stage begins by reusing the strongest valid prior result and relevant established literature (R2D2, DreamerV3, TD-MPC2, modern recurrent/SSM baselines, offline-policy identification). Component ablations are **conditional failure diagnosis**, not default milestones.

**Gate 0 — Evidence reconciliation and real-data authorization (no new architecture claim).**
- Reconcile **separate lineages**: ME054 native, ME056 *Concept-CED* PR332, ME056 *anchored-native* PR334, ME057 fused PR338, ME058 source preflight PR336 and independent-task ME059 PR337. Do not conflate the duplicate ME056 names, source/test exposure, FLOP receipts or scores. Read their latest branch result artifacts before updating this plan.
- Preserve original ME054 developmental robot MAE 0.07880869, GRU 0.09231612, persistence 0.06694929; the task's test was already exposed and full training-FLOP parity is not certified. **Do not call this a matched independent TAC win.** Independently verify any Concept-CED MAE attribution against actual training artifacts before promoting it.
- ME057 PR338 reports 38,268 TAC vs 38,295 GRU parameters and profiled-update 42,640,106 vs 42,784,699 FLOPs. Its completed 80-episode TEST means are TAC 0.663284, GRU 0.658526 and persistence 0.672681 original-unit MAE. TAC won one of five seeds; the episode-clustered TAC-minus-GRU interval was +0.000285 to +0.009266, so its registered gate failed in GRU's favor. ME059 PR337 remains separate and must not inherit this result. The ME058 PR336 source preflight is not ME059 final training; partial profiler FLOPs are not a complete audited training-compute certificate.
- Before a new fit: freeze dataset licence, authentic source hash/receipt, original chronological entity IDs, known action availability, missing/censored outcomes and exact source-specific time-cut. Mark **prediction-only** datasets separately from supported real-policy datasets. Confirm PR status and original artifact outcome rather than inferring from a workflow YAML.
- **Accept:** auditable qualification ledger and real observations with eligible target definition. **STOP:** identity/actions/costs missing for the proposed estimand; missing parity manifest; invalid/test-exposed confirmation.

**Gate 1 — Prove factual real-world prediction and stable state (P0).**
- Use independent authentic multistep physical-robot trajectories as the immediate first test, following the sealed ME059 training/validation contract only when every preregistration, original data and true compute receipt is qualified. Evaluate future physical state (and human takeover only with enough positive support), *pre-feedback only*. Compare against last-state persistence, servo-only physical prior, GRU and a contemporary appropriately budgeted recurrent predictor.
- Score MAE/MSE per physical channel and horizon, divergence/exploding state, calibration, long-history degradation, missingness and independent new-task generalization. Include factual action conditioning with the *executed recorded action*, never label potential outcomes as observed.
- **Accept:** predeclared independent holdout, repeatable improvement over strongest practical reference and fully qualified matched neural comparison where claimed. **STOP:** still worse than persistence/servo reference, post-feedback leakage or unequal fairness budgets; diagnose root failure before building another architecture.

**Gate 2 — Show real decision-relevant memory and learned entity structure (P0/P1).**
- On a new authentic longitudinal entity dataset with verified IDs/times, test whether earlier facts materially change later observed outcome/decision quality. Compare identical-input equally resourced structured GRU, memory-free and direct-history baselines; include hard interference, ID permutation, unseen IDs and delayed contradictions.
- Favor *future utility* over reconstructing an old summary. Only activate proposed direct associative reads, retention/responsibility writes, entity-local indexing, cross-concept messages or extra hierarchy if a prespecified failure demands them. Preserve separate native and Concept-CED arms.
- **Accept:** consistent sealed heldout improvement in factual prediction or supported decision utility, not merely probe accuracy or synthetic retention.

**Gate 3 — Establish action value, uncertainty and safe intervention (P0).**
- First acquire/qualify authentic supported **action → measured outcome → actual cost → attribution** chains with known available choices and propensities or justified identification. If such data are absent, continue **prediction-only** reporting without claiming policy intelligence. Historical ME049 RCT was valid randomized one-step context but native selected the same fixed Mens email arm for every test case (no personalized policy gain).
- Estimate uncertainty/calibration, expected costs/benefits, information value for inspect/wait, hard action support and abstention. Optimize over the **governed available-actions set**; compare best fixed action, no action, strongest bandit/GRU/policy references with appropriate off-policy variance checks. Distinguish predictive association, policy value and causal effect.
- **Accept:** out-of-time/entity or independently randomized real policy net utility gain, cost distribution, risk coverage, confidence bounds, no unauthorized actions. **STOP:** unsupported counterfactual extrapolation or fictional decision prices.

**Gate 4 — Verified closed-loop learning and useful deliberative depth (P0/P1).**
- With actual outcome receipts, verify that correction changes only beliefs that evidence contradicts, preserves authoritative facts, and improves *subsequent* registered real predictions/actions. Demonstrate delayed/out-of-order outcome handling and bounded failures.
- Then test fixed versus adaptive repeated latent refinements or hierarchical/global/entity/event representations **only if** they improve measured external predictions or net decisions under the same compute/data rules. A four-pass decoder, latent movement or teacher imitation is **not** System-2 intelligence.
- **Accept:** independent temporal improvement after feedback with bounded unsafe/costly actions and demonstrated incremental value from extra reasoning.

**Gate 5 — Transfer and scalable serving (P1/P2).**
- Evaluate independently unseen sites, customers and relevant data modalities. Separate zero-update transfer from few-shot adaptation and from continual weight updates; preregister unseen shifts, data rights, retraining allowance, baseline and forgetting checks. ME047's existing cross-site ability to execute is **not** transfer success.
- Only once capability thresholds pass, assess genuine relevant-subset compute scaling (**C5** including addressing/index build, reads, execution and verification), CPU/GPU memory, latency, energy and portability. Measure real throughput without demanding equal CPU wall time in scientific comparison.
- **Accept:** independent transferable utility, stable calibration and operational service-level qualification; otherwise preserve the failed gate.

**Gate 6 — Company pilot and product qualification (after scientific gates).**
- Select **one** legally authorized customer/workflow wedge from Woyengi/KoreFrame/Veritas, logistics or Nsat. Use a time-cut read-only integration first; then shadow recommendations; then small approved governed interventions with independent outcome collection. Maintain product-specific SLAs, rollback, human escalation, data minimization and signed audit receipts through the appropriate external platform.
- Measure improvement against the incumbent workflow **including full cost of collecting data, model execution, human inspections and failed actions**. A working demo or synthetic Veritas world is not commercial proof.
- **Accept:** multi-period customer-relevant improvement, lawful source rights, verified real actions/outcomes and secure rollout; only then discuss commercial readiness.

### Minimal common reporting card for EVERY stage

**Additionally enforce the mandatory repository checkpoint/replay gate**: every reported fit has tracked restorable checkpoint bytes in a pinned TAC-OSM commit plus environment, exact loader, score/replay script, SHA manifest, permitted data reacquisition and independent clean-checkout validation; no ephemeral-only assets. Missing pieces are BLOCKED, and a local directory or Actions artifact never makes a result complete.

Record: task question and owner; branch and exact commit; original external data licence/source/hash; entity/time cutoff and actual accessible features; preregistration; risk/support/censoring; strongest practical baselines; exact trainable/total parameters; real training data/order/tokens/updates; **actual audited training compute and tuning**; sample counts, >=2 seeds where the policy requires; sealed DEV/TEST access chronology; predictive metrics and interval; real decision cost/policy identification when applicable; calibration/abstention; full inference cost; raw checkpoints/logs; all failed attempts; reproducible CI; PASS/FAIL/UNKNOWN for each scientific gate. An unmeasured output is UNKNOWN, not PASS.

**No automatic model training, merge or deployment is authorized by this planning document.**

## Research provenance, interface decisions and ownership

The company-facing mapping above is a **research/product hypothesis**, not permission to modify those repositories. Product teams own canonical state, agent execution, security and customer-specific data. TAC research owns belief dynamics, action predictions, loss contracts, uncertainty and model scores. Veritas may audit deterministic replay, but only authentic operational data can qualify real performance. Keep model recommendations distinguishable from authorized execution in code and evidence.

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

### ME-057 fused predictive-anchor training completed — useful dynamics, no TAC advantage

[Draft PR #338](https://github.com/FPC-effortless/TAC-OSM/pull/338) completed five paired 600-update fits per architecture on the independently reserved `clean_stains_off_tabletop` task. All ten checkpoints froze before DEV/TEST opened. Both trained recurrent systems beat physical persistence on the 80-episode TEST cohort, showing that this task contains useful action-conditioned temporal structure.

| TEST original-unit physical MAE | Result |
|---|---:|
| Native fused TAC | 0.663284 |
| Matched GRU | 0.658526 |
| Persistence | 0.672681 |

TAC improved 1.40% over persistence but was 0.72% worse than GRU, won one of five paired seeds, and had a +0.000285 to +0.009266 episode-clustered TAC-minus-GRU 95% interval. Its preregistered requirement of at least 2% improvement over both GRU and persistence therefore failed. Stable eager TAC also required about 56% more optimizer CPU than GRU in the completed local fits.

The Actions run evaluated heldout data and then failed only while serializing a NumPy seed-win scalar. The committed result was recovered by rescoring hash-verified checkpoints without refitting; the workspace could not independently download the Azure-hosted Actions artifact, so local checkpoint hashes are not presented as a byte-for-byte audit of that archive. See the PR's completed-results receipt and retain this provenance distinction.

**Architecture consequence:** preserve the bounded, causal physical feedback and action-conditioned rollout that now beat persistence. Do not promote the fused anchor as the competitive core. The next complete candidate must improve the *forecasting computation itself*: make the strong servo/physical prior and current observation a direct base prediction, then learn a bounded horizon-conditioned mixture of fast-state, slow-state and associative residual experts with calibrated uncertainty/abstention. Train the entire system against factual multihorizon physical outcomes under matched parameters, authentic windows and profiled FLOPs. Do not spend another stage merely probing ME057's state; use ablations only if the stronger integrated candidate fails.

### 2026-10-23: ME060/ME061 implemented and trained capability tranche

[ME060 draft PR #339](https://github.com/FPC-effortless/TAC-OSM/pull/339) and
[ME061 draft PR #340](https://github.com/FPC-effortless/TAC-OSM/pull/340) implement
and train complete native fast/slow/associative physical world models. Twenty
original fits completed across two integrated recipes, with five paired seeds
per recipe. All failures remain preserved. This is a two-design development
program with historical source DEV exposure, not a single selection-free
independent source test.

ME060 introduced bounded 8x8 associative delta memory with norm projection,
direct horizon/action-conditioned reads, a SwiGLU residual trajectory decoder
and positive Laplace scales. Its source DEV TAC MAE .845300 lost to persistence
.702952. Frozen clothes-hanging TAC .373251 beat GRU .381716 but lost to
persistence .351410. Both gates failed; merely adding experts and a physical
prior did not make the system useful.

ME061 retained native state dynamics, added shared causal velocity/innovation
statistics, initialized forecasts exactly at persistence, learned signed
reliability-attenuated physical corrections and aligned its primary loss with
original-unit MAE. This is an integrated architecture/training improvement;
individual component effects and expert specialization have not been proven.

| ME061 factual physical MAE | Native TAC | GRU | Persistence | Servo |
|---|---:|---:|---:|---:|
| Historical source DEV,76 episodes | .603894 | .602600 | .702952 | 1.105522 |
| Frozen tea-pouring task,80 episodes | .493206 | .509535 | .455992 | 1.421244 |

Source TAC improves 14.09% over persistence and 28.56% over the preceding
ME060 recipe, with changed capacity/features/loss/seeds disclosed. It still
loses narrowly to the matched GRU, so the source gate fails. On the separately
registered different task, TAC beats GRU by 3.20%, wins all five paired seeds,
and has TAC-minus-GRU episode interval [-.021299,-.012083]. TAC remains
8.16% worse than persistence, so the transfer gate also fails. Target trajectories
opened only after models froze; no target fitting or preprocessing adaptation
occurred. Same robot interface does not establish different-site/operator
generalization. Intervals condition on fitted seeds and are not multiplicity
adjusted across this program.

ME061 uncertainty is implemented and measured, not qualified: 90% nominal
whole-episode intervals attain 85.79% source DEV coverage with mean physical
width 11.116. No observed inspection-cost utility, policy value, causal control,
iterative reasoning benefit or online weight adaptation was evaluated.

ME061 parameters 33,548 vs 33,802 and common operator-profile estimates 51.22M
vs 50.85M per update pass 1% gates. Mean optimizer CPU 18.810 vs 7.905 seconds
is an efficiency limitation. Supported-operator estimates do not certify true
complete hardware FLOPs. Source error arrays and frozen target results replay
exactly; 978 tests pass. All 20 weights/optimizer checkpoints and detailed scores
are retained in a checksummed workspace archive (PR #340 ARTIFACT.md), not
misrepresented as an independently downloaded GitHub Actions training artifact.

**Next construction objective:** retain ME061 as a development candidate and
learn reliable residual utility under real task shift, so the integrated
predictor can preserve persistence when physical/learned corrections are
unhelpful. Train against authentic future losses and calibration/risk targets;
require a newly registered target task to beat the strongest practical prior
as well as matched learned rivals. This is the next capability implementation,
not a return to an open-ended sequence of memory probes. Cost-sensitive
actions remain blocked on actual supported action/outcome/cost evidence.

### 2026-10-23: ME-062 selective forecasting and governed Woyengi handoff — SOURCE-ONLY

[Draft PR #341](https://github.com/FPC-effortless/TAC-OSM/pull/341) is a successor to ME-061, retaining bounded fast/slow/associative recurrence while adding a shared (TAC and GRU) causal horizon-conditioned trust gate over the learned physical residual, plus a positive heteroscedastic Gaussian uncertainty head and physical Gaussian training loss. Untrained model source and synthetic software contract tests only: **TRAINING=NOT_RUN; MODEL_MAE=UNKNOWN; REPO_REPLAY=BLOCKED; SUPERIORITY=UNKNOWN**. The new source code is not an observed improvement.

The branch also defines a versioned read-only Woyengi → TAC envelope for principal/workspace/entity-scoped, bitemporal (valid time vs recorded time), provenance-carrying authorized state histories and at most five candidate action sequences. TAC emits non-authoritative prediction proposals only, with checkpoint, state revision, trace and replay IDs, mandatory human review while uncertainty is unqualified; no model output authorizes an execution. Woyengi must authenticate, authorize and confirm snapshot freshness before and after inference. Veritas can stress-test RL/policies offline, but its generated worlds are **synthetic**, and model-simulated counterfactual scores cannot establish real action utility. Woyengi feature adapters, action governance integration, and Veritas trainer/evaluator adapters are NOT yet built by this TAC PR; absorbing SafeEdge/KoreFrame into Woyengi remains a separate authorized integration task.

ME-062 must be preregistered for a new untouched authentic physical task before any target labels are opened; compare equally trained/parametered/data/computed GRU, persistence, servo, and matched no-memory control; retain historical DEV as exploratory only and avoid old exposed TEST, clothes-hanging and tea-pouring as untouched confirmation. Measure 90% coverage and risk-width/coverage with new CAL-only Gaussian calibration, plus reset-state and separately trained k=1/4/16/64 memory ablations. Primary qualification requires beating the strongest practical baseline on the unseen real task with uncertainty-supported advantage. Simulated Veritas rollout success never substitutes. CPU runtime is disclosed independently, not required equal.

**New compulsory gate already effective:** no ME-062 fit can be declared complete unless original weights, optimizer and replay evidence for every compared arm/seed are in the TAC Git repository under `research/replay-bundles/ME062/`, retrievable and independently verified from a clean checkout. Before running any expensive fits, reserve that storage, register FIT provenance/schedules/compute/selection, and verify legal replay source access. The previously retained ME-061 local archive is historical evidence, not proof of newly required in-repository replay. See PR #341's [protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/me062-reliability-woyengi-handoff-2026-10-23/research/mathematical-engineering/me062/PROTOCOL.md) and [governed handoff](https://github.com/FPC-effortless/TAC-OSM/blob/research/me062-reliability-woyengi-handoff-2026-10-23/research/mathematical-engineering/me062/WOYENGI_HANDOFF.md).


### 2026-10-23: ME062 genuine original FIT training, repo checkpoint preservation, and prospective transfer

[Draft PR #341](https://github.com/FPC-effortless/TAC-OSM/pull/341) now has actual real original robot FIT-only training, **not merely source code**. [Original-FIT-only numeric preflight](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39881743894) audited 208 authentic episodes, parameter counts 37,427 TAC / 37,681 GRU (0.674% gap), and same-real-batch supported-operator update estimates 57,191,440 TAC / 56,827,806 GRU FLOPs (0.636% gap). All active parameters had finite preflight gradients; zero optimizer steps or heldout trajectories were opened in the preflight.

[Original ten-checkpoint FIT training run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39882865345) fitted 5 paired seeds 62001–62005, 600 AdamW updates per model, original source-sha `b55d0e1f894072c5738ef8e45aa283a3b464c1a1dc8f60c63594666d16b30a49`, all training sample orders frozen, followed by final numerical parity PASS. Physical validation data remained unopened during fitting. Original model+optimizer/RNG/checkpoint bytes for each seed/arm are now in **this TAC Git repository** under `research/replay-bundles/ME062/checkpoints/`; a separate GitHub Actions clean checkout verified the 10 SHA-256 identities and reloaded all models. `MODEL_REPLAY=PASS`. Fit CPU (efficiency only) ~30.46 s TAC vs ~13.83 s GRU; ~2.20x penalty. Final profiled FLOP receipts per model 34.314864 B TAC / 34.0966836 B GRU; unsupported operations and complete hardware FLOPs remain uncounted/uncertified.

The genuine different-task evaluation was prospectively registered using **metadata only**, selecting `line_trash_can_with_trash_bag` and 80 episodes at pinned publisher revision `9290f5a96dc91ea58256e81d9ce2dd9621890b51`. This task selection excludes historical source, clothes-hanging and tea-pouring research uses and did not read target trajectories, interventions or outcomes. It is the same robot interface and does not demonstrate operator/site independence.

**Scientific qualification remains pending:** trained checkpoints and model restore are separate from measured future-state MAE, uncertainty calibration, original heldout source DEV and registered transfer. Source DEV has historical development exposure and is not independent TEST. The [source+target scoring and independent replay workflow](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39883815447) subsequently completed successfully, reproducing original published real scores byte-for-byte from a distinct clean checkout with all ten Git original checkpoint bytes and **zero additional fits**: `MODEL_REPLAY=PASS`, `SCORE_REPLAY=PASS`, but `SCIENTIFIC_SUPERIORITY=FAIL`. Source DEV is not independent TEST, and same-robot different-task transfer is not independent site/operator generalization. Do not promote any ME062 architecture/transfer/decision claim until exact outputs, bootstrap intervals, strongest-prior comparisons, source calibration, score replay and replay receipts are independently verified and preserved in Git. The matched no-memory and k=1/4/16/64 context-length experiments are still unrun; unique associative/slow-memory importance is not proven. Veritas RL simulations and Woyengi governance interfaces remain separate from real intervention evidence. 


### ME062 exact scored negative result — frozen original FIT models, no post-test refit

The **five-seed ME062 registered accuracy gates FAILED**. Both original real cohort score files were generated from the ten previously Git-frozen and independently model-restored checkpoints: [source CAL/DEV and different-task scored summary](https://github.com/FPC-effortless/TAC-OSM/blob/research/me062-reliability-woyengi-handoff-2026-10-23/research/replay-bundles/ME062/REAL_WORLD_EVALUATION_SUMMARY.json). Source DEV (historically exposed, 76 episodes) physical MAE: native TAC **0.609373443944**, matched trained GRU **0.605164397358**, last-state persistence **0.702951822257**, servo **1.105522184081**; TAC wins **0/5 paired seeds**, TAC-minus-GRU episode-bootstrap interval **[+0.00195320, +0.00662016]**. The explicit reliability gate preserves a 13.31% source advantage over persistence but is inferior to GRU; do NOT credit native memory superiority or an independent test.

The frozen, metadata-preregistered *line_trash_can_with_trash_bag* target (80 original episodes) physical MAE: TAC **1.256390114265**, GRU **1.223529804823**, persistence **1.295032144382**, servo **1.031359254918**. TAC wins **0/5** paired seeds, TAC-minus-GRU interval **[+0.02227347,+0.04447820]**, and also loses to the *strongest practical servo baseline*. This is same robot interface with unverified site/operator independence; no target fitting, preprocessing, or calibration was permitted. Both superiority gates **FAIL**; preserve the negative result. The task-dependent switch from persistence-strong in ME061 tea pouring to servo-strong in ME062 trash bag is an observed baseline ranking change, **not** proof of causal source-shift mechanism.

Nominal 90% CAL-based whole-episode intervals for TAC achieve only **87.37% source DEV coverage** (mean physical width 12.6399) and **30.0% target transfer coverage** (mean width 12.2252); GRU whole-episode coverage is 85.53% source and 20.50% target. Observed transfer calibration catastrophically fails, and observational exchangeability remains unverified. Do NOT promote confidence-based auto-execution/abstention and do not tune on this now-exposed transfer. Gaussian trust-gating + shared physical recipe did NOT solve generalization; the stronger servo baseline makes the shift clearer. Matched no-memory and independently retrained context-length causal memory tests remain unrun. Research next must target robust physics/persistence selection under **a new untouched task**, with a genuinely independent heldout and action-support evidence before commercial/action claims. Preserve original in-repo weights, source failure receipts and subsequent independently replayed scores. 


**Post-evaluation replay completion:** [ME062 Git-tracked independent real-world score replay receipt](https://github.com/FPC-effortless/TAC-OSM/blob/research/me062-reliability-woyengi-handoff-2026-10-23/research/replay-bundles/ME062/EVALUATION_REPLAY_RECEIPT.json) records original ten checkpoints restored and CAL/DEV/80-target source results exactly recomputed; this satisfies the repository-owned preservation and *this dataset's* legal reproducibility lane. Do **not** confuse an independent computational replay with an untouched scientific independent holdout or qualified predictive superiority. Keep all negative scores and old frozen models available in the research branch.


### 2026-10-23: ME063 real FIT three-regime mixture result — both registered gates FAILED

[Draft PR #342](https://github.com/FPC-effortless/TAC-OSM/pull/342) (stacked on ME062, not merged) actually trained ten native TAC/GRU physical predictive models on publisher-original real FIT (208 episodes, frozen SHA b55d0e1f894072c5738ef8e45aa283a3b464c1a1dc8f60c63594666d16b30a49, five seeds 63001-63005 and 600 updates each), retaining original checkpoint/optimizer/RNG bytes directly in `research/replay-bundles/ME063/checkpoints/`. Supported-operator prefit resource parity: 37,177 TAC versus 37,431 GRU trainable params (**0.679% gap**); 56,895,406 TAC versus 56,531,772 GRU profiled forward/backward FLOPs/update (**0.639% gap**). Actual training final parity PASS; CPU update totals ~20.06 s TAC versus ~9.30 s GRU (efficiency limitation only; no equal CPU requirement). A separate clean GitHub Actions checkout verified all ten checkpoint hash identities and loadability **before any scoring**. The true full hardware/optimizer FLOP accounting remains uncertified.

**Architecture:** shared across TAC/GRU, a causal horizon-conditioned 3-way convex mixture of last-state persistence, original-FIT-identified physical servo state prediction and native recurrent neural prediction. Its uncertainty head moment-matches physical-prior spread, learned neural uncertainty, and component disagreement, then CAL-only source-episode conformal calibration. This is **not** a new general mixture-of-experts or adaptive calibration theory (see PMLR 2026 Dynamic TMoE and JMLR 2024 Gibbs/Candès); novelty question is empirical usefulness of native fast/slow/associative state with matched controls. Actions are logged factual covariates, not an intervention.

| Real original physical MAE (lower better) | TAC | Matched trained GRU | Last-state persistence | Original FIT servo |
| --- | ---: | ---: | ---: | ---: |
| Exposed source DEV, 76 episodes | **0.572882** | 0.573576 | 0.702952 | 1.105522 |
| Newly metadata-registered `fold_and_stack_scattered_clothes_neatly`, 80 episodes | 0.828290 | 0.829905 | 0.895391 | **0.793413** |

TAC narrowly wins 4/5 source seeds and 3/5 frozen transfer seeds, with **recording-block** TAC-minus-GRU 95% intervals [-0.003577, +0.001314] and [-0.008709, +0.005304], respectively. **Both intervals include zero, so neither TAC-GRU advantage establishes superiority.** On the untouched clothes-folding task, the servo forecast is better than both trained neural models (TAC 4.4% worse than servo); primary predeclared best-comparator requirement **FAILS**. No ME062-to-ME063 controlled paired improvement claim: architectures, training recipes and seeds changed, and the prior exposed trash-bag target cannot serve as new confirmation. This is a different recorded task on the same robot interface, not independently verified site/operator generalization.

Mean TAC router weights source DEV: persistence 7.33%, servo 40.89%, neural 51.78%; frozen transfer: persistence 10.36%, servo 55.09%, neural 34.55%. **Causal-input routing shifts toward servo in the new task without target fitting**, but the aggregate blend fails to match servo alone, and learned weights must not be called calibrated confidence. Source-trained nominal 90% episode conformal coverage: TAC **83.95% source DEV**, **72.25% new transfer**, with mean physical widths ~27.69 and 16.74. Both coverage and sharpness remain unqualified; exchangeability and independent-site claims unsupported. [Actual score source and checkpoint evidence](https://github.com/FPC-effortless/TAC-OSM/blob/research/me063-causal-regime-mixture-real-robot-2026-10-23/research/replay-bundles/ME063/REAL_WORLD_EVALUATION_SUMMARY.json).

**Research interpretation:** ME063 empirically shows learned *task-dependent weighting* and competitive but not demonstrably better TAC performance; physics prior selection remains unreliable. Scientific gate **FAIL**; memory ablation, long-history entity retention, interventional decision-cost utility, online verified adaptation, and calibrated policy abstention NOT demonstrated. Preserve all genuine original checkpoints and negative labels; do not tune on exposed source DEV or the now-open new transfer. The next experiment should isolate **causal forecast-correction necessity and out-of-support detection** on a separately registered, truly untouched real task, with proper stronger-prior/fair-GRU controls, rather than rediscovering mixtures or conformal methods.

**ME063 evaluation-replay audit complete:** [GitHub Actions run 39886390227](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39886390227) completed both original real evaluation and a separate clean-checkout replay. [Git-owned EVALUATION_REPLAY_RECEIPT.json](https://github.com/FPC-effortless/TAC-OSM/blob/research/me063-causal-regime-mixture-real-robot-2026-10-23/research/replay-bundles/ME063/EVALUATION_REPLAY_RECEIPT.json) reports source DEV and frozen clothes-folding transfer score files recreated **byte-for-byte by SHA-256** from the original ten Git-owned checkpoint bytes and legally accessed pinned publisher recordings; NO refits. Thus `MODEL_REPLAY=PASS`, `REAL_EVALUATION_REPLAY=PASS`, but `ARCHITECTURE_SUPERIORITY=FAIL`, `CALIBRATED_UNCERTAINTY=FAIL`, and `ACTUAL_POLICY_VALUE=NOT_TESTED`. Preserve all negative results and do not merge this branch as a model capability upgrade.


### 2026-10-23: ME064 true task-level leave-one-task-out training registration

[ME064 draft PR #343](https://github.com/FPC-effortless/TAC-OSM/pull/343) uses **five authentic previously development-exposed physical robot tasks**, sharing a 14D r1_lite recorded action/state interface, for *a single common jointly trained native TAC and matched GRU per paired seed*. Frozen source task cohort: clean-stains, fold-and-stack clothes, hang clothes, line-trash-bag, tea-pouring, with **32 whole FIT and 8 whole CAL episodes per task** selected by original publisher metadata hash, 4-episode recording block separation, 160 FIT original total, 40 CAL real total. Original clean-stains source TEST is forbidden from the new training cohort by reusing the original source FIT restriction. Original metadata-only selection of never previously evaluated task `install_spray_nozzle_onto_bottle` froze an 80-episode target before reading a single target trajectory; plug-charger and tissue-box remain unvisited.

Training is single five-task **multi-source, leave-one-previously-unseen-task-out** (NOT 5-fold task cross-validation nor a new hardware/site/operator split). Each of new paired seeds 64001..64005 receives the same 600 optimizer updates, batch eight, 64 state/action history and 16 future observed-action labels, **round-robin task** assignment (120 updates per source task per seed), independently sampled eligible origins within each source task, identical target/authority censoring, FIT-only normalization and servo prior identification, and a shared causal 3-way physical mixture head. Five seeds x TAC/GRU = 10 planned models. No learned target preprocessing, target labels in training, post-hoc checkpoint selection, source-calibration leakage or heldout retuning.

[Original five-task real FIT preflight](https://github.com/FPC-effortless/TAC-OSM/blob/research/me064-multitask-leave-one-task-out-2026-10-23/research/mathematical-engineering/me064/PREFLIGHT_MEASURED.json) measured: original 160 FIT robot episode source digest `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`, **37,177 TAC / 37,431 GRU parameters** (gap 0.679%), **56,895,406 TAC / 56,531,772 GRU PyTorch-supported operator FLOPs/update** (gap 0.639%). Parameter, supported-operator FLOP and all active-gradient source checks PASS. Whole optimizer/hardware FLOPs remain uncounted; CPU wall-clock equality NOT required. [Registered FAIR_COMPARISON](https://github.com/FPC-effortless/TAC-OSM/blob/research/me064-multitask-leave-one-task-out-2026-10-23/research/mathematical-engineering/me064/FAIR_COMPARISON.json) is frozen before any optimizer update. The scientific registered target requires TAC lower MAE >=2% versus each trained GRU, persistence and FIT servo with 4-episode source block-clustered paired CI upper bound below zero, and source CAL derived empirical nominal 90% whole-episode coverage/risk width reported. Same robot is NOT proven independent domain/operator/site.

Original checkpoint **bytes** (10 trainable model/optimizer/RNG and original source/schedule hashes) MUST be stored in TAC Git under `research/replay-bundles/ME064/`, clean-checkout restored before CAL/target opened, then independently reproduced from original legally available publisher trajectories. Preserve negative result or exact failure. No simulator task result qualifies real-world success. Production Woyengi integration forbidden absent independent model calibration and action utility.

### Woyengi guardrails verified alongside ME064

[Woyengi draft PR #53](https://github.com/FPC-effortless/Woyengi/pull/53) is a separate **source-only** TypeScript pure function and ADR based on the actual Woyengi `AgentGateway`, `CapabilityEngine`, `LocalComputeProvider` and governed effects semantics. It verifies Woyengi-origin read permission and snapshot freshness, task/request/checkpoint fingerprint and finite physical forecast receipt. Even a valid checkpoint forecast is `ADVISORY_REVIEW_REQUIRED` and `executionPermitted=false`; an unqualified ME062/063/064 forecast is `UNQUALIFIED_REVIEW_ONLY`; invalid references, stale/revoked permissions, malicious self-granted execute, and unknown model are blocked. This is NOT YET connected to a live API or production `AgentGateway.act()`; separate runtime reauthorization before any real effect remains mandatory, and no forecast can count as authority or canonical factual state. Woyengi CI Quality/Conformance and Container Smoke checked and PASSED on the draft PR. Veritas synthetic tests remain offline and never convert inferred counterfactual forecasts into real intervention evidence.


### 2026-10-23: ME064 real untouched spray-nozzle result — strongest practical baseline gate FAILED

[Draft ME064 PR #343](https://github.com/FPC-effortless/TAC-OSM/pull/343) trained 10 real original 5-task FIT-only model/optimizer/RNG checkpoints for paired seeds 64001–64005 under resource matching (37,177 TAC / 37,431 GRU parameters, gap 0.679%; 56,895,406 TAC / 56,531,772 GRU supported-operator FLOPs/update, gap 0.639%) and passed final actual-train preflight/compute parity. Five previously development-exposed tasks supplied 32 original FIT episodes each (160 total), with a separate 40 original CAL episodes (8 per task), identical round-robin task/window schedules and 600 optimizer updates per arm/seed. Original source FIT digest `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`. Ten original checkpoint bytes with optimizer/RNG source identifiers were committed in `research/replay-bundles/ME064/checkpoints/` BEFORE target outcomes, and an independent clean checkout verified 10 SHA-256 and restored 10 models in `CLEAN_CHECKOUT_MODEL_REPLAY.json`. Training is genuine original data, not synthetic or imagined.

The metadata-only preregistered entire untouched target `install_spray_nozzle_onto_bottle` (80 genuine publisher episodes, 20 four-episode blocks) was chosen from never-evaluated available tasks before opening ANY target trajectory. It was excluded from training, task normalization, source CAL and model selection. It is a different task on the SAME robot platform, not independent site or operator.

**Exact initial authentic physical holdout MAE** (lower better):
- **Last-state persistence 0.0123712880507810 (BEST)**
- **Native five-task TAC 0.135153214889579**
- Matched five-task GRU 0.148456336071249
- Source FIT-identified servo 0.918423870205879

The native TAC average score is about 9% lower than GRU, with paired original recording-block bootstrap 95% interval for TAC minus GRU **[-0.01809460,-0.00862360]** (uncertainty over the fixed five-model averaged episode predictions), yet TAC wins only **2/5** independent training-seed pairs, so superiority across independent model fits is NOT established. TAC versus persistence block bootstrap **[+0.11124416,+0.13384909]**; the last-state forecaster is approximately **10.9x lower-error than TAC**, definitively failing the registered all-baselines superiority gate. TAC's router assigned mean 51.06% persistence, 22.40% servo, 26.55% neural. This is not enough to avoid severe overcorrection during nearly static source task conditions. Mean source-CAL-based nominal 90% **whole-episode** coverage on untouched target reaches **100%** for both TAC and GRU, but is **extremely conservative**, with mean physical interval widths ~17.07 (TAC) and ~15.90 (GRU) versus TAC physical MAE ~0.135. Never imply this qualifies accurate, sharp risk estimates or safe autonomous decisions. Native memory necessity, independent-site transfer and real counterfactual utility remain NOT demonstrated.

**Honest reproducibility caveat:** initial [GitHub real outcome scoring run #39891151580](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39891151580) preserved original score receipts and returned correct negative gate; separate clean-checkout reproduction generated virtually identical figures but score JSON SHA-256 differs by small CPU floating-point differences (example TAC MAE 0.135153213376 vs 0.135153214890). Consequently the earlier **bitwise numerical score replay gate correctly FAILED**, despite 10/10 exact model checkpoint SHA and strict load PASS. A new separate objective numerical-tolerance replay audit is registered in `scripts/me064_validate_numeric_replay.py`, with exact nonfloating/source/schema identity, frozen original vs recomputed SHA receipts, absolute 1e-4/relative 1e-5 tolerances on individual score floats, tighter 1e-5 primary MAE/CI summary tolerance; this must **PASS** before `REAL_SCORE_REPLAY=PASS_WITH_FLOAT_TOLERANCE`. Do NOT falsely label byte-identical score replay, do NOT use unseen task for threshold tuning or refitting. The PR stays draft and unmerged. Preserve failed results and original checkpoint/source bytes.

**Research lesson:** ME062/ME063/ME064 confirm the strongest physical baseline changes by task and current mixture is not trustworthy even when routing weights shift. Stop treating a three-way blend as intrinsically calibrated. Before Woyengi production action use, native TAC must demonstrate useful selective **non-intervention**, reliable dynamics change detection, sharply bounded risk coverage, and decision value on separate authentic independent episodes/tasks with externally verified actions/outcomes. Woyengi's CI-tested advisory `tac-forecast-guard.ts` in draft [PR #53](https://github.com/FPC-effortless/Woyengi/pull/53) remains source-only, fail-closed for execution and not yet wired to production AgentGateway; Veritas simulations remain diagnostic only.


**ME064 frozen score-only posthoc diagnostic (NOT a new independent test):** [Git-owned original task error decomposition](https://github.com/FPC-effortless/TAC-OSM/blob/research/me064-multitask-leave-one-task-out-2026-10-23/research/replay-bundles/ME064/POSTHOC_FAILURE_DIAGNOSTIC.json) was generated strictly from preexisting original per-episode score receipts, without model inference, new target publisher downloads, optimizer steps, weight selection or recalibration. Using fixed *exploratory, after-exposure* physical persistence-MAE boundaries (<0.025 near stationary; 0.025–0.1 limited motion; >=0.1 moving), **78/80** original nozzle-install episodes are near-stationary (persistence MAE mean 0.011458 vs TAC mean 0.134010), **2/80** limited motion (persistence mean 0.048003 vs TAC mean 0.179749), **0/80** substantial motion by this band. TAC beats persistence **0/80** episodes, versus GRU **55/80** episodes. The per-seed original TAC/GRU mean errors show **2/5 TAC seed wins** with substantial outlier sensitivity: seed 64005 TAC 0.337241 vs GRU 0.491191, despite three GRU-favoring seeds. The episode-cluster CI for five-model averaged differences therefore cannot establish superiority across independent training seeds. This diagnostic's thresholds were selected AFTER the independent holdout's aggregate error was known and MUST NOT be used as model inference thresholds, policy gates, architecture tuning evidence or fresh generalization proof. It strongly motivates a **prospectively registered** real motion/nonmotion distinction on another untouched source, rather than additional post-hoc tuning on this target.


### 2026-10-23: ME065 comparative micro-world-model controls — source preflight only

[Draft ME065 PR #344](https://github.com/FPC-effortless/TAC-OSM/pull/344) establishes new **untrained** compact Gaussian stochastic RSSM and selective-diagonal-SSM recurrent controls, both using ME063's same 14D observed physical state/actions, persistence/servo/neural experts, causal history observer, horizon-conditioned forecast decoder and risk interface. Fully trained native fast/slow/associative TAC and matched GRU remain the original benchmark arms. This is NOT a full DreamerV3 agent, not Mamba, and does not turn native TAC into an SSM. Real-source parity has **not** yet been verified: the original five-task FIT-only PyTorch parameter/FLOP/gradient preflight is underway, must enforce <1% differences for every trained arm, and **must stop all full fits** if failed. ME056 concept-swarm/CED remains a separate fifth candidate and cannot join a fair comparison until its previously failed ~61.32% real-FIT FLOP preflight is corrected without dummy operations. No ME065 training, heldout test, skill, new superiority or agent-control claim follows from this scaffold. See `research/mathematical-engineering/me065/PROTOCOL.md` on the PR branch.

**Compulsory follow-on only after successful original source preflight:** preregister a new untouched genuine original publisher robot task using **metadata only**, never the already exposed ME064 spray-nozzle target; match original FIT data, trainable counts, optimizer-update/supported-compute budget, paired seeds and calibration protocol; preserve all original full model/optimizer/RNG checkpoint **bytes in Git** before opening new outcome labels; independently restore and score; compare native TAC and each eligible competitor against strongest physical persistence/servo as well as the neural alternatives and report **both episode-block uncertainty and independent-seed consistency**. No automatic Woyengi execution or reward/counterfactual intelligence assertion.

**Frozen failure diagnosis (exploratory, not qualifying):** [ME064 score-only posthoc diagnostic](https://github.com/FPC-effortless/TAC-OSM/blob/research/me064-multitask-leave-one-task-out-2026-10-23/research/replay-bundles/ME064/POSTHOC_FAILURE_DIAGNOSTIC.json) counted **78/80 near-stationary** original heldout episodes under its explicitly posthoc observed-future motion bands; mean TAC physical MAE ~0.1340 versus persistence ~0.01146 in those episodes. TAC beat GRU on **55/80** individual episode means but persistence on **0/80**. The routing weight to persistence was only ~51%; it failed to prevent large nonzero forecasts when little happened. These diagnostic classes are derived from FUTURE outcomes and must never feed a production gate or be reused to tune on those 80 original episodes. Future work must evaluate real observable forecast-time change evidence and selective non-intervention on another independently reserved authentic task. Previously reported CAL-based interval coverage was unhelpfully wide.



### 2026-10-23: ME066 four-arm authenticated real FIT training

**Observed training result, not a capability claim.** [Draft PR #345](https://github.com/FPC-effortless/TAC-OSM/pull/345) created and source-registered a four-architecture same-original-data challenge: native TAC ME063 (37,177 params), fully trained GRU (37,431), compact Gaussian deterministic+stochastic RSSM (37,319), and input-selective stable diagonal-SSM control (37,415, **not official Mamba**). ME065 [draft PR #344](https://github.com/FPC-effortless/TAC-OSM/pull/344) independently measured all on original real five-task FIT (160 authentic source episodes); reported supported-operator FLOPs per update were **56,895,406 / 56,531,772 / 56,423,631 / 56,841,027**, maximum 0.679% parameter and 0.829% FLOP disparities, all active gradients present, numeric prefit PASS. These are partial profiler FLOPs and **do not certify all-kernel hardware compute**, and CPU wall-clock equality is NOT compulsory. All active recurrent bodies share the genuine observed state/action and physical persistence/servo/neural forecast router/head, with disclosed .005 RSSM KL auxiliary.

ME066 separately registered 5 prospective paired original seeds **65001–65005**, **600 source-real AdamW optimizer updates × 4 architectures = 20 actual full FIT models** using exactly the same original FIT task-window order within each paired seed; source FIT SHA `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`. Git owns **all 20 original trained weight/optimizer/RNG checkpoint bytes**, their per-seed original hashes, and postfit numeric parity receipts under `research/replay-bundles/ME066/`. **Actual original postfit final parity PASS** for active parameters, source data/order and PyTorch-supported forward/backward estimate. The metadata-only, prefit-frozen new 80-episode original target is **`plug_charger_into_power_strip`**, explicitly different from the exposed ME064 spray-nozzle; **`replace_tissue_box_and_pull_out_first_tissue`** remains reserved. No source CAL or new target trajectory was opened during prefit or any of the 20 original optimizer fits.

**Exact model restoration still has a required audit:** first independent clean Git model load job failed *before loading any checkpoint* due absent `pyarrow` Python module in its CI verifier environment, **not** changed weights or failed metric. The separate repair installs the dependency and re-runs frozen checkpoint restoration **without retraining any model**. Until clean-Git SHA+load audit produces PASS, **ME066 new heldout scoring is strictly blocked**, and no performance/superiority/transfer/decision claim exists. Model and immutable physical publisher source bytes, original fitting and negative score evidence must stay Git-replayable. Once independently verified, score only the original metadata-reserved real charger task; compare TAC versus RSSM/SSM/GRU/persistence/servo, motion forecasting, source-CAL-only uncertainty/sharpness, 4-episode original recording-block bootstrap and all five seed pairs. Checkpoint hash replay and score replay are distinct: label any non-bitwise independent numeric score agreement with explicitly frozen valid float tolerance, never pretend bitwise equality. Real factual recorded actions do not constitute intervention policy utility; Woyengi remains advisory only. The previously unqualified **ME056 Concept-CED** (61.32% profiled FLOP mismatch) is not presented as a fair trained fifth arm.



### 2026-10-23: ME066 genuine sixth-task results against compact world-model controls

**First frozen authentic new-task result published, all-baselines superiority FAIL.** [Draft PR #345](https://github.com/FPC-effortless/TAC-OSM/pull/345), original scoring from [GitHub workflow #39894504075](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39894504075) on metadata-registered original publisher `plug_charger_into_power_strip` (80 genuine episodes, 20 four-episode recording blocks; same robot platform, distinct never-before-opened task). The unchanged twenty original real-FIT checkpoint bytes were source-trained on five original tasks and independently strict-hash and model-loaded **20/20 PASS** from clean Git checkout before any CAL or new target trajectory was opened.

**Frozen physical MAE** (lower is better):
- last-observed physical persistence **0.2453838432** (**best overall**);
- trained matched GRU **0.3815281845** (**best learned**);
- compact input-selective diagonal SSM control **0.3989837375**;
- compact Gaussian RSSM control **0.4034837283**;
- **native TAC fast/slow/associative 0.4105902042** (**worst learned arm**);
- source-FIT identified physical servo **1.2005298074**.

Native TAC was ~7.62% higher-error than GRU and ~67.33% higher-error than persistence; it beat GRU in only **1/5** independent trained seed pairs, while episode-block 95% CI of native TAC minus GRU **[+0.01766173,+0.04044166]** excludes parity against TAC. TAC relative to persistence original block 95% CI **[+0.11451411,+0.20985601]**, establishing factual holdout inferiority. TAC vs selective SSM CI **[+0.00276894,+0.01948439]**, unfavorable to TAC. TAC vs Gaussian RSSM CI **[-0.00715383,+0.02097459]** spans zero, insufficient to claim a reliable RSSM win over TAC. **Registered TAC superiority gate FALSE**. Seed-pair training variability is substantial and must not be hidden by episode-block resampling of averaged predictions.

Original source-CAL-only nominal 90% whole-episode uncertainty actual target coverage: TAC **77.75%**, GRU **79.0%**, RSSM **81.25%**, selective SSM **74.25%**. Mean physical interval width TAC ~**23.18** versus error ~**0.411** (width/error ~**56.45**), GRU ~**19.72**, RSSM ~**20.64**, selective SSM ~**20.25**. Hence all demonstrated poor interval sharpness and unqualified transfer coverage; neither any policy effect nor safe abstention follows. New task motion bands are **posthoc outcomes** (27 near-stationary, 30 limited, 23 moving) and may diagnose failure, not tune/test a new forecast gate on these same observations. The router assigned native TAC persistence/servo/neural **28.18% / 33.30% / 38.52%**, not a demonstrated correct regime classifier.

**Research decision:** no promotion or merging of ME066 as positive TAC architecture. Preserve the full negative outcome and all Git-owned original weights/optimizer/RNG with source episode/source hashing, original CAL/test scores, frozen budgets and source code. This matched four-arm original robot experiment does **not** show native tri-state memory superiority over GRU/RSSM/selective recurrence or physical persistence. Do not characterize the diagonal selective control as official Mamba, nor the Gaussian RSSM as full DreamerV3. ME056 Concept-CED remains independently **UNTRAINED / computational parity unqualified**, not an unseen challenger result. Independent separate original score recomputation is required; initial first-frozen score receipt is valid factual evidence but **do not say numeric replay PASS** until receipt is committed. Do not use these 80 original target episodes for model selection/tuning after exposure.

**Next research priority:** investigate why matched learning produces false motion and unsound uncertainty across real physical regimes, without rediscovering known mixtures/SSM math or adding cosmetic heads. Require a change/no-change detector supervised only on original FIT or freshly reserved authentic episodes, compare a stronger *honest* persistence/physics/robust-learning reference, test genuinely long-horizon and action-sensitive factual information in independent tasks, then verify memory mechanism against matched no-memory and bounded-history controls. Any new architecture/optimizer objective demands another **unused** originally authentic task and all usual 1% active parameter/profiler compute parity and Git-owned replay gates. Robotic logged-action forecasts remain observations, not evidence of identified do(a) counterfactual treatment effects or autonomous control value. Woyengi remains advisory-only.



**Independent ME066 original real score replay FINAL RESULT (2026-10-23): PASS_WITH_FLOAT_TOLERANCE.** A separate clean GitHub Actions checkout independently reloaded all 20 source-trained checkpoint bytes, refetched original legal publisher source CAL and untouched charger task trajectories, recomputed and audited all original score JSON schemas and source/target metadata (strict equality for nonfloat data). Original score JSONs were **NOT bitwise identical** due CPU variation: 2,168 of 579,784 float fields differed, maximum abs float difference **0.0000346899032593**, within prospectively declared abs 1e-4/rel 1e-5 per float and 1e-5 primary physical-MAE/CI summary limits. Verified receipt: [ME066 EVALUATION_REPLAY_RECEIPT.json](https://github.com/FPC-effortless/TAC-OSM/blob/research/me066-four-arm-original-robot-training-2026-10-23/research/replay-bundles/ME066/EVALUATION_REPLAY_RECEIPT.json); independent original [Actions run 39894504075](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39894504075) completed successfully. This is **numerical replay under an explicit bounded float tolerance, never exact score-byte replay**; original 20 trained checkpoint hashes/load PASS remained exact. Registered TAC scientific superiority FAIL is reproduced and the original negative physical result is final for this untouched task. Do not retune, reselect or retrain on these exposed results; PR remains draft and unmerged.


### 2026-10-23: ME067 frozen memory lesion result — fast/associative help, reliability shift hurts

**Original GitHub run [ME067 #39898699817](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39898699817) and draft [PR #346](https://github.com/FPC-effortless/TAC-OSM/pull/346).** The new registered `noncomparative_diagnostic` passed compulsory research governance, independently reused 20 original ME066 trained checkpoint bytes and the PASS_WITH_FLOAT_TOLERANCE frozen actual publisher physical scores, reread **only 40 already-exposed source CAL + 80 already-exposed charger task episodes** with verified original per-parquet SHA, retrained **zero** models and changed **zero** weights. All unlesioned per-model episode forecasts reproduced the original outcome origins, live-step masks and score within predeclared absolute 1e-4. This was a strict *posthoc latent-state readout intervention*, not another heldout task or trained variant.

**Original previously exposed charger task frozen native TAC MAE** 0.4105902042 vs persist 0.2453838432, GRU full 0.3815281844. Readout lesions (mean five independently originally trained checkpoints, 80 original episodes, lower is better):
- zero velocity AND innovation reliability traces: **0.2952071325**; delta to full **-0.1153830717**, favorable on **5/5** frozen seeds, block-bootstrap fixed-model 95% interval **[-0.14527494, -0.08512809]**.
- zero slow state only: **0.3811334452**; delta **-0.0294567591**, favorable **3/5** frozen seeds, fixed-model original-block CI **[-0.04103664,-0.01572630]**.
- zero 8×8 associative state only: **0.4428551179**; delta **+0.0322649137**, worse on **5/5**, fixed original-block CI **[+0.02438969,+0.04093000]**.
- zero fast state only: **0.6001573078**; delta **+0.1895671035**, worse on **5/5**, fixed original-block CI **[+0.16665579,+0.21248935]**.
- zero both fast and slow: **0.4844864206**; delta **+0.0738962163**, worse on **5/5**.
- zero ALL hidden/reliability: **0.4043588681**, delta **-0.0062313361**, interval spans zero.

**Exposed original source CAL (40 genuine episodes) acts as a counter-indicator against simplistic pruning.** Full native TAC MAE **0.7572043383**; zero fast **0.7930257016**, zero slow **0.7920301288**, zero association **0.7628351577**, zero reliability/kinematics **0.7590431777**. Fast/slow appear useful in source CAL while slow and reliability induce adverse transfer to the exposed charger task. These findings are consistent with a *domain-sensitive reliability/innovation-conditioning failure*, rather than a globally useless native memory. The frozen unmodified trained GRU also improved from 0.3815281844 to 0.3227768677 when the same reliability traces were zeroed on the charger target, with mixed seed directions; the vulnerability is **not exclusively TAC**.

**Interpretation and restrictions:** Each lesion is an out-of-training-distribution edit to the trained hidden belief, so it does NOT prove causality in physical dynamics, identify the correct new model weights, or qualify deployment performance. Posthoc ranking of multiple lesions on the *already revealed* charger labels cannot constitute a confirmatory gate. The known original trained winner remains matched GRU and the original physical baseline winner remains persistence. Do not use charger/spray-nozzle for fitting or threshold/model selection; preserve original negative ME066 outcome. New physical models require a prospective original FIT-only objective and equal-data/parameter/profiler-FLOP preflight, source CAL selection, complete repository-owned original checkpoints with clean-Git model+score replay, and a **genuinely never-opened third task** `replace_tissue_box_and_pull_out_first_tissue` (or a genuinely independent external physical source).

**High-value next integrated hypothesis, not another arbitrary 1-variable tweak:** native TAC's fast/slow/associative state can be useful but should learn *whether physical motion follows the current action*; separate observation-reliability/authority from the state-prediction readout, enforce an explicit train-only persistence-referenced **conditional innovation / selective no-change control**, and train jointly on observed real FIT action transitions. Compare matched native TAC + identical GRU control, stable persistence and servo on a newly sealed real task. Make the action-alignment timing explicit, stress long horizons and change/no-change, track sharp physically meaningful intervals and seed variance. Existing switching-state filters / innovation diagnostics are prior art; do not claim novelty simply for a mixture router. Independent physical reward/authorized intervention evidence is required for commercial control; Woyengi remains read-only advisory.


### 2026-10-23: ME068 source-physical action timing — 257,971 genuine autonomous transitions

**GitHub source-only action audit complete and governance PASS**, [draft PR #347](https://github.com/FPC-effortless/TAC-OSM/pull/347), [source-owned original result](https://github.com/FPC-effortless/TAC-OSM/blob/research/me068-original-robot-action-timing-audit-2026-10-23/research/mathematical-engineering/me068/SOURCE_FIT_ACTION_ALIGNMENT.json), [original Actions audit run #39899562213](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39899562213). ME068 read **ONLY the 160 original source FIT robot episodes** across all five previously registered physical tasks, identical original source digest `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`. No new model fit (zero optimizer steps), no source CAL, no previously exposed spray or charger task, no remaining reserved tissue-box task. Human-free shared measurement windows t−8..t+2 gave **257,971 authentic measured autonomous one-step robot transitions**. In those original FIT windows **91.2269%** had mean 14D physical joint change below an explicit hindsight diagnostic threshold 0.025. These are highly autocorrelated frames, NOT independent samples or a success rate.

**Real original FIT physical next-joint one-step MAE** (lower better, same common real windows): last observation **0.07019513195**, FIT-identified original servo with action at t **0.12046094284**, past action at t−1 **0.11914191368**, t−2 **0.12019357838**, t−4 **0.12590195195**, t−8 **0.13624281188**. The best sampled *causally already available* action offset **t−1** was also the minimum among all tested previous/current and future logged offsets in each of five original tasks, but improvement relative to t of only about 0.001319; it does **NOT** certify physical actuator delay or that action timestamps should be changed. Pure in-FIT action-lag optimization is not a heldout capability result, and FIT-identification of servo gain is itself optimistic in-FIT. Temporally FUTURE logged actions (negative k in registered convention) were used only as retrospective timestamp diagnostics and **are forbidden as real-time features**. Every tested servo lag remained worse than simple last-state persistence on the same authentic source windows.

**Cumulative inference from ME064→ME068:** cross-task physics cannot be repaired merely by more recurrent capacity. Native TAC fast/associative state genuinely influences the trained predictor on source/transfer, while the learned velocity/innovation reliability readouts show brittle domain shift (ME067); source original command-based servo is predictively poor even on original FIT (ME068); and rare meaningful motion is underrepresented in uniformly sampled frames. No independent decision utility, exact action-effect identification or robust calibrated risk is shown. The long-horizon ME066 evaluation conditions on original *realized* logged future action prefixes: valid as a factual action-conditioned rollout comparison, but do NOT represent the information available to an online agent unless that exact future proposed command sequence was already authorized and known. Future real decision benchmarks must explicitly pin decision-time knowledge, command/ack time, human authority and outcome availability, avoiding any claimed online prediction from unavailable realized future commands.

### Research next: integrated movement-preserving conditional physical dynamics (proposal, NOT trained)

Avoid rediscovering known switching filters, mixture-of-experts or state-space recurrence just to relabel them. Keep native fast/slow/associative state and verified postfeedback correction as the TAC research identity. Fix the specific real-world failure with **jointly trained, causal motion/no-motion prediction** and an exact persistence path, not another posthoc latent reset:

```text
 z_t  = NativeTAC.observe_only_past_verified_state_and_actions()
 q_{t,h,j} = sigmoid(motion_head(z_t, known_or_proposed_action_prefix, h))
 d_{t,h,j} = residual_head(z_t, known_or_proposed_action_prefix, h)
 xhat_{t+h,j} = x_{t,j} + q_{t,h,j} * d_{t,h,j}
 # if motion probability is low, mean prediction approaches exact x_t
 # never inject unverified postdecision outcomes or future unproposed commands
```

Model a conditional zero/near-zero versus actual-motion distribution using a proper calibrated likelihood or joint movement prediction, with physical sensor precision/noise and motion-target semantics fixed **from source FIT only**, not the exposed ME064/066 evaluations. The action-timing variable must be estimated/verified from actual command timestamps or action receipt metadata where available. Separate verified reliability/innovation feedback **writes** from uncertainty in **predictive readouts**; never give this candidate a missing action/authority signal its matched GRU control does not receive. No fixed domain oracle or hand-scripted target-aware expert switch. Retain true original physical persistence, causal AR/constant velocity and train-identified servo references; a strong matched GRU must share identical motion head/objective/source windows, plus eligible RSSM/SSM controls. Task time signatures, horizon errors, false change predictions, motion-conditioned scoring and sharp risk intervals are **primary planned metrics**, not opportunistic later analysis.

**Hard progression gate:** first independent physical publisher data/timing qualification, then authentic original FIT-only model numeric active-parameter/partial-FLOP/source preflight with <1% gaps, same original real data and optimizer budgets paired seeds, full original weights+optimizer+RNG at pinned retrievable Git paths before any final new target opening, clean independent checkout restoration, source-CAL-only selection, then a reserved never-opened real target (remaining `replace_tissue_box_and_pull_out_first_tissue` or stronger independent robot/site). Do not spend this last clean holdout to compare another untrained speculative router; select a sufficiently substantial causal physical model candidate and a competitive learned control in advance. CPU wall-clock inequality is report-only, not a fairness veto. A positive result must win against **persistence** and the best learned control across real tasks, 5 paired seed stability and original recording-block uncertainty, and useful calibrated interval widths. Factual logged actions do not establish do(a) intervention value or policy success. Woyengi remains fail-closed advisory, no runtime authority promotion.

Known foundations: [Yong et al. 2021 switched mode/input/state estimation](https://doi.org/10.1002/rnc.5306); [TD-MPC2 genuine model-based control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html). These mathematical ideas are prior art; the experimental question is whether a native TAC memory/controller can beat strong simple physical references under appropriately qualified actual data.


## 2026-10-23 ME069/ME070 evidence correction: no native-memory advantage yet

**ME069 (draft PR #348) is NOT a TAC-specific win.** On authentic BPI2012 (2,618 TEST cases, five paired seeds), native TAC and matched GRU share FIT empirical priors and learned mixture/wait heads. Native next-event NLL 0.122626 versus GRU 0.119182 (TAC **2.89% worse**; paired bootstrap interval positive), accuracy both 94.77%, native time MAE 6.3561h versus GRU 6.3618h, mean training CPU 35.251s versus 13.777s (~2.56x worse efficiency). Both beat the 0.348946 first-order prior, but that **does not isolate TAC memory**. Positive value over an underfit simple reference is not architectural superiority. Real stateful workflow inference up to 32 events is implemented, **not** persistent 64/256-event capability. PR #348 retains all ten original Git model/optimizer/RNG checkpoints; local/GitHub independent replay evidence includes two documented fixed-tolerance failures and later passes—no universal deterministic promise.

**Retrospective original TEST audit** now measured in [draft ME070 PR #349](https://github.com/FPC-effortless/TAC-OSM/pull/349), with all seven diagnostic software tests passing in [Actions run 39920681790](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39920681790). Paired correct/wrong disagreements are only 11–36 cases per 2,618 per seed (exact McNemar p 0.227–1.0). Mean 230h waiting-time interval width conceals a pathological mixture: median **4.8–10.6 seconds**, p90 **109–197 hours**, p99 **2,134–4,605 hours (89–192 days)**, maximum **21,087–33,638 hours (~2.4–3.8 years)**. This is evidence of **severely uneven predictive dispersion**, not evidence that all cases have 230h uncertainty, nor that tighter intervals are automatically better. Direct interval/Winkler/pinball scores and next-activity lifecycle label frequencies cannot be computed from currently saved per-case statistics alone; require legal original-source replay and full endpoints. This diagnostic uses **already exposed** ME069 TEST; no model selection, refitting or new holdout scientific promotion from this information.

**Next source-first capability hypothesis:** BPIC2018 ([original DOI](https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972)) has published ~2.51M events/43,809 cases and reported max trace length 2,973, but independent eligible 256-event **case cohort counts and source licence/hash must be verified**. [ME070 protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/PROTOCOL.md) requires >=50 FIT and >=25 TEST distinct real cases with a factual next event following 256 observed events, 60/20/20 chronological original-case splits and no crossing-case leakage. Otherwise **STOP** and qualify another authentic source. No new model training until this gate and a **new fully preregistered architecture-comparison FAIR_COMPARISON** pass. The current ME070 registration explicitly identifies **noncomparative source/retrospective diagnostics only**.

**One falsifiable 64-event kill gate, not another endless ablation loop:** compare native TAC, matched GRU, meaningfully parameter/compute-matched native-associative-memory ablation; include shared-head/transition-only, smoothed n-gram, and a genuine process-monitoring LSTM or SuTraN comparator. At real online-observable prefixes (not a retrospectively selected midpoint), pre-register rejection of a broad native memory advantage unless TAC earns >=1% case-mean NLL improvement vs GRU at >=64 original events, clustered paired-case 95% upper CI <0, >=4/5 paired seed wins, and a positive incremental benefit of native associative memory. A 256-history claim also independently needs adequate true >=256-event evidence. Measure calibrated and sharp waiting intervals by quantile/Winkler proper scores, positive-gap risk, coverage and width distributions against empirical conditional quantiles, plus rare/cross-lifecycle transitions. Preserve full in-repository checkpoints and independent clean-checkout replays; CPU is **efficiency only**, not a fairness parity condition. ME055's older 5% CPU acceptance bound was historical and superseded for prospective science by the 2026-10-22 owner rule; its original recorded failed gate is not erased. No unsupported operator cost, causal benefit, Woyengi authority or commercial readiness claim.


**ME070 verified source-lifecycle supplement (2026-10-23):** original publisher-byte-hash-verified BPI2012 was reloaded using the ME069 source adapter and analyzed in [Actions run 39921891424](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39921891424). Its original true case-length maxima are FIT **161**, DEV **94**, TEST **109** events (FIT mean 17.80, TEST mean 18.27); therefore **zero TEST cases can provide 256 previous events**. On the retrospective ME069 TEST midpoint targets, **38.43%** remain in the same underlying activity and **13.06%** are exactly START→COMPLETE of that activity, while the majority target label is only **28.23%**. These lifecycle effects require difficult-transition and lifecycle-stratified baselines; they are not proof that all 94.77% accuracy is trivial. Original checksummed aggregate results were committed under [ME070 EXPOSED_ME069_LIFECYCLE_AUDIT.json](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/EXPOSED_ME069_LIFECYCLE_AUDIT.json). The original BPIC2018 long-source downloader attempted three connections in [run 39920980632](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39920980632) and timed out; dataset hash/256-event case eligibility remain **UNVERIFIED**, so ME070 training is **STOPPED** until lawfully accessible real data qualify. This is data-access failure, not a measured insufficient-cohort outcome.


### 2026-10-23 executed ME070 non-neural higher-order controls — original real BPI2012

[Draft ME070 PR #349](https://github.com/FPC-effortless/TAC-OSM/pull/349) now contains **actual FIT-only trained and Git-preserved 1/2/3/5-event-count predictors**, not merely a proposed control. From 7,096 original FIT cases (126,285 authentic events), fixed smoothed backoff (alpha=5) was fitted exclusively on FIT; original Git-owned count state was frozen at commit `6b711ff` **before** any historical DEV/TEST scoring, scored at `6f34537`, and the identical scores independently replayed from a clean repository checkout in [Actions run 39924308857](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39924308857). The original state/checkpoint and score bytes remain in `research/replay-bundles/ME070/practical-ngrams/` on the draft ME070 branch. Nine fixture regression tests passed. BPI2012 TEST is **already exposed** by ME069; this is **retrospective real-data diagnostic** rather than new independent qualification.

On the SAME 2,618 original ME069 TEST cases at the original hindsight-midpoint selection, empirical first-order Markov NLL/accuracy **0.348946 / 84.76%**, second-order **0.198247 / 92.25%**, third-order **0.172033 / 93.58%**, fifth-order **0.159743 / 93.96%**. Historical native TAC **0.122626 / 94.77%**, historical matched GRU **0.119182 / 94.77%**. Fifth-order FIT count memory explains **~54.2% of the first-order log-loss gap without a neural model**, so never credit improvement vs first-order baseline to native TAC memory. Both neural models still beat the fifth-order practical baseline but **GRU remains stronger than TAC**, and relative architectural contributions are unresolved. N-grams are explicitly unmatched practical references, not capacity-matched architecture comparisons. The original source also has only FIT=174 / DEV=15 / exposed TEST=61 cases with actual next labels after 64 observed events; after 128: FIT=4, DEV=0, TEST=0; after 256: zero across all. [Source eligibility receipt](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/EXPOSED_BPI2012_HISTORY_ELIGIBILITY.json).

Original BPI2018 source **metadata**, not its bytes, has now been independently resolved from the official Figshare article **12688355** (file **24025820**, size **158270992** bytes, publisher MD5 `66e54ddaa6b50d81a0166766d6f7a5d4`), but the original 4TU URL and ALL tested official Figshare downloader/API file routes returned timeouts/HTTP **504** in GitHub Actions. Source SHA256 and honest 256-event cohort counts remain **UNKNOWN**, thus the fully matched new ME070 neural model train/test remains **BLOCKED**. Never replace missing real source with simulated data or report a published maximum history as cohort evidence. Stop automatic failed downloader workflow dispatches until original access changes.

**Capability status unchanged:** observational real enterprise forecasting exists; native long-memory advantage, 256-event context, real counterfactual action utility, and enterprise deployment **NOT DEMONSTRATED**. Continue the pre-registered 64-event kill test only with independently qualified real data and repository-original fitted checkpoints plus clean-clone replay.
