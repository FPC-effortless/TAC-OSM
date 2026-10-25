# NEXT-REPRESENT-ADAPT — Emergent World Representations, Experience Memory, and Predictive Control

**Registered:** 2026-10-25. **Status:** RESEARCH PROPOSAL / NOT IMPLEMENTED / NOT TRAINED / NO TAC-OSM CAPABILITY CLAIM.
**Placement:** A dependent work package of [NEXT-SHARED-SELECTIVE — Geometry-Constrained, Entity-Selective Parallel World-State Intelligence](NEXT-SHARED-SELECTIVE-GEOMETRY-CED.md), NOT a second shared core, a seventh model, a replacement for the six families, or authorization for live device/customer actions. The [master plan](../MASTER_RESEARCH_PLAN.md), [ideas registry](../IDEAS_REGISTRY.md), [authentic evidence policy](../REAL-WORLD-EVIDENCE-RULES.md), [capability truth standard](../CAPABILITY_PROGRESS_TRUTH_STANDARD.md), and PNDS source-to-math obligations retain precedence.

## 1. The scientific question and evidence tiers

**Primary question:** Can a small, finite-state, parallel-trainable TAC-OSM learn internal representations with **causal predictive and decision value** from genuine longitudinal source data, and then use **source-verified experience** to improve forecasting and action choice, beyond matched short-context, structured-memory and native-core references?

Distinguish four properties that must NOT be conflated:
1. **Recoverable state:** a probe can decode a fact from latent activations. This alone does not prove the network uses that fact; a powerful probe may learn reconstruction itself.
2. **Causally useful representation:** controlled latent intervention changes the model's predictions in the preregistered, directionally expected way with bounded collateral damage. This is evidence of model-internal causality, **not** automatically identification of real-world action effects.
3. **Experience-based adaptation:** a frozen model improves later forecasts or choices by updating accessible, provenance-checked experience; neural weights remain fixed. This is **non-parametric/contextual improvement**, not gradient training of the underlying network.
4. **Improved real-world decisions:** an authorized decision under actual action availability yields verified lower physical/operational cost, compared with a strong real baseline and a valid observational or prospective causal identification design. Predicting counterfactuals from observational logs does not by itself establish policy benefit.

Success on a language-agent simulated environment or Othello demonstration is **literature evidence only**; it has no authority to establish a TAC-OSM real-world performance claim.

## 2. Literature and exact transfer limits

### 2.1 Lu et al. — Self-Improving Agents with World Model Predictive Paths

Primary: https://arxiv.org/abs/2610.28945 (preprint submitted 2026-10-20).
The paper proposes candidate action generation, one-step textual next-observation prediction, model-evaluated action costs, MPPI-inspired exponential cost weighting, final selection, and observation-based experience updates. Its **planner memory** supports proposing/selecting actions; **world-model memory** stores environment-transition lessons. Base-model weights remain frozen and improvement is via available textual experience, **not parameter self-training**.

Reported benchmark context: ALFWorld and ScienceWorld are simulated textual environments; three language-model backbones are studied. In the reported Gemma-4-26B-A4B ALFWorld SR@1 memory ablation, no memory=56.7, planner-only=49.3, world-only=68.7, both=65.7 percent; ScienceWorld corresponding values are 38.9, 44.4, 36.7, 46.7 percent. Thus the memory branches can help *or interfere*, with no monotonic guarantee. The reported Gemma step latency of roughly 6.72 versus 0.51 seconds for ReAct is an illustrative cost in that setup, **not TAC timing**. The decision-weighting effect remains insufficiently isolated from the candidate predictions/costs/final-controller behavior; pre-register its own matched ablation here before claiming any benefit.

Transfer to TAC-OSM: treat planner versus dynamics experience as **separate information functions and controlled read/write interfaces**, not necessarily two full independent recurrent tensors or copied LLM notebook prompts. World memory may include provenance-linked transition episodes and qualified semantic regularities; planner memory may store verified choice failures, feasibility, constraint and risk lessons. Never promote a model-inferred textual rule to an observed fact. Log predictions before receiving an outcome.

### 2.2 Li et al. — Emergent World Representations

Primary: https://arxiv.org/abs/2210.13382 (ICLR 2023).
Othello-GPT was trained to predict legal next moves and was not directly given a board-state representation. Nonlinear probes decoded board information; activation interventions changed model outputs consistently with altered board states. This is evidence that **predictive sequence objectives can induce functional latent state representations in a controlled synthetic domain**. It does not show that any native TAC-OSM layer already represents physical entities, nor that generic state prediction alone guarantees useful emergent structure.

Follow-up about player-relative representations: https://arxiv.org/abs/2309.00941 (linear board-state views can depend on representation/coordinate choice; verify details when importing experiments). Control for probe capacity, state reconstruction from simple shortcuts, and player/source-relative coordinate choices.

### 2.3 Anthropic — Tracing the Thoughts of a Large Language Model

Primary: https://www.anthropic.com/research/tracing-thoughts-language-model (2025-03-27). Interpretability interventions in poetry suggest mechanisms sometimes represent a prospective rhyme target before emitting intervening tokens. **Training output horizon need not equal internal planning horizon**, but this is limited, model- and task-specific causal evidence, not proof of generic deliberative planning. For TAC-OSM distinguish learned forward-looking representation from an explicit multi-branch decision algorithm.

### 2.4 Schaeffer, Miranda, Koyejo — Are Emergent Abilities of Large Language Models a Mirage?

Primary: https://arxiv.org/abs/2304.15004 (NeurIPS 2023). Exact-match or otherwise discontinuous success metrics can create sharp apparent phase transitions from smooth underlying improvements. This does **not** rule out functionally useful latent mechanisms. Always report continuous metrics and per-case curves alongside pass/fail, across history lengths, entity counts, horizons, physical regimes and seeds.

### 2.5 Distinct prior art and original contribution boundary

The MPPI-style Boltzmann weighting / KL-regularized minimization and activation probes are established techniques. Original TAC-OSM research hypotheses here are the **specific fixed-memory, causal source-authorized, scan-compatible implementation**, prediction-damage-based admission and its demonstrated interaction with latent state learning, model-internal interventions, calibration, decision-cost, and real-life sequential experience updates. No novelty, formal source equivalence, improved intelligence, or emerging human-like understanding is assumed.

## 3. Proposed integrated computation graph

At time t, inputs are only observed-at-time evidence o_t (and observation mask), authorized source/identity/clock data e_t, previous executed action a_(t-1), and goal/constraints g_t. Never use outcome o_(t+1), withheld truth, retrospective timestamps or a final-sequence representation in an earlier query.

1. **Input and representation:** x_t = Encode_theta(o_t, e_t, a_(t-1), g_t); a learned feature/attention/readout can represent dynamic state, entities, relations, and latent concepts, while preserving source-authorized identity.
2. **Shared causal memory update:** input-only scheduled P_t, Q_t yield M_(t+1) = M_t P_t + Q_t (or the exact existing source-convention counterpart), composed by the original associative prefix-scan algebra. Mandatory deterministic serial-versus-scan equivalence and prefix-causal tests; no pre-read update/eviction that depends on the previous dynamic memory in this schedule.
3. **Postscan read and optional CED:** z_t = Read_theta(M_t, x_(<=t), authorized external memory); optional bounded geometry and original **Concept Exchange Decoder** refinement use causal z_t **after** the scan. Four CED iterations, if selected, remain sequential in depth. Do not confuse TAC's Concept Exchange Decoder (PR331, previously untrained) with DeepSeek's Causal Encoder–Decoder. Report extra state bytes for 6×16 concept slots, maps, external episodes and ring buffers separately from the 608-byte native FP32 core.
4. **Predict actual futures:** p_theta(o_(t+h) | z_t, a_(t:t+h-1), h), with correct conditional availability; use coherent output distributions, calibrated uncertainty and genuine time-ordered outcomes. For unobserved hypothetical actions, label forecasts as **model forecasts**, not identified causal do-operator effects.
5. **Option generation and evaluation (later gate):** propose K available eligible actions, forecast one or several conditional trajectories, score specified expected cost, delay, constraint violations, verified risks and uncertainty. Compare MPPI/KL weighting against argmin cost, same-candidate controller without weights, temperature variants, uniform/random, short-context and no-world-model arms. Missing good candidates cannot be recovered by reweighting.
6. **Execute only if authorized; verify:** under real permission and safety boundaries, log selected action and predicted outcomes BEFORE execution; when actual outcomes arrive, measure residuals and update authorized experience. Distinguish observed, inferred, disputed, censored, and corrected evidence; record clock/version and avoid overwriting observed ground truth.
7. **Two logical experience functions:** dynamics memory retains observed state transitions, prediction residuals, valid time/rules with source; policy memory retains observed action feasibility, decision costs and lessons. Separate admission, deduplication, retention and verification rules; maintain bounded memory, expiry/revision, causal retrieval, and anti-interference safeguards. Both are optional ablation arms; neither is assumed always helpful.

**Critical boundary:** synchronous source-dependent memory edits, final-action selection, real execution, outcome feedback and CED refinement are not automatically parallel sequence-scan operations. Preserve the fast offline parallel scan and exact recurrent online step for the input-only core and separately measure remaining sequential stages. If read-dependent writes are proposed, derive a different legal operator or report a serial stage.

## 4. Mathematical hypotheses and proof obligations

### 4.1 World prediction and latent state

Let z_t = f_theta(x_(<=t)) be strictly causal, with z_(t+1)^pred = F_theta(z_t, a_t, delta_t) and o_(t+1)^pred = G_theta(z_(t+1)^pred). Train only on actual recorded and temporally eligible observation/transition targets. A possible *candidate* objective is:

$$
L = \lambda_p L_{\mathrm{proper\ forecast}} + \lambda_s L_{\mathrm{state}} + \lambda_a L_{\mathrm{eligible\ action}} + \lambda_r L_{\mathrm{retention}} + \lambda_v L_{\mathrm{verified\ correction}}.
$$

No term is included automatically. Every term requires data eligibility, implementable tensor shapes, numerical gradients and preregistered ablations. A latent representation is useful only if it improves authentic held-out forecasting/decisions or survives an appropriate causal intervention test.

### 4.2 Controlled feature contribution and retention

For feature/slot i, define intervention damage relative to a valid matching control:

$$
I_i = E_{(case,t)\sim D_{\mathrm{heldout}}}[L_{\mathrm{task}}(z_t^{\setminus i})-L_{\mathrm{task}}(z_t)].
$$

Intervene at the same prediction cut, with feasible activation controls matched for norm/distribution and slot dimension. Also evaluate swap, random-direction, decoy feature, off-target, and null interventions to distinguish genuine task-relevant representation from intervention artifacts. This quantity is an **association to model-internal causal use under the intervention**, not ground-truth real-world counterfactual importance. Estimating I_i naively for every step is expensive; any learned proxy for admission must be trained strictly on development and measured for compute, bias, forgetting and over-servicing.

### 4.3 MPPI-inspired decision weights

For K eligible actions with costs c_i estimated under the same evidence:

$$
w_i=\frac{\exp[-(c_i-\min_j c_j)/\tau]}{\sum_j \exp[-(c_j-\min_k c_k)/\tau]},
\qquad
w^*=\arg\min_{w\in\Delta_K}\left[\sum_i w_i c_i +\tau D_{\mathrm{KL}}(w\Vert u)\right],
$$

where u is uniform over **eligible** actions, tau>0, and exact equivalence assumes a fixed cost vector and finite feasible set. The weights are **relative decision preferences, not the probability of action success**. Subtract minimum cost for numerical stability; test deterministic tie/invalid-mask handling, FP32/BF16 finite precision, extreme temperatures and logsumexp behavior. If using a learned evaluator, its errors must be independently calibrated/scored. Evaluate whether the weights change final decisions *and* real outcomes; one-step feedback may be insufficient for delayed benefits.

### 4.4 Theory versus checked implementation

PNDS must eventually specify tensor shapes, mask semantics, state-update associativity, time/authority identity conditions, stability constraints, observation/action conditioning, intervention operators, regularization domain, differentiability, bounded write/retention and history-composition proofs where applicable. Run independent Python kernels, finite differences/autodiff gradient comparisons, bitwise/provenance tests and finite-precision discrepancy bounds before equating equations with PyTorch. A successful Lean proof of an idealized affine operator **does not certify a complete neural training graph or prove emergent representations**.

## 5. Prespecified real-data research sequence

| Gate | Hypothesis | Authentic primary target & controls | Outcome / stop |
| --- | --- | --- | --- |
| R0 — source/retention headroom | Earlier source-authorized entity information matters beyond short history | Time-cut-authenticated real longitudinal entities, genuine future outcomes; short-window 8 vs long 256, structured state, simple physical/practical predictors, native frozen checkpoint | If no old-information signal, **NO HEADROOM**; do not fit a complex retrieval/experience layer just to raise write frequency |
| R1 — latent state | Native state contains decodable held-out facts beyond shortcuts | Source-split real entity/physical variable labels, probe controls, linear/nonlinear probe matched capacity, shuffled/recency-only/reader-alone references | Report probe strength, calibration and leakage; do not claim model uses information from probe alone |
| R2 — intervention | State representations causally influence model predictions | Same real cases/prediction cut; state swap, targeted ablation, off-target perturbation, norm-matched randomized controls; actual future scores | Directionally correct selective effects with bounded collateral cost; no claim about real-world do(a) |
| R3 — memory functions | Dynamics and policy experience improve *separately* or interfere | Frozen shared native model, none/dynamics-only/policy-only/both with equal **accessible evidence, token/byte budget and retrieval compute** wherever possible; cross-episode splits that prevent carryover leakage | Report negative transfer/interference, stale lesson damage, latency, source provenance; do not assume two memories always beat one |
| R4 — weighting/branching | Prediction-guided actions beat cost-only selection | Identical proposals, predicted consequences, evaluator cost and action availability; explicit argmin, no-weight controller, MPPI weights, uniform, random and no-forecast alternatives | Prefer safe logged counterfactual estimates only where identification/overlap permits; otherwise **no real causal-policy claim** and seek authorized prospective evaluation |
| R5 — experience adaptation | Frozen model becomes more accurate/calibrated on later genuinely independent real observations | Prequential online timeline: forecast -> freeze prediction -> observe -> append verified lessons -> next forecast; memory-free, equal-size unstructured store, structured memory, stale/shuffled controls | Demonstrate improving **future** real scores without future leakage, catastrophic forgetting, growing state, hidden parameter updates or unexplained inference overhead |
| R6 — integrated six-family rollout | Shared learned representations improve actual lane-specific intelligence | Only after R0/R1/R2 and applicable R3–R5 pass; independently qualified System1, Physical, Enterprise, Temporal, Edge-Lite, Counterfactual outputs and economics | Never combine incompatible lane metrics or claim all six have been trained by one successful lane |

R0 and the existing NEXT-SHARED-SELECTIVE entity-admission study remain the priority. R1–R3 can be diagnostic adjuncts **without replacing authentic capability experiments or forcing endless ablations**. R4 is conditional on adequate authorized action/outcome evidence. For observational action logs without propensities, overlap or valid identification, record prediction accuracy only, not action-policy improvement.

## 6. Mandatory reporting, falsifiers and resource policy

- **Fair architecture comparison:** active trained/deployed parameters in the standing **95k–105k** prospective window and **≤1% difference** between competing neural arms, equal real source, time-cut, eligible examples, exposure, seed policy, training FLOP budget and tuning opportunities; CPU wall time need NOT be matched. When a frozen-LLM agent baseline is used descriptively, do NOT claim a fair neural architecture head-to-head at mismatched parameter/training budgets.
- **Real primary evaluation:** no scripted/synthetic success may be used to declare intelligence or real-world progress. Diagnostic synthetic tests are permissible for algebra/causality unit checks only. Never open protected test cohorts to tune.
- **Metrics:** proper NLL/CRPS/Brier or relevant real-task score, MAE/regime-stratified error, coverage/calibration, continuous probe quality, intervention effect and collateral shift, delayed retrieval by age/entity count, verified real action cost when identifiable, p50/p95/p99 latency, resident bytes and writes. Include exact-match/task success only *alongside* continuous metrics.
- **No emergent-ability marketing:** no declaration that a larger model suddenly gained intelligence from a discontinuous pass threshold alone; state what internal mechanism was measured, whether intervention succeeded, whether real task performance improved, and uncertainty per seed/case.
- **No positive self-improvement claim** if later improvement arises only from bigger context, more experience data, evaluation episodes contaminating one another, better prompting, evaluator changes, unverified textual rules, or repeated retries that cost more than the reported benefit.
- **Falsifiers:** if interventions don't alter downstream predictions selectively, retain only "decodable"; if planner experience harms a domain, disable it there; if one-step forecasts misrank real actions, don't expand MPPI weighting; if memory retrieves stale or unauthorized facts, invalidate that run; if long context provides no information advantage, stop that path.
- **Status/provenance:** all previous ME/DEV results including ME099 failed full gate, ESWAM012 long-memory negative and PR331 untrained status remain immutable. Cite original experiment commit/PR before claiming a follow-on lineage; no result in this proposal is a newly trained or independently verified TAC result.

## 7. Minimal next engineering deliverable (not evidence of intelligence)

Produce a source-pinned **R0/R1 pre-registration** with the actual current shared core's tensor/interface map and authentic case identity/time checks, plus independent mathematical operator tests. Then train and measure a full candidate only after the data/compute/parameter gates pass. Any implementation must retain **exact six-family registry identity** and a single shared research core. This document is a registered hypothesis and experiment design; its completion does not require claiming that self-learning or causal world-state emergence is already achieved.

## 8. Source provenance

Original inputs for this review: user-supplied transcripts **"Predictive World Models for Self Learning AI Multi-memory.txt"** and **"Is It Just a Fancy Dictionary, or Is Something Emerging.txt"**. Primary sources linked above supersede narrator conjecture where the transcripts describe mathematical novelty, emergence or empirical scope. No uploaded transcript is itself an implementation or original TAC experimental receipt.
