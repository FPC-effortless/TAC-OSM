# Unified Native Model v1

Status: research architecture and pre-capability implementation. Historical TAC-OSM measurements remain frozen.

## 1. Complete computational cycle

I_t -> D_t -> Z_t -> M_t -> P_t -> R_t -> Pi_t -> C_t -> A_t -> O_(t+1) -> V_t -> W_t -> L_t -> (S_(t+1), K_(t+1), E_(t+1))

I_t is the learner-visible information. D_t discovers candidate structure. Z_t is the predictive/action-sufficient representation. M_t addresses persistent state. P_t proposes a bounded candidate set. R_t chooses the relevant structures. Pi_t plans over candidate computations using the learned transition model. C_t executes an explicit program. A_t is the committed action or prediction. O_(t+1) is the observed consequence. V_t verifies the executed trace and returns evidence/confidence. W_t decides what should persist. L_t updates parameters and computational structure.

The persistent substrate is:

P_t = (S_t, K_t, E_t, V_t)

S_t: durable world/task state.
K_t: reusable operators/programs/abstractions.
E_t: verified experience and provenance.
V_t: verifier constraints, confidence, contradictions, repair and retraction state.

## 2. Identifiability is a first-class gate

Representability alone is insufficient.

For a target Y and learner-visible observation map Obs:

Obs(x1) = Obs(x2) => Y(x1) = Y(x2)

must hold for the registered task family. Otherwise the learner cannot distinguish worlds that require different actions.

REP-001 demonstrated this exact failure mode: an analytic witness solved the relation while the learned router stayed near chance. Therefore every learned mechanism gets both a representability test and an identifiability test before training and capability inference.

## 3. Predictive and action-sufficient state

Z_t is not defined as a generic embedding. It must preserve information that predicts future observations and changes action consequences:

Z_t = f_theta(I_(<=t), D_t)

hat Z_(t+1) = T_theta(Z_t, a_t, K_t)
hat O_(t+1) = G_theta(hat Z_(t+1))

A representation earns scientific relevance when removing or perturbing it changes registered predictive or action endpoints, not merely reconstruction loss.

## 4. Structure discovery

The native structure learner decomposes Z_t into slots or executable factors. The first reference implementation uses a differentiable slot bottleneck. Future variants may learn graph topology, typed relational factors, programs or operator schemas.

A structure is useful only when it survives held-out predictive/action tests and verification.

## 5. Addressing, proposal and routing are separate

M_t = Address(Q_t, S_t)
P_t = Proposal(Z_t, M_t, K_t)
R_t = TopK_B Score(Z_t, P_t, K_t)

The proposal stage has its own coverage endpoint. A reranker cannot recover a target removed by proposal. The correct decomposition is:

P(success) = P(target in P_t) * P(success | target in P_t)

Current TAC-OSM product-key evidence establishes a real sparsity/capability frontier, but fixed-factor product-key retrieval still scores a growing fraction of M. It is not an asymptotic sublinear result.

## 6. Predictive planning

The transition model supports counterfactual rollouts:

hat Z_(t+k+1) = T_theta(hat Z_(t+k), a_(t+k), K_t)

Planning is constrained by a computation budget. External candidate retrieval and internal program routing are different problems and have separate accounting. Any goal-conditioned action-selection experiment must make the goal observation an explicit learner-visible input; using an unobserved future target to construct the proposal or route is leakage and invalidates the measurement.

## 7. Executable structures

Executable topology is explicit. The system must not infer hidden program wiring from an unidentifiable substrate. Exact execution is the reference semantic layer; learned execution strength is an ablation.

## 8. Verification, repair and uncertainty

Verification is post-execution. It returns validity, confidence, failed constraints, counterexamples and repair targets.

Repair is bounded: verify -> localize -> select -> patch -> re-execute -> re-verify.

Confidence controls write eligibility. Contradictions can trigger invalidation/retraction rather than silent overwriting.

## 9. Write, consolidation and experience

The write policy selects among ADD, UPDATE, INVALIDATE, DELETE, CONSOLIDATE and NOOP.

Only verified experience is durable by default. Recurring verified transition laws may be consolidated into K_t as parameterized operators. This is distinct from ordinary memory retrieval.

## 10. Objective

L = L_obs + lambda_pred L_pred + lambda_route L_route + lambda_struct L_struct + lambda_verify L_verify + lambda_reuse L_reuse + lambda_memory L_memory + lambda_align L_align + L_noncollapse

with hard resource constraints:

C_proposal + C_route + C_exec + C_verify + C_write <= B_compute
|S_t| + |K_t| <= B_memory

The objective is future verified utility under bounded computation and memory.

## 11. Multimodal instantiation

The shared core uses modality-specific observation adapters and decoders, but a shared predictive structural latent, action-conditioned transition mechanism, routing/planning interface, verifier and persistent substrate.

Language: token sequence prediction and action-conditioned sequence prediction.
Image: patch/pixel prediction and action-conditioned future-frame/state prediction.
Audio: waveform/spectral-frame prediction and action-conditioned future-frame prediction.

The confirmatory benchmark deliberately uses procedurally generated paired views with held-out semantic combinations. The hidden semantic factors never enter model inputs. Cross-modal alignment uses paired observations, not supplied semantic labels.

This is motivated by independent research showing latent-prediction architectures can support multimodal alignment and temporal prediction. M3-JEPA applies JEPA in multimodal latent space; LLM-JEPA applies embedding-space prediction to language; Audio-JEPA applies masked latent prediction to audio; V-JEPA 2 demonstrates latent video prediction combined with planning; and ER-JEPA adds episodic replay to language JEPA. These are prior-art motivations, not inherited evidence for TAC-OSM.

## 12. Claims boundary

The benchmark can establish only that the reference implementation learns the registered synthetic tasks. It cannot establish human-level language, image or audio understanding, broad multimodal reasoning, natural-world world modeling, general intelligence, or asymptotic compute scaling.
