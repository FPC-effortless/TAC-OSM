# Unified Native Model — Mathematical Specification v1

## State and information

Let the environment have latent world state \(X_t\), but let the learner observe only

\[
I_t=\operatorname{Obs}(X_t,Q_t,S_t)
\]

where \(Q_t\) is an explicit task/goal observation and \(S_t\) is the model's durable
world/task state. Hidden environment truth, gold actions and future observations
are not components of \(I_t\).

The complete persistent substrate is

\[
\mathcal P_t=(S_t,\mathcal K_t,\mathcal E_t,\mathcal V_t)
\]

with world state \(S\), executable knowledge/library \(\mathcal K\), verified
experience/evidence \(\mathcal E\), and verification/uncertainty/retraction
state \(\mathcal V\).

## Optional active information acquisition

When the current information is insufficient,

\[
U_t = u_\phi(I_t,\mathcal P_t)
\]

estimates uncertainty or unresolved observational equivalence. If the environment
permits probes,

\[
B_t=\operatorname{ProbePolicy}(U_t,\mathcal P_t,B_{\rm probe})
\]

selects a bounded information-gathering action. The new observation is

\[
I_t^+=\operatorname{Obs}(X_t,B_t).
\]

The probe receives no hidden truth. Its value is measured by the reduction in
uncertainty and improvement in downstream held-out prediction/action.

## Structure discovery and representation

\[
D_t=\operatorname{Discover}_\theta(I_t^+,\mathcal P_t)
\]

produces latent factors, slots, graph relations, executable candidates or other
structures.

\[
Z_t=f_\theta(I_{\le t}^+,D_t,\mathcal P_t)
\]

is the predictive/action-sufficient state.

Representation gates require:

1. representability;
2. identifiability;
3. non-collapse;
4. predictive sufficiency;
5. action sufficiency.

For target relation \(Y\),

\[
\operatorname{Obs}(x_1)=\operatorname{Obs}(x_2)
\Rightarrow
Y(x_1)=Y(x_2)
\]

must hold on the registered task family.

## Addressing, proposal and route

\[
M_t=\operatorname{Address}(Z_t,S_t)
\]

returns the state region relevant to the query.

\[
P_t=\operatorname{Proposal}(Z_t,M_t,\mathcal K_t;B_P)
\]

is a cheap bounded candidate set.

\[
R_t=\operatorname{TopK}_{B_R}
\operatorname{Score}(Z_t,M_t,P_t,\mathcal K_t).
\]

Proposal coverage and conditional route quality must remain separate:

\[
P(\operatorname{success})
=
P(y^\ast\in P_t)
P(\operatorname{success}\mid y^\ast\in P_t).
\]

No proposal miss may invoke hidden-truth exhaustive fallback.

## Predictive world model and planning

The transition model is action-conditioned and structure-conditioned:

\[
\hat Z_{t+1}
=
T_\theta(Z_t,A_t,\mathcal K_t).
\]

Multiple future projections may be predicted from the same latent:

\[
\hat Y_{t+1}^{(m)}
=
G_m(\hat Z_{t+1}), \qquad m\in\{\text{text},\text{image},\text{audio},\text{structure}\}.
\]

A bounded planner searches the learned model:

\[
\Pi_t=
\operatorname{Search}
(T_\theta,\mathcal K_t,Z_t,R_t,B_{\rm plan}).
\]

The same transition model is used for all planning arms; only search depth/budget
changes in registered interventions.

## Executable computation

A plan contains explicit executable structures.

\[
C_t=\operatorname{Execute}(\Pi_t,\mathcal K_t)
\]

produces a trace

\[
\tau_t=(z_1,\ldots,z_n,y_t)
\]

and work accounting \(W_t^{\rm exec}\).

Exact semantics are the reference mode. Learned soft computation strength or
learned internal topology are separate ablations and cannot define the reference
semantics.

## Outcome and verification

The environment evolves only after an action is committed:

\[
X_{t+1}=F_{\rm env}(X_t,A_t),
\qquad
O_{t+1}=\operatorname{Obs}(X_{t+1},Q_{t+1},S_t).
\]

Verification is post-execution:

\[
V_t=
\operatorname{Verify}(Z_t,C_t,O_{t+1},\mathcal V_t)
\]

returning validity, confidence, failed constraints, counterexample and evidence.

Repair is bounded:

\[
C_t^{(0)}
\rightarrow
V_t^{(0)}
\rightarrow
\operatorname{Localize}
\rightarrow
C_t^{(1)}
\rightarrow
V_t^{(1)}
\rightarrow\cdots
\]

with at most \(B_{\rm repair}\) attempts.

## Persistence and consolidation

The write policy decides

\[
W_t^{\rm mem}
=
\operatorname{WritePolicy}
(Z_t,C_t,O_{t+1},V_t,\mathcal P_t)
\]

with operations

\[
\{\text{ADD, UPDATE, INVALIDATE, DELETE, CONSOLIDATE, NOOP}\}.
\]

Verified experience is committed only after the final accepted verification
state. Contradictions can increment evidence counters and trigger retraction.

Executable knowledge evolves as

\[
\mathcal K_{t+1}
=
\operatorname{Consolidate}
(\mathcal K_t,\mathcal E_{t+1})
\]

through recurring verified transition laws, operators and compositions.

## Unified learning objective

The reference multi-objective is

\[
\mathcal L =
\lambda_{\rm obs}\mathcal L_{\rm obs}
+
\lambda_{\rm pred}\mathcal L_{\rm pred}
+
\lambda_{\rm trans}\mathcal L_{\rm trans}
+
\lambda_{\rm route}\mathcal L_{\rm route}
+
\lambda_{\rm struct}\mathcal L_{\rm struct}
+
\lambda_{\rm action}\mathcal L_{\rm action}
+
\lambda_{\rm verify}\mathcal L_{\rm verify}
+
\lambda_{\rm reuse}\mathcal L_{\rm reuse}
+
\lambda_{\rm xmodal}\mathcal L_{\rm xmodal}
+
\lambda_{\rm noncollapse}\mathcal L_{\rm noncollapse}.
\]

Resource constraints are explicit:

\[
C_{\rm probe}+C_{\rm proposal}+C_{\rm route}
+C_{\rm plan}+C_{\rm exec}+C_{\rm verify}+C_{\rm write}
\le B_C
\]

and

\[
|\mathcal P_t|\le B_M.
\]

## Causal ablation definition

A mechanism is load-bearing only if disabling it changes a registered future
endpoint while the other causal interfaces remain fixed.

The preferred endpoint hierarchy is:

1. held-out future prediction;
2. action-conditioned prediction/selection;
3. target proposal recall;
4. conditional route quality;
5. verified execution success;
6. future performance after raw-observation removal;
7. retained executable knowledge;
8. computation and storage cost.

## Multimodal requirement

For modality \(m\),

\[
Z_t^{(m)}=E_m(X_t^{(m)})
\]

must enter a shared latent interface. The transition and operator substrate are
shared in the primary arm.

A modality is considered modeled only after it beats its registered training-only
baseline on held-out data. Cross-modal grounding is a separate criterion and
cannot be inferred from within-modality prediction.

## Key falsifiers

The model fails as a unified mechanism if:

- the latent is non-identifiable from allowed observations;
- predictive state collapses task-relevant distinctions;
- proposal coverage falls to chance;
- routing requires hidden truth;
- execution correctness depends on inferred oracle topology;
- verification can be bypassed by writing before the outcome;
- persistent memory improves training but harms future held-out performance;
- operator consolidation retains unverified or contradicted structures;
- compute reduction is obtained by moving the same work into an unreported stage.
