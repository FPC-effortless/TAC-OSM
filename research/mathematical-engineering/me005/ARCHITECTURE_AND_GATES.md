# ME-005 — unified native TAC-OSM mathematical architecture (2026-10-20)

**Status:** integrated reference mathematics + exploratory software gates; all individual research ideas preserved; **NOT** a reclassification of TAC-OSM as an SSM. The research thesis is **predictive, persistent, executable world-state learning**. This branch is stacked on the authentic real-data ME004 trained-predictor branch, NOT on an invented new backbone.

## Evidence-driven bottleneck register

| Existing evidence | Limitation we must address | Native mathematical mechanism | Falsifier |
|---|---|---|---|
| DEV017/020, SEQ005–008 | learned long-history entity isolation vs update interference; a routed GRU ties TAC | exchangeable entity-like assemblies, temporal correspondence, sparse learned graph, explicit identity oracle control | strong matched routed GRU wins |
| ME001B / SEQ010 | native learned forecaster previously lost persistence | inherit actual pretrained two-timescale ME004 native recurrence; output-preserving residual addition | combined system increases real validation/test error |
| ME001C–ME002 | bounded state ≠ nonvanishing credit | identity-preserving fast/slow state, separate plastic/episodic storage | long-horizon real retention fails |
| ME004F | native two-timescale forecast beats tested GRU in magnitude but worse change direction, ~2.18× train CPU | calibrated multi-horizon forecast, explicit change/risk head, capacity/latency-matched GRU and boosted baseline | direction/cost gates fail |
| CDL Stage B/R13 | costly conditional-NLL teacher, hard top-k collapse, verifier can hurt | local Gaussian conditional relevance + learned router, separate signed-outcome-modulated eligibility, optional soft or sparsemax allocation | loses conditional relevance or adds net cost |
| CASM L2 heldout 0/18; DEV064B | compositions/control attribution not established | learned operator library with role–filler binding, identity/no-op and held-out family×argument tests | controlled compositional transfer still fails |
| PNDS proposal | E→P consolidation and temporal boundary unreadied | provisional memory and independent receipt-gated persistence | missing receipts promote |
| BDH/NADOC proposals | coactivation specialization, plasticity and oscillatory phase unproved | bounded associative weights, sparse neural communities, independent write/read/think/plastic gates | matched ordinary learned gates perform as well |
| Continuous symbols/VSA/CAN | manifold attractors/logic remain conceptual | bipolar exact role unbinding, attractor energy as differentiable consistency penalty, learned iterative refinement | no compositional or causal refinement advantage |
| Veritas safety and critic | learned energy not a guarantee | learned internal risk signal **plus independent external authorization and outcome verification** | any unauthorized action permitted |
| SYS1–SYS2, real action logs | actual planning-specific advantage not proven | optional action/goal-conditioned head **OFF without verified real logged actions/outcomes**; trained native action/STOP policy from DEV067 remains separate | no identified action-effect/cost gain |

## Unified typed state and recurrence

The complete world state is \(\mathcal{S}_t=(H_t^{fast},H_t^{slow},Z_t^{assemblies},A_t^{fast},E_t^{elig},\phi_t,\mathcal M^E_t,\mathcal M^P_t,\Xi_t)\).

1. **Existing learned native dynamics (fixed pretrained initial checkpoint):** \(f_t=(1-g_t)f_{t-1}+g_t \tanh(Ux_t+Vf_{t-1})\), \(s_t=(1-\bar g_t)s_{t-1}+\bar g_t \tanh(W[s_{t-1},f_t])\). Both are convex coordinate-bounded updates when initialized in [-1,1], **not** an SSM conversion, gradient guarantee, nor proof of correct real-world state.
2. **Entity-like learned assemblies:** content-address observation tokens \(k_j\), exchangeable learnable anchors \(c_i\), soft token-to-assembly including a null route \(a_{ij}=\mathrm{softmax}_{i,\varnothing}(\langle q_i,k_j\rangle/\sqrt d)\). Sparse cross-assembly graph \(G_{ij}=\mathrm{sparsemax}_j(\langle q_i^G,k_j^G\rangle/\sqrt d)\). Learned convex updates to slots. These are latent groupings, NOT semantically established identities.
3. **CDL-PHR predictive relevance:** \(R_{local}=-(1/2)\|u-\hat u(\mathcal M)\|^2\); compare to null memory \(R_{null}=-(1/2)\|u\|^2\). Separate prediction-gradient eligibility and later genuine signed, receipt-verified *decision* advantage; do not reverse an accurate predictor merely because an action had low utility.
4. **Plastic local association:** \(A_{t+1}=\Pi_{\|A\|_F\le c}((1-\lambda)A_t+\eta g^{plastic}_t (u_t-\hat u_t)p_t^\top /d)\). Enforce norm by projection and preserve autograd over current unroll; no update of frozen native transition matrices at inference.
5. **Executable memory/compositional algebra:** bipolar VSA roles \(r\in\{-1,1\}^d\) yield \(r\odot(r\odot x)=x\) exactly; compose shared learned operator families \(f_i\) under soft routing, with an identity/no-op family. Factor global operator parameters from instance bindings. Causal execution correctness and generalization remain unproven.
6. **Neural-assembly oscillations:** bounded learned \(\phi_{t+1}=(\phi_t+\omega_\theta)\bmod 2\pi\) and independent gates \(g^{write},g^{retrieve},g^{reason},g^{plastic}=\sigma(f(h_t)+a\sin(\phi_t+\psi))\). No claim about actual theta/gamma LFP rates.
7. **Continuous-symbol attractors and computation:** choose learned target \(c(h)=\mathrm{softmax}(q(h)\cdot C/\sqrt d)C\); iterate \(h^{k+1}=(1-\alpha)h^k+\alpha c\), \(\alpha\in(0,1)\), with learned halting weights. For **fixed** c, \(E(h,c)=\|h-c\|^2/2\) decreases; this does NOT prove symbolic reasoning or correct world facts.
8. **Action and safety boundaries:** optional action-conditioned predictive heads only when recorded action data is supplied; critics score estimated risk, while Veritas/independent validators retain real tool permissions, physical constraints and post-action receipts. Internal energy topologies cannot guarantee safety.
9. **Ephemeral/persistent evidence lifecycle:** record observation immediately in current belief; store candidate memories in E, and promote to P only with independently verified receipts across different contexts and untouched scientific tests.

## Training and research order — all ideas included, not all claims conflated

- **ME005-A integration/software:** Verify exact preservation of the SHA256-verified *trained* ME004 baseline at zero residual initialization; masked-value isolation, streaming equivalence, convex hidden bounds, norm-bounded fast association, finite gradients, role binding/unbinding, no receipt→no promotion, action head disabled when actions absent, deterministic module ablations. This may use random numerical unit diagnostics; **not** an empirical success claim.
- **ME005-B authentic predictive training:** Same official MetroPT real TRAIN history from ME004. Train integrated adapter only, keeping the verified two-timescale backbone frozen. Compare base, base+unified, base+graph-only, base+plastic/CDL-only, base+operators/refinement, and a parameter/compute-matched GRU; train-/calibration-only selection, keep newer future period reserved. Report wall time, memory, multi-horizon error, large-change direction and calibration.
- **ME005-C authenticity/domain transfer:** Frozen checkpoints on newly preregistered later week or independent installation; do not reuse already-opened March 2–9 as an untouched test. Entity identity, composition, and action utility each require corresponding real data; structural synthetic probes alone never qualify world-model performance.
- **ME005-D decision integration:** after an independent benefit, expose world-state outputs to already trained neural CASM action/STOP controller, preserve historical DEV/SEQ versions and compare no-CASM/simple controls. Require credible logged action propensities/coverage, outcomes, independently verified costs before any counterfactual/commercial benefit claim.

## Mathematical obligations, efficiency, and safety

Forward invariance is separate from nonvanishing gradient and correct overwrite; input masks must eliminate unobserved values. With N assemblies, F observed feature tokens, d feature width and K learned operators, dense soft binding is O(NFd) per time step; sparse graph attention is O(N²d); fast full-rank W is O(d²), per-step; K operator evaluations add O(Kd²). **Neither constant-cost scanning nor Mamba-equivalent O(T) kernels are claimed.** Evaluate low-rank plasticity, sparse graph caching, associative grouped scans, phase simplification, and CPU/GPU budgets after demonstrating real utility. Report parameter/compute budgets, original seed lists, prospective source/exposure registry and no score-driven retuning.

**Selection policy:** do not proclaim architecture novelty, emergent semantics, learned reasoning, independent real safety, full multimodality, or generalized counterfactual competence from unit tests or a single sensor stream. Preserve EVERY source proposal, historical failed study, and untested hypothesis. Favor model-first measured progress over rediscovering established mechanisms. No automatic PR merge, no replacing original weights, and no hidden data exposure.

**Starting checkpoint:** ME004 original frozen trained real model https://github.com/FPC-effortless/TAC-OSM/pull/248; follow-up true-time validation https://github.com/FPC-effortless/TAC-OSM/pull/249. The two-timescale native TAC forecasting result is *limited evidence*, not a qualified full agent.
