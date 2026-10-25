# NEXT-SELECTIVE-BELIEF-SSD — Transformer/Mamba-2-Informed Routing, Probabilistic Beliefs, and Channel-Specialized State

**Date:** 2026-10-25. **Status:** proposed technical research / unimplemented in this document / no newly trained checkpoint or measured improvement. **Placement:** subordinate to [Unified Master Research Plan](../MASTER_RESEARCH_PLAN.md), [NEXT-SHARED-SELECTIVE](NEXT-SHARED-SELECTIVE-GEOMETRY-CED.md) and [NEXT-REPRESENT-ADAPT](NEXT-REPRESENT-ADAPT-WORLD-STATE-EXPERIENCE.md). **Not** a seventh TAC model, replacement for six-family core, or justification to erase the original 8×8 associative memories or failed results.

## 0. Owner-supplied design brief: preserve ideas; verify the mechanisms

The owner proposes addressing (i) "ghost memory": weak/ignored readout and alleged high write rejection; (ii) inaccurate/undercovered gap uncertainty; and (iii) cross-channel pollution. The proposal combines content-addressed attention, Mamba-2 structured state-space duality (SSD), explicit mean/covariance belief propagation, multi-head channel specialization and an exact parallel training / streaming implementation, ideally within the existing 608-byte FP32 core.

**Retain this as a falsifiable design brief, not confirmed causal diagnosis.** "77% rejected writes" was reported as 205/266 in an earlier hypothesis and is still **UNVERIFIED from the pinned relevant source and target regime**. The 8×8 associative memory was not proved unused merely because a model underperforms; in ME100 source admission and time-trace changes also affect memory readout, making attribution open. A decoupled *input-only* affine schedule remains end-to-end differentiable and does NOT mathematically imply the decoder ignores the memory. Long-history degradation in certain ME100 CO/O3 scores is evidence of a domain-specific failure, not a proof of literal tensor "bleeding." Do not diagnose from metaphor when more discriminating lesions/gradients/conditional data controls are available.

### Distinguish source-grounded mechanisms from extrapolations

| Owner claim | Verified prior art | Unsupported extrapolation / experimental correction |
| --- | --- | --- |
| Transformers avoid fixed compressed historical memory via attention | Attention uses data-dependent Q/K/V scores and can access a growing causal KV cache | Cache size and lookup work grow with context; not a free non-parametric dictionary, guaranteed exact retrieval or automatic usefulness. Every token stored does not imply every token attended correctly; serving variants may compress/evict KV. |
| Attention entropy automatically expresses posterior uncertainty during missing data | Attention weights are normalized affinities, so entropy is computable | **False as a general implication.** High/low attention entropy has no guaranteed relationship to predictive variance, proper NLL or 90% interval coverage. An explicitly trained uncertainty model and real gap calibration are needed. |
| Mamba-2 SSD makes content-sensitive writes and reads efficient | Input-conditioned B_t/C_t and scalar-times-identity A_t with semiseparable dual form permit block-matmul/chunkwise methods | No guaranteed no-rejection, no cure for unused memory or calibration. Mamba-2's A,B,C can be computed from current input; making them functions of **previous recurrent state/readout** usually destroys the *same* globally input-only associative schedule, notwithstanding chunked execution. |
| Mamba-2 decays automatically widen uncertainty on missing tokens | Input/time-conditioned decays and persistent state can represent different timescales | Deterministic decay is **not covariance propagation**, and it can shrink state magnitude while predictive confidence incorrectly rises. Need actual PSD uncertainty state/output law. |
| Multi-head attention/SSM guarantees cross-channel isolation | Per-head state partitioning and masks allow specialized computation | Heads can mix through shared encoders, output projections, gates and losses; full isolation can remove useful real cross-variable coupling. Specialization must be measured, not presumed. |
| MoE creates enormous capacity at low cost | DeepSeek-V3 uses 671B total / 37B activated parameters per token with DeepSeekMoE and MLA | Training/storage still account for total params, expert communication and capacity; active ≠ total and conditional routing is not a proof of knowledge organization. Not an off-the-shelf transplant to a 100k-param budget. |
| Deep compositional layers yield high-level reasoning | Stacked nonlinear projections support compositional representations | Early/middle/deep semantic interpretations are useful heuristics, not guaranteed layer-by-layer functions. New depth/expressiveness may hurt latency or overfit small real datasets. |
| "O(log N) GPU speedups" | Parallel prefix scan admits O(log N) *critical path depth* for associative update summaries under suitable parallel scheduling | Speedup is hardware-, layout- and batch-dependent; total work, memory transfers, chunk-matmul and kernel overhead matter. Logarithmic depth does not imply logarithmic wall-time improvement or O(log N) total work. |

**Primary literature:** Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762); Dao & Gu (ICML 2024), [Transformers are SSMs / SSD / Mamba-2](https://proceedings.mlr.press/v235/dao24a.html), [paper](https://arxiv.org/abs/2405.21060); [official Mamba implementation](https://github.com/state-spaces/mamba); [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) and [official model repository](https://github.com/deepseek-ai/DeepSeek-V3). The preceding review's real-representation and experience-learning literature remains in NEXT-REPRESENT-ADAPT.

## 1. Binding present-source evidence — ME100 and ME101, not generic analogies

**ME100, draft PR395:** five complete 97,235-parameter TAC-Temporal source-admission models, six pollutants six hours ahead, real station data, 256-hour histories. Exposed 2015 natural-gap PM2.5 MAE 38.986917 full-history vs 46.073106 8-hour on the same trained models; clustered interval [-15.369833, +1.868555] overlaps zero. Nominal 90% gap coverage **67.69%**; full capability gate **FAIL 0/5**. Some long-context CO/O3 point errors and density metrics regress. These observations do NOT isolate the original 8×8 matrix as the cause.

**ME101, draft PR397:** five new 97,259-parameter models introduce source-age/evidence-mass conditional current-state variance and coherent affine emission, without adding persistent FP32 tensors (the 608-byte primary core remains, external staleness/metadata counted separately). Ordinary PM2.5 MAE improves from 35.689034 to 32.770872; natural-gap mean 38.986917 to 37.747018, but **three of five mapped parent comparisons worsen**. Gap nominal 90% coverage reaches only **68.67%**, below unchanged 85% gate; six-channel gap NLL worsens **5.769106→6.234504**, PM2.5 proper interval score **279.347305→301.220560**. The predictive intervals narrow; the proposed conditional belief design DID NOT resolve gap undercoverage. **Full gate FAIL 0/5, MIXED_TRADEOFF.** ME101 used **twice ME100's training-update budget and a changed loss**; no matched architectural superiority or parameter/data-efficiency claim is permitted.

ME100 and ME101 are research PRs with original receipts, not artifacts changed by this document. The guarded 2016–2017 numerical outcomes remain sealed; 2015 is exposed development, not fresh independent qualification. Use source original result documents [ME100](https://github.com/FPC-effortless/TAC-OSM/blob/research/me100-entity-sensor-world-2026-10-25/research/mathematical-engineering/me100/RESULTS.md) and [ME101](https://github.com/FPC-effortless/TAC-OSM/blob/research/me101-belief-sensor-world-2026-10-25/research/mathematical-engineering/me101/RESULTS.md), plus linked PNDS PR12/13 partial math. Preserve historical negative ESWAM012, ME099 and untrained PR331 CED sources.

## 2. Proposed selective belief-state graph

Sequence inputs o_t, observation mask m_t, real elapsed time \delta_t, station/entity/source identity/id permissions e_t and observed/eligible previous actions a_(t-1) must be available **by prediction time**; future outcomes and cached final-sequence states must never leak into earlier reads. Shared encoder -> typed source-aware per-channel evidence/gates -> H separated or grouped input-only SSD/affine scans -> causal content query/read + bounded **postscan** cross-head mixer and optional original CED -> current belief mean/structured covariance -> coherent multi-horizon predictive distribution -> properly scored forecast and authorized action-choice branch -> subsequent verified outcome and source-dated experience correction. CED K=4 remains serial across depth. Do not invent extra current action/feedback labels for UCI sensor data.

### 2.1 Multi-head input-only SSD / scan-compatible memory

For h=1,...,H heads, current-time *input-only* projections generate:

\[
\alpha_t^{(h)}=\exp[-\operatorname{softplus}(\rho_\theta^{(h)}(x_t))\,\delta_t],\quad
b_t^{(h)}=B_\theta^{(h)}(x_t,m_t,e_t),\quad
c_t^{(h)}=C_\theta^{(h)}(x_t,m_t,e_t).
\]

A minimal candidate writes

\[
S_t^{(h)}=\alpha_t^{(h)}S_{t-1}^{(h)}+b_t^{(h)}v_t^{(h)\top},\qquad
r_t^{(h)}=c_t^{(h)\top}S_t^{(h)},\quad v_t^{(h)}=V_\theta^{(h)}(x_t),
\]

with dimensions explicitly chosen and source/observation mask gating enforced. Equivalent fixed-memory affine TAC form:
\[
M_{t+1}=M_tP_t+Q_t,\quad (P_t,Q_t)=f_\theta(x_t,m_t,\delta_t,e_t),
\]
where the exact multiplication orientation is aligned to the current source implementation. The elementary affine pair composition (for *right multiplication*) obeys:
\[
(P_1,Q_1)\circ(P_2,Q_2)=(P_1P_2,\ Q_1P_2+Q_2).
\]
This operation is associative over exact arithmetic. Confirm with PNDS and independent PyTorch serial/prefix/chunk gradients, including masked no-evidence identity, finite-precision tolerance and causal authorized-source invariants.

**Critical reconciliation of owner's "previous state projection" request:** If \alpha_t, B_t, P_t or Q_t depend on r_(t-1) or S_(t-1), the input-only update summaries cannot be independently precomputed, so exact original global affine scan generally fails. Candidate alternatives: (a) make B/C/read/write projections share parameters and end-to-end forecast gradients while P/Q depend only on causal input-side features; (b) compute read-conditioned gate *after* the scan for a distinct bounded stream-time correction, honestly sequential; or (c) develop a new provably associative augmented operator, with extra state/work and complete proof obligations. Mamba-2 SSD is **not** evidence that arbitrary previous-state-dependent gates remain scan-parallel.

A simple per-head scalar \alpha may shrink expressive transition dynamics relative to an 8×8 learned P_t. Evaluate learned transition rank, timescale diversity, long-context recall and conditional forecast loss, including dense/recurrent native controls. A multi-head "SSD" proposal is not guaranteed more expressive at equal state and trained parameter budgets.

### 2.2 Content routing without a full Transformer KV cache

Candidate optional causal read:
\[
q_t=W_q x_t;\quad
a_{t,i}\propto\exp(q_t^\top k_i/\sqrt{d_k})\mathbf1[i\text{ available at }t],
\quad r_t=\sum_{i\in\mathcal M_t}a_{t,i}v_i.
\]

Explicitly distinguish (i) exact causal retrieval from an authorized history/KV memory with O(window × width) memory and O(window) per-query scoring, versus (ii) selective bounded anchor/ring/content-addressed slots with a *measured finite* memory budget and possible irretrievable evicted detail. Exact arbitrary-token attention cannot be promised at O(1) fixed stored state without additional strong input assumptions. Compare against identity-key retrieval and structured source-aware baselines on **authentic delayed relevance cases**, not synthetic keyed-recall headline tasks. Bound external cache and charge all memory separately.

To test "ghost memory" rather than infer it from a rejection proportion, measure actual forward read contribution, read-vs-no-read loss, learned admission/selectivity, gradients back to relevant B/C/P/Q and source events, response to targeted memory lesion/swap, long-vs-short same-checkpoint history and spillover into unrelated channels. A low admission rate can be optimal if source events are redundant; a high admission rate can be harmful. No fixed 85% acceptance-rate target.

### 2.3 Channel specialization while preserving physical relationships

Owner's numerical illustrative split **88 = 4×22** is a candidate **only if the pinned source verifies an allocatable 88-dimensional substate with the right layout**. **608 FP32 bytes = 152 scalar floats**, not a guarantee that 88 specific floats are freely available; do not treat 4×22 as a verified shape or reassign the native 8×8 state silently.

For source-qualified pollutant sensors, possible heads are pollutant-specific or chemically/temporally related groups, with inter-head transfer controlled by observed-at masks and bounded gates. For the distinct Physical/Enterprise/System1/Temporal/Edge/Counterfactual families, use domain-specific projections/heads around the shared core; do not merge robot gas, obstacle and loan variables in a single 88D tensor merely to label the heads. Proposed constrained mixer:
\[
\tilde r_t^{(h)}=r_t^{(h)}
+\sum_{j\ne h}\sigma(g_\theta^{hj}(x_t,e_t,m_t))\,W_{hj}r_t^{(j)},
\]
where cross-domain/source-authority masks require forbidden links to vanish; spectral-norm or bounded-gain assumptions must be checked, not presumed. Run equal-budget dense-vs-block-diagonal-vs-learned-sparse group comparisons, channel-specific interventions and source-swap controls. Total head memory + mixer parameters/compute must fit the prospective rules. Strict isolation may regress because PM2.5, PM10, CO, SO2, NO2 and O3 and weather can carry real interdependencies; the goal is **controlled conditional sharing** with better forecasts and calibration, not universal decorrelation.

### 2.4 Explicit mean–variance belief propagation with missingness

**Goal:** predict a calibrated distribution conditional on observed source history and missingness, not mere attention entropy or deterministic decay. Proposed state law under an explicit linearization/local-dynamics assumption:

\[
\mu_{t|t-1}=F_\theta(\mu_{t-1|t-1},a_{t-1},\delta_t),
\quad J_t=\left.\frac{\partial F_\theta}{\partial x}\right|_{\mu_{t-1|t-1}},
\]
\[
\Sigma_{t|t-1}=J_t\Sigma_{t-1|t-1}J_t^\top+Q_\theta(\delta_t,\text{regime}_t),
\qquad Q_\theta\succeq0.
\]

For observation y_t, mask H_t and trusted measurement noise R_t\succeq0, a **linear-Gaussian conditional update** candidate (not an identity for nonlinear/missing-not-at-random worlds) is:
\[
K_t=\Sigma^-H_t^\top(H_t\Sigma^-H_t^\top+R_t)^{-1},
\quad\mu^+=\mu^-+K_t(y_t-H_t\mu^-),
\]
\[
\Sigma^+=(I-K_tH_t)\Sigma^-(I-K_tH_t)^\top+K_tR_tK_t^\top.
\]
Use masks: when no new authorized observation exists, omit the measurement update, **not** treat zero/missing values as observations. A Kalman-style K_t depending on evolving covariance/state does **not** preserve the original input-only affine full-history scan unless proven with an augmented associative operator or computed in a separate sequential/postscan stage.

A cheaper probabilistic *output* law that retains the existing primary scan is:
\[
\mu_{y,h}=g_{\theta,h}(z_t,\text{age}_t,\text{mask}_t),
\quad \Sigma_{y,h}=J_h\Sigma_{\mathrm{anchor},t}J_h^\top+L_hL_h^\top+\operatorname{diag}(\operatorname{softplus}(d_h)+\varepsilon),
\]
with L low-rank and d learned only from FIT-eligible source inputs. If J_h is a Jacobian, compute from **the actual chosen forecast mapping**, not a hand-waved ME098 matrix; an exactly affine decoder is a special case. Gram and strictly positive diagonal ensure PSD and potentially SPD under eps>0; this does not guarantee correctly calibrated posterior uncertainty or nominal 90% empirical coverage.

**ME101 already tested one related conditional variance proposal and failed the gap calibration gate.** Future model experiments must explicitly go beyond merely asserting variance increases with sensor age: source-conditioned **residual-based training**, coherent aleatoric versus epistemic explanations where identifiable, valid missingness mechanisms, multi-step gap stratification, real NLL/CRPS/proper intervals, shift robustness, and untouched external cohort. A covariance monotonicity claim requires stated conditions (e.g. PSD Q and no observation, bounded J) and may not hold componentwise under arbitrary Jacobians; don't force it by multiplying every width with posthoc constants.

### 2.5 State accounting and depth expressivity

For float32, 608 bytes = 152 floats **only for the pinned native recurrent primary state as currently measured**. It does not include model weights, source IDs, timestamps, source-age bookkeeping, masks, caches, external memory, per-head covariances or runtime activation tensors. A full symmetric 6×6 covariance requires 21 FP32 scalars = 84 bytes; its six diagonal elements require 24 bytes, not counting means/masses or unobserved cross-horizon covariance. To remain literally 608 bytes, replace or factor existing allocations and prove shape/storage invariance. Otherwise report the expanded state explicitly as **608 bytes plus X**, accepting a trade-off rather than falsely promising zero-cost uncertainty. ME101 explicitly avoided adding persistent FP32 covariance tensors and nevertheless failed calibration.

A 4-head 22-dimensional split is an illustrative **hypothesis, not source-derived guaranteed 88D state**. A deeper stack of bounded postscan residual/nonlinear blocks can add expressiveness but also adds latency/parameters and sequential depth; original TAC CED is a distinct concept-refinement mechanism. MoE-like sparse routing is OPTIONAL and must be compared at the same **total trained/deployed** and active budget with no argument from 671B DeepSeek scale to a 100k TAC model. At unchanged state width, test finite parameter allocation to heads versus deeper decoder versus stronger source-sensitive representation, not all mechanisms at once.

## 3. Mathematical engineering and Lean/Python source-equivalence obligations (PNDS)

1. **Source tensor audit:** pin the exact original 8×8 memory update, actual core width and 608-byte calculation, source trace availability and all currently active ME100/ME101 definitions. Reconcile existing operator orientation and state-quantization assumptions BEFORE replacing equations.
2. **Affine and semiseparable law:** prove associative composition for each proposed per-head update in exact arithmetic; prove scan-vs-stream outputs at arbitrary causally masked prefix cuts; prove no hindsight or unauthorized source writes. Prove that decoupled multi-head block-diagonal recurrences do not mix state within heads, and separately prove conditional postscan mix-mask properties.
3. **Probabilistic validity:** enforce symmetry, PSD/strict positivity, no-observation behavior, finite source-age dynamics and legal posterior correction conditions; do not claim generic learned attention entropy equals uncertainty. For low-rank Gram covariance, derive memory complexity and stabilize Cholesky/likelihoods under small variances.
4. **Independent executable graph:** Python reference implementation for every tensor/normalization/mask/nonlinearity, forward gradients/autodiff/finite-difference and full fitted-network serial/prefix/chunk FP32/FP64 comparisons. Lean proof of an idealized recurrence is **not** an IEEE or original PyTorch graph proof.
5. **Expressivity/resource accounting:** measure rank/feature interactions, parameter allocation, total FLOPs, peak GPU/CPU resident memory, p50/p95/p99 training and online latency. Serial and parallel algebra can be equivalent yet realize very different costs.
6. **Causality/action status:** learned F(x,a) on passive station readings is not an identified action-conditioned world model. For real decision claims require recorded available actions, observed outcomes and identification or authorized prospective testing, matching NEXT-REPRESENT-ADAPT rules.

## 4. Preregistered capability-first integrated experiment program

**Do not fit a separate architecture for every small mathematical option.** Start with a complete capability-bearing candidate, source audit and prespecified real task. Register a small set of meaningful matched native references to identify whether proposed structural modifications add real value, without treating control-only tests as a substitute for a better model.

| Work package | Main capability target / comparison | Pass/fail evidence and stop |
| --- | --- | --- |
| B0 — source/theory preflight | Current exact ME100/ME101 graph and state accounting; authentic old-information signal; retrieve actual write/read measurements and original source tokens | Confirm whether 77% "rejected writes", 88D and "ghost reads" are real; if not, withdraw diagnosis. Headroom must beat short observed-context control |
| B1 — strong selective-SSD read/write | One source-aware multi-head input-only SSD candidate plus a parameter-matched existing native fast/slow/8×8 model, structured and practical real baselines | Actual held-out point/probabilistic forecasts, memory lesions, gradient/read contribution and stable long-context benefit; preserve scan and step parity |
| B2 — missing-observation probabilistic correction | Integrate **one** well-defined source-conditioned latent/observational uncertainty propagation on complete model, comparing actual frozen ME101 as a *historical unequal-budget reference* and equal-budget newly trained controls for architecture claims | Real gap 90% nominal empirical coverage clears the **unchanged 85% qualification floor** without worsening required NLL/proper scores and daily/rare regimes; additional 2016–17 only under registered protected evaluation rules |
| B3 — bounded cross-channel specialization | In otherwise matched full models, dense mixing vs channel-blocked vs bounded gated inter-head mixing | Improve PM2.5 without harming other gases, joint six-channel NLL, coverage, tail misses or source/time authorization. Report per-pollutant and real cohort confidence bounds |
| B4 — expressivity/operating point | Head count/state width/depth/MLP/optional sparse expert trade-offs with strictly fixed eligible total trainable/deployed parameters/data/compute | Actual Pareto of real capability, inference latency, resident bytes, FLOPs and calibration; no automatic O(log N) speedup or 608-byte claim |
| B5 — integration with representation and actions | After qualified prediction/retention: intervention-tested causal latent reads and real prequential adaptation, then candidate-guided decisions only with action-authorized datasets | Actual independently verified future predictive/decision gains; do not claim business or robotics autonomy from passive pollutant forecasting |

**Hard policy:** all new architecture-superiority arms use the standing *95k–105k real trained/deployed learned parameter size*, ≤1% paired param gap, matched authentic source/time splits, model exposure, training compute, tuning budget, seeds, case IDs and independent sealed test conditions. CPU wall time must be reported, not forced equal. ME100 (97,235) and ME101 (97,259) remain historical references without retrospectively equalizing their training. No repeated 2015 selection after exposure, no synthetic headline research, no 2016–17 test access until sealed candidate/predictor/source rights/replay thresholds are satisfied. New input masks must not create synthetic unsupported source labels. Fail closed for pre-registered guardrails: uncertainty coverage, joint likelihood, interval scores, other-channel harm, safety/time validity and long-history value.

## 5. Capability objective: what would count as a meaningful advance?

Not "Transformer-like intelligence", "automatic 90% coverage", "zero bleeding", "77% recovered writes", "all heads perfectly isolated", or "still 608 bytes" without proof. A qualified success would show, on genuinely independent real source histories under fair training, that the shared compact TAC family gains **selectively useful old-state information**, preserves it across gaps, learns conditionally calibrated coherent joint futures, and does not degrade related signals. Internal memory interventions must causally change forecast outcomes in preregistered ways; actual action utility is a **later separate** gate.

The research remains original by its verified operator/source design and measured capability, not by relabeling standard attention/SSD/Kalman/MoE mathematics. All failed historical results and frozen original checkpoint bytes remain intact.

## 6. Mapping to the six existing families

- **Temporal (priority):** ME100/ME101 authentic station gaps, source-age belief, six-channel coherent dynamics. Track channel-specific calibration and fixed-state source-qualified history.
- **Physical:** kinematic / motion / servo-prior structured state, actually observed actions, world transition uncertainty; evaluate real executed actions and stationary-vs-moving trade-offs.
- **System1:** low-latency incremental reads and source-valid fast actions with no unsafe automatic outputs; read contribution and p95/p99 measured on hardware when available.
- **Enterprise:** separated authorized event/source histories, temporal semantics, state-to-workflow prediction and verified outcomes (not retrospective case-end proxy).
- **Edge-Lite:** independently measured additional memory/compute/quantization; do not assert a microcontroller SLA without deployment evidence.
- **Counterfactual:** a later stage only where genuine action availability, outcome measurement, and causally supportable policy estimation exist.

**No seventh architecture.** These are capability-specific uses of a single shared mathematical operator and independent evaluation contracts, not six copied untrained memory engines.
