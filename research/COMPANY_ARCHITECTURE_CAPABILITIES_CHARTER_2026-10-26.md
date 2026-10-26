# TAC-OSM company architecture and capability charter
**Date:** 2026-10-26. **Status:** proposed owner-directed research/product prioritization, not a qualified checkpoint, architectural-superiority finding, formal whole-model proof, product release, or customer validation.
**Authority:** [Unified Master Research Plan](MASTER_RESEARCH_PLAN.md), [Company Strategy and Daily Operating System](COMPANY_STRATEGY_AND_DAILY_OPERATING_SYSTEM.md), [Real-World Evidence Rules](REAL-WORLD-EVIDENCE-RULES.md), [PNDS Mathematical Engineering Rule](PNDS-MATHEMATICAL-ENGINEERING-RULE.md), and repository AGENTS.md. Technical/evidence/release rules and original experimental artifacts override strategic objectives where they conflict.

## First-class ecosystem design dependency — 2026-10-26

The [source-grounded ecosystem interoperability and evidence contract](ECOSYSTEM_INTEROPERABILITY_AND_EVIDENCE_BOUNDARIES_2026-10-26.md) is a required companion to future TAC-OSM state, memory, typed-decision and verification research. Existing Woyengi state/reconstruction/authority/effects semantics and Veritas's qualification distinctions shape model interfaces at design time. Do not implement a competing mutable "truth store" in TAC-OSM, bypass Woyengi authorization or claim Veritas attestation from a model score. Forge, TrainLab and PNDS provide authentic legal input, reproducible run evidence and mathematical specifications. A constructed boundary validator can enforce only structural nonpromotion and as-of access; genuine interop must be separately source-authorized and independently tested.

## 0. Binding company direction, scope, and distinctive hypothesis

**Research mission:** develop efficient, stateful, predictive and probabilistic intelligence that maintains useful representations of changing world state, tracks authorized identities, forecasts genuine future outcomes, and emits calibrated machine-readable decisions. Design for fixed-size **neural working state**, parallelizable sequence learning where mathematically valid, fast recurrent online execution, and mobile/edge devices as first-class deployment targets.

**Developer business:** alongside the flagship predictive-state research, publish genuinely useful small task-specialist models and open tooling. Initial adoption may come from utility models and Veritas/Forge/TrainLab, even if TAC-OSM's own scientific gates remain failed. Don't require a new neural architecture to perform ordinary language detection, PII classification, audio event detection, or image classification when simpler validated models are better. A product catalog does not imply extra TAC-OSM research families.

**Company inspirations, not imported authority:**
- Desert Ant Labs: compact specialist models, private/offline deployment, easy language-native SDKs, device benchmarking, and discoverability. Its software/model licensing and reported performance do not establish TAC-OSM quality; preserve source/model licenses and never train on third-party outputs where prohibited.
- TypeSafe/Jev: native structured decisions (Choice, Score, Boolean probability), multi-question inference on one shared encoded context, machine-native outputs rather than generated text. Their proprietary approach or marketing is not evidence that TAC-OSM has implemented or matched it. Call our primitives our own; do not imply license to copy internals or a proprietary RLCD method.
- Liquid AI and state-space models: treat end-to-end efficiency, selective recurrence, small-model deployment and hardware-native kernels as engineering disciplines, not inherent TAC novelty.
- TAC-OSM + PNDS original scientific hypothesis: reliable retained *evolving* state + source/entity-aware memory + future prediction + calibrated decisions in one compact parallel/streaming graph; distinctive only if better independent outcomes and reproducible analysis demonstrate it.

**Positioning:** reliable operational agents and software predictions today where independently supportable; mobile/edge native persistent predictive-state intelligence tomorrow. Avoid a general-chatbot-first strategy, fabricated pilots/revenue, and claims that a roughly 100k-parameter state model equals a full speech/vision/language foundation model.

## 1. Six non-overlapping research families; common six-verb capability contract

Maintain **exactly six** active families under the shared, source-backed TAC-OSM mathematical architecture. Do not silently add a seventh model family; do not relabel historical checkpoints or promote unqualified lines.

| Family | Native scientific goal | Candidate mobile/developer outcome | Non-negotiable risk |
| --- | --- | --- | --- |
| System 1 | Fast, calibrated immediate event/choice/risk inference from genuine persistent context | Embedded stateful typed-decision API; agent/app routing | Existing S1 real-data memory/motion failures; no autonomous permission |
| Enterprise | Real logged entity-state transitions, outcome likelihood/time, decision cost | Workflow-risk and verified action/status advisory | Honest temporal availability, case/subject leakage, no invented verified completions |
| Temporal | Long-horizon, irregular-time and as-of/bitemporal forecasts, coherent uncertainty | Offline sensor/event forecasts with calibrated intervals | ME110 point/interval tradeoff; protected target access |
| Physical | Recorded physical action-conditioned dynamics and sensor changes | Robot or equipment telemetry and monitored advisory | Strong measured-velocity/persistence/servo baselines and takeover evidence |
| Edge-Lite | Source-equivalent export, fixed-state inference, quantization, latency/RAM/energy | On-device Android/iOS SDK and offline specialist packages | Real devices and numerical parity are required, not just ONNX export |
| Counterfactual | Eligible alternative action consequences and decision costs | Guarded scenario evaluation after causal identification | Observational action conditionals are not intervention effects |

**Six verbs as a shared contract** (not six more models): **Remember** source-available history; **Understand** entity and relational state; **Predict** future events and distributions; **Decide** with typed task heads; **Verify** against independently obtained facts and permissions; **Deploy** within measured compute/memory/hardware limits.

Keep CED where actual source-implemented and mathematically specified; do not assume any CED layer adds causality, retrieval ability, or novelty without controlled evidence.

## 2. Ten prioritized architectural capabilities

| Priority | Capability | Scientific/engineering requirement and falsifier |
| --- | --- | --- |
| P0 | 1. Stateful world representation | Compact recurrent state tracks real changes; fail if current snapshots or structured last-value baseline are equally good |
| P0 | 2. Reliable associative, temporal and entity memory | Retrieve useful distant/identity-specific facts; fail on cross-entity interference or no genuine downstream advantage |
| P0 | 3. Parallel-compatible training plus recurrent execution | Same source graph admits audited bounded/chunkwise scan and step mode; fail if actual operators break equivalence |
| P0 | 4. Probabilistic future-state prediction | Properly scored future value/event/time distributions and physical support; fail on critical gaps, calibration or simple baselines |
| P0 | 5. Native typed machine decisions | Schema-safe Choice, Predicate, Score, Forecast, Defer heads without text parser; fail if head cannot use verified state |
| P1 | 6. Calibrated uncertainty and adaptive deferral | Per-domain/subgroup calibration and meaningful abstention cost; fail if confidence is uncalibrated |
| P1 | 7. Independent verification and state correction | Prediction vs actual facts; correct beliefs after authorized outcomes, never silently mutate operational truth |
| P1 | 8. Edge/mobile execution | Offline export, fixed neural-state budget and measured p50/p95/peak RAM/energy on physical hardware |
| P1 | 9. Modular multimodal encoders | Plug structured logs, sensors, text, audio, vision into source-timed encoded observations; task-qualified models may remain separate |
| P2 | 10. Counterfactual planning and cost | Model defensible alternatives from identifiable intervention data, account for confounding, off-policy reliability and costs |

Dependency policy: P0 memory and prediction cannot be declared solved by architectural component activation or fixture CI; P1/P2 expansion needs actual upstream capability evidence. Build one integrated capability, not ten unrelated architectures.

## 3. Memory and temporal information architecture

Working memory components: (a) short-response fast state; (b) slower rate-dependent retention; (c) source-indexed entity slots and relational/global state; (d) associative retrieval; (e) time/source-availability-aware correction; and (f) optional permissioned external history managed outside fixed neural working memory.

**Memory acceptance:** actual source chronology, delayed facts and genuine entity IDs, interfering distractor entities, old-history ablations, cost-aware retained fact usefulness, multi-horizon impact, and verified held-out outcomes. Compare against structured schemas, retrieval baselines, GRU/SSM/Transformer as appropriate; enforce matched parameters and training/data/compute for claims of architectural advantage. External storage size, retrieved context, normalizers, weights, execution workspace and cache must be measured separately: **fixed-size neural state != fixed-size complete application memory**.

**Negative findings remain binding:** DEV-017 structured 41-value memory strongly outperformed older TAC long-history prediction; ME-074 keyed recall problems; EWA001/002 memory utility worse than similarity-only controls; recent S1TEMP006/007 and S1MOTION006 real-source full gates **FAIL 0/3**. Inactive/misaddressed memory, limited reader contribution, over-servicing and source leakage are candidate causes to test, not conclusions proven by these failures. S1TEMP008 source-indexed entity memory and PNDS exact-real isolation lemmas are promising *engineering* but its genuine BPI2017 outcome training and held-out qualification remained unproven as reviewed.

**Retention objective:** prospective ability to maintain facts whose removal measurably increases future prediction error, decision cost or factual correction rate. Do not claim counterfactual causation from observational ablation alone. Admission and retention must be bounded, stable, source authorized and interpretable.

## 4. Mathematical computational graph: source-to-output

The full model must have exact typed shapes, update order, numerical operations, allowed observations/actions, outputs, and trained parameter counts. A candidate specification is:

1. As-of authorized encoder: \(e_t=E_\theta(o_t,a^{observed}_{t-1},\Delta t,s_t)\). \(s_t\) includes only truly available identity/authority/feature masks; source/knowledge time remains explicit.
2. Parallel-compatible affine state component: \(h_{t+1}=A_t h_t+b_t\), where \(A_t,b_t\) depend only on available inputs or other separately proven parallel components. Associative composition permits a prefix scan **for this component**; arbitrary nonlinear state dependence does not.
3. Source-eligible memory update: \(M_{t+1}=U_\theta(M_t,e_t,\mathrm{id}_t)\) with entity isolation and bounded retention, proofs for actual update semantics.
4. Readout: \(z_t=R_\theta(h_t,M_t,e_t)\) using only source-allowed snapshots.
5. Distribution: \(P_\theta(Y_{t+h}\mid z_t,a^{observed}_{\le t},h)\), reporting coherent physical/logical support, censoring and interval law. Hypothetical \(do(a)\) remains an independently qualified causal capability, not the default inference.
6. Typed task head: \(d_j=D_{\theta,j}(z_t,q_j)\), enabling multiple authorized queries to share one encoded state. Heads must not manufacture permission or bypass validator/authority boundaries.
7. External verification/correction: observed source outcome updates beliefs only via an authorized source; Woyengi/application policy owns authoritative changes.

**Design invariant:** batch, bounded-block and step update compute the same defined function up to preregistered floating tolerances; no claim all recursive nonlinear work is perfectly parallelizable. Preserve the existing native core and CED source semantics as actually implemented; this graph is a target contract, not a claim every version already fits it.

## 5. Typed machine-output architecture

Recommended output primitives:
- **Predicate:** a typed Boolean event probability with domain/horizon/reference truth definition.
- **Choice:** a probability distribution over an explicitly allowlisted set of permitted labels; malformed label mapping is a hard error.
- **Score:** real value in a documented range or unit, with a calibrated meaning.
- **Forecast:** coherent distribution over event, state, severity or arrival time at declared horizon; support, missingness, coverage and censoring tested.
- **Defer:** calibrated decision to abstain, escalate or request valid additional information; always subject to application policy.

Support multiple concurrently evaluated task questions from one causal state, with versioned schema, no text parsing requirement, type/range validation, observed evidence reference, model/checkpoint hash, temporal cutoff and optionally auditable uncertainty. Schema validity **does not imply** factual correctness, calibrated probability, safe action or verified outcome.

**Illustrative product questions:** "Which recorded workflow is at higher delay risk?", "What is the next sensor state and interval?", "Is an operator inspection justified?", "Does the state evidence support a request for more information?" A numerical answer to any of these is unqualified until proper outcome evaluation.

## 6. Joint learned prediction, uncertainty, deferral and cost

A proposed, **unvalidated** integrated research objective is

\[
\mathcal{L}=\lambda_s\mathcal{L}_{future-state}
+\lambda_d\mathcal{L}_{decision}
+\lambda_p\mathcal{L}_{proper-probability}
+\lambda_m\mathcal{L}_{memory-utility}
+\lambda_v\mathcal{L}_{verification}
+\lambda_b\mathcal{L}_{resource-and-decision-cost}.
\]

Separate point/state pathways from calibrated uncertainty heads where supported by observed gradient conflicts. The original trained ME110 five-seed study showed better development point error (gap/general PM2.5 roughly 7.22%/7.28% versus mapped parents) but poorer gap interval score; complete registered qualification **FAIL 0/5**. Higher point accuracy therefore cannot be promoted as calibrated predictive success.

Research requirement: fully preregister point, NLL, proper interval scores, coverage both minimum/maximum, miscoverage by channel/source/long gaps, uncertainty-width tails, deferral coverage-risk curves and decision cost. Inspect gradients at common trainable blocks, head and source availability masks. Any PCGrad/projection or typed-role routing must show realized optimizer-step effects and source-held-out benefits, not only local analytic dot products or convergence of the scalar joint loss.

For reward or decision-based optimization, require independent external verified task goals, source/outcome receipts, trusted evaluator separation and model-agnostic anti-reward-hacking checks. Synthetic adversarial tests validate the *detector logic only*. Refer to TAC draft PR430; unmerged runner framework is not presently independent real-model attestation.

## 7. PNDS mathematical and formal engineering requirements

Develop a versioned source-to-output specification, with:
- Shapes of every learned tensor and actual live parameters; model-specific state bytes and external memory counters.
- Exact causal access graph and conditional availability/identity masks; mathematical initial and reset states.
- Affine or otherwise associative scan proofs only for operations satisfying stated algebraic premises; exact composition and source isolation lemmas.
- Full differentiability/JVP/VJP/adjoint contracts for learned operators and loss routing, compared with PyTorch autograd and independent Python references.
- Floating-point roundoff, overflow, reduction order and nonlinearity envelopes for actual FP32/FP64/quantized export kernels, not unspecified abstract-real arithmetic alone.
- Probability normalization, quantile monotonicity, censoring atoms, physical support, score/likelihood equivalence and calibration evaluation interpretation.
- Explicit distinction between a mathematical theorem, verified source-level refinement, numerical finite tests, software CI, learned scientific performance and hardware deployment.

ME110/PNDS #35 and earlier source-pinned PNDS lemmas provide **partial exact-real/source checks**, not a Lean theorem for every possible TAC-OSM operation, trained stochastic gradient or IEEE implementation.

## 8. Six-family development acceptance gates

**General independent capability gate:** meaningful original real-world complete-model skills on unseen longitudinal source with fit/validation/test isolation, preregistered metrics/thresholds/seeds, original checkpoints+optimizer+RNG+source/split/rights receipts, independent clean-checkout replay and all critical subtype/decision costs. No invented outcome truth, protected TEST tuning or synthetic success promotion.

**Matched superiority gate:** approximately 95k–105k active/deployed learned neural parameters for prospective TAC research unless formally amended; <=1% parameter differences among claimed matched neural arms; original identical source inputs, training examples/splits/exposure, budgeted compute and comparable tuning rights. CPU wall-time is a measured product-efficiency metric, not a mandated equality of training wall clock.

**System 1:** future-event/risk and decision ranking under legitimate current observations; measure old-history and entity interference, calibration/deferral, source-availability as-of and cost. No autonomous tool execution claims.

**Enterprise:** real multistep events and verified outcomes, individual case/subject isolation, delays/time-to-event, operator decision costs, no unearned "completed" states. Real observational action histories cannot justify causal intervention policy on their own.

**Temporal:** genuine missingness/as-of data, multi-horizon point/interval/NLL and gap robustness across channels with all original gating denominators; do not reuse exposed ME110 development outcomes as new unseen validation.

**Physical:** genuine action/observation trajectories, transition errors and practical baselines (measured velocity, persistence, servo/physics), no post-takeover leakage or success invented from measured telemetry.

**Edge-Lite:** original checkpoint identity, source-specific export and numerical equivalence over histories, real Android/iOS hardware reports (RAM, p50/p95 latency, cold start, energy), offline behavior and backend fallbacks. Artifact manifest-only tests are identity preflight, not device or learning results (draft PR432).

**Counterfactual:** evidence-supported identifying assumptions, causal effect validity, actual action eligibility/overlap, confounding sensitivity, counterfactual uncertainty, off-policy testing and risk/cost before deployment or action-selection claims. Do not conflate \(P(Y|A=a)\) with \(P(Y|do(A=a))\).

## 9. Edge/mobile engineering as a standing horizontal requirement

**First target:** physically available low-end and mid-range Android CPU device; expand to iOS and hardware accelerator backends only after source-equivalent export is valid. LiteRT, ExecuTorch, ONNX and Core ML are candidate execution formats, not interchangeable proof that all model operators accelerate.

For every qualified export measure:
1. Exact original model checkpoint/weights/hash and spec, exported artifact/checksum and live network state interfaces.
2. Numerical model-output AND carried-state parity through long genuine histories, reset, long-gap and missingness cases; quantization-induced differences.
3. Parameter/model bytes and explicit maximum total live RAM including workspace, normalization, externally retrieved context and runtime libraries; separately distinguish bounded neural state from variable data.
4. Real-device p50/p95 per-step latency under cold/warm runs, energy/battery methodology, offline/network-disabled operation, thermal behavior and fallback details. Targets (e.g. <1 MB small weights, <=20 ms p95 update, <=1pp applicable metric drop) are **proposals**, never measurements.
5. SDK acceptance: simple Kotlin then Swift API for initialization, permitted input, update, query, reset, access restriction and source provenance, graceful unsupported-device behavior, typed outputs and safe deferral.

A 100,000-parameter FP32 weight payload is approximately 400,000 decimal bytes before activations and runtime; raw INT8 weights ~100,000 bytes before scales/metadata. This is *not* a comprehensive general language/audio/vision model size or total app footprint. Publish appropriately sized independent utility models (text intent/language/PII, lightweight audio event/VAD, narrow visual classification/anomaly and event/sensor classifiers) only after their own source, license, privacy, accuracy and device benchmarks.

## 10. Model catalog, software interfaces and authority separation

**Flagship research product:** stateful typed-decision and future-state API/SDK backed by source-authorized evolving memory. A single encoded state answers multiple typed predicates/choices/forecasts; evaluation must verify historical contribution and forecast/decision reliability.

**Near-term independently valuable utilities:** on-device task-specialist classification, anomaly, privacy, audio/video event recognition and safe structured output. Prefer small established pretrained encoders and task-specific architectures when TAC-OSM is unqualified or uneconomical; license model weights/data properly. Utility products do not imply ten foundational TAC model lines.

**Forge:** source rights, actual availability timestamps, verified outputs vs unknown labels, deterministic grouped temporal splits, hashes, leakage tests and reproducible data release. The standalone MVP in Veritas PR420 is draft; source authenticity, independent reviewed licensed publisher data and security release remain open.

**TrainLab:** reproducible fit manifests, original seed/run/compute ledger, checkpoint and data provenance, fair comparison receipts and continuous replay. Existing TAC training infrastructure should be reused before claiming a separate deployed TrainLab application.

**Veritas:** independently verify intended vs actual authorized state changes, outcomes, reward, safety and agent/tool action lineage; support strict anti-reward hacking and regression CI. Correct product boundary differs from confident model answer.

**Woyengi:** durable canonical facts, policy permissions, bitemporal state and actual authorized actions, while predictive TAC outputs remain shadow/advisory unless separate safety, calibration and authority controls approve action. No neural confidence grants write permission.

**Open-source commercialization:** transparent independently useful open weights/tools/docs/SDKs where legally possible, with hosted dataset/qualification/train-run and device-test services, integration/support and enterprise customization as *hypotheses*. No invented usage, customer pilots, licensing facts or ROI.

## 11. Research sequence and planned milestones

**Phase 1 - actual reliability (priority):**
- Validate one real, source-authorized useful memory skill with genuine long-history and entity grouping. Use meaningful structured-state, last-value and neural baselines and exact source-time cuts.
- Repair the joint point/interval prediction conflict, verify calibration on genuine new source cohorts without leaking protected outcomes.
- Keep learned capability, software/Lean proofs, and replay as separate reporting lanes.
- Stop if source or independent outcome/reward evidence is missing; no artificial "passed" gates.

**Phase 2 - machine-native decisions:**
- Implement and score typed Predicate/Choice/Score/Forecast/Defer queries on shared state. Compare against static typed classifiers and appropriately matched stateful controls; evaluate value of history and deferral risks.
- Preserve policy boundaries and human review.

**Phase 3 - device deployment:**
- Complete original-model export, repeated authentic-sequence numerical replay and real Android p50/p95/peak-memory tests, then publish narrow SDK and model cards only if empirical gates pass.
- Validate adoption from third-party installs and genuinely observed developer feedback.

**Phase 4 - advanced intelligence:**
- Add evidence-qualified perception modalities, verification-based correction, authentic long-horizon dynamics, identifiable counterfactual planning and practical robot/agent advisory.

Do not force these as calendar guarantees; prioritize by falsified hypotheses, independent source access and actual user/developer value. Daily select exactly three measurable outcomes (science evidence blocker, usable software artifact, real customer/developer validation). Weekly evaluate technical and adoption gates, not number of branches/PRs.

## 12. Immediate falsifiable research sequence from 2026-10-26

**Experiment A - Memory usefulness and identity (System 1 / Enterprise):**
- Prerequisite: source-authorized historical BPI/multi-entity data; trusted protected-case partition; actual FIT-only events and independently verified delayed outcomes.
- Candidate: source-indexed associative+slow shared architecture. Fair controls: source-ID-oblivious model and strong structured/entity-store, GRU/SSM or other appropriate baseline; no superiority without full FLOP/exposure/parameter parity.
- Primary skill: genuinely improved long-history source-specific event/time prediction and downstream decision cost, across subject/horizon subgroups. Fail explicitly if any critical original gate fails or attribution relies only on active slots.
- Current: S1TEMP008 **NO BPI TRAINING/SCORING**. Fix publisher and archive isolation first.

**Experiment B - Joint predictive distribution (Temporal):**
- Prerequisite: previously untouched authorized longitudinal source/cohort distinct from already-exposed 2015 DEV and protected 2016-17 TEST unless expressly approved for sealed independent scoring.
- Candidate: availability-conditioned per-channel point/uncertainty optimization and proper distribution law, not another arbitrary larger FFN.
- Primary success: point + interval score improvements with calibrated coverage and no critical channel/gap regressions across registered seeds. Fail if any required gate misses.
- Current: ME110 **FAIL 0/5** despite point improvement.

**Experiment C - Real phone inference (Edge-Lite):**
- Prerequisite: source-authenticated checkpoint/exporter and state-in/state-out manifest, numerical parity on authentic histories.
- Target: Android offline CPU, device/OS/SoC/backend specified; p50/p95, energy, RAM and reset/state consistency. Fail if no genuine physical evidence.
- Current: PR432 *preflight only*, **NO QUALIFIED DEVICE EXECUTION**.

**Commercial parallel track:**
- Veritas dependency security exact-head independent review and Forge original publisher import must pass before promoting a release. Preserve current blocker; do not bypass source/reviewer gates.
- One buyer/use-case interview or independent developer installation requires real consent and evidence; no invented outreach.
- Cross-family reward-signature CI tests from draft PR430 do not replace independent evaluator, source ACL and original learned-outcome provenance for a trained runner.

## 13. Operational and archival policy

Historical original experiments (including negative and mixed outcomes), PRs, proof attempts and reviewer findings remain immutable and addressable. This charter governs **prospective direction** only. Existing draft branches do not merge automatically; each requires applicable review/CI and explicit integration decision. No new protected evaluation, cloud training expenditure, customer contact or autonomous deployment is authorized by this document alone.

**Evidence tiers to use in every handoff:** SPEC_ONLY; SOFTWARE/CONSTRUCTED_PASS; FORMAL_PARTIAL; TRAINED_REAL_DEVELOPMENT; INDEPENDENT_UNTOUCHED_REAL; REAL_DEVICE_QUALIFIED; PRODUCT_RELEASE_QUALIFIED; EXTERNAL_CUSTOMER_VALUE. Never combine these in one "passed" label.

**North-star falsifier:** if persistent state fails to improve genuine long-history, entity-specific, calibrated outcome/decision quality over meaningful fair baselines at practical mobile costs, the central architectural hypothesis remains unsupported, regardless of elegant proofs, generated outputs, attractive model branding, or number of integrated modules.
