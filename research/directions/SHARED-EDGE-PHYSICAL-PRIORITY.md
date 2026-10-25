# Priority A — Shared Edge Physical / Sensor World-State Engine

**Research decision proposal, 2026-10-25.** A prioritized direction within the existing TAC-OSM six-family research and PNDS mathematical governance; **not** a trained multi-domain checkpoint, an additional seventh family, a merged historical variant, a new fit authorization, or verified hardware deployment.

## Decision: retain six interfaces but NOT six compulsory neural models

Keep the original six identifiers and their historical provenance as source-evidence contracts. Reorganize the work as **one core operator stack + three capability programs + two infrastructure layers**, without forcing six separately trained neural networks or asserting that the same pretrained weights work across unlike domains.

| Existing interface/family | Future role | Fusion rule | Preserved hard boundary |
|---|---|---|---|
| System1 | low-horizon reactive forecast/risk readout of Physical/Temporal | same core + optional quick readout | no actuator authority or real safety proof |
| Physical, incl. ES-WAM | **priority A** dynamics, trajectories, observed actuator conditioning and tracked entities | shared scan + physical-specific encoder, dynamics law, forecast distribution | real persistence/servo baselines, no do(action) claim |
| Temporal sensor (ME100/101) | **priority A** source-aware IoT observation, missingness, entity/clock/context, six-channel forecasting | reusable per-channel source trace, masks, learned belief front end and sensor-specific head | real missing sensor ≠ verified delayed arrival; no assumed kinematic law |
| Temporal bitemporal records | external **evidence / time-authority service**, reused for Enterprise and sensor revisions only when authentic valid/known-at stamps exist | typed deterministic as-of adapter, not a distinct neural memory engine | preserve corrected/as-known-at truth and no future leakage |
| Enterprise | **priority B** event/time/lifecycle forecasting with categorical/survival head | reuse source-aware scan, but independent target law and domain data | workflow end-of-record ≠ business completion; no forced physical x+v transition |
| Edge-Lite | **cross-cutting export/deployment profile** for physically qualified checkpoints | same weights if precision export preserves skill | real on-target p50/p95/p99, RAM/flash/energy required |
| Counterfactual | **priority C, optional and gated** factual scenario query / candidate-action advisory | read-only projection atop qualified source state, future dynamics only with authorized action context | logged action does not identify causal effect; no autonomous execution |
| PNDS | common typed math, exact-real operator obligations, numerical reference and Lean source counterpart | engineering authority, not a predictive model | mathematical validity ≠ trained generalization or hardware safety |

The three capability programs are A. physical/sensor world-state intelligence, B. enterprise event-state forecasting, and C. optional identified action valuation. The infrastructure layers are (i) the shared causal evidence/parallel-affine engine and (ii) provenanced deployment/formalization. **Do not misrepresent these groups as newly trained models.**

**Selected TOP-PRIORITY research direction:** General-Purpose Edge Physical Intelligence **as a domain-adaptable architecture**, not as proven one-weight-fits-all intelligence. Target real pollution and industrial IoT forecasting first (current authentic training data), then authentic robot joint/trajectory experiments, then licensed traffic actor motion. Vehicle/robot/drone *control*, real-time hazards and certified safety are future unqualified extensions. Multiple domains may share operator equations but require different units, input encoders, dynamics, event horizons, downstream heads, source metadata and independent evidence. Do not apply ME098's x(h)=x0+g_h*v0+(h-g_h)*u as a mandatory equation for pollution concentrations or categorical enterprise events.

## Source-verified motivation / current negative evidence

- The prospective six-family factory (PR390) sets every family `qualified_head=false`, `qualified_real_skill=false` and `authorized_to_act=false`. It offers core candidates by configuration; it **does not** demonstrate six separately trained, jointly trained or deployable neural families.
- **ME098/099 physical:** genuine 1/4/16-frame forecasting and coherent dynamics on exposed ALOHA DEV; ME099 full gate 4/5 passed (one H4 coverage failure) and NLL lost to FIT reference. Not independent or equally budgeted comparator superiority.
- **ME100/101 Temporal sensor:** five 97,235 / 97,259-parameter original models on authentic Beijing station observations. ME101 mean daily PM2.5 32.7709 vs 35.6890 ME100 (different optimizer exposure: 3072 vs 1536 updates); 2015 already exposed, 2016–17 protected. Gap 90% coverage ~68.67% vs >=85% required; ME101 full gate 0/5. The belief formula is not a calibrated latent-state posterior.
- **ESWAM012/QUAL001:** on the exposed original traffic location, long-native 256 observed frames was worse than 8, with no independent 256-context qualified information win. Historical observed window history is not indefinite fixed-state retention.
- **ME069/073 Enterprise:** authentic event forecasting exists in older models, including old native-vs-GRU failure and weak early history; cannot claim skill in consolidated shared engine.
- **System1/Edge-Lite/Counterfactual:** current domain adapters principally encode source and authority contracts, not trained qualified policies, real microcontroller deployment or identified causal effects.
- **CED PR331, DeepSeek-style selective lookup/mHC/MoE and Muon:** optional capacity/compute hypotheses only, not essential to the first useful physical forecasting gate. Additional concepts/multi-head routing add learned parameters, activation memory and latency even when primary streaming memory remains 608 B.

The 608 B is **152 FP32 persistent source/matrix values PER tracked sample/entity** (plus counters, clocks/owners, metadata, all model weights/optimizer/activation/lookup memory). A 97,259-parameter FP32 model stores about **389,036 bytes of weights alone**. Neither microseconds/sub-milliseconds, microcontroller fit, edge battery life nor safety certification has been measured and qualified. An associative prefix can have O(log T) theoretical scan span at fixed matrix dimensions; the entire neural graph, GPU/CPU implementation and optional CED/replay do not inherit that property automatically.

## Proposed source-to-output computational architecture

Input: original station/robot/traffic observations with measured units, source ID, chronological timestamp, masks and **past** executed actions only when genuinely recorded. Verify field authority and as-of time. Map to normalized source-specific e_t and feature channels.

**Shared causal state engine:** source/mass affine scan and causal evidence prefix (optional per authentic source) -> provisional fast/slow features -> input-only 8x8 gated-delta affine update `M_t = M_(t-1)P_t + Q_t` -> chronological associative scan for batched training / one physical rank-one update per online observation -> causal read-before-write retrieved state. Keep owner/version validation and source masks. Source gating may not secretly depend on the previous matrix read if exact affine prefix equivalence is claimed.

**Domain conditional readouts** (all optional individually learned or jointly trained *only with an honest full multi-domain fit and parameter budget*):
- `sensor`: missing-current-state representation, channel-specific observation uncertainty and calibrated observational multi-horizon forecast (no imaginary observed values).
- `physical`: source-known executed motor commands and physical trajectory dynamics; compare actual persistence, velocity and servo before claiming neural gains.
- `tracking`: source-grounded traffic actor identities/reappearances with meaningful real long-history headroom.
- `reactive`: hazard/risk score inferred from qualified factual state, **read-only**; authority and actuation remain outside the network.
- `enterprise`: categorical event/event-time forecasts, conditional on authentic event histories, as an independent typed head in priority B.

If pursuing a single multi-task checkpoint, pre-register shared trunk + domain-specific adapters/heads with all weights counted within 95k–105k deployed AND trainable parameters. Compare against separately trained models and strong per-domain simple baselines under equal original data access/observations, compute allocation and per-domain sample exposures. It is a **hypothesis** that parameter sharing will improve anything; pollution and robot labels cannot be spliced into fake joint trajectories.

## Research phases and falsifiable gates

**0. Source/benchmark preflight (no fitting).** Freeze previously exposed ALOHA, Beijing2015, traffic original DEV evidence; keep Beijing 2016–17 untouched, and locate truly independent original robot/sensor/traffic sites or periods. Declare genuine source-action availability, overlap, missingness, measured horizon units, domain score and licensed/sourced assets. No synthetic success claims.

**1. Unified operator and adaptor engineering (no fitted superiority).** Reuse one mathematical source/matrix prefix implementation, typed mask/identity/clock contract and PNDS proofs; derive proper observation/future uncertainty. Set source-dependent heads without silently copying incompatible physical laws. Prove exact-real associativity and no-future-leakage under explicit premises; test source-to-output/gradient FP32/FP64, carrying chunks, multiple entities and immutable checkpoint replay. Whole-network Lean/IEEE remains OPEN.

**2. Capability-first real experiments.** Register and actually train complete models in two **distinct** authentic domain cohorts—start source sensors (because they already have recorded gap failures), next robot motion/contact with known executed action timing—under 95k–105k actual trained/deployed learned params, <=1% matched head-to-head model size, identical real data/splits/presentations, optimizer/tuning privileges and **training compute budgets**; CPU wall-clock runtime is an OUTCOME not a parity gate. At least five original models and all failed checks; no protected TEST until preregistered comparison. For sensors: conditional NLL, calibrated gap/daily intervals, gap/daily six-channel proper scores and practicality vs last-value, ridge, appropriate classical filter, matched GRU/SSM/Transformer. For robots/traffic: movement vs stillness, source identity, true long-memory read benefit, original physical priors and matching strong controllers' *forecasts* only; no autonomous action claim.

**3. Multi-task sharing versus specialised checkpoints.** Only after baseline skill, compare a genuine shared-weights multitask candidate with per-domain single-task counterparts on same original observation/data budgets and equal compute/params; publish negative transfer, calibration loss, interference and parameter/latency overhead, not just mean averaged metrics. Keep weights specialised if joint sharing is worse. Prove useful long history vs 8-hour/frame context before crediting attention/associative-memory compression.

**4. Edge empirical qualification.** Reuse frozen qualified weights; compare FP32, FP16 and INT8 where supported on named physically measured device(s). Record p50/p95/p99 per-event latency, burst behavior, CPU/GPU usage, activation RAM, flash/model bytes, energy, model input source latency, numerical error and **full-domain heldout predictive quality**. Do not assert real-time or submillisecond speed until met. Keep all external ring/map/KV/metadata memory included. Genuine hazard avoidance requires independent physical safeguards and separately authorized testing.

**5. Expand only when evidence warrants.** If both physical/sensor tasks qualify with no unexplained negative transfer, add ES-WAM traffic tracking; later typed enterprise event forecast and intervention-specific counterfactual decision work. Optional geometric CED, conditional lookup, richer uncertainty regimes and hardware fused kernels are competing ablations, NOT compulsory permanent submodules.

## Governance, decision and archival safeguards

- Retain exactly six externally visible original family identifiers and version references; this is **architecture consolidation** not deletion of research or merging checkpoint tensors. Do not change code registry/PNDS manifests yet.
- Keep every old original ME/DEV/ESWAM/DIAGKEY training/proof/result/checkpoint and failure; the original weak/failing gates stay failed.
- Preserve PR393 ME099 full gate FAIL and PR397 ME101 full gate FAIL; no new fits are performed by this direction.
- User-facing capability description: **predictive state estimation and distributional forecasting from real temporal evidence**. Do not claim general reasoning, causal alternative-action outcomes, verified actuator safety, device-class supremacy, trained domain transfer or an LLM replacement.
- Experimental progression and promotion remain subject to REAL-WORLD-EVIDENCE-RULES.md, CAPABILITY_PROGRESS_TRUTH_STANDARD.md and PNDS dual-language mathematical engineering. One active domain family or profile may configure the same core but cannot borrow evaluation wins from another domain.

Original sources: [six-family registry](../parallel-six/active-variants.json), [consolidation](../parallel-six/CONSOLIDATION.md), [ME099](../mathematical-engineering/me099/RESULTS.md), [ME101 PR397](https://github.com/FPC-effortless/TAC-OSM/pull/397), [ME100 PR395](https://github.com/FPC-effortless/TAC-OSM/pull/395), [ESWAM QUAL001 PR386](https://github.com/FPC-effortless/TAC-OSM/pull/386), [Concept CED PR331](https://github.com/FPC-effortless/TAC-OSM/pull/331) and the [Unified Master Research Plan](../MASTER_RESEARCH_PLAN.md).
