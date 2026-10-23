# ME075 — System 1 native fast recurrence + independent gated-delta core prototype

**Effective 2026-10-23. Status: CODE IMPLEMENTED / NUMERICAL TESTS, NO ORIGINAL ME075 TRAINED FITS.** Strictly separate (A) a viable architecture for honest native one-step real-data forecasting and (B) optional gated delta associative memory mathematical repair. These are **not** yet two scientifically validated capabilities or a 25k inference product.

## Evidence starting point

[DIAGKEY001 draft PR #357](https://github.com/FPC-effortless/TAC-OSM/pull/357) has 20 guarded authentic **synthetic-mechanism diagnostic** original trained checkpoints and an independent replay report, with five paired seeds and heldout random-key fixtures. Means for native TAC on distractor lengths16/64/256: **3.75%/4.492%/3.4766%** versus causal transformer **99.0625%/98.7891%/98.8672%**, near 3.125% chance. The older pre-live-guard failed run is separately quarantined. All architectures' whole-compute budgets still differed ~18–21% under partial profiler coverage, so **not a fair budget-matched architecture superiority study**. This shows a conspicuous failure of the tested TAC training/feature architecture at controlled key binding, not a universal memory theorem or real-world inability of every variant.

ME074 original BPI2012 enterprise frozen readout lesions showed fast zero *helps* early exposed h1 but *hurts* late h1, and 8x8 association-zero had small mixed effect. That diagnosis does NOT validate immediate policy intelligence or prove safely removing associative memory from all systems. Enterprise/Physical can perform genuine context-conditioned forecasting even if they lack arbitrary 256-step random key recall. Do not block unrelated real predictive work due solely to a synthetic mechanism deficit.

### A. System1NativeFast candidate, not 25k trainable

- **Code:** `src/tac_osm/variants/system1_native.py`.
- **Original future-fit size:** **100,541 active trainable/deployed** parameters for original recorded 14D joint/source-state and 14D issued-action interface; no dummy padding or disconnected heads. This respects current **95,000–105,000** new-model budget; the owner's suggested **~25k** fast-executor is a future size-policy amendment / derivative candidate, **not** silently approved for a new fit.
- **Observed input at time t:** original continuous `x_t`, **actually available** recorded issued `a_t`, elapsed time since the *previous known observation* (nonnegative). No `x_(t+1)`, future issued action, status determined after takeover, or oracle event in current features. Existing ME068 found original action/observation lag interpretation potentially fragile; do not represent the logged command as a proven causal physical force or executable policy.
- **State:** 96D bounded fast state; no 8x8 associative memory, slow state, attention cache or multistep simulated planner. Per-entity streaming state must never cross independent episodes or identifiers.
- **Readouts:** anchored one-step continuous mean `x_t + delta_t`, positive predictive scale, and an **unqualified auxiliary event logit**. The latter has **no authentic learned risk semantics** until specifically sourced risk/event labels and supervised score evaluation exist. No ALLOW/ABORT permission. No SLA latency inferred from small parameter size.
- **Training/score contract once qualified:** actual valid `x_(t+1)` at physically continuous authority windows; Laplace proper NLL, coordinate MAE, movement vs quasi-static slices and valid-next-state denominators, source FIT-only last-state/servo reference, entity-level paired confidence and seeds, per-original episode heldout sequence and hardware p50/p95/p99 inference timings. Real policy outcome/utility and causal identification remain **separate additional gates**. No real fit/scores in this refactor.

### B. Gated delta 8x8 fast-weight memory math

[ICLR 2025 Gated Delta Networks](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html) already supplies this mathematical approach. The new `gated_delta_memory.py` is an isolated **nonfitted** module:
```
M_t = alpha_t * M_(t-1)
      + beta_t * [v_t - (alpha_t * M_(t-1)) k_t] outer k_t
q_t = M_t k_query
```
where `k` is normalized and `alpha,beta` are distinct [0,1] retention/write gates from known input. It explicitly updates only what the old key would read and can overwrite a binding in an orthogonal key space. A random 16-key target set projected into eight dimensions and 48 distractor keys are **not orthogonal**; neither 8x8 capacity nor training objective guarantees reliable arbitrary retrieval at delay256. The full name `GatedDelta8x8` refers to arithmetic and candidate parameterization, not a demonstrated improvement to TAC or guaranteed Transformer parity.

**Numerical unit tests only:** correctness of exact unit-key value overwrite, no cross-key corruption for orthogonal keys, ability to erase, rejecting invalid/missing gates, query event not written, finite gradients, check 100541 native S1 params and no future-label input. These tests do **not** train a model or score 16/64/256 keyed recall.

## Hard scientific and commercial decision rules

- **ME075 engineering PASS:** all new numerical/contracts tests pass on a clean GitHub Actions checkout, and the actual System1 model has 95k–105k live count. Call this `ENGINEERING_VERIFIED`; stop there unless a real primary task is scored.
- **ME075 trained real capability:** only after lawful available original source FIT dataset and at-time action/exposure qualification, actual live-guarded `budgeted_optimizer` training with real actual source-derived outcomes, five seeds if used in gain assertion, **all original .pt model+optimizer+RNG / raw genuine forecast/score bundle committed in Git**, independent clean checkout original scoring replay, honest real-case denominator. Follow [research evidence policy](../../REAL-WORLD-EVIDENCE-RULES.md). No synthetic training counts as this.
- **Before any new synthetic DIAGKEY re-training:** existing single synthetic-mechanism owner exception is registered for **DIAGKEY001**, not a transferable license to new unlimited synthetic fitting. Require separate explicit prospective exception; then a *frozen preregistration*, genuine trained new 100k GatedDelta model including all active heads, exact source schedule, original checkpoints, clean-replay and compute qualification. A >90% keyed score on synthetic demonstrates a **mechanism learned**, **never** real-world intelligence, physical policy, memory superiority or investor proof.
- **If claiming a superior architecture:** matching real sources, event-age/target, trainable and deployed parameters within **1%**, same real FIT data/exposure, optimizer/tuning compute/FLOPs, actual identical test conditions. CPU time is efficiency not a parity gate. For developmental improvements use the strict [truth standard](https://github.com/FPC-effortless/TAC-OSM/blob/research/master-research-plan/research/CAPABILITY_PROGRESS_TRUTH_STANDARD.md).
- **System 1 readiness remains UNKNOWN:** no real fitted native fast-only model, no healthy-risk calibration, no authoritative action permission, and no measured sub-ms latency here. The 6-lane code/advisory PR remains independent and draft; this proposal branches from it without mutating historical ME075 negative source or Woyengi/Veritas.
- **YC statement must not claim:** “our gated delta model now matches Transformer at fraction of compute.” No such model-trained/compute-matched result exists. Accurate statement: “We identified a narrowly scoped synthetic associative-recall failure, preserved all original models, and are engineering a revised memory mechanism and a separate honest one-step state predictor for real-data validation.”

## First real training preflight (NOT executed)

Use source-qualified original [ME068 physical FIT-only audit](../me068/PROTOCOL.md) and genuine publisher ME064/066 episodic time/action validation **only if** a correct record-time `x_t,a_t -> x_(t+1)` contract can be independently established. Registered original source SHA `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776` is historical verification, **not newly downloaded original source bytes**. Original 160 FIT episodes / ME068 reported observations do not produce a fresh untouched trial cohort. Acquire lawful original source and audited new future-outcome qualified episodes before making any generalization claim; never open final reserved tissue-box data during engineering. Use FIT for construction/tuning and a new pre-registered untouched real task or source for qualification; do not run an action-permission experiment without external authority.

**End of present PR scope:** safe engineering, no new fit, no action deployment, no proof of increased predictive intelligence.
