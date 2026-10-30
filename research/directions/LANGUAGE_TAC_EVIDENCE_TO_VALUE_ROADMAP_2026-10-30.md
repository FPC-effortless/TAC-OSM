# TAC-OSM Language Research: Unified Evidence-to-Value Roadmap (2026-10-30)

**Status:** PROPOSED documentation-only strategy; no new training, protected evaluation, model promotion, merge, deployment, spending or commercial claim.  
**Integration target:** the existing six-family `research/master-research-plan` program. Language is the **shared architecture testbed**, not a seventh family. Earlier model versions, proofs, checkpoints and negative results remain immutable.

## Decision and falsifiable thesis

Prioritize TAC-OSM language modeling as the fastest source-authentic environment for developing and comparing an efficient recurrent/parallel, persistent-state architecture. Success is **not** merely low parameter count or subquadratic asymptotic cost; it is a demonstrable **quality/latency/memory/total-cost Pareto improvement** against strong implemented alternatives under equivalent information and budgets.

- Short-run goal: close an actual generalization and code-utility gap versus GRU and strong count/raw-history baselines while reducing TAC decode overhead.
- Medium-run goal: establish an actually trained and independently replayed 10M model that reaches a meaningful Pareto point against 10M GRU, selective-SSM and Transformer controls on legitimate mixed-language data.
- Long-run goal: qualify an efficient specialist model for real phone or fast cloud use, and only then consider 100M+/foundation-model-scale investments.
- The six existing TAC families (System 1, Enterprise, Temporal, Physical, Edge-Lite, Counterfactual) share underlying mechanisms/PNDS mathematics where source-compatible, but remain separate real-world capability and deployment qualifications.

## Locked facts as of October 30, 2026 (no cross-study leaderboard mixing)

**CODEVALUE004** — [TAC draft PR657](https://github.com/FPC-effortless/TAC-OSM/pull/657):
- 24 genuine T4 fits, 120 candidate checkpoints authenticated, 24 DEV selections, 280 scores and 143,360 suggestion records preserved/replayed.
- TAC small+constant transfer NLL 2.3380, accuracy 48.85%; broad+cosine TAC NLL 2.0279, accuracy 52.67%. Paired NLL improvement 0.3101 with 95% CI 0.2048–0.4231, positive across five publishers and three seeds; corpus size and diversity both changed.
- Under the broad corpus, GRU NLL 1.8373 and accuracy 55.42%; WB6 count predictor accuracy 56.01% but NLL 2.4094. Different metrics select different winners. All TAC code-completion utility gates FAILED.
- The isolated cosine-schedule NLL effect is uncertain (CI crosses zero). This is **internal DEV** generalization, not external blind TEST, commercial usefulness or a qualified architecture winner.
- Comparisons between differently evaluated studies (e.g. prior CODEVALUE003) must not be represented as one exact paired gain.

**LANG10M001** — [draft PR658](https://github.com/FPC-effortless/TAC-OSM/pull/658), [tracker659](https://github.com/FPC-effortless/TAC-OSM/issues/659):
- 10,001,188-parameter candidate is **UNTRAINED**. 4,096-piece byte-fallback tokenizer roundtrips 219 authenticated FIT files, about 3.63MB, 1.17M one-pass tokens.
- Two physical T4s and 20 repaired capacity/gradient shapes verified. At batch 2 x length 1,024: FP32 forward+backward 26,622 tokens/sec, peak allocated GPU memory 0.627 GiB; AMP-FP16 27,542 tokens/sec, 0.414 GiB.
- Those measurements exclude optimizer stepping, dataloading, synchronization/distributed overhead, and actual full training; peak **allocated** GPU memory is not complete process/system VRAM.
- The original AMP concept-solve dtype failure was preserved and repaired in the scoped 10M candidate. Unscaled AMP still has entirely zero gradient tensors in 8/10 shapes; GradScaler/full-gradient parity and actual optimizer stability remain OPEN. FP32 is conservative.
- PNDS PR94 partial checks do not constitute full source-forward/gradient/IEEE formal equivalence. No 10M language-quality result, English benchmark, real mobile deployment or customer benefit is established.

**Historical, not prospectively confirmed here:** LANGLAYER040 graph-free 48D layered candidate is a useful ~500K branch; reports of roughly 3.92 ms/byte TAC versus 0.33 SSM and 0.17 GRU are a hypothesis requiring exact-code, matched-host, tokenizer-aware reproduction before use as a funding statistic. Historical fast/slow/8x8 associative and memory results must be version-identified; do not assume the 10M graph is identical. Original 1,344-byte raw-history ring strongly outperformed same-size lossy hash tables on a historical diagnostic; treat raw storage as an essential baseline. Cache/raw-history advantage is not yet autonomous address accuracy.

## Research portfolio and work order

### Gate A — Evidence, exact graph and resource truth (P0, no fit)
1. Capture source SHA, model class, parameter decomposition, actual fast/slow/8x8/concept/cache pathways, tokenizer, history length, KV/cache/state bytes, head and output objective of the specific 500K candidate and 10M candidate. Verify rather than infer retained mechanisms.
2. Freeze all source rights, source-boundary separation, FIT / tuning DEV / reporting DEV / transfer publisher / untouched TEST rules; exact tokenizer fitting scope and cross-source dedup/leakage checks.
3. Register legal scale exceptions **explicitly**: the general TAC prospective 95K–105K fit rule is binding by default; a documented owner-scoped qualified language exception is required before any 500K/10M optimizer training. A prior trained 500K result does not itself grant a 10M training exception.
4. Prepare fair-comparison manifests, per-source and per-seed score tables, full original model/optimizer/RNG checkpoints, independent clean-checkout replay and source-pinned PNDS Python+Lean obligations. Use anti-cheat and reward-integrity protections where applicable.
**Exit:** audit-ready preregistration and source/model manifests; no implicit authorization to train.

### Gate B — Decode cost engineering (P0/P1, no new model training)
- Re-run native incremental one-token decode on the exact saved 500K checkpoint and matched GRU, selective SSM, Transformer and raw-count controls on **the same device, thread limits, precision, state reset, input history, tokenizer and sampling policy**; report model differences clearly when byte/token comparisons are not semantically equivalent.
- Collect warmed p50/p95 ms/token, normalized ms/byte, cold-start, tokens/s, temporary peak RSS, persisted state bytes and model weights, CPU cycles/operator trace, context-length sweep (128, 1K, 4K, 16K where executable), batched and single-stream paths. Use a physical Android phone for device qualification, not CPU simulation.
- Diagnose Python launch/dispatch, per-token allocation, repeated tiny GEMMs, concept solver, recurrent carry, memory reads, normalization and logits; optimize **without changing predictions** first through preallocation, fusion, compiled/static loops and appropriate numeric parity. Evaluate custom C++/Triton only against a measured dominant bottleneck, not as a prestige requirement.
- Exploratory objective: TAC below 1.0 ms/byte on the same single-thread CPU environment; **target, not achieved claim or sufficient competitive criterion**. A slower but more accurate/low-RAM Pareto solution remains possible. Report the strongest baseline as well as aspirational threshold.
**Exit:** exact model/logit/state parity, independent repeatable time/energy/memory profile, with verified bottleneck attribution; do not confuse PyTorch compile speed with Android viability.

### Gate C — Mathematical state, memory and prediction engineering (P1)
- Maintain a stable recurrent/parallel native baseline. Prove/validate exact causal transition, stability, boundedness, update ordering, gradient propagation, precision behavior and cost in PNDS (Lean 4 and executable Python matching actual source tensors).
- Affine input-only transitions may admit associative scans; state-dependent nonlinear operations generally do **not** become associative by declaration. Separate parallel precompute/scan components from genuinely recurrent steps, with exact sequential-streaming equivalence tests.
- Use source-pinned ablations for fast/slow state, the 8x8 associative matrix, upper recurrent layers, concepts and graph. Do not reintroduce a symmetric graph whose aggregate cancels at readout. Prefer 48D graph-free layered candidate unless stronger, matched evidence displaces it.
- Test a **dual cheap memories + selective expensive adjudication** proposal only if the current failure is a real autonomous-retrieval weakness. Compare two specialized bounded histories (e.g. short raw recency ring plus longer keyed/associative evidence) against raw-ring-only, cache-only, GRU/SSM state and an equal-budget combination. Report query/address precision, wrong-key negative control, age 8–31 and older, retrieval provenance, saved information, predictor NLL AND actual useful suggestions, cost of adjudication and accuracy-cost frontier.
- Use conditioned expensive verification selectively with a learned calibrated confidence gate; random verification at the same trigger rate must be a control. Historical uncalibrated verification and lossy table failures are negative priors, not features to assume solved.
**Exit:** matched, source-authentic gains in future prediction/utility and independent source-graph/math tests; no raw-memory inflation concealed in persistent-state claims.

### Gate D — 500K matched prediction and code utility (P1; training only after gate A)
**Primary prospective protocol:** start with the best source-pinned eligible graph-free layered candidate and CODEVALUE004-proven broader data recipe. Compare TAC, GRU, selective-SSM, causal Transformer, strong WB6/count and raw-history retrieval control. Count/trainable parameters within 1% for neural arms, include all active model weights, equal original source/splits/tokenizer and exposure, equal tuning opportunity; report exact hardware, implementation/kernel efficiency, optimizer behavior and estimated+measured FLOPs.
- Use **two clearly separated comparison budgets** when feasible: equal tokens/updates and equal actual training FLOPs. Do not present unequal-compute rows as matched-FLOP wins. CPU wall time remains an efficiency outcome, not a scientific matching requirement.
- Minimum five matched seeds for definitive 500K model comparisons, multiple independent held-out publishers, training curves, paired confidence intervals and no selection on sealed TEST.
- Score NLL, Brier/calibration, top-1, exact-4, real incremental editor-completion acceptance/compile+test or other verifiable real-user usefulness, decode p50/p95, persistent bytes, and cost per accepted suggestion. Measure whether a model can improve utility, not just perplexity.
**Exit:** credible demonstration of quality parity or superiority at competitive device/cloud cost, or a documented negative result and diagnosis. No architectural promotion merely from one metric.

### Gate E — 10M capacity to **trained** scaling study (P1/P2, data and authorization first)
**Do not launch a full pretrain directly from the 1.17M token pilot.** Incrementally authenticate a licensed, deduplicated FIT corpus of diverse English, code, documentation and structured instructions, with immutable source-manifests and publisher/time-grouped DEV and held-out transfer cohorts. A 200M+ token budget is a **planning scenario**, not a scientifically determined minimum or guaranteed enough for TAC. Trace learning curves and data/compute-optimal tradeoffs; compare at 10M, intermediate 25M–50M and larger exposure *only if justified*.
- Freeze tokenizer and fit only on FIT. Audit differing token-per-byte rates and include byte-normalized quality; any alternative tokenizer is a separate controlled study.
- Implement actual 10M neural GRU, selective SSM and Transformer controls within 1% active parameters (as far as feasible), frozen data/exposure/tuning, equivalent source/holdout protection and fair compute accounting.
- Validate FP32 optimizer-step reproducibility, optimizer+activation+peak VRAM and sustained measured tokens/sec with data loading. Investigate the AMP zero gradients with per-parameter norms, loss scale, underflow and source-node AD; require forward/backward/full-gradient parity across relevant shapes and successful short update stability before using FP16. Do not silently substitute BF16 on T4.
- Run a limited authorized seed pilot and scale to full matched 3–5 seeds only if the registered uncertainty, quality and cost thresholds justify it. Preserve early-stop losers and complete replay, including failed runs. Do not extrapolate hardware profiles to full training costs without measuring full optimizer steps.
- Evaluate long-context retrieval/state tracking at source-verified real sequence lengths and measure memory state-size scaling; 10M upper-layer capacity alone does not prove long-term memory.
**Exit:** independently replayed actual 10M trained TAC and controls, measured quality/compute curves; no claim from forward/backward capacity tests alone.

### Gate F — Product, paper and funding claims (conditional, not automatic)
- Edge: Android offline actual native/runtime checkpoint, byte-exact or documented numerical equivalence, single-core/batched p50/p95, total RAM, startup, energy, heat, model-file size, privacy/state isolation and consistency across runs.
- Cloud: concurrent load-test, full request p50/p95, throughput, total cost per 1,000 **useful** outputs, recovery, state isolation, numerical/evaluation parity.
- Product candidate: opt-in editor/code suggestion or narrow language prediction capability, grounded developer workflow, allowed licensed data, baseline replacements, observed suggestion utility and developer time saved; do not claim customers, pilots, security or revenue from unit tests.
- Paper can report architecture, full equations, source-pinned multi-seed experiments, negative controls and scoped Lean proofs even before superiority; a *positive efficiency/prediction paper* requires replicated real evidence. Prepare a pre-seed/grant narrative only with dated verified facts, credible differentiator and cost model; no promised funding amount or presumed valuation.
- Broader 100M/1B/7B scale is **not a current prerequisite** for credible pre-seed research and is not authorized simply because 10M fits in T4 memory. Consider it only when 10M improvement, data, cost and bottleneck evidence justify it.

## Priority queue for the research agent

**Now (do not fit):** (1) canonical model/version/parameter/memory graph inventory; (2) reproducible decode microprofile and matching baselines; (3) legitimate source expansion + dedup/split registration; (4) 10M FP32 step and FP16 full-gradient root-cause diagnosis; (5) unified matched-budget/proof/evidence manifests.

**Next gated fits:** conduct 500K broad-data model-vs-baselines five-seed study and 10M short authorized matched-compute pilot only after each has a complete, valid and independently reviewed registration plus data sufficiency.

**Do not prioritize:** arbitrary 16x16 associative growth, graph reintroduction without readout effect, retraining on tiny code pilot as a foundation model, investor-grade speed claims from unmatched microbenchmarks, speculative 1B scaling, or more optimizer variations absent demonstrated integrated gain.

## Success scorecard and stop/continue rules

| Dimension | Primary evidence | Stop or adjust when |
| --- | --- | --- |
| Prediction | Paired held-out cross-publisher NLL and top-1, calibration | Strong simple baselines keep winning after controlled training/data improvements |
| Task value | Compile/test-validated accepted completions, real workflow costs | Better NLL fails to yield usable decisions after evaluation |
| Retrieval | Address precision, long-delay factual recovery, wrong-key controls | Raw ring or cache dominates dual-memory design at same bytes/cost |
| Runtime | Same-host p50/p95 CPU/Android time, RSS/state bytes, energy | TAC remains orders slower without compensating value |
| Scaling | Sustained full-step throughput, 10M multi-seed learning/compute curves | Parameter growth consumes cost without accuracy/transfer benefit |
| Integrity | Model/source/checkpoint/independent replay, true full graph/precision status | Leakage, unequal budgets, missing checkpoint, unknown math boundary |
| Commercial | Third-party pilot task benefit and reproducible deployment economics | No measurable buyer/job value |

## External engineering context (not evidence for TAC)

Mamba-2 structured state-space duality emphasizes algorithms tied to hardware-efficient kernels: https://proceedings.mlr.press/v235/dao24a.html . Mamba-3 (ICLR 2026) explicitly optimizes **inference** and state tracking: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html . Liquid AI's Oct 27, 2026 LFM3 release is a much larger **~1.7B-parameter, reported 22T-pretraining-token** hybrid family: https://www.liquid.ai/blog/lfm3 . These serve as design principles and distant reference targets, **not** direct apples-to-apples 10M benchmark controls.

## Cross-repository ownership

- **TAC-OSM**: implementations, real dataset manifests, preregistration, trained+replayed comparisons, hardware results.
- **PNDS**: exact versioned source-matching Lean/Python models, associative-scan/gradient/stability/IEEE claims with explicit OPEN obligations.
- **Veritas**: independent source and checkpoint replay, evaluation integrity and eventual user-task outcomes.
- **Woyengi**: authorized operational facts/permissions when a future real product needs state/actions; TAC never self-authorizes business effects.

This document proposes a direction and gates; it deliberately **does not** override binding science, security, runner, scale or owner approval policies. Historical original results, code and negative reports remain preserved.
