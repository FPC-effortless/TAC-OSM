# Research Ideas Registry — TAC-OSM and related programs

This is a **starter registry in TAC-OSM**, pending creation of a separate cross-project ideas repository. The registry does not claim that proposals were implemented or evaluated.

| ID | Idea | Origin | Status | Evidence or next test |
|---|---|---|---|---|
| TAC-001 | Direct associative retrieval for physical forecast | ME-054 review | Proposed | Ablate on authentic robot trajectories |
| TAC-002 | Multiscale learned retention | MTSK transfer | Proposed | 128-state matched comparison |
| TAC-003 | Prediction-error responsibility writes | MTSK transfer | Proposed | Key collisions and long-horizon error |
| TAC-004 | Entity-local memory updates | MTSK transfer | Data-dependent | Requires reliable persistent IDs |
| TAC-005 | Bounded norm projection versus tanh compression | ME-054 mathematics | Proposed | Memory stability and forecast MAE |
| TAC-006 | Concept routing diversity and specialization | ME-056 | Diagnostic proposed | Effective rank, entropy, error attribution |
| TAC-007 | Cross-context concept attention | LLM routing analogy | Proposed | Equal-budget routing ablation |
| TAC-008 | Adaptive recurrent decoder depth | LLM latent scratchpad / CED | Proposed | Per-pass held-out forecast and cost |
| TAC-009 | Hierarchical latent-first dynamics | JEPA/world models | Proposed | Action-conditioned multihorizon real forecasts |
| TAC-010 | Elapsed-time liquid-inspired recurrence | LTC/CfC | Conditional proposal | Irregular timing analysis first |
| TAC-011 | Memory/working-state role separation | KV-cache analogy | Proposed | Retrieval and interference tests |
| TAC-012 | Counterfactual decision-preserving retention | TAC-OSM research | Research program | Logged actions, costs, identification limits |
| TAC-013 | Bounded verification and correction | ME-054; DEV-035 | Existing principle | Compare real correction errors |
| TAC-014 | Hardware-efficient recurrent execution | LLM/SSM engineering | Proposed | FLOPs and wall time, secondary to capability |
| TAC-015 | Structured entity-factorized state | DEV-017/020 | Historical baseline evidence | Verify protocols and replicate real-data relevance |
| TAC-016 | Fully learned neural action/stop control | DEV-058/061 | Synthetic evidence only | Real decision-cost validation |
| TAC-017 | Native and Concept-CED real-world benchmarking | ME-054/056 | Developmental results | Persistence beats both reported MAEs |
| TAC-018 | Retrieval versus retained-state diagnostic | Berkeley stale binding analogy | Proposed | Probe memory contents vs actual readouts |
| TAC-019 | Unproductive refinement loop detection | RADAR analogy | Proposed | Pass-wise prediction gain |
| TAC-020 | Long-horizon execution reliability | NVIDIA long transduction analogy | Proposed | Real long-history error and action robustness |
| CROSS-001 | CDL/Hebbian-style relevance learning | Cross-project idea | Unverified | Formalize update and benchmark |
| CROSS-002 | CASM decision mechanism outside ML | Cross-project idea | Unverified | Define domain-specific prototypes |
| VER-001 | Absorb MiMo RL-environment design into Veritas | Veritas research | Proposal | Verify repo design and dependencies |
| CROSS-003 | Unified reusable GitHub skills | Cross-repo workflow | Proposal | Repo-wide compatibility audit |

## 2026-10-22 portfolio-aligned capability work items

Canonical sequence and acceptance gates: [MASTER_RESEARCH_PLAN.md](MASTER_RESEARCH_PLAN.md#ordered-integrated-capability-program--supersedes-the-previous-component-first-sequence). These items supersede any implicit demand to run TAC-001 through TAC-020 sequentially. Existing TAC research ideas remain optional and **only** activate for a diagnosed integrated capability blocker.

| ID | Rank / target | Candidate external integration or data lane | State | Falsifiable proof required / do not claim |
| --- | --- | --- | --- | --- |
| CAP-001 | P0 / real persistent state + factual action-conditioned forecasting | Authentic independent robot time-series and ME059 registered comparison | Partially implemented, not yet qualified | Sealed real future-state performance exceeds persistence/servo reference and a fully matched GRU; correct pre-feedback state, parity receipts |
| CAP-002 | P0 / uncertainty, inspect, abstain and cost-sensitive intervention | Real supported action/outcome/cost logs, possibly approved Woyengi operational pilot | Not established in real operation | Lower net observed cost than best fixed/no-action policy, justified causal support, calibrated uncertainty, safe escalation |
| CAP-003 | P0/P1 / decision-relevant long-term entity memory | Verified identity-linked temporal history (not MovieLens coordinate recursion) | Partial synthetic / mixed real negatives | Future heldout prediction or decision benefit robust across unseen IDs, long histories and seeds |
| CAP-004 | P0 / verified feedback-correction loop | Woyengi authorized timestamped event/receipt read; Veritas engineering replay | Mechanism partially implemented; real benefit not established | Measured later real error/cost reduction, strict no-leakage and preservation of independently verified facts |
| CAP-005 | P1 / hierarchical abstractions and extra thinking depth | Concept-CED, TAC-Transformer, TAC-Prime, ATLAS-CASSA, Neural-Field-Dynamics | Research ideas / candidate modules | Adaptive computation or learned hierarchy causes statistically supported downstream real gains under matched budgets; no System-2 assertion by recurrence alone |
| CAP-006 | P1 / transfer and adaptation | Nsat regional/site forecasting; other legally sourced operations | Not established | Independent unseen-site performance advantage and forgetting/calibration controls; distinguish frozen transfer from online learning |
| CAP-007 | P2 / native multimodal temporal grounding | Actual time-aligned observation streams from authorized systems | Not established | Additional modality's heldout predictive/decision lift with matched sources and rights; external encoders are separately credited |
| CAP-008 | P2 / genuinely selective compute at scale | Native CDL/routing/CASM research; real executable service traces | Mechanisms exist; C5 not established | End-to-end retrieval/index, routing, execution and verification cost scales with relevant subset while meeting utility gates |
| CAP-009 | Product / governed autonomous operations | Woyengi + KoreFrame + SafeEdge/Aegis + Veritas, after real data qualification | Integration proposal only | Traceable prediction→authorization→actual action→verified outcome; independent measurable customer benefit and strict permission boundaries |
| CAP-010 | Vertical options / logistics, workforce, Nigeria environment | Shipping, Scrappy/HCSE, Nsat, BusinessData | Proposed evaluation pathways | Confirm legally usable genuine entities, histories and targets first; prediction-only datasets cannot validate intervention |
| CAP-015 | P0 / native categorical enterprise state and event-time prediction | [ME069 draft PR #348](https://github.com/FPC-effortless/TAC-OSM/pull/348), authentic BPI2012 loan-process log; five paired seeds, original Git-owned checkpoints and complete clean-checkout real score replay | **TRAINED/REAL-SCORED; local REPO_REPLAY PASS; cross-CPU numerical exception retained; joint superiority gate FAIL** | Native TEST next-event log loss .122626 vs empirical .348946 and matched GRU .119182; accuracy 94.77%, TAC 1/5 seed wins vs GRU. Waiting-time MAE 6.3561h vs empirical 6.4953h; wide 230.12h nominal90 intervals. One GitHub replay failed fixed timing tolerance; a subsequent run passed unchanged. Improve integrated history/timing model on FIT/DEV, qualify on a new authentic cohort; no actual SLA, intervention, lifetime memory or commercial claim. |
| CAP-011 | P0 / competitive action-conditioned physical world model | ME060/ME061 native trajectory/uncertainty models, PRs #339/#340 | Implemented and trained; source improvement, both final gates failed | ME061 source .603894 vs persistence .702952; new task .493206 vs GRU .509535 but persistence .455992. Next: learn residual utility under shift while preserving strong fallback forecasts; qualify uncertainty and exceed strongest prior on a newly registered task. CPU reported independently |
| CAP-012 | P0 / selective reliable physical forecast and governed Woyengi handoff | ME062 draft PR #341, original FIT preflight/trained checkpoint and frozen real transfer receipts | **TRAINED/REAL-EVALUATED; both superiority gates FAILED**; original 10 checkpoints in Git; independent model-load and real score replay PASS; science superiority FAIL | Source DEV TAC .609373 vs GRU .605164 vs persistence .702952; new task TAC 1.256390 vs GRU 1.223530, servo 1.031359, persistence 1.295032; 0/5 TAC seed wins in each; transfer nominal90 episode coverage 30%. No decision/online learning/independent-site claim. Next newly untouched task and task-shift robust strongest-prior selection, calibrated risk, plus matched no-memory/long-history tests |
| CAP-013 | P0 / causal task-shift regime arbitration with native recurrent memory | ME063 draft PR #342; original FIT 208 episodes, 10 repo checkpoints, 80 new real clothes-folding transfer | TRAINED/REAL-EVALUATED; **both superiority gates FAILED**; independent original model checkpoint audit PASS and frozen real score SHA-256 bitwise replay PASS | Source TAC .572882 vs GRU .573576 vs persistence .702952, source TAC 4/5 seed wins but block CI spans 0. Transfer TAC .828290 vs GRU .829905 vs stronger servo .793413, TAC 3/5 wins but block CI spans 0. Router raised servo weight from 40.9% to 55.1% on transfer but failed strongest-prior test; source-cal 90% interval coverage 72.25% target. No unique memory, independent-site, policy or causal claims. Next isolate genuine memory retention and better shifts with new untouched cohorts. |
| CAP-014 | P0 / multi-task real physical world-model generalization and Woyengi uncalibrated forecast guard | ME064 draft PR #343, real publisher five-task FIT 160 original episodes / source CAL 40 / untouched 80-episode spray-nozzle transfer, ten original matched Git checkpoints; Woyengi guard PR #53 | **TRAINED AND REAL-SCORED; strongest-persistence gate FAILED**. Final parameter/data/supported-op FLOP parity PASS, ten Git checkpoint SHA/load audit PASS; first independent score JSON byte match FAIL on tiny CPU float variance, strict numerical-tolerance replay pending. Woyengi CI PASS but runtime wiring not yet complete | Untouched target physical MAE: TAC .135153, GRU .148456, servo .918424, **persistence .012371** (10.9x better than TAC). TAC 2/5 paired seed wins; episode-cluster ensemble TAC-GRU CI [-.01809,-.00862] but persistence CI strongly positive. Nominal90 source-CAL target episode coverage 100% with overly wide interval mean 17.07; not qualified. Requires independent narrow-float replay, real change/intervention detection, matched memory-free/long-history controls and true policy value before any production action. |

**Audit caution:** ME-054 original recovered DEVELOPMENTAL physical MAE (TAC ~0.07881; GRU ~0.09232) loses to persistence ~0.06695 and is neither sealed independent confirmation nor fully true-FLOP-certified architecture superiority. Concept-CED ME-056 PR #332 originally failed prefit FLOP parity; its score claims require exact artifact reconciliation. Distinct anchored ME-056 PR #334, fused ME-057 PR #338 and independent-task ME-059 PR #337 must never be conflated. PR #338 is now scored: TAC 0.663284, GRU 0.658526 and persistence 0.672681 TEST MAE, with the registered TAC superiority gate failed. PR #337 still requires its own exact artifact result. No synthetic task, code-level test, audit receipt or README can promote a real-world or commercial ability.

### Hard stop and handoff to execution

For every CAP item, preserve external source licences, real identifiers and time cuts, source hashes, entity-linked original data, missing/censored outcomes, scientific preregistration and frozen tests; enforce [REAL-WORLD-EVIDENCE-RULES](REAL-WORLD-EVIDENCE-RULES.md) and [CAPABILITY-FIRST-RESEARCH-RESET](CAPABILITY-FIRST-RESEARCH-RESET-2026-10-22.md). Match **trainable parameters, real data/exposure, training and tuning compute budgets, and evaluation** before claiming architecture superiority. Report CPU speed independently (no equal CPU-time requirement). Record PASS/FAIL/UNKNOWN, negative results and independent artifact links. Data/project code availability is not proof of outcome coverage. This registry neither starts training nor authorizes writes outside TAC-OSM.

## Historical experiment index (verify source evidence before interpreting)
| Experiment | Reported result | Limitations |
|---|---|---|
| DEV-016 | Joint 73.64%, frozen writer 59.32% | Predictive reliability gate failed |
| DEV-017 | Structured 91.84%, TAC 57.90% future prediction | Experimental scope must be retained |
| DEV-020 | Structured probes 100%, TAC ~50-51% | Probe failure does not identify unique cause |
| DEV-043 | Pairwise ranking 100%, hard accuracy 50% | Calibration/representation concern |
| DEV-058 | 576/576 default, 560/576 repeat | Synthetic, no real-world claim |
| DEV-061/R1 | Reported 100% primary completion | Synthetic; persistence wins real forecasting |
| ME-054 | MAE 0.07881 vs GRU 0.09232 | Persistence 0.06695; parity needs audit |
| ME-056 | MAE 0.08037 vs GRU 0.09082 | Persistence 0.06695; parity needs audit |

## Mandatory artifact availability for any implemented CAP/TAC study

A trained research idea is **not completed** until original TAC/GRU/other model checkpoints for each seed and each recoverable failed run, plus all lawful data/metadata, config, code, dependencies, outcome predictions, score scripts, hashes and clean-checkout replay receipts are **available from this repository**. See [repository checkpoint/replay rule](REPOSITORY-REPLAY-AND-CHECKPOINT-RULE.md). Temporary GitHub Actions links or locally ignored `results/` do not count; external-provider runs have exactly the same requirement. Track `REPO_REPLAY=PASS/FAIL/BLOCKED` separately from real-world scientific and parity gates.

## Record schema for future ideas
Each idea should include: ID; title; problem; proposed mathematics; origin; relevant repositories; relation to existing ideas; hypotheses and failure modes; status (proposed/diagnostic/implemented/trained/verified/rejected); commit/PR; datasets and splits; parameter count; training budget; baselines; seeds; metrics; confidence and limitations; next experiment.

No synthetic result may be labeled real-world success. Never erase negative findings. Source code, checkpoints, reports and runs should be linked rather than duplicated without provenance.


## 2026-10-23 real robotic world-model diagnostic evidence: ME067 / ME068

These are **real-world mechanistic and source-data diagnostics only**, not new independently heldout capability successes. Both are implemented and independently GitHub Actions verified, with zero new model fits; successor evaluation must preserve a sealed physical holdout.

| ID | Evidence | Honest research status | Next decisive test |
| --- | --- | --- | --- |
| TAC-021 | [ME067 native frozen readout lesions](https://github.com/FPC-effortless/TAC-OSM/pull/346): 80 already exposed original charger episodes; full native physical MAE 0.41059; zero fast 0.60016; zero associative 0.44286; zero velocity+innovation reliability 0.29521. 20 existing original checkpoint bytes intact. | **Exposed-only mechanistic sensitivity:** native fast/associative state influences prediction, while motion reliability conditioning transfers poorly. Posthoc out-of-training-distribution readout lesions are not independent model improvement or proof of memory necessity. | New original source FIT-only, equal-resource trained causal motion/no-motion and persistence-preserving native TAC plus equally informed GRU, genuine NEW heldout original robot task. |
| TAC-022 | [ME068 real original robot action-time audit](https://github.com/FPC-effortless/TAC-OSM/pull/347): 160 original FIT episodes, 257,971 genuine autonomous transitions, about 91.23% with posthoc mean per-joint movement <0.025; one-step last-state MAE 0.07020 versus original FIT-identified servo action at t 0.12046 and t−1 0.11914. | **In-FIT source measurement, not independently heldout forecasting.** Every tested servo lag loses to persistence; one-frame past action modestly scores better across all five original FIT tasks. It does NOT establish a causal actuator delay or authorize an online future logged action feature. | Verify physical command/ack semantics; model conditional motion from actually available causal action and verified outcomes; score false-positive motion, mean physical MAE and interval sharpness on new real tasks. |
| CAP-001b | ME064/066 independent original robot transfer failed strongest persistence baseline; ME067/068 indicate potentially useful native state but fragile reliability/movement prediction. | No physical intervention-policy utility or independently calibrated safe production autonomy. Woyengi PR53 still advisory only. | Shared physical conditional dynamics head and training objective for native TAC and strong GRU, same authentic data/parameter/optimizer budget and <1% supported-operator profiled FLOPs, full original Git weight/optimizer/RNG checkpoint and independent numeric score replay, untouched physical target. |

Do not re-run historical control proofs or retroactively promote exposed-task lesions. The integrated successor must **beat the simplest strong physical predictor**, not merely match another recurrent neural network.

## ME070 new evidence and falsifier (2026-10-23)

| ID | Status | Non-negotiable next evidence |
| --- | --- | --- |
| CAP-011 — authentic 64/256-event memory over GRU | **Source blocked / no fits**: original BPI2012 TEST longest case 109 events; BPIC2018 publisher source timed out in GitHub Actions; all 256-event cohort counts UNKNOWN | Qualify legally accessible real XES with >=50 FIT and >=25 independent TEST cases with 256 observed events before original next label; register new matched `FAIR_COMPARISON.json`; kill broad memory claim if at 64 events TAC does not beat matched GRU by >=1% NLL, clustered upper CI <0, >=4/5 seeds, and learned associative memory does not add value |
| CAP-012 — original enterprise score/error overlap | **Real retrospective diagnostic measured; not new TEST**: 11–36 paired correct/wrong disagreements / 2618 cases; McNemar p .227–1.000, no accuracy edge | Preserve [original scored diagnostic](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/EXPOSED_ME069_DIAGNOSTICS.md); independent untouched score needed for promotion |
| CAP-013 — calibrated useful waiting time | **Real retrospective tail failure**: native median 90% width 4.8–10.6 sec but p99 89–192 days and maximum 2.4–3.8 years | On new qualified source record proper interval score, median/p90/p99 width, pinball loss and conditional empirical-quantile baseline; demand sharpness AND calibrated coverage |
| CAP-014 — lifecycle difficulty and true process history | **Real source audited**: TEST 38.43% same base activity, 13.06% same-activity START→COMPLETE; TEST longest original case 109 events | Use real decision-time prefixes, lifecycle-collapsed separate task, higher-order n-gram and process-specific specialist, rare transitions; no hindsight-midpoint promotion |

See [ME070 draft PR #349](https://github.com/FPC-effortless/TAC-OSM/pull/349), [source access blocker](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/SOURCE_ACCESS_BLOCKER.md), and [master-plan corrected evidence](MASTER_RESEARCH_PLAN.md). Zero new ME070 trained models or new memory advantage is claimed.


| CAP-015 — observed-prior attribution control, actual original real fits | **COMPLETED (retrospective, not scientific promotion)**: original FIT-only 1/2/3/5-event count-reference state and original exposed score file are Git-owned, independently clean-clone replayed; 5-event NLL 0.159743 vs first-order 0.348946, TAC 0.122626, GRU 0.119182 on the historically exposed ME069 TEST. Demonstrates much of improvement over first-order is generic history context, not TAC memory. | [ME070 executed ngram results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/PRAGMATIC_NGRAM_RESULTS.md); next unique-memory claim requires authentic new 64/256-event supported source, matched associative ablation, strong GRU and preregistered untouched TEST. |

**Publisher source access updated:** BPIC2018 original record Figshare article 12688355, file 24025820, size 158270992, publisher MD5 `66e54ddaa6b50d81a0166766d6f7a5d4`; multiple official delivery endpoints HTTP 504. This is metadata verification only, NOT model input source verification or actual 256-history counts. No new ME070 neural fits, no capability promotion. Historical BPI2012 after-64 histories: FIT174/DEV15/exposed TEST61; after-128: FIT4, DEV0, TEST0. Original source receipt preserved in PR #349.
