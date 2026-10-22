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

**Audit caution:** ME-054 original recovered DEVELOPMENTAL physical MAE (TAC ~0.07881; GRU ~0.09232) loses to persistence ~0.06695 and is neither sealed independent confirmation nor fully true-FLOP-certified architecture superiority. Concept-CED ME-056 PR #332 originally failed prefit FLOP parity; its score claims require exact artifact reconciliation. Distinct anchored ME-056 PR #334, fused ME-057 PR #338 and independent-task ME-059 PR #337 must never be conflated. As of the master-plan edit, PR #338 and #337 contain no fully scored independent trained qualification result. No synthetic task, code-level test, audit receipt or README can promote a real-world or commercial ability.

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

## Record schema for future ideas
Each idea should include: ID; title; problem; proposed mathematics; origin; relevant repositories; relation to existing ideas; hypotheses and failure modes; status (proposed/diagnostic/implemented/trained/verified/rejected); commit/PR; datasets and splits; parameter count; training budget; baselines; seeds; metrics; confidence and limitations; next experiment.

No synthetic result may be labeled real-world success. Never erase negative findings. Source code, checkpoints, reports and runs should be linked rather than duplicated without provenance.
