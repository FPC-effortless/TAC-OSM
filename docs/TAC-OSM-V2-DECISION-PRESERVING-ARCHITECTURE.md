# TAC-OSM v2 — decision-preserving predictive-state architecture (RESEARCH RFC)

Status: **PROPOSED, NOT SCIENTIFICALLY VALIDATED**. First staged implementation tracks DEV-044–048. This RFC does not alter the original PLM/CDL/CASM/OSM APIs, any baseline numbers, or the archived DEV-015–043 results.

## Empirical starting point

* DEV-017/029/040A: structured/known-entity state and explicit public-history baselines often beat unconstrained recurrent variants; over-service can hide prediction errors.
* DEV-041: a trainable bounded MTSK recurrent writer beats **that initialized** frozen writer, on a benchmark a last-event table also solves. Tuned frozen reservoir and optimized GRU/gated-delta were not tested.
* DEV-042/043: original recurrent scores rank 96/96 paired cancellation futures directionally, but fail hard classification and selective cost. Positive Platt/temperature transformations and 0/10/50% causal-order head-only fits do not solve action selection. One head retraining seed reverses rank. Neither state absence nor sole calibration defect is proven.
* Research priority: use **cost from actual interventions** as the unambiguous endpoint, with explicit reference costs and compute. Do not promote architectural components from probe accuracy.

## Operational interfaces

1. **Local temporal encoder** `encode(events, public_entity_id, elapsed_ticks)`. Preserve order of schedule/change/inspect; do not take hidden simulator generations or supervisor labels at inference.
2. **PLM persistent state** `state = update(state, encoded_event, time_gap, entity_address, commit_verification)`. Proposed bank structure `entity: fast/slow/event`, plus optional bounded relational slots. Identity update for unrelated entity on local writes and for rejected commits. Cost budget measured as bytes and FLOPs, not just scalars.
3. **CDL relevance routing** `retrieve(state, public_query, budget)`. Logs retrieved slots/latency and avoids leaking evaluator's target ID except when publicly observed.
4. **OSM action-conditioned forecasting** `predict(state, do(action), horizon) -> distribution + uncertainty`. For maintenance, separate WAIT, SERVICE and INSPECT receipt-conditioned forecasts. Conditioning is a hypothesis, **not causal identification** without simulator interventions.
5. **Contrastive cost-sensitive policy** `cost(state, action) = failure_cost*P(failure|action) + action_cost + information_cost`. Rank action alternatives, then choose the minimum expected cost among CASM-admissible actions; logged regrets against simulator's exact independent hindsight.
6. **CASM execution** `validate(action, constraints) -> accepted/rejected`. An uncertain forecast never silently maps to WAIT; INSPECT or human review are costed fallback options.
7. **Verifier and corrective write**: log observed outcome, prediction error and intervention receipt; apply verified updates. Keep provenance and rejection masks.

The central loop is `observe → local encode → verified PLM update → CDL retrieve → OSM action-counterfactuals → CASM constraint check → cost-aware select → act → verify → corrective learning`.

## Formal training hypotheses

* Forecast loss `L_future`: proper scoring on all candidate action outcomes, not only target WAIT.
* Decision regret `L_decision`: differentiable surrogate comparing model policy to simulator-computed costs. Discrete cost `4 failures + services + 0.1 inspections` stays the independent eval.
* Pair ranking `L_rank`: margin-based cancellation-vs-pending risk and explicit hard-negative action choice.
* Retention `L_retention`: ablate/perturb memory item only using public-history-derived state; importance `E[Δ forecast loss + lambda * Δ physical decision cost]`. Never inject exact future labels into the deployed memory update.
* Calibration `L_calibration`: Brier/NLL across WAIT/SERVICE/INSPECT; separately from cost minimization.
* Budget `L_budget`: actual state bytes, update and routing latency; 768-scalar fixed memory and 41-scalar table are only specific benchmark regimes.

Register each coefficient, batch construction and optimization budget **before** the corresponding experiment. Ablate each loss against a single-task and a structured state baseline.

## Scientific gates

| Stage | Intervention / comparator | Required falsifier |
| --- | --- | --- |
| DEV-044A | initial retention / key-value scales, trainable vs frozen MTSK, with GRU768, gated-delta768 and schema41 | Frozen gain/decay configuration catches up or beats learned writer at comparable developer selection budget |
| DEV-044B | fresh seeds, convergence-matched frozen reservoir and other recurrent controls | MTSK advantage disappears with enough baseline tuning |
| DEV-045 | frozen representations: cost-trained policy, contrastive state-action head, simple rule table and calibrated threshold | Learned action head fails to beat calibrated threshold/last-event on failure-cost guard |
| DEV-046 | train-only prediction vs action-only vs joint/counterfactual vs simulator imitation; writer frozen/trainable | Better teacher agreement does not reduce real physical cost |
| DEV-047 | temporal encoder + fast/slow/event memory vs entity schema, selective SSM and masked-local TAC | Learned memory fails on unseen entity permutations, 16/32-entity stress or long delayed order cases |
| DEV-048 | verified end-to-end PLM/CDL/OSM/CASM with vs without routing, verification, GEPA dev optimizer | Full system is more costly or less reliable than tuned simplified reference |

Keep untouched confirmatory worlds, group both members of a causal pair together, maintain an external SHA-256 split ledger, archive failed/partial jobs, and count optimizer/search trials including failed ones. Three-seed pilots are NOT confirmation; require 10+ fresh initialization seeds and independently varied simulator configurations before strong claims.

## External mechanism attribution — hypotheses, not copied discoveries

* Direction-selective retinal computations motivate local temporal-order detectors, not proof of learned causal memory.
* E. coli non-genetic physical memory motivates event-persistence ablations, not neural scaling evidence.
* Selective state-space models (Mamba) motivate efficient streaming alternatives but no guaranteed event retention.
* [ActionOPD](https://arxiv.org/abs/2610.19820) motivates supervision of semantic **actions at encountered states**, with simulator cost the primary reward instead of teacher agreement.
* [Contrastive Language Models](https://github.com/Contrastive-LM/CLM) motivate disjoint state/action embeddings and hard negatives; this is a distinct proposed adaptation to non-language maintenance.
* [GEPA](https://arxiv.org/abs/2507.19457) motivates reflective search **only within training/development splits**. Its prompt optimization outcomes are not validated memory or policy improvements.

## Conditions for promoting a component

A component is not promoted without: exact code/weights/history provenance; disjoint train/calibration/test; thresholds chosen before test; independent action-physical audit; comparison with last-event, exact public-history, tuned GRU/gated delta and schema; confidence/reliability across fresh seeds; and total compute cost. An internal pair-ranking diagnostic must never stand in for economically correct decisions. Maintain all PRs as drafts until these gates are satisfied.
