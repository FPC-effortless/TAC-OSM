# TAC-OSM six research regimes — codebase refactor and evidence map

**Status 2026-10-23:** implemented **interfaces, safety boundaries, bitemporal reconstruction,
physical forecast fallback and lane registry only** in draft research. Not six
trained models or validated deployments. This does **not** move/replace original
ME054–ME074 fitted weights, frozen test cohorts, configurations or source code.
Use the latest [canonical master plan](https://github.com/FPC-effortless/TAC-OSM/blob/research/master-research-plan/research/MASTER_RESEARCH_PLAN.md)
and [Capability Progress Truth Standard](https://github.com/FPC-effortless/TAC-OSM/blob/research/master-research-plan/research/CAPABILITY_PROGRESS_TRUTH_STANDARD.md).
This branch is stacked above the live 100k guard overlay, not the main branch.

## Strategic research decision

**Shared mathematical lineage; six distinct evidence contracts.** Reuse native
fast/slow recurrence, bounded residual/delta updates, selective associative
readout and prior-aware action/event conditioning as **candidate shared
mathematics**, not an assertion that all lanes require an identical memory
shape. The present v1 adapters use the existing source modules without changing
their originally fitted tensor shapes. Different regimes may use different
memories/heads after honest training and independent real-data qualification.

- Do not force one enterprise event model to learn contact dynamics or bitemporal
  compliance rules via the same output head.
- Do not equate design specialization, speed, quantization or module removal with
  improved intelligence. Each distinct lane gets its own authentic primary
  evaluation and the next qualified checkpoint.
- Keep an operator-facing workflow product wedge as a **hypothesis**;
  no artificial single “six-variant intelligence score” or invented ROI.
- No cross-lane headline leaderboard. A log-loss percentage from a categorical
  loan workflow is not comparable to robot MAE or execution latency.
- New fitted weights in **every** lane and any trained comparator must obey
  `95,000–105,000` actual trainable AND deployed learned parameters unless
  owner formally amends the prospective policy. Compare architectures with
  additional ≤1% parameter parity, same real-source/cohort, data presentations,
  tuning privileges and training compute. CPU runtime is not an equality gate.
  Historical 37k/123k models remain immutable historical evidence.

## Variant table: purpose, actual implementation, independent qualifications

| Lane | Code/legacy source | Target measurement on original real data | What is actually established | Missing evidence / release gate |
| --- | --- | --- | --- | --- |
| **System 1 / reactive** | `variants/system1.py`; fast/slow source in ME054, ME061, ME074 | Actual observed reflex/risk proxy, first-step regret/error, false-positive intervention harm, **real measured device p50/p95/p99 latency** under actual load | ME074 fast-state readout lesion reduced NLL at age2 but harmed age6/16; shows history sensitivity, **not** high-frequency trained action utility | No direct learned execution policy, no sub-ms SLA, no trustworthy ALLOW/ABORT authority; adapter only returns risk and measured single invocation duration |
| **Physical / robot** | `variants/physical.py`; `real_robot_me060_trajectory`, `real_robot_me061_reliable`, `real_robot_me063_regime` | Genuine multi-step 14D source joint/sensor MAE/NLL, motion-event quality, contact/servo/persistence transfer under known-at-time executed actions | ME061–ME068 real source/physical experiments; ME064/ME066 transfer loses dramatically to physical persistence; frozen ME067 physical memory lesions descriptive | No proved contact-force oracle, torque response guarantee or hard-switch improvement. Default physically justified FIT-selected *persistence or servo*, residual only after own real transfer qualification |
| **Enterprise / workflow** | `variants/enterprise.py`; `workflow_me069`, `workflow_me071`, `workflow_me072`, `workflow_me073` | Real online-available 1/2/4/8-event forecasts, genuine waiting-time NLL/Brier/coverage/interval score, early branch mistakes, operator lead-time, observable verified business status | ME069 genuine BPI2012 next-event train; ME071 persistent streaming; ME072 real multi-horizon; ME073 original 99,781-param autoregressive path model; ME074 exposed comparison shows practical strong early 3-gram and small/inconsistent associative lesions | Early 2-event stage ≈40%; tail undercoverage and broad far horizon; no new untouched enterprise-site qualification, true SLA labels or actual customer ROI |
| **Temporal / bitemporal** | `variants/temporal.py`, existing `woyengi_forecast_contract.py` | Reconstruct *as-known-at* vs *as-valid-at* at any cut from real authorized revisions, correct late/backdated amendments without future data, preserve original sources | Deterministic dual-clock reconstruction and software time-cut fixture now implemented | No independently scored trained bitemporal model, no audited authentic correction cohort, no claim that Woyengi is changed or source-of-truth; rewind/replay recomputes from records, **not** a logarithmic checkpoint tree |
| **Edge-Lite / embedded** | `variants/edge_lite.py`, existing live `parameter_budget.py` | Original checkpoint ↔ quantized numerics equivalence, actual ARM/embedded hardware p50/p95/p99 latency, memory/power envelope, real sensor anomaly quality | Artifact/evidence audit interface only; historical ME055/ME056 CPU profiling concerns | No INT8 compiled model, microcontroller benchmark, <50ms guarantee or 10k/15k trained model. Quantization *reduces storage*, not learned parameter count. Smaller learned models need separate explicit owner policy amendment |
| **Counterfactual / scenario** | `variants/counterfactual.py`, existing CASM and ME073 imagined-rollout math | Real action-context conditional outcome, logged action availability, verified outcomes/propensities; prospective policy effect/regret/cost with identification | Branching simulated/observational **advisory** estimators; synthetic Veritas diagnostic experience and passive ME073 trajectories | No causal alternative-action effects, reliable action-value advantage, offline propensity correction or authorized MCTS controller; simulations cannot qualify real-world policy intelligence |

**The six-lane refactor is architecture organization, not 6 learned breakthroughs.**
ME074 only tested **frozen out-of-distribution lesions** on already exposed BPI2012
DEV. Its associative readout change is ~0.0006 NLL at age6 and ~0.0064 at
age16 (as reported in study); this is *not* proof of universal dead weight or
a prescription to delete a trained 8x8 tensor and pretend the old checkpoint
remains compatible. Fast-state readout at age2 made predictions worse with the
original trained state, but is neither a low-latency executor test nor an
argument for unconditional global fast-state removal. Any early gating,
association deletion or memory conversion would require **a newly fitted**
model under the active 100k training gate; preserve old ME073 weights.

Additional published methods and authentic evaluation designs are indexed in
[the six-lane peer-reviewed literature-to-experiment map](SIX_VARIANT_LITERATURE_TO_EXPERIMENTS.md).
Published results are prior art, **not** demonstrated TAC capability gains.

## Historical experiments attached to each research lane

| Lane | Source studies / records | Correct interpretation |
| --- | --- | --- |
| System1 | [ME054 archived study](https://github.com/FPC-effortless/TAC-OSM/tree/governance/me073-live-parameter-budget-2026-10-23/research/mathematical-engineering/me054), [ME061](https://github.com/FPC-effortless/TAC-OSM/pull/340), [ME074](https://github.com/FPC-effortless/TAC-OSM/pull/356) | Native recurrence/fast-state readout works as forecast representation under conditions; **no verified policy or edge latency** |
| Physical | [ME054 archived study](https://github.com/FPC-effortless/TAC-OSM/tree/governance/me073-live-parameter-budget-2026-10-23/research/mathematical-engineering/me054), [ME056](https://github.com/FPC-effortless/TAC-OSM/pull/334), [ME061](https://github.com/FPC-effortless/TAC-OSM/pull/340), [ME062](https://github.com/FPC-effortless/TAC-OSM/pull/341), [ME063](https://github.com/FPC-effortless/TAC-OSM/pull/342), [ME064](https://github.com/FPC-effortless/TAC-OSM/pull/343), [ME066](https://github.com/FPC-effortless/TAC-OSM/pull/345), [ME067](https://github.com/FPC-effortless/TAC-OSM/pull/346), [ME068](https://github.com/FPC-effortless/TAC-OSM/pull/347) | Real physical trajectories but persistent losses to strongest persistence and no production regime-switch gain |
| Enterprise | [ME069](https://github.com/FPC-effortless/TAC-OSM/pull/348), [ME070](https://github.com/FPC-effortless/TAC-OSM/pull/349), [ME071](https://github.com/FPC-effortless/TAC-OSM/pull/350), [ME072](https://github.com/FPC-effortless/TAC-OSM/pull/351), [ME073](https://github.com/FPC-effortless/TAC-OSM/pull/353), [ME074](https://github.com/FPC-effortless/TAC-OSM/pull/356) | Genuine exposed-development learned event and trajectory abilities; no verified downstream SLA or business completion |
| Temporal | ME069/071/073 authentic event-time ingestion, Woyengi contract proposals | **Not** evidence of real late-record revisions or as-of compliance generalization |
| Edge-Lite | ME055–ME056 compute/performance probes and ongoing [100k guard](https://github.com/FPC-effortless/TAC-OSM/pull/355) | Profiling and honest parameter inventory, **not** INT8 edge deployment |
| Counterfactual | ME049 historical action-proxy diagnoses, ME061 observed actions, ME073 passive imagined paths; DEV016/017/020 structured/memory studies | **Not** identified causal intervention effects, prospective policy utility or proven Veritas integration |

Links are cross-study navigation, not claims that historical versions have been
merged into one deployable checkpoint. ME054/ME056 forks and ME056's anchored PR #334 have
multiple historical versions; inspect the exact protocol and pinned source
before using an old result in a cross-variant comparison.

## Next engineering work in three controlled phases

### Phase A — now: isolate and prove safe interfaces (this PR)

- Shared `variants/contracts.py` and `registry.py`; isolate six entrypoints;
  no destructive moves or refits. Keep original ME069/071/072/073 model modules
  and source equations intact. All trained checkpoint hashes remain unchanged.
- Software fixtures: no lookahead, bitemporal correction cannot alter earlier
  known-as-of histories, correct persistence/servo fallback, readonly
  prediction envelopes, no causal/authorized action impersonation, and no
  false tiny-parameter deployment.
- Map authentic old experiments to lanes and add a documented
  **Capability Progress Scorecard per newly fitted lane**, including negative
  metrics and exact matched query cohorts.
- The adapters do not currently use trained neural weights autonomously: an
  authorized caller must supply an actually loaded checkpoint predictor and
  original source features. This is interface scaffolding only.

### Phase B — core trinity: prioritized NEW trained variants (future studies)

1. **Enterprise first (commercial hypothesis):** explicitly register early
   2-observation and varied-history questions; test FIT-only early history
   priors, metadata available *at record time*, learned memory gating and
   direct vs free-running event/time coupling **in one integrated native model**.
   Metadata that includes post-event or hindsight labels is forbidden. Do
   **not** cut association merely because ME074's frozen lesion is small.
   New 95–105k parameters, original 5-seed FIT checkpoints, different fresh
   authentic sealed qualification cohort and proper all-horizon scoring. New
   Woyengi read-only adapter after actual authorization, no unsupervised SLA.
2. **Physical in its own lane:** test genuine changes/contact events from
   timestamped measured sensor/motor data with observed actions. Keep actual
   persistence and FIT-selected causal servo references; test any learned
   regime switch on new untouched real task. Do not presuppose servo beats
   persistence or that task contains labeled high-contact forces. Original
   100k fit/controls when superiority is claimed; Git checkpoint/replay.
3. **System1 only after real policy/telemetry support:** one-step hazard
   predictor and explicit operator authorization boundary. Proposed 20–35k
   neural scale is **not authorized** by the 100k rule; treat low-latency
   inference as a separate optimization or seek a prospective signed policy
   amendment. Never let model confidence serve as an ALLOW/ABORT approval.

### Phase C — Temporal, Edge, Counterfactual (only upon evidence/source readiness)

- **Temporal:** fit/prove actual known-at revision streams and date-as-of
  outcomes from authorized Woyengi-like records; verify reconstitution
  invariants before measuring future state and compliance loss.
- **Edge:** qualify hardware source and power, then freeze a 95–105k parent,
  use approved quantization with exact score preservation; smaller learned
  deployment requires explicit rule amendment before new fit.
- **Counterfactual:** use verified action availability and outcomes; report
  factual action-conditioned model accuracy first. Synthetic Veritas
  policies support failure exploration only. For causal difference / net
  decision value, preregister propensity/randomization and authorized pilot;
  no automatic real execution from TAC scores.

## Traceability and scientific acceptance

Each lane's next ME study **must** register original source/version/hash,
approved true-time observation and target contract, independent cases and
censoring, sealed/test exposure, trainable/deployed parameter budget,
original controls and strong practical reference, active 100k pre-step
guard, original checkpoint bytes + optimizer/RNG and independent checkout
score replay. Architectural superiority retains ≤1% parameters, matched
data/exposures and training/tuning compute; single-architecture capability
development may proceed without routine component ablations. No blanket
System1 sub-ms, Physical torque, Enterprise SLA, Temporal regulatory,
Edge embedded-speed or Counterfactual causal claims until measured.

**Investor ledger** remains orthogonal: independent customer discovery,
authorized read-only demonstration, genuine prospective pilot and measured
operator costs; never promote a source-log END into verified business
closure or a synthetic simulation into customer ROI.

**Merge policy:** draft/isolated six-lane code PR only. Not merged into
default branch, not a signed deployment of Woyengi/Veritas, and no trained
six-lane model weights created by this refactor.

## ME078/ME079 trained physical capability update

Two integrated recipes,30original real fits including matched GRU/causal-attention controls. Native motion forecasting and factual movement discrimination improved on exposed source data. Stationary preservation and cross-architecture superiority remain failed; independent transfer/control/customer value unqualified. Read the original [ME078 results](mathematical-engineering/me078/RESULTS.md) and [ME079 results](mathematical-engineering/me079/RESULTS.md), with every model/optimizer/RNG checkpoint in Git. Both local clean-clone replays pass; both external strict replays retain small derived interval-score exceptions. These are physical research models, not six deployed domain engines or Woyengi authorization.
