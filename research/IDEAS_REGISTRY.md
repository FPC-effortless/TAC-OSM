# Research Ideas Registry — TAC-OSM and related programs

## EDGEACTION001 — real logged-setpoint training, complete 5-seed physical gate FAIL (2026-10-25)

**Implemented and actually trained in [TAC draft PR401](https://github.com/FPC-effortless/TAC-OSM/pull/401).** Five 99,760-param `ActionQueryEdgeWorld` models, each 256 real-data optimizer steps (batch8), logged source state/action inputs and **read-only hypothetical-command retrieval** atop one input-scheduled associative prefix scan. 49 source FIT episodes for fitting, 10 separate **historically accessible FIT-only** episodes for original internal score (1,840 authentic measured one-frame next states). Source action is a logged joint-position target, *not verified physical command latency or `do(action)`*. All original model/optimizer/RNG/schedule/checksum bytes committed BEFORE scoring, after a no-retrain recovery from a branch-update push conflict. Old DEV59–71 and protected TEST>=72 remain CLOSED; original models of ME099, ME100 and ME101 unchanged.

**Original measured 14D MSE:** velocity reference **4.8720e-6**; FIT-only servo **2.12093e-5**; **TAC five-seed mean 4.84951e-5**; persistence **8.78959e-5**; raw published target **3.91193e-3**. New TAC is 44.83% below persistence but **9.95× worse than velocity**, 2.29× worse than FIT-only servo, 13.57× worse moving MSE than velocity. ALL FIVE original per-seed preregistered full gates **FAIL**, despite clean original score/replay [Actions40188456161](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40188456161) PASS. No controlled neural architecture superiority, robot brain, car autonomy, simulation-based research success, policy benefit, independent site/test or calibrated future uncertainty claim. Original newer measured results: [EDGEACTION001 report](https://github.com/FPC-effortless/TAC-OSM/blob/research/shared-edge-action-implementation-2026-10-25/research/mathematical-engineering/edgeaction001/RESULTS.md).

**Required next scientific direction:** Preserve simple causal velocity/servo motion predictors as strong physical priors and test a **trained corrective residual with source-confirmed executed-command timing**, against fully matched neural models on authentic independently held-out real-device/task outcomes. No more EDGEACTION001 fits against these now exposed ten internal episodes; no TEST access. Keep shared edge physical domain priority and six original family contracts, PNDS as mathematical authority, and retain the prior SIM001 synthetic diagnostics without promoting them.

---

## SIM001 + EDGE-ACTION-001 — corrected OOD simulation and original real-FIT command-query eligibility (2026-10-25)

**Original SIM001 draft PR399/run40185878515: SIMULATION ONLY, NEGATIVE zero-shot transfer.** Five checksum-verified original ME099 trained ALOHA checkpoints in synthetic MuJoCo 2-joint arm and planar force/yaw vehicle proxy (not full car). The invalid `valid=False` padded-output/zero-width test is explicitly WITHDRAWN as uncertainty evidence. Corrected last-valid h2..51 blackout: arm forecast/persistence MAE1.763/1.820 with nominal90 coverage1.0%; planar proxy .320/.179 with coverage11.4%. Arm assisted blackout final tracking improved0.691→0.505, mean worsened1.151→1.250. Neither real safety nor neural control established; original ME099 real-data full gate stays FAILED. Total runner3–4.5ms includes Python and physics, not embedded inference.

**Implemented [draft PR401](https://github.com/FPC-effortless/TAC-OSM/pull/401):** read-only candidate-action query atop full native causal associative scan, with strict recorded owner/hour/version/history masks, tested chunk equivalence and full gradient path. Factory shares mathematical core between Physical/System1 action-query and Temporal ME101 sensor profiles but DOES NOT claim shared fitted weights. New original pinned source-only FIT audit [GitHub40187233556](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40187233556) PASS:59 authentic ALOHA FIT episodes/88,382 adjacent-frame action/state triples,14D states/commands,median frame interval~20ms, no protected TEST scoring. **Recorded `action[t]` actual motor-execution meaning and causal identifiability NOT VERIFIED**. The new candidate is 99,760 learned parameters but **UNTRAINED**; zero capability claims. Actual real action-conditioned fitting requires distinct preregistration, source semantics, matched parameter/data/compute and genuine future outcomes. Full original report preserved in PR399; source-engineering specification in PR401.

These original simulation results are diagnostics only, not proof of real model performance or a license to train on a fake physics world. Preserve all six original families and negative gates; physical/edge priority remains RESEARCH-ONLY.

---
## PRIORITY-A SHARED-EDGE-PHYSICAL — one engine, task-adaptable physical and sensor predictions (2026-10-25)

**Status: prioritized RESEARCH DESIGN only; no new fits or code registry changes.** The six original family identifiers remain for compatibility/provenance, but six independent learned cores are NOT compulsory. Physical+ESWAM trajectory intelligence, Temporal ME100/ME101 sensor/missing-source evidence, and System1 short-horizon risk/forecast share one mathematical parallel/streaming core and use domain-specific truthful observation, units, dynamics and probabilistic heads. Edge-Lite is a hardware export profile, valid/known-at Temporal provenance a deterministic authority service, Enterprise retains specialized event-time semantics, and Counterfactual remains gated nonexecuting scenario research. Exactly six documented family identities persist; PNDS remains mathematical foundation, not a seventh model.

**Compulsory evidence:** genuine different-domain trained capability, not model/weight transfer by assertion. ME099 and ME101 original full gates stay FAILED; ESWAM012 long-history negative, ME100/101 gap uncertainty and poorer NLL remain central blockers. Complete neural models must have 95k–105k real active/deployed parameters; architecture comparisons additionally require <=1% param mismatch, same observed data, matched training budget/FLOPs and untouched source/time/site cohorts. CPU runtime is an efficiency outcome only. Formal Lean kernel checks do NOT prove whole-system safety. The 608-byte core is not the device total (model weights, source clocks and optional external caches count).

**Primary research order:** sensor gap+ordinary forecast with reliable beliefs, genuine robot/contact motion forecast, traffic actor memory with independent evidence, then optional actual shared-weights transfer test versus matched specialist training and real edge hardware p50/p95/p99/energy/flash/RAM. Autonomous vehicle/drone safety, microsecond speed and hardware deployment remain unqualified. Existing enterprise and counterfactual research is archived and independently pursued only where data/outcomes justify it. CED/geometry/MoE/lookup/Muon are optional controlled mechanisms, never automatic compulsory additions.

[Master-priority decision](directions/SHARED-EDGE-PHYSICAL-PRIORITY.md) · [Master plan](MASTER_RESEARCH_PLAN.md).

---


## ME081 — learned horizon-query forecasting: stronger averages, quiet-state trade-off

[Draft PR #375](https://github.com/FPC-effortless/TAC-OSM/pull/375) trains **five complete 100,165-parameter native models**. Fast/slow/8×8 associative dynamics remain intact. New training uses 32 actual observation frames with full gradient credit, a shared horizon-query distribution decoder, and properly scored CRPS plus a small log score and factual motion supervision. This is actual end-to-end learning, not an execution-only result or attribution study.

Primary evaluation preserves the exact original eight-frame task: 40 already-exposed real REPORT episodes, 960/956/928 factual pairs at 1/4/16 frames. All figures below are means of individual models, not an ensemble deployment.

| Physical MAE, lower is better | Frozen ME080 | ME081 | Persistence |
| --- | ---: | ---: | ---: |
| 1 frame | 0.059393 | **0.055968** | 0.068668 |
| 4 frames | 0.219619 | **0.166113** | 0.302400 |
| 16 frames | 0.883994 | **0.737952** | 0.975966 |

**Improved:** mean errors fall **5.77%, 24.36% and 16.52%** versus ME080, and 18.49%, 45.07% and 24.39% versus persistence. Four-frame MAE is also 10.67% below the stronger ME079 native reference. Actual-moving one-frame MAE falls from 0.533420 to **0.408457**, below ME078's 0.421182 and clipped velocity's 0.482509 on the same 84 moving queries. Proper interval scores fall **51.57%, 38.10% and 50.50%**, with nominal 90% coordinate coverage 92.92%, 92.88% and 91.88%.

**Regressed:** stationary MAE rises **59.04%, 55.65% and 162.28%** versus ME080 at the three horizons. One-frame stationary MAE is 0.022168, about 13 times ME077's 0.001701. Physical marginal NLL also worsens at every horizon. One-frame risk AP/AUC (0.54992/0.91466) improves over ME080 but remains below the stronger historical ME078/079 rankings. Better interval scores do not establish uniformly better calibration.

All five fits improve primary one-frame mean error over their ME080 counterparts, but the fixed-model paired 40-episode bootstrap 95% interval for new-minus-old error is **[−0.012166, +0.005730]**, including zero. Quiet preservation and this confidence gate fail. **Full gate FAIL / MIXED_TRADEOFF; no further ME081 fits.** There is no independently confirmed intelligence or architecture-superiority claim.

**Actual longer-context readout:** 904 registered genuine 32-frame query histories provide 904/900/872 future pairs. Mean errors are 0.055557/0.157857/0.740098. On exactly those IDs, the same model with eight frames scores 0.055775/0.159718/0.748874. These small descriptive differences do not demonstrate a material memory advantage or arbitrary-history retention. Both information conditions and all adverse cases are preserved.

**Resource/source scope:** unchanged licensed BodenAI/RW-RL-HIL archive, 120 FIT / 40 exposed REPORT, source revision 9290f5a96dc91ea58256e81d9ce2dd9621890b51. Five cold starts, 2,048 updates and batch 24, final-only, no sweep/restart/warm start. Each model sees 1,572,864 observed prefix presentations, **four times ME080**; no matched-compute, sample-efficiency or architectural-winner claim. Protected tissue-box, source CAL and TEST remain closed. Factual action inputs are not identified causal control.

**Original proof:** prefit `b5698c86`; all five whole-model/optimizer/RNG/live receipts frozen at `dd71fee2` before either readout; original scores `bf23f52b`. Fresh remote sparse checkout restores and replays **27,600 original model forecast records** and recomputes the full summary exactly locally. These remain repeated forecasts over 40 episode clusters. The initial disposable full clone hit disk capacity; original worktrees/checkpoints were untouched.

**Strict external replay FAIL:** [run 40021655456](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40021655456) completed the all-model/both-history census. Out-of-tolerance record counts: point prediction 8, MAE 5, lower quantile 8, upper quantile 6, width 12, interval score 11; counts overlap. Max absolute point drift 1.50443e-5 and MAE drift 4.18854e-5. Targets, coverage, risk probabilities/Brier and NLL meet the frozen contract, but full replay remains failed. No old/new tolerance or original output changed. Local full suite: **1,169 tests PASS**.

[Full results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me081-horizon-query-world-2026-10-23/research/mathematical-engineering/me081/RESULTS.md) · [Original checkpoints and forecasts](https://github.com/FPC-effortless/TAC-OSM/tree/research/me081-horizon-query-world-2026-10-23/research/replay-bundles/ME081).

**Next capability requirement:** retain these real moving-state/mid-horizon skills while recovering quiet precision and density quality, then independently qualify the complete model. New fitting requires a new substantive prospective registration. No default attribution campaign, hidden retry, protected TEST access, universal-intelligence claim, merge or deployment. Enterprise/Temporal/Edge/Counterfactual, causal policy and customer/operator value remain separately unqualified.

## Native computation reuse — formal architecture and measured execution improvement

[Draft PR #372](https://github.com/FPC-effortless/TAC-OSM/pull/372) implements the owner's request for mathematical notation and an optimized computational workflow. The [specific native specification](https://github.com/FPC-effortless/TAC-OSM/blob/research/native-computation-reuse-2026-10-23/research/computation-reuse/MATHEMATICAL_ARCHITECTURE.md) defines a shared encoder tensor DAG, explicit fast/slow/associative transition, conditional mixture decoder and versioned caller-owned inference effects. It preserves the trained ME080 architecture and all 99,816 parameters; **no model was fitted or predictive-intelligence improvement claimed**.

The execution-only implementation batches independent decoder positions, decodes only the final prefix when requested, shares three quantile inversions, and rejects stale/out-of-order/conflicting cached events. Ordinary parameter/buffer/mode changes invalidate the inference handler. Cache state never substitutes for a training graph. No SSM conversion, associative nonlinear scan, Triton/GPU/compiler fusion or category-theory novelty is claimed.

Original registered CPU timing, five frozen models, batch 24/history 8/one thread: median paired speedups across fits **1.457× all-prefix forward, 1.369× forward/backward, 1.845× final-prefix inference and 1.958× three quantiles**. Five warmups and 31 randomized paired measurements per workflow/model, raw times/IQRs/bootstrap intervals preserved. The fixed timing batch repeats one eligible FIT transition; this is a shape-specific local execution result, not workload-wide latency, independent hardware or energy qualification.

All five models matched the reference bit for bit locally on all 960 original exposed query windows each, including final optimized quantiles. Float64 software tests also check every model parameter gradient. **1,166 local tests passed.** External optimized/reference agreement is recorded separately; this never changes ME080's original three lower-quantile and one interval-score cross-platform exceptions. Original checkpoints, scores, tolerance and failed full capability gate stay intact.

The [results and boundaries](https://github.com/FPC-effortless/TAC-OSM/blob/research/native-computation-reuse-2026-10-23/research/computation-reuse/RESULTS.md) distinguish reusable execution from useful memory. Eight-frame training still does not qualify arbitrary long-history prediction. The next capability-first training registration must address real motion, mid-horizon and quiet-state trade-offs; execution efficiency does not satisfy an intelligence or customer-value gate. No protected source, new fit, main merge or deployment.

Independent [GitHub verification 40012640343](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40012640343) **PASS**: software values/gradients/cache tests, original model/receipt integrity, all five models and every original real query, including final quantiles. This compares optimized and reference execution on the external host; external timing was not measured, and historical ME080 cross-CPU replay exceptions remain FAIL and unchanged. Engineering tier only; no new fit or learned-intelligence gain.

## ME080 — native mixtures and reusable state: mixed real capability

[Draft PR #371](https://github.com/FPC-effortless/TAC-OSM/pull/371) trains **five original models with 99,816 parameters each**, retaining native 32D fast state, 32D slow state and an 8×8 associative matrix. The integrated model learns quiet/movement Laplace mixtures at 1, 4 and 16 frames; medians and marginal intervals come from the same CDF. Observation-only projections are shared within chunks, and explicit caller-owned state reuses prior computation. TAC's nonlinear dynamics remain intact. No SSM conversion, associative scan, fused GPU kernel or category-theoretic novelty is claimed.

The same licensed source supplies 120 FIT and 40 **already exposed** REPORT episodes: 960, 956 and 928 actual query–target pairs at the three horizons per seed. Each cold start receives 2,048 updates, batches of 24 windows and eight supervised causal prefixes. One fixed likelihood/BCE recipe, no search, restart or warm start. This is single-architecture capability development, **not** a matched-compute architecture comparison or a five-model deployment. No protected tissue-box, source CAL or TEST access.

| Physical MAE, lower is better | Frozen ME079 native | ME080 native | Persistence |
| --- | ---: | ---: | ---: |
| 1 frame | 0.060595 | 0.059393 | 0.068668 |
| 4 frames | 0.185953 | 0.219619 | 0.302400 |
| 16 frames | 0.947424 | 0.883994 | 0.975966 |

**Improved:** relative to ME079, one-frame mean MAE falls 1.98%, 16-frame MAE falls 6.69%, stationary one-frame MAE falls 43.79% (0.024795 → 0.013938), and 16-frame proper interval score falls 12.62%. Mean errors are 13.51%, 27.37% and 9.42% below persistence at the three horizons.

**Regressed:** four-frame error rises 18.10%; actual-moving one-frame error rises 22.93% (0.433934 → 0.533420); risk AP/AUC falls to 0.53443/0.89689 from 0.60775/0.91971; one/four-frame interval scores worsen. Stationary error remains 8.19 times ME077 and 3.50 times persistence. Overall one-frame error also remains slightly worse than clipped velocity. Nominal 90% coordinate coverage of 93.46%, 91.96% and 91.91% does not establish sharp calibration.

The paired 40-episode bootstrap 95% interval for one-frame new-minus-ME079 MAE is **[−0.009942, +0.006327]**, including zero. Four of five seeds improve, but the registered 5% mean threshold and several guardrails fail. **Full gate: FAIL. Tier: MIXED_TRADEOFF. No further fits under ME080.**

**Reuse scope:** the trained models' raw outputs agree for full eight-frame, split 3+5 and one-observation execution on all original 960 queries per seed on the local host. State tensor payload is 628 bytes per entity, excluding weights, activations and container/allocator overhead. Actual training and accuracy scoring reset after eight frames; 40-step software equivalence is **not** long-history prediction qualification. Callers supply identity/known commands and invalidate caches when weights or preprocessing change. Policy, causal action, safety, enterprise transfer and customer value remain unqualified.

**Preservation:** prefit `4c7309ff`; all five original model/optimizer/RNG and live receipts frozen at `e37a4b88` **before** scoring; original scores at `ea6cdd5e`. A fresh remote clone restored all five models, replayed 14,220 raw forecast records and recomputed the entire summary exactly locally. These are repeated forecasts over 40 episode clusters. Strict external replay `40005926076` **fails at seed 80002's lower quantile**; the full read-only census is recorded separately without changing tolerance or original outputs. No all-platform-green claim. A separate helper repairs the missing original-registration reference in final delivery metadata; original model bytes, registration, runner and forecasts remain unchanged, and the initial failure remains documented.

[Full results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me080-streaming-mixture-world-2026-10-23/research/mathematical-engineering/me080/RESULTS.md) · [Original checkpoints](https://github.com/FPC-effortless/TAC-OSM/tree/research/me080-streaming-mixture-world-2026-10-23/research/replay-bundles/ME080).


Full external census [40006593903](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40006593903) restored all five models and 14,220 forecasts. **Three lower-quantile records and one interval score exceed the original tolerance; every other recorded field passes**, including point predictions, actual targets, physical MAE, NLL, risk probabilities/Brier, upper quantiles, widths and coverage. Max lower drift is 8.3226e-6; max interval-score drift is 1.09736e-4. Counts can overlap. The audit process completed, but strict full replay remains FAIL. Original outputs and tolerances stay unchanged. The final local suite passes 1,161 tests.

**Next requirement:** develop a complete model that preserves quiet precision **and** changing-state/mid-horizon skill. Any additional fitting needs a new substantive prospective registration; no hidden retry, default attribution campaign or protected TEST tuning. Independent-source qualification and operator usefulness remain outstanding. Enterprise, Temporal, Edge and Counterfactual evidence stays separate.

## ME078–ME079 — trained real motion capability advances; no native architecture winner

Two complete integrated physical-model recipes are now trained and preserved:
[ME078 draft PR#368](https://github.com/FPC-effortless/TAC-OSM/pull/368) and
[ME079 draft PR#369](https://github.com/FPC-effortless/TAC-OSM/pull/369).
**30original fits**: each recipe has native fast32/slow32/8×8associative TAC,
GRU56 and compact causal RMSNorm Transformer, five paired seeds per family.
Each family received both recipe opportunities, identical actual source data,
sampling/importance weights,8causal-prefix supervision and1/4/16frame targets,
2048updates/batch24, final-only checkpoints, no sweeps or restarts. All actual
trainable/deployed model counts remain95k–105k; pairwise gaps<0.78%.
Audited forward/backward/clip plus analytical AdamW scalar-operation estimates
match within0.66%, with zero unsupported floating ops and independent matrix
profiler confirmation. This is **estimated algorithmic work**, not hardware
FLOP/instruction/time/energy or foundation-scale equivalence.

Same original licensed robot source:120FIT/40already-exposedREPORT episodes;
960H1queries,956H4pairs,928H16pairs. No source CAL/TEST/protected tissue-box.
Scores are means of individual model errors, not deployment of five-model
ensembles. Source-ending/takeover censoring is explicit.85,320raw forecast
records across the two recipes are repeated model/seed evaluations, **not**
85,320independent real observations or newly gathered trajectories.

| Mean physical MAE | Frozen ME077 | ME078 native | ME079 native | ME079 GRU | ME079 Transformer | Persistence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1,960queries | 0.066476 | 0.061221 | 0.060595 | **0.058553** | 0.059470 | 0.068668 |
| H4,956pairs | not trained | 0.195231 | 0.185953 | **0.181330** | 0.188688 | 0.302400 |
| H16,928pairs | not trained | 0.950255 | 0.947424 | 0.937354 | **0.923996** | 0.975966 |

**Real skill gains:** ME078 native actual-moving H1MAE0.421182 versus ME077
0.741981, persistence0.743241 and strongest tested fixed physical reference
(clipped velocity)0.482509: **43.24%below ME077 and12.71%below velocity**.
ME079moving0.433934 is slightly worse than ME078 but still substantially better
than those physical references. ME079 overall improves persistence by
**11.76%/38.51%/2.92% at1/4/16frames**. Mean individual native-fit motion AP/AUC
rises from ME077**0.16222/0.63493** to ME078**0.61717/0.92603** and ME079
**0.60775/0.91971** on the same84positive/876negative factual cases. This is
actual developmental movement discrimination, not safety, policy or customer
utility. Separate mean-probability ensemble diagnostics are not100kmodel claims.

**Material failures:** stationary H1MAE regresses from ME0770.001701 to ME078
0.026704 and ME0790.024795. ME079's added within10%stationary-preservation gate
FAILS. Both registered dual learned-control superiority gates FAIL. ME079's
aggregate horizon-relative error is0.822702 vs GRU0.804255 and Transformer
0.812258; adjusted97.5%episode-cluster intervals favor controls on this exposed
cohort. Full gates remain FAIL and evidence tier **MIXED_TRADEOFF**, not a
superior native architecture, independent intelligence or general world model.
ME077 remains the better stationary specialist; do not silently stack multiple
100kmodels and advertise a single100kdeployment. No further fit is authorized
by either completed registration; all attempts and adverse results stay visible.

**Preservation:** ME078 prefitsdd824ffe, all15original checkpoints frozen0352faba
before scoring, scores8b1dfb8c. ME079 prefits5fe54ce4, all15originals frozenaabb05b4
before scoring, scoresd1c8e2ed. Original model/optimizer/RNG/live receipts and raw
forecasts/targets/probabilities/intervals are Git-owned. Independent fresh remote
clone replays PASS for all30originals and original metric summaries, without
refits.1156local tests PASS in the final branch. **Strict external replay FAILS**
on54ME078 and29ME079 derived interval scores. Full external audits restore all
models and confirm every other forecast/score field meets the frozen tolerance;
no original model/score or tolerance changed. No all-platform-green claim.

Read the full[ME078 report](https://github.com/FPC-effortless/TAC-OSM/blob/research/me078-native-kinematic-world-2026-10-23/research/mathematical-engineering/me078/RESULTS.md)
and[ME079 report](https://github.com/FPC-effortless/TAC-OSM/blob/research/me079-risk-conditioned-physical-2026-10-23/research/mathematical-engineering/me079/RESULTS.md).

**Next capability priority:** a stronger integrated native predictive-state model
that retains stationary fidelity while forecasting genuine motion and longer
future trajectories, using authentic available context and established methods.
No attribution campaign is a prerequisite, but no extra small-gate retry may hide
these failures. Independent source/condition qualification remains necessary
before broader claims; the protected target is not automatically authorized.
Physical learning does not qualify the separate Enterprise/Temporal/Edge/
Counterfactual lanes. All outputs remain advisory; action/control authority,
operator benefit, investment/customer value and deployment are NOT_MEASURED.


## ME077 — native selective physical prediction, mixed authentic development result (2026-10-23)

[Draft PR #364](https://github.com/FPC-effortless/TAC-OSM/pull/364) trains the next
**whole native System1 model**, not another attribution study. Five original
102,503-active/deployed-parameter models combine bounded fast recurrence,
already-observed velocity context, a persistence/motion Laplace mixture,
mixture-median prediction and coherent uncertainty. Natural within-task outcome
frequency replaces ME076's enriched motion sampling. Larger training budget is
explicit: 1,024×24 transition presentations per seed versus ME076's 160×12.
This is a **practical integrated-system development comparison**, not matched
architecture, sample-efficiency, new-source transfer or control superiority.

Same exact authentic 40 exposed REPORT episodes / 960 genuine next-state queries:

| Cohort | Frozen ME076 MAE | New ME077 MAE | Persistence MAE | FIT servo MAE |
| --- | ---: | ---: | ---: | ---: |
| All 960 | 0.080158 | **0.066476** | 0.068668 | 0.097457 |
| Actual moving 84 | 0.748786 | 0.741981 | 0.743241 | **0.704517** |
| Near-stationary 876 | 0.016043 | **0.001701** | 0.003983 | 0.039245 |

Five-seed means. Overall error decreased **17.07% vs ME076**, **3.19% vs
persistence**; stationary error decreased **89.39% vs ME076**. All five seed and
task means improved overall. Paired episode-cluster 95% CI for new-minus-
persistence MAE [-0.002446,-0.001945], conditional on the fixed fitted models and
this exposed population. **Full registered gain gate FAILED**: persistence gain
missed 5%; moving-state gain was only 0.17%, and servo remains stronger there.
Tier **MIXED_TRADEOFF**, not independent intelligence, reliable motion anticipation,
causal action utility, safety or investor/customer qualification. Original
ME076 negative evidence and protected tissue-box holdout remain unchanged.

Original prefit/source freeze `1396564b`; all five original weights+optimizer+RNG
and live receipts frozen at `1b358218` **before** REPORT scoring; raw forecasts
and scores at `3cabf74c`. Fresh remote Git clone restored and replayed all five
models and raw real scores within the unchanged numerical contract, with no
refit. Separate [GitHub original-model/source replay39992505986](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39992505986) and normal CI passed at score commit3cabf74c; the final local suite passed1,142tests. No universal platform guarantee.
[Full results and limits](https://github.com/FPC-effortless/TAC-OSM/blob/research/me077-selective-physical-state-2026-10-23/research/mathematical-engineering/me077/RESULTS.md).

**Next physical capability priority:** improve genuine changing-state prediction
without losing stationary gains, through an integrated model trained on authentic
multistep outcomes and strictly available action context. No automatic new fit,
sealed-task opening, attribution campaign or policy deployment. Enterprise,
Temporal, Edge and Counterfactual lanes retain their separate evidence status.
Operational/customer value is UNKNOWN/NOT_MEASURED.


This is a **starter registry in TAC-OSM**, pending creation of a separate cross-project ideas repository. The registry does not claim that proposals were implemented or evaluated.


## DIAGKEY004 native memory interface repair — synthetic-only

[Draft #362](https://github.com/FPC-effortless/TAC-OSM/pull/362): original guarded
101,527-parameter native model, shared write/query addresses and tied value
codes, decoupled gates and protected queries. Five final synthetic accuracies
99.80/100/99.41/100/100%; all-five90% immediate-recall gate passes. Same-task
640-update mean97.93% versus unchanged native30.31%; at4096 unchanged native
improves to94.02% but one seed77.15%. All25 original fits and50 milestone
checkpoints/logits preserved. Full FLOP parity is unverified. Fresh local clone
replay passed; GitHub predicted classes/NLL match, with2/819,200 logit elements
failing the original tolerance. That strict external exception remains visible.
**ENGINEERING_VERIFIED synthetic mechanism**, not a real skill/architecture winner.
Next interference rung is not trained or automatically authorized. Preserve
independent authentic System1/Physical/Enterprise work and ME076 negatives.

## ME076 — Actual authentic System 1 native predictive failure (2026-10-23)

**Real original FIT training and replay completed, strict skill gain FAILED.** [Draft PR #360](https://github.com/FPC-effortless/TAC-OSM/pull/360) now includes **five fitted 100,476-parameter** native fast-only whole models, 160 real source optimizer updates per seed, original source FIT 120 episodes / source-exposed episode-disjoint 40 report, five original checkpoint+optimizer+RNG files and 5×960 original real per-case predictions. Independent clean-runner checkpoint+score replay **PASS all five**; independently audited original training budgets 100,476 learned parameters and 160 updates/seed **PASS**, no untrained-head camouflage. Original source 160-episode SHA `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`; original CAL, original TEST and the reserved protected task **never opened**.

| Source-exposed REAL next-state evaluation | Original persistence MAE | FIT-identified servo lag1 | Five-seed native mean MAE | Verdict |
| --- | ---: | ---: | ---: | --- |
| 960 observations/seed on 40 original report episodes | **0.068668** | 0.097457 | 0.080158 | **FAIL**: native +16.7% error versus persistence, all five seeds |
| Actual 84 motion-rich observed cases | 0.743241 | **0.704517** | 0.748786 | **FAIL** versus both practical controls |
| 876 near-stationary observed cases | **0.003983** | 0.039245 | 0.016043 | **FAIL**: excessive corrections |

See [full original real negative report](https://github.com/FPC-effortless/TAC-OSM/blob/research/me075-system1-native-gated-delta-2026-10-23/research/mathematical-engineering/me076/RESULTS.md) and its [posthoc per-real-episode uncertainty](https://github.com/FPC-effortless/TAC-OSM/blob/research/me075-system1-native-gated-delta-2026-10-23/research/mathematical-engineering/me076/POSTHOC_REAL_SCORE_DIAGNOSTIC.json). The 40-case episode-cluster posthoc 95% interval for **native minus persistence** physical MAE is **+0.011116 to +0.011862** on the **already exposed DEV** population. Worse performance was also observed for all five genuine source tasks separately. This is real negative evidence, **not** independent heldout transfer, module attribution, causal action intelligence or an investor-worthy System1 product advantage.

**Next capability lane `S1-002` (not fitted or approved as a gain):** jointly learned, uncertainty-aware movement gate + sparse conditional bounded physical residual, keeping factual `x_t` persistence as an explicit candidate and accounting for realistic ∼9% genuine movement prevalence. Study early identification of actually changing states, stationary false corrections, physical proper NLL and tail/abstention risks; no target-derived movement decision feature. Five fitted original ME076 models and same-case persistence/servo are now **frozen comparisons on development data**; a truly untouched real task/source must be preregistered for `INDEPENDENT_GAIN`. Preserve owner 95–105k live model gate, genuine real source, original Git checkpoint+independent replay, and no default component-ablation cycle.

**Investor policy:** distinguish authentic training/replay integrity (achieved) from physical predictive advantage (failed), device latency (unmeasured), safe action control (unverified) and willingness-to-pay (unknown). No 25k System1 extraction, hardware production or ALLOW/ABORT deployment is justified by this record.

## ME075 actual System 1 engineering status — 2026-10-23

**Current owner priority:** focus the next capable-model design cycle on truthful real one-step System1 performance and a **separate** mathematically grounded Gated Delta associative memory candidate. DIAGKEY001 PR #357 guarded original synthetic recall fits (TAC 3.75/4.49/3.48% versus Transformer 99.06/98.79/98.87% at16/64/256) reveal *this fitted mechanism/recipe* did not learn the synthetic key/value task, but their ~18–21% partial FLOP budget disparity and artificial sources preclude universal architectural inferiority or real-world prediction conclusions. Do **not** block proven ability in five other domain lanes merely because random-key recall failed.

| Registration | Implemented now | Measured evidence / exact tier | Biggest missing capability |
| --- | --- | --- | --- |
| `ME075-S1` | [Draft PR #360](https://github.com/FPC-effortless/TAC-OSM/pull/360) `System1FastModel`: 96D fast-only recurrence, exactly **100,476** active trainable/deployed parameters on physical registered 14D streams, bounded residual one-step next-state distribution, no ungrounded risk head, no authority or rollout | Engineering **only**, [test workflow](https://github.com/FPC-effortless/TAC-OSM/actions/workflows/me075-system1-engineering.yml); **zero original new ME075 fitted native checkpoints or real scored cases** | Actual training and honest real one-step prediction on source verified physical motion/action windows; strong persistence, motion-rich cases, hardware p95/p99, authorization boundary |
| `ME075-GDM` | `gated_delta_memory.py`: optional corrective 8×8 `alpha M + beta(v - alpha M k)⊗k` with separate retention/write gates and no-write query | Known published mathematics, analytical unit invariants only; **not** a new trained keyed-recall pass or 90%/Transformer-matching model | New explicit lawful diagnostic preregistration if synthetic fitting; genuine real long-context evidence for any memory-intelligence claim |

Historical genuine ME068 FIT-only robotic audit scores **257,971** autonomous original transitions, near-stationary diagnostic fraction **91.23%**, one-step persistence MAE **0.070195** versus FIT-only servo best tested causal lag1 MAE **0.119142**. These are previously measured original FIT-only reference statistics **not** a new independent held-out System1 evaluation, physical policy success or causal actuator alignment. They show why a naive one-step accuracy/MAE could appear high on static trajectories; select and report **movement-rich** original factual outcomes as well as all-case results.

**Hard constraints preserved:** Future trained models across all variants still must have **95k–105k** actual learned parameters (25k System1 would require an explicit prospective owner amendment); real original outcomes and observed-at causal fields; original Git checkpoint/optimizer/RNG and independent replay; same-task ability improvement and appropriate heldout confidence; no trained memory or System1 intelligence claim from synthetic/CI alone. `ME075` branch stacks on six-variant draft #358; master plan is separately updated, neither branch merged.

## New six-lane TAC-OSM capability research registry — 2026-10-23

The integrated six-regime architecture is now the prospective research portfolio; it is **not** evidence of six trained neural models. Common native recurrent and bounded residual mathematics can be shared while event, physical, temporal and policy targets remain separate. Read the binding [six-lane master plan section](MASTER_RESEARCH_PLAN.md) and [isolated six-variant branch](https://github.com/FPC-effortless/TAC-OSM/tree/research/six-variant-core-lanes-2026-10-23/src/tac_osm/variants).

| Lane ID | Research focus | Historical original evidence | Next real capability gate | Current tier |
| --- | --- | --- | --- | --- |
| `S1-001` | One-step native reactive risk/action advice | ME054/061 fast dynamics; ME074 exposed fast readout lesion, not policy or sub-ms evidence | Authentic reactive/action outcomes; guarded human decisions; measured hardware p95/p99 | `ENGINEERING_VERIFIED` adapter only |
| `PHY-001` | Action-conditioned actual physical dynamics and robust fallback | ME054/056, ME061–068; real transfer to persistence failed ME064/066 | Genuine heldout robot motion/contact, proper joint dynamics vs strongest physical prior | `DEV_CAPABILITY_OBSERVED` in preserved historical native; new lane adapter `ENGINEERING_VERIFIED` |
| `ENT-001` | Enterprise early branch, action/time path and entity forecast | ME069–074, five original ME073 trajectory models; exposed easy/hard negatives | Early rare transition + calibrated time/END-aware future on genuinely new lawful real cases | `DEV_CAPABILITY_OBSERVED`; original unmerged weights remain preserved |
| `TMP-001` | Bitemporal known-at/valid-at predictions and late corrections | Woyengi adapter proposals; no actual bitemporal correction-trained model | Real authorized audit revisions with as-of nonleakage and independently tested forecast correction | `ENGINEERING_VERIFIED` time-cut adapter only |
| `EDG-001` | INT8/frozen model on actual embedded hardware | ME055/056 CPU & operation profiles; no original embedded speed run | Exact quantized score fidelity and device memory/energy and p95/p99 latency, policy-compliant learned parameters | `DESIGN_ONLY` trained edge skill; export audit software |
| `CF-001` | Conditional observed action consequences and separate identifiable policy value | Historical CASM, ME049/061/073 and Veritas simulations; not real counterfactual benefit | Real actions/propensities, observed outcomes and randomized/identifiable policy utility with authorized pilot | `ENGINEERING_VERIFIED` scenario adapter; no causal capability |

**Parameter governance conflict:** user-suggested System1 20k–35k, Physical 50k–80k and Edge-Lite 10k–15k learned architectures are **not authorized** by the current future-live-model 95k–105k requirement. Preserve them only as potential future *design alternatives*. No new tiny model fit or hidden distillation until an explicit policy amendment; quantization by itself does not change learned parameter count. Architecture comparisons additionally require ≤1% parameter and equal authentic data/exposure/training/tuning compute parity; CPU wall time is not a matching constraint.

**ME074 interpretation:** exposed original associative readout lesion small/inconsistent, not universally useless or safe to delete from frozen ME073; fast zero improves earliest predictions and harms later ones, not direct proof of System1 execution benefits. Physical router must not assume servo is strongest, since real persistence won previous transfer. EOR in BPI is a published suffix proxy, **not** verified business completion. Temporal replay is not a new neural trained skill and Veritas simulated branches are diagnostics, not true causal evidence.

**Work order:** safe six-lane interface refactor (NO NEW FIT) → Enterprise/Pysical integrated full-native capability training in separate genuine datasets → a real reaction task for System1 → authorized temporal and hardware qualifications → verified action-rich Counterfactual model. Maintain the compulsory Capability Progress Scorecard and investor-value ledger independently for every trained study. Do not promote different task accuracies into an invented overall intelligence percentage.

## Compulsory evidence tier for every capability claim — 2026-10-23

All future CAP/ME/DEV research records must link to the canonical [Capability Progress and Intelligence Evidence Standard](CAPABILITY_PROGRESS_TRUTH_STANDARD.md) and name the fixed `tier`. Merely integrating, training and replaying a model is **not** a measured improvement. Development tasks and previously exposed TEST are not fresh confirmation. For a before/after gain, compare frozen complete systems on the exact same real cases, targets, prediction ages and horizons, retaining primary proper scores, original per-case outputs, statistical uncertainty and all worse outcomes. Independently confirmed improvement additionally needs a genuine unseen real cohort and predefined meaningful effect, no severe risk/calibration regression and independent replay. Mark any tradeoff honestly; never invent a scalar “intelligence increase”. Comparative architecture winners require separate matched parameter/data/exposure/compute evidence, while single-architecture capability development may proceed without default ablations.

Every new registry entry records `what_improved`, `what_regressed`, `what_is_unknown`, case counts, frozen old/new model identities, result tier, investor/customer meaning or `NOT_MEASURED`. Historical ME071–ME074 interpretation is specified in the canonical policy without modifying their actual checkpoints or score receipts.

## Compulsory runtime size-policy integration — 2026-10-23

Before the next capability fit, integrate the universal live-model guard from
[draft #354](https://github.com/FPC-effortless/TAC-OSM/pull/354) or its
[ME073 overlay #355](https://github.com/FPC-effortless/TAC-OSM/pull/355).
Both remain unmerged; an old study ID or declared manifest count cannot authorize
new fitting. Capability development and deferred-source studies require the same
95k–105k live trainable/deployed counts, pre-update runtime receipt and original
checkpoint verification. Fair comparisons retain the independent 1% and real
data/compute conditions; CPU is efficiency only. Preserve original models and
registrations through explicit immutable replay, never retroactive attestations.
This closes software-governance gaps; it adds no predictive capability claim.

## Completed native trajectory capability: ME073

**ME073 trained and independently replayed, 2026-10-23:** five original99,781-parameter models, six authentic FIT epochs, shared eight-step native imagined-state rollout and activity-conditional bounded incremental clocks. At six observations, conditional1/2/4/8-event accuracy81.85/68.83/62.11/53.44% over367/326/275/189 actual future records; one-step NLL.372472, proper interval score18.97h, nominal90% coverage85.99%. **Early goal failed:** two-observation accuracy40% in every seed and NLL1.416298; distant8-step probability scoring regresses relative to the historical ME072 recipe. Age16/h8 mean width636.94h remains broad. Different budgets/objectives/inference prevent an architecture-winner claim. Original checkpoints frozen100671c7 before scores190c2bc8; [independent replay39948180370](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39948180370) restored all five and recomputed8,495 case/query vectors,33,980 horizon marginals and67,960 representative weighted paths plus actual scores. These are repeated forecasts over925 exposed DEV cases. See [ME073 results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me073-autoregressive-native-trajectories-2026-10-23/research/mathematical-engineering/me073/RESULTS.md). Original TEST remains unscored by ME073; no external source/customer, long-memory, causal action or production qualification.

**Next integrated skill:** stronger early and distant probability forecasts with useful timing tails; original FIT-only changes and a newly registered readout before independent qualification. No additional attribution matrix is required. Reuse existing persistent carry rather than call it new unlimited memory. A selective linear scan would need a separate core fork; a lowest-variance selector and Student-t/hard72h cap are unproven fixes and were not substituted for measured predictive quality.

## Commercially guided capability-to-value program — investor-readiness companion (2026-10-23)

**Owner's actual objective:** TAC-OSM should get *more capable* as a predictive world model **while** building evidence for credible deep-tech investment and a specific independent customer use case. Do not turn this into an endless component-ablation, architecture-attribution, marketing, or fundraising-only project. The canonical, binding [investability and capability master plan](MASTER_RESEARCH_PLAN.md) defines evaluation and stage gates. The current code may be sufficient for **exploratory research pre-seed conversations**, but enterprise-scale product value, validated paying customers, commercial seed readiness and competitive IP rights are **NOT VERIFIED** from research repository artifacts.

| ID | Rank / capability-to-customer milestone | What would genuinely count as completion | Status / boundaries |
| --- | --- | --- | --- |
| CAP-022 | **P0 model:** useful multi-horizon future workflow-state forecasting | Integrate and train native next 1/2/4/8 observed-event forecasts and genuine timing from real online prefixes; record eligible cases at each query age, difficult event classes, proper NLL/Brier/coverage/calibration and end-to-end real checkpoint replay | **ME072 AND ME073 TRAINED/REPLAYED**: ME073 adds dependent weighted paths; six-prefix4-event accuracy62.11%, but early accuracy40% and distant probability losses remain failures. Historical ME072: five original ME072 models; six-event-prefix conditional accuracy82.02/68.47/59.64/54.18% at1/2/4/8 future events. Exposed DEV only; early branches and timing remain weak; no independent operational qualification |
| CAP-023 | **P0 model:** useful event-time and early-warning confidence | Actually measured sharp, trustworthy forecast intervals and decision-time warning lead, error/abstention costs when real outcomes/SLAs exist; calibrated proper scores, p50/p90/p99 widths and missed/false risks | **Open**: ME073 one-step coverage85.99%, proper interval score18.97h; age16/h8 width636.94h, still broad. Historical ME072 coherent clock trained, but one-step nominal90 coverage84.74%; age16/horizon8 mean width3,536h. ME071 postprocessing failures preserved; external warning cost unknown |
| CAP-024 | **P0 product:** minimal safe enterprise workflow prediction interface | Authorized case/event streaming, real-before-outcome forecasts, entity state updates, provenance and uncertainty in an advisory dashboard/API; access/privacy, no automatic authoritative writes | **Not verified in TAC repo**; Woyengi/other portfolio code can be proposed integration, not proof of deployment |
| CAP-025 | **P0 customer:** discover one buyer and validate a pilot | Document authentic buyer pain, price/budget and baseline, lawful case stream, explicit consent, ideally 2–3 design partners plus >=1 structured external pilot as a *suggested* discovery target | **UNKNOWN**, no claimed signed customer/pilot/revenue; an internal prototype does not satisfy |
| CAP-026 | **P0 commercial:** observed operator value | Precommitted prospectively observed time-to-warning, true false alerts/missed deadlines, actual costs and net workflow benefit against customer existing process; human approval | **Not established**; process-label accuracy is not cost saving or causal action value |
| CAP-027 | **P1 investment diligence:** ethical/licensed business and reproducible research | Technical claims/data/source licenses, IP and contributor assignments, planned legal entity/cap table, commercial usage rights, privacy/security, financial model and milestone use of capital verified with appropriate advisors | **Incomplete/unknown**, not evidenced merely by public code; never infer exclusive IP rights |
| CAP-028 | **P1 transferred intelligence:** generalization beyond single organization/domain | Untouched authentic external case/cohort / time/site shift, persistent state, missing events, verified feedback, real action-outcome if available, bounded error and meaningful end-user utility | **Not qualified**; long-horizon dataset access and causal-action evidence remain separate gates |

**Every new ME/DEV whole-model study should link to the right CAP ID and include a research-first Capability-to-Value Card** (model skill, original legal source/temporal availability, factual metrics, customer decision hypothesis, observed pilot/customer evidence or UNKNOWN, genuine checkpoint/replay, stage R0/R1/P0/C0/C1/I0). Allow exploratory foundational model research with customer impact marked unknown. Do not claim completion of product/customer stages from original FIT/DEV trainability alone. Research comparison parity and memory-specific rejection gates remain intact *for claims of superiority*; they are not prerequisites for integrated capability development.

**Commercial sequencing:** valid native model and real scored evidence → investor-legible full-model skill → permission-bound advisory product → authorized independent pilot → measured operational benefit → seed-quality customer diligence. These are evidence stages, not calendar, valuation, revenue or fundraising guarantees. Founder's investor deck may convey honest early research potential now; never fabricate commercial traction.

## Investor-readiness capability-to-value registry (owner direction 2026-10-23)

**Two north stars:** (A) develop an intelligent native world model that predicts and updates authentic states; (B) demonstrate customer-understandable, measured operational usefulness capable of supporting a credible deep-tech investment thesis. **Do not replace (A) with superficial dashboard/product work** or conflate synthetic demo scenarios with experimental evidence. Follow the full [investor-readiness roadmap](INVESTOR_READINESS_AND_CAPABILITY_ROADMAP.md) and [master plan](MASTER_RESEARCH_PLAN.md). No disclosed paying customers, real external pilot/ROI, completed investor round or independently verified commercial product has been established in this research record.

| ID | Capability-to-value objective | Required honest evidence / minimum deliverable | Status as of registry addition |
| --- | --- | --- | --- |
| **CAP-INV-001** | Integrated **multi-horizon world-state forecasting** for operators | Genuine frozen original native model scores for supported 1/2/4/8 event horizons, query-age coverage and proper evaluation with actual event-time available labels. [ME072 draft #351](https://github.com/FPC-effortless/TAC-OSM/pull/351) has five original trained models and independently replayed1/2/4/8-event scores; retain proper timing failures and conditional-case denominators. | **ME072/ME073 TRAINED/REPLAYED ON EXPOSED DEV; early/distant forecast failures and independent operator value unqualified** |
| **CAP-INV-002** | Coherent event/exception/delay confidence that supports decisions | Conditional empirical real-data logs and horizon delays, Brier/ECE/NLL, quantile pinball and proper interval score, p50/p90/p99 width, alarm lead time / false alert rate and abstention on all eligible real observational cases | **RESEARCH GAP**: ME071 sharp NLL but worsened calibration and broad interval tails |
| **CAP-INV-003** | Reliable current persistent entity state and verified new-outcome corrections | Factual read-only streaming state by verified ID and timestamp, no future information, source-eligible long histories and recovery/missingness probes; later score observed correction benefits, not code-path existence alone | **PARTIAL MECHANICS; capability benefit unproven** |
| **CAP-INV-004** | Evidence-backed **advisory** enterprise workflow demo/API | Predictions from actual checkpoint-backed model, real or licensed/anonymized demo stream, uncertainty and provenance, no automated approvals, no real-customer performance claims from synthetic UI fixture | **NOT VERIFIED PRODUCT READY** |
| **CAP-INV-005** | Independent validation of an actual operator problem | Real stakeholder discovery and documented frequency/cost of operational exception, lawful prospectively agreed pilot protocol and proposed 2–3 design partners / >=1 pilot; observed warning lead time, false alarm, net decision/time/cost savings; pricing validated rather than invented | **NOT VERIFIED; targets only** |
| **CAP-INV-006** | Investor diligence readiness | Entity/ownership and collaborator assignments, data rights and license strategy, security/privacy/infrastructure, source/weight/data room, team/company context, realistic financing use-of-funds and honest commercial traction | **NOT VERIFIED; legal/human authority required** |
| **CAP-INV-007** | Independent repeatable real-world capability and product value | New legally available untouched source from a different process/site/task, pre-registered learned ability and prospective pilot validation. Preserve original checkpoints/splits/negative results and distinguish robust evidence from research DEV | **NOT QUALIFIED** |

Every future completed CAP/ME/DEV cycle records **both ledgers**: `model_capability: actual metric / source / sample / frozen Git model / independent replay / exposure status`, and `commercial_relevance: exact buyer problem / candidate operational action / observed or conjectured lead time / actual pilot-evidence status / investor-readiness gap`. Treat unknown revenue, company formation, customer evidence, market size and claimed IP exclusivity as **UNKNOWN** rather than zero or success. Preserve the user's hard real-data and matched-architecture-comparison constraints; unrelated attribution studies are optional, not the research objective.

## Owner-ordered next capability work — 2026-10-23

The research owner is **more interested in model capabilities than ablation/attribution studies**. Prioritize integrated trained model progress over proving every TAC submodule novel or better than GRU. A comparator remains mandatory **if claiming architectural superiority**, but is not a prerequisite for a real single-architecture capability-development cycle. Existing held-out data, fair-budget gates for comparisons, authentic-only success claims, and full original Git checkpoint/replay rules are unchanged. ME070's 64-event scientific rejection applies only to a *memory-superiority claim*, not to whether TAC-OSM should continue learning useful skills.

| Capability ID | Whole-model capability to develop | Measured development evidence required (not an untrained proposal) |
| --- | --- | --- |
| CAP-017 | Native multi-horizon workflow forecasting from observed prefixes | Actual original real-case queries at eligible horizons; case coverage, next-event/next-several-event metrics and proper time scores; no hindsight midpoint; record rare branch and status-change errors |
| CAP-018 | Informative reliable waiting-time, event/stop uncertainty | Jointly trained zero/positive event and temporal distributions, proper log loss/Brier/ECE/pinball/interval score, coverage with median/p90/p99 width, real censoring and terminal-case boundary |
| CAP-019 | Persistent, updateable, missingness-robust factual belief | Real observational entity continuity, state carry/read/revision over genuinely supported histories, verified-outcome feedback, time gaps and missing observation behavior; execution mechanism alone is not a predictive gain |
| CAP-020 | Action-conditioned future state and event outcome representation | **Only when authentic records include actions available at prediction time and observed consequences**; multi-horizon action-conditioned factual outcomes, error/decision costs, safe uncertainty; no causal-treatment promise without identification |
| CAP-021 | Native hierarchical / iterative state correction and generalization | Integrated mathematical engineering only where errors warrant it, train/checkpoint full original variants with real outcomes, and qualify on independently sourced real domain/site/task before external-product claims |

**Execution:** extend frozen ME071 into an integrated next candidate and actually train it on legally available real data. Keep original checkpoints and source-replay provenance available inside TAC-OSM's repository. If novel long-history source acquisition remains blocked, improve genuinely supported enterprise forecasting and outcome/time calibration rather than staging more model-comparison diagnostics or manufacturing artificial long cases. **Update:** CAP-017 multi-horizon event forecasting and CAP-018 coherent time modeling now have real trained ME072 developmental evidence, with undercoverage/wide-tail failures; CAP-019–021 broader benefits remain unqualified. [ME072 results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me072-marked-multihorizon-world-model-2026-10-23/research/mathematical-engineering/me072/RESULTS.md); [capability-to-value card](https://github.com/FPC-effortless/TAC-OSM/blob/research/me072-marked-multihorizon-world-model-2026-10-23/research/mathematical-engineering/me072/CAPABILITY_TO_VALUE.md).

## Completed integrated capability development: ME072

**ME072 completed, 2026-10-23:** five original 123,377-parameter native models trained six authentic BPI2012 FIT epochs and independently replayed from Git. At six observed events, 555/925 exposed DEV REPORT cases reach the query; actual future-record denominators at horizons1/2/4/8 are367/326/275/189. Conditional activity accuracy is **82.02/68.47/59.64/54.18%**, NLL **.383476/.712665/1.067113/1.378439**. Nominal90% one-step clock coverage is **84.74%** (undercoverage); distant tails remain broad (age16/horizon8 mean width3,536h). Original weights were frozen before scoring; independent GitHub run39941010981 restored all five and recomputed all registered real-case outputs. See [ME072 results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me072-marked-multihorizon-world-model-2026-10-23/research/mathematical-engineering/me072/RESULTS.md) and [capability-to-value card](https://github.com/FPC-effortless/TAC-OSM/blob/research/me072-marked-multihorizon-world-model-2026-10-23/research/mathematical-engineering/me072/CAPABILITY_TO_VALUE.md). This demonstrates multi-horizon developmental forecasting, not an architecture winner, business completion, long-memory qualification or external operational value.

## Current integrated capability work: ME071

| ID | Capability | Delivered evidence | Remaining objective |
| --- | --- | --- | --- |
| CAP-016 | Persistent, probability-aware native enterprise event model | [ME071 PR #350](https://github.com/FPC-effortless/TAC-OSM/pull/350): five original 93,770-parameter native fits; six real FIT epochs/715,134 targets each. Full-case detached fast/slow/associative carry, history log pooling, smoothing, ordered waiting quantiles, disjoint temperature/CQR, independent entity streams. Fixed-six-event DEV REPORT accuracy81.58%, fitted NLL .401741 vs raw .420459; Brier/ECE worsen and calibrated mean intervals51.69h. Original Git checkpoints, forecast arrays and independent five-model real-score replay preserved; later preprocessing mismatch preserved and loader repaired to restore exact frozen values. | Improve complete model's probability quality and interval usefulness, then qualify on an untouched authentic source. No new baseline matrix, attribution fit, fresh TEST, 256-memory, action-policy or commercial claim. State carry is not longer gradient credit. Preserve ME070's long-memory rejection rule. |

The owner's current emphasis is integrated architectural power before detailed
mechanism attribution. This single-architecture development lane cannot be
used to bypass the matched-resource contract for future architecture winners.
Synthetic fixtures/interleaving are supporting engineering checks only.

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


| CAP-024 — factual all-case END-aware future scoring and state attribution | **ME074 EXECUTED, EXPOSED DEV ONLY; no new neural fit.** FIT-anchored original 1st/3rd/5th order non-neural Markov/survival controls scored at query ages2,6,16 and horizons1/2/4/8 with all eligible-case categorical-or-END NLL/Brier. At age6 on 555 original real cases, native five-seed mean h1/h8 NLL **0.24668 / 0.88572** versus fifth-order **0.56627 / 1.06720**, but at age2 on 925 real cases fifth-order h1/h8 **1.14613 / 1.40970** beats TAC **1.41638 / 1.67668**; 3gram h1/h8 **1.14613 / 1.38203** also beats TAC. This is an activity-only unmatched practical baseline, not a neural architecture winner. | [ME074 original real baseline](https://github.com/FPC-effortless/TAC-OSM/blob/research/me074-real-baseline-memory-diagnostics-2026-10-23/research/replay-bundles/ME074/REAL_DEV_END_AWARE_BASELINE.json), [draft PR #356](https://github.com/FPC-effortless/TAC-OSM/pull/356). Require equal 100k GRU autoregressive and direct multi-horizon heads with trained original checkpoints and genuinely unexposed real evaluation. |
| CAP-025 — native fast/slow/associative contribution on genuine workflow histories | **ME074 5 frozen source-trained ME073 seeds, posthoc original exposed cases; zero fitted models.** One-step joint NLL query age2/6/16 full **1.41638/0.24668/0.23178**; fast-zero **1.19197/1.16326/0.58114** (helps early but harms mature cases, all five trained seeds in each direction); association-zero **1.42889/0.24727/0.23817** (small mixed effect). Full original Git-owned probabilities replayed exactly before each lesion. **No causal physical action, long-memory ability, learned associative necessity or superior recurrent capacity proven by OOD state ablation.** | [Original real memory diagnosis](https://github.com/FPC-effortless/TAC-OSM/blob/research/me074-real-baseline-memory-diagnostics-2026-10-23/research/replay-bundles/ME074/EXPOSED_MEMORY_LESIONS.json). Next use prospective trained parameter/compute-matched memory ablations and independent 64+ and 256+ source entity records, not synthetic-success promotion. |


| CAP-MOTION-001 — true real movement discrimination separate from mean-state error | **ME077A EXECUTED NO-FIT ON ACTUAL EXPOSED ORIGINAL ROBOT DATA:** original five ME077 frozen 102,503-param models, same 960 real next states / 40 authentic recording episodes (84 actual moving). Source-case exact SHA and all original model prediction bytes preserved. Auxiliary motion risk had ROC-AUC **0.64412**, episode-bootstrap 95% CI **[0.59382,0.69674]**, AP **0.17056** vs real source prevalence **0.0875**, Brier **0.07870** versus hindsight REPORT prevalence-oracle **0.07984** (NOT deployable). Mean p_moving **0.1084**, p_stationary **0.0982**; threshold0.25 **missed all 84 actual movements**. Top 10% highest scores captured20/84 genuine movements, 401 false alarms at threshold0.10. Mixture 14D component probability is NOT the global event head. No trained new model, no untouched external test or policy value; ME077 originally improved exposed all-state MAE to0.066476 but failed its own >=5% gate and this auxiliary head is not a reliable anticipatory detector. | [Original fully paired real data score](https://github.com/FPC-effortless/TAC-OSM/blob/research/me077a-real-motion-risk-latest-base-2026-10-23/research/replay-bundles/ME077A/ORIGINAL_REAL_MOTION_RANKING.json), [no-fit reproducible PR #367](https://github.com/FPC-effortless/TAC-OSM/pull/367). Next register 100k integrated change-risk and conditional residual **jointly evaluated** on authentic independent robot data; keep real motion false alarms and missed movements separate from physical MAE and do not force DIAGKEY004 memory into System1 without same-task empirical gain. |

| CAP-GRAPH-082 — native parallel local dynamics plus quiet precision | ME082/083/084:15authentic guarded fits completed, original Git checkpoints and paired scores preserved; all full gatesFAIL/MIXED_TRADEOFF. Mean errors improve but quiet/H4/clusterCI fail. | Preserve complete attempt ledger in draftPR377. No extra fits under those registrations. |
| CAP-CLOCK-VALUE-085 — time-aware native state with precision-preserving value retrieval | Separately preregistered99731-param integrated model; first5-seed4096-update recipe training, second conditional recipe frozen. Exported pure graph2579nodes, all20parameter tensors connected; learned gain NOT YET SCORED. | Keep original full gate, explicit exposedDEV/clock feature disclosure, all original weights beforeREPORT and independent strict replay. Do not present engineering as intelligence, scan-parallel recurrence or verified product value. |

| CAP-CLOCK-VALUE-085 outcome | Five real fits completed; H1/H4/H16MAE0.053390/0.157590/0.681905, stronger mid/long forecasts but quiet0.022057 and episodeCI fail. Full gateFAIL; independent original27600forecast replay and compiled-profile restorationPASS. | Preserve originals and exposedDEV status; ME086's separately frozen second recipe runs unchanged. |
| CAP-BOUNDED-VALUE-087 — coherent median learning and robust quiet value retrieval | Separately frozen five-seed complete-model training started:99731params, FIT-noise-bounded quiet borrowing plus correct implicit CDF gradient of the actual mixture median. No REPORT score yet. Existing gates unchanged; all6attempts visible. | Whole-model improvement first, no attribution prerequisite, no future-conditioned inference, no original-checkpoint rewrite or protected cohort opening. |

| CAP-BOUNDED-VALUE-087 outcome | Five guarded originals completed, H1/H4/H16MAE0.050888/0.148123/0.682106; movingH1 0.363536. Original95%pairedCIcleared but quiet0.020909 and stronger repeated-attemptCI failed. FullFAIL; independent original and compiled restorationPASS. | No extra fits under ME087. Preserve all original fields and cohort exposure; no memory attribution required to continue native capability development. |
| CAP-PRECISION-088 — coordinated precision/movement fine tuning | Fixed ALL5ME087 warm starts, all99731weights trainable;2048new+4096priorupdates per lineage, two independent seed processes. Predictive noise units, near-step residual trust, factual coordinate routing and explicit quiet precision objective; no REPORT score yet. | Frozenbbf30a57; all prior gates unchanged, all7attempts visible, all5new originals before scoring. Explicitly cost-sensitive development, not automatic probability calibration or independent general intelligence. |

### Native visual predictive-state capability — source qualification, 2026-10-24

ME088–ME090 full joint gates FAIL, originals and all attempts preserved. Quiet
precision and moving accuracy remain in conflict. Qualify synchronized current/past
RGB observations from the existing licensed FIT/REPORT episodes before building
an integrated <=105k native visual world model. Preserve native fast/slow/delta
memory, causal cutoff, fixed physical gate and original replay. No future images,
new sealed episodes, causal policy, architecture-winner or customer-value claims.
This is a capability direction, not evidence of a trained visual ability.

### ME091 — integrated observed vision and native predictive location (training)

One whole-model recipe,104225active weights; shared overhead-image CNN, native
fast/slow/association and parallel horizon readout. Shared bounded translation
moves both conditional mixture means together without new state rollout. All
licensed same-role input pixels are Git-owned; source/frame SHA verification
passed for160episodes. All29parameter tensors active in a real-FIT graph. Five
fixed-parent fits running, no REPORT result or gate pass yet. Existing quiet/
moving/multi-horizon gate and protected sources unchanged; no model restarts,
matched winner, policy control, site transfer or customer-value claim.

### ME092 — train the integrated model against compatible task gradients

ME091 full gateFAIL, all five originals/replays retained. New104225parameter
whole-model recipe uses fixed paired native/visual initializers, shared proper
forecast terms, factual quietH1/movingH1/long-horizon loss partition and batched
VJP/PCGrad. No extra task models/forward passes, no model reset or winner claim.
One registered2048update recipe per five seeds, no REPORT yet. All50prior trained
versions remain visible; same protected sources and joint gate unchanged. This
is an optimization/capability attempt, not proof of a gradient-conflict cause.

### ME093 — integrated local-noise and transition distribution (2026-10-24)

Motivation: ME092 improved movement/long forecasts but quiet accuracy and paired
intervals failed. Frozen output trace suggests a narrowhold/broadtransition mixture
may conflate small observation noise with actual movement; descriptive hypothesis,
not established cause. Build one three-Laplace native visual predictive distribution
with causal robust local center, ordered scales, exact proper scoring and shared
quantile inversion.96,595active parameters, ALL5indexedME092parents,3,072newsteps,
whole-model training, unchanged gate and source cutoffs. Registered51120067;
training underway, no skill-success claim. Attribution secondary; no extra fits
under exhaustedME092. Commercial/customer evidence remains NOT_MEASURED.

### ME094 — explicit median and coherent multimodal shape (2026-10-24)

ME093's extra noise regime did not meet quietH1/stricterCI gates. New whole model
separates a learned predictive median from shape mixing/scales, translates the full
multimodal CDF with correct implicit derivative, and adds causal eight-value time/
content relevance.101,173parameters,4,096newupdates, ALL5fixedME093parents,
unchangedgate; registered c87dc63c, fits underway, no result yet. Standard quantile/
mixture/retrieval mathematics reused, no new theorem, extra ablation campaign or
architectural-superiority claim. Genuine customer/decision value NOT_MEASURED.

### ME094 query-selective computation reuse — ENGINEERING only

Implemented and first-original real-FIT checked: request point/risk/state from the
same complete median-anchored model, prune unused probability roots by compilation.
FixedB2incremental0.74ms vs3.43msfulloutput, sharedCPU; no new learned arm or added
skill/superiority claim. Full probabilistic query remains available and model still
training. Exact original score replay and joint predictive gate remain compulsory.

### ME095 — bounded selective state and known context

Whole model registered00295abc,104,689params,5fixedME094parents,2,048newupdates.
Adds input-selective affine level/velocity memory with a verified parallel prefix
scan alongside unchanged nonlinear native fast/slow/M, known operation priors and
current clock phases. Real-source causality/gradient/stream/cache checks pass;
training underway, no predictive result. Same fixed numerical gate, stricterfamily
interval, no protected data or future commands. Extra known context/cost disclosed;
not a controlled architecture win or proof that a particular memory block helps.


### SHARED-EVIDENCE-ME097 — trained contextual retrieval candidate (2026-10-25)

Status: EXPOSED_DEV_GATE_PASS; INDEPENDENT_QUALIFICATION_OPEN. TACPR391/PNDSPR9.
One95,874parameter core shared by six family configurations; one physical lane
trained. Separate context-conditioned query + direct retrieved correction;
input-only affine schedule preserves scan/stream equivalence. Five original
replication seeds clear5%whole/late persistence criteria and moving no-regression:
meanMSE0.000301391vs0.000467679, moving0.001496158vs0.002507034. Original archive
freeze1901d395 before scoresaaa58ad5; independent replay40163334508PASS.
No component attribution, compute-matched architecture winner, independent-site,
long-memory, control or customer utility claim. All10ME096/097 originals remain;
ME096 archive-order exceptionBLOCKED, ME097 unchanged new-seed replication.
ME0954completed+1interrupted/noREPORT retainedb30b5aaf; no hidden retries.
Next capability: coherent multi-step outcomes/uncertainty and independently
qualified source-supported delayed/identity-aware forecasting on this shared
engine. Do not simply lengthen retention again. Full neural Lean refinement OPEN.


### ME098 / ME099 — composed trajectory forecasts and latent uncertainty regimes (2026-10-25)

Implemented/trained: one shared TAC conditioning graph, fixed-coefficient future
mean semigroup, response-Gram joint scale, shared uncertainty regime and proper
interval objective.103,322/103,380parameters;10original real-data fits preserved.
TACPR392/393, PNDS10/11. ME099 MSE39.36/40.67/41.95%below persistence at1/4/16frames
on13exposedDEVepisodes; interval scores21–25%better thanME098, widths45–53%wider.
Four/fiveseeds fullpass; seed99001H4coverage84.192562<85 =>FULLGATEFAIL.
No morefits/changedgates in this correction protocol. Independent originalreplay
and actual-source values/gradientsPASS. FullNNLean/IEEE, independent cohorts,
otherdomainheads, usefullongmemory and action/customerutility remainOPEN.

Proposed NEXT-SHARED-NOISE: nonnegative state-conditioned future-noise envelope
within the response Gram; test middle-horizon reliability on new real data.
Proposed NEXT-SHARED-CLOCK: actual elapsed-time/entity corrections and typed
workflow time/lifecycle heads, no false completion or missingness claims.
Proposed NEXT-SHARED-QUERY: reusable coherent path sampling, square-root density
kernels and measured chunk/hardware processing. No new implemented skill claimed.


### ME100 / NEXT-SHARED-SELECTIVE — implemented sensor admission pilot

[TAC draft PR395](https://github.com/FPC-effortless/TAC-OSM/pull/395) and
[PNDS draft PR12](https://github.com/FPC-effortless/pnds-lean/pull/12). Five
97,235-parameter original models trained on UCI501 FIT 2013–2014 and were saved
before 2015 REPORT scoring. Native Temporal masked source sum/mass traces,
affine associative memory and six-pollutant coherent forecasts are implemented.
Long-context gap PM2.5 MAE is 38.986917 versus 46.073106 with recent history.
All five directional history gates pass, but the clustered interval includes zero.
Full gate FAIL: 0/5 pass, including gap coverage 67.69% below 85%. General PM2.5
MAE is 35.689 versus LOCF 34.516. The 967 local tests and original clean/GitHub
replay pass. Protected 2016–2017 outcomes remain unscored. No seventh family,
action intelligence, broad gain or Phase2 qualification is claimed.

### NEXT-SHARED-BELIEF — proposed uncertainty over missing world state

ME100 source retention improves gap point forecasts but fails uncertainty on the
cases needing retained observations. Learn a coherent latent belief over unobserved
current state and future change, conditioned on causal source age, evidence mass
and innovations. Consider bounded second-moment traces and a jointly trained
latent-regime emission mixture. Count state/covariance bytes and preserve
input-conditioned scan scheduling. Require actual 95k–105k parameters and a new
registration before training; retain all original models and ME100 failed gates.
Do not use the protected cohort for model selection. This is an unimplemented,
untrained hypothesis with no gain promised. Geometry, CED and sparse lookup remain
draft PR394 proposals, not prerequisites for this step.


### ME101 / NEXT-SHARED-BELIEF — implemented and trained; MIXED_TRADEOFF

TAC draft PR397 / PNDS draft PR13. Five fresh 97,259-parameter models; source-age
belief variance marginalized through a conditional affine emission; unchanged
native scan and 608-byte FP32 state. Ordinary PM2.5 MAE35.689→32.771 (8.18% lower),
with lower ordinary mean error for all six pollutants. Gap PM2.5 MAE38.987→37.747,
but gap density and interval scores worsen; coverage68.67% below85%. Full gate
FAIL0/5. All originals frozen3752a98e before scores5430d4b1; clean/GitHub replay
PASS40178388335; PNDS source/math PASS40177277628. 970 local tests. Protected
2016–2017 unscored. No further ME101 fits or changed gates.

### Capability idea queue — implemented advisory research workflow

PR397 adds scripts/research_idea_queue.py, CAPABILITY_QUEUE.json and a fast-loop
guide. It lists all preflight-valid registrations, excludes completed trials from
next-fit candidates, recommends bounded concurrent compute and keeps missing future
registrations distinct. It executes no fit or arbitrary shell. Live source, math,
resource and data-access gates are separate; a queue entry is not authorization.
Two lower-cost delegated tasks and two CPU fit workers were used in ME101; no
speedup or dollar-saving claim. Attribution remains deferred.

### NEXT-SHARED-BELIEF-LEARNING — proposed complete-model development

ME101 did not establish calibrated missing-state beliefs. Consider jointly learning
current-state representations and future outcomes, with authentic observed-label
reconstruction and coherent uncertainty regimes. Any generated observation masking
must be explicitly registered training augmentation over genuine labels, never
sold as independent natural-gap evidence. Use a bounded FIT-only selection budget
before the frozen complete-model study. Keep source-specific channel selection,
multi-horizon sensor paths, typed enterprise event heads and optional useful CED
in the shared queue. These are proposals, not new trained capabilities. Do not
consume protected TEST in selection or rerun the failed ME101 protocol.
