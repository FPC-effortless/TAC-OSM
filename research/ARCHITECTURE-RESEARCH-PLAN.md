# TAC-OSM architecture research — 2026-10-21

This dated addendum preserves the historical roadmap and every failed study.
It directs native architecture improvement. Standing real-world/parity rules
remain in force; draft studies are not merged or commercially qualified.

## Current direction: complete compute-adaptive models (ME043)

Architecture-first development supersedes the diagnostic prerequisites in the
historical sections below. Build/train complete models, measure accuracy and
actual resource costs, then iterate. Mechanistic attribution is deferred;
real-source integrity, frozen test access, numerical correctness and honest
comparison contracts remain compulsory.

ME040 shortened observed-token native processing; ME041/042 each trained an
11-architecture portfolio on authentic historical development data. The
published workflow outcomes favor different architectures across domains and
resource budgets, and no new portfolio candidate passed both gates. See
[ME041](https://github.com/FPC-effortless/TAC-OSM/pull/306) and
[ME042](https://github.com/FPC-effortless/TAC-OSM/pull/307).

[ME043](mathematical-engineering/me043/PROTOCOL.md) implements a context-first,
input-routed native fast/slow/associative-memory expert with shared prediction
heads and learned compute preferences. Registered 96 real full-model fits,
including unchanged native/context/GRU references on both historical domains.
The 96 A +64 separately registered B fits are complete. B preserves full context
capacity and executes native training every eighth update; this reduces measured
training CPU versus the dense adaptive design, but no candidate meets the joint
cross-domain accuracy gates. [All original results](mathematical-engineering/me043/RESULTS.md)
remain exploratory and unpromoted. New model development continues without
synthetic capability claims or reopening
previously consumed final tests. Only observed results can establish whether
this design improves a practical frontier; independent new-site qualification
remains separate.

## Established findings and immediate engineering

ME024 RMS readout improved optimization; ME025 timescale additions failed the
materiality gate. ME026 improved equivalent execution. ME027 exposed strong
nonlinear common-feature controls and inconsistent native-state value. ME028's
nonlinear observable readout improved prediction but did not qualify state;
ME029's chronology-vs-sorted and chronology-vs-pool gates both failed. Its
chronological mean Brier .164198272 is historical DEV engineering evidence.
Repeated cohort use is not independent confirmation.

ME030 tests a native limitation: an additive state/context logit cannot adapt
state sensitivity to the query. A joint 12-unit tanh head preserves recurrence
and is compared with the additive head and an equally expressive joint pooled
head. See [protocol](mathematical-engineering/me030/PROTOCOL.md) and
[derivation](mathematical-engineering/me030/MATHEMATICS.md). This is readout
engineering, not conditional-information identification or chronology qualification.

## Subsequent gates

1. Freeze representations and test conditional residual prediction with FIT-user
   cross-fitting, identical nonlinear readout interfaces, capacity, sampling and
   selection controls. A user's features must come from a representation trained
   without that user's records. Estimate finite-family predictive gain, not mutual
   information. New reserved-cohort access requires a separate preregistration
   after design/preprocessing freeze; current future cohorts stay closed.
2. Only after useful state is established, test event-boundary compression at fixed
   total state capacity. Include observed-boundary, random-boundary and fixed-window
   controls with equal information and exposures. Preserve predictive distinctions,
   not reconstruction alone. The earlier proposed ME028 compression stage was
   deferred: ME028 became readout engineering, so compression needs a future ID.
3. After compression earns its cost, test global/episodic/local state allocation
   at fixed coordinates and parameter budgets. Isolate allocation from extra head
   expressiveness or prediction supervision.
4. Separate observed-world updates from internal computation: hold world state s
   fixed while updating reasoning state r. Adaptive depth must improve observed
   outcomes enough to pay measured compute costs. A reasoning step is neither a
   new observation nor an external action. KuaiRec factual watch outcomes do not
   supply identified intervention effects or actual decision costs.

## Literature-informed hypotheses

- [GenRec](https://arxiv.org/abs/2608.10257): align context and readout to downstream
  prediction. TAC hypothesis: query conditioning may matter more than extra memory.
  Event verbalization is optional; Netflix results are not TAC evidence.
- [Sep-Latent](https://arxiv.org/abs/2610.22924): concentrate predictive information
  at meaningful boundaries. Test genuine event anchors against controls; language
  reasoning compression results do not validate physical memory.
- [Next-Concept Modeling](https://arxiv.org/abs/2610.14162): investigate multiple
  predictive resolutions after state value is established. This does not prescribe
  vector quantization, Transformer conversion or a parallel text decoder.
- [Hybrid Reasoning Agents](https://arxiv.org/abs/2610.18861): distinguish internal
  latent computation from external interaction. A TAC adaptation needs authentic
  outcome/cost evidence, instrumentation and a tested stopping rule.

These papers motivate hypotheses; their reported metrics are not reproduced or
used as TAC thresholds. Native TAC can use query conditioning, memory structure,
credit assignment and computation allocation without becoming an SSM or LLM.

## Evidence discipline

Freeze complete matrices before fits; retain every attempt, checkpoint, optimizer/
RNG, curve, source digest and actual resource receipt. Separate numerical proofs
from capability evidence. Require calibration, entity/time shift checks and closed
final cohorts. Equal caps are not equal spend. Any parameter, data, actual-compute,
selection or evaluation parity failure blocks architecture-superiority claims.
Keep practical simple references while each experiment diagnoses a native mechanism.

## ME030 completion

All nine registered fits are complete; the joint native head failed all gates.
Brier .167361674 vs additive .164198272; calibration worsened, and mean-state
replacement helped every joint-native seed. The head is rejected under this
recipe. See [verified results](mathematical-engineering/me030/RESULTS.md).
Conditional residual cross-fitting remains proposed and unexecuted. No compression,
hierarchical state or internal reasoning mechanism is promoted by this failure.

## ME031 completion — entity-isolated residual probe

The proposed cross-fit diagnostic is now executed:36fits,3rotationsx3seeds, separate
representation/readout/evaluation users, no DEV scoring or reserved cohort access.
State residual Brier .164980411 vs common-only residual .162310484 and frozen base
.162070604. State loses all9cells; calibration guard passes, predictive-value gates
fail. This is finite-family prediction on previously used FIT, not an information
estimate or independent architecture qualification. See
[full results and limitations](mathematical-engineering/me031/RESULTS.md).

The next unexecuted native architecture hypothesis targets representation learning:
train the state to reduce genuine response residuals beyond a separately trained,
frozen common predictor. Ensure gradients reach memory, preserve user separation,
and give control encoders equal feature/target exposures and resource allowances.
Preregister separately before any new fitting. ME031 froze state during probing;
it did not execute this objective change. Compression/hierarchy/internal reasoning
remain gated; none is justified by this negative result.

## ME032 completion — centered conditional memory training

The proposed representation-training change is now executed:9R-only common-head
fits+27new memory fits, all with user-separated fitting/evaluation. New centered
correction delta(h,C)=w'h+v'[tanh(Bh+AC+a)-tanh(AC+a)] guarantees delta(0,C)=0.
All27memories receive nonzero credit after initialization; conditional common
heads remain exact while joint control adapts them. Native recurrence unchanged.

Conditional native Brier.165824448 improves the R-only teacher.170203796, but
joint native.165650858 and pooled.163399886 are stronger within this matrix.
Teacher-gain/calibration pass; trained-control/fold/bootstrap gates fail. No
candidate promoted. [Verified results](mathematical-engineering/me032/RESULTS.md).
Source remains previously used FIT, not an independent or prospective temporal
qualification. A learned nonzero constant state can still supply context correction.

Next unexecuted native-engineering test: isolate S-specific common/embedding
adaptation from ordered-memory effects, with an S-adapted common reference and
freeze/unfreeze embedding controls. The R teacher's lack of S fitting is not
proof of memory value. No post-hoc fit is added to ME032. Only after useful native
state is established should fixed-capacity learned summaries, compression,
hierarchy or adaptive reasoning advance. Existing plan hypotheses remain intact.

## ME033 completion — semantic adaptation isolated; exact storage optimization

54new fits test history/query identity path x frozen/adapted item embeddings,
plus native/common adaptation controls. Common-head-only S adaptation is strongest
(Brier.161558311); history frozen.162330508,history adapted.163599877,native adapted
.166925030. Updating embeddings worsens BOTH factorial paths in all3pooled seeds.
History/native candidate gates and adaptation-specific interaction fail. Frozen
latents are reconstructable from existing C, so their gain is finite readout
fitting, not additional ideal historical information. See
[verified results](mathematical-engineering/me033/RESULTS.md).

Component moment states are48/16float32bytes, checkpoint-fixed; their sums/counts
compose and reproduce real prefix forecasts within1.19e-7. They are not full
forecast state, and embedding updates invalidate sums. A separate no-fit exact
storage wrapper shares identical frozen item tables:512000tensor-weight bytes
saved per eligible model, all18models/64656forecasts bit exact. This is storage
engineering, not native predictive superiority or model promotion.

Next unexecuted native optimization: freeze the R-pretrained input embedding while
learning native dynamics/readout, with strong common-only S adaptation included.
ME033 did not fit native_frozen. Separate registration before fitting; no post-hoc
addition. New independent/out-of-time qualification and state-value gates remain
necessary before compression,hierarchy or adaptive reasoning.

## ME034 completion — stable inputs help; new-user gate fails and recipe closes

The formerly unexecuted frozen-input proposal above is now completed in
[ME034](mathematical-engineering/me034/RESULTS.md),72new real-data fits.
R-only pretraining provenance is verified: no all-FIT/evaluation-user embedding
pretraining within a rotation. On5841later observations/438new remainder6users,
native frozen Brier.157755632 improves adapted native.158996207 but loses to
common adaptation.156919508 and boosted.155062852. The registered adjusted
Brier/AUC gates both FAIL. Source freeze260fd62b precedes fitting and final access.

**Retire positive native-memory claims on this KuaiRec task and close this stable
input recipe.** Remainder6 is exposed; no tuning, new-gate test, hierarchy,
compression or latent-reasoning extension on it. Existing negative results and
native mathematics remain preserved. No model promoted or merged. A stronger
readout result does not prove recurrence is universally useless or that its exact
failure mechanism has been isolated. Future work needs a genuinely new real task,
objective or source rationale, registered before outcomes, with a train-only
demonstration that ordered history offers headroom over strong summaries.

Frozen-table sharing extends to27native/pooled/GRU runtimes,512000tensor-weight
bytes saved each with exact forecasts. 971tests+12subtests;594279neural forecasts
exact through checkpoint/script/raw pipelines,84897tree forecasts exact. Full
compute parity remains unqualified despite native/GRU parameter gap0.097% and
operator-subtotal gaps<1%; no architectural-superiority claim.

## User-directed architecture-first phase — ME035/ME035B

The user's current priority supersedes the proposed diagnostic-first headroom
sequence: implement and train model improvements now. Use diagnostics only to
resolve a concrete integration/correctness problem or guide the next fix;
proving unique state value is not a prerequisite for bounded real-data model
development. Keep original stopping decisions and qualification boundaries intact.

[ME035/ME035B](mathematical-engineering/me035/RESULTS.md) implements a joint
trainable prediction path/SwiGLU readout, causal normalization with retained
level/volatility, residual slow-state updates and multiscale initialization.
All first-round TAC stages18924weights, same training/exposure. On authentic
ETTh1 multivariate1/6/24h development forecasting, same250-step MSE improves
from.497194 to.432309 (13.05%, all3seeds). Following the remaining ridge gap,
a separately registered second build adds a trainable full-history AR backbone,
training-only ridge initialization and anchored joint adaptation. The selected
32154-weight model scores.369106, comparable with ridge.370001; the tiny mean
edge is not consistent across seeds or independently qualified.

15neural fits+1ridge fit, all attempts/checkpoints retained. The final candidate
adds capacity and warm-start training; no matched superiority claim. Selected
raw-unit TorchScript models are available. One strict initial float32/float64
equivalence tolerance fails by1.41e-7 and remains recorded; all509796checkpoint
forecast values replay bit-exactly. No promotion/merge or KuaiRec access.

Next engineering should target lower error, efficient normalization/streaming
and stronger trainable predictive paths, retaining useful components instead of
requiring another standalone diagnostic study. Allocate new bounded development
budgets transparently and qualify generalization on separately registered real
cohorts when advancing a candidate. Do not retune on ME034's closed cohort.

## ME036 completion — full-horizon native patch forecasting

Architecture-first construction continues in [ME036](mathematical-engineering/me036/RESULTS.md):
ordered eight-step patches, native fast32/slow32 memory per channel, residual slow
updates, trainable multiscale rates, SwiGLU/32-basis prediction and a joint AR path.
All models start from scratch. Five authenticated benchmark files; full sequences
at96/192/336/720steps; standard chronological splits and every test anchor.
ETTh1/ETTm1 are resolutions of the same equipment records, not independent sites.

280neural trajectories/two seeds each trained1000updates; AR-only, matched context
and GRU corrections, compact DLinear/PatchTST/iTransformer-style controls. Macro
MSE native.346956, AR.348944, context.343989, PatchTST-style.345084. Native's0.57%
AR gain misses the registered2% gate. At720steps native regresses17.20%/11.02%
against AR on ETTh1/ETTh2. No promotion, paper parity or matched-compute superiority.

The original locally normalized ridge fit optimized a different residual objective;
retain its scores. A separately recorded **post hoc** fixed variance-weight correction
(alpha100, no tuning/TAC changes) scores.389767 macro MSE. This is corrected
reference evidence, not a fresh prospective confirmation or a revised gate.

1680checkpoints/81962496validation forecast values replay exactly. Forty selected
native models have both standardized and physical-input standalone exports;
3902976export forecast values replay exactly. Whole-window normalization still
requires full96-step replay. The old ME035B precision exception remains unchanged.
995tests+12subtests pass. Original models, optimizer/RNG states, data/provenance
and logs are retained; no KuaiRec access or synthetic capability fits.

**Next construction, not a diagnostic gate:** preserve strong linear and nonlinear
context prediction paths while integrating native state, add horizon-conditioned
residual control and richer channel coupling, and train the complete model with
transparent budgets. Target the long-horizon regressions directly. Detailed
mechanistic attribution remains deferred; retain competitive controls as columns.
Use a new registration and disclose benchmark exposure before fitting; this
completed test matrix is no longer a fresh qualification cohort.

## ME037 completion — bounded horizon control and integrated context

[ME037](mathematical-engineering/me037/RESULTS.md) builds and trains a complete
native fast/slow forecaster with a learned full-history context branch, joint AR
backbone, SwiGLU readout and context-dependent monotone horizon gate. A bounded
`tanh` correction prevents unlimited head rescaling from defeating attenuation;
the resulting amplitude envelope is proved, without claiming a loss guarantee.
ME036 already provided explicit level/scale context; 720-step outputs are direct
forecasts, not 720 recurrent updates. State dilution is not an established cause.

All 192 adaptation trajectories and eight new-house AR parents completed. Native
macro MSE on the five exposed benchmarks is .339944 vs AR adaptation .347832:
2.27% lower, passing the registered development gate. Context .341093 and GRU
.341475 remain close. At H720 native versus AR is +0.13% ETTh1 / -0.28% ETTh2,
substantially smaller regressions than ME036. Historical native .346956 vs .339944
is an unmatched system-development improvement, with more capacity/pretraining.

A separately reserved temporal test from UCI374, one Belgian house with real
energy/temperature/humidity measurements, scores .862641 native vs .870339 AR:
0.88% better, **failing** its separate 2% gate. Energy-only MSE .547785 vs .589338
is reported separately; individual MAE/channel outcomes vary. This is training on
the house's past, not zero-shot unseen-house transfer. New test outcomes are now
exposed; no tuning on them or retrospective change to the registered gate.

864 original parent/adaptation checkpoints replay 37,062,144 validation scalars
bit exactly; 48 native models have standard/raw exports, 4,118,016 exact values.
1,005 tests plus 12 subtests pass. All old 1,680 ME036 checkpoints are unchanged;
ME035B's historical precision failure stays visible. Actual compute differs, so
no architectural-superiority, modern-paper parity or mechanistic attribution claim.

Continue architecture-first construction in a separately registered next build:
add efficient cross-channel prediction and input-conditioned temporal feature
selection while preserving the strong context/linear paths and bounded forecast
correction. Train complete candidates; use matched controls as table columns,
not as a prerequisite mechanistic study. Reserve a new real cohort before any
further qualification claim. Forecast improvement does not demonstrate action
control: an action-conditioned extension needs genuine actions, costs and observed
outcomes. PR300 remains a draft task-specific foundation-model comparison, not
proof of TAC control strengths or a prohibition on forecasting research.

## ME038 completion — innovation-coupled native state transition

[ME038](mathematical-engineering/me038/RESULTS.md) changes the architecture's
native dynamics: an 8x8 associative state uses a normalized-key latent prediction
error write and retrieves into BOTH fast32 and slow32 updates. Shared historical
channel context feeds the coupled state and all neural controls. It is a new
implemented TAC variant, not a first-in-literature claim for established delta
writes or associative memory. Conditional write contraction and absolute state
bounds are specified; full coupled Jacobian contraction is not claimed.

All 192equal-update and 48separate five-second training CPU model fits completed,
plus 8new AR parents. Native parameters match context/GRU within 1%; recurrent state
is 128 float32 values/channel, twice ME037. Actual CPU in the exposure lane differs;
CPU-budget lane consumes different examples and total pipeline costs differ.
Neither qualifies architectural superiority. Numerical correctness is not
mechanistic attribution, and attribution remains deferred until competitiveness.

New UCI849 Tetouan temporal test, one city/three distribution zones: exposure-lane
power MSE native.153310, context.156326, GRU.156362, AR.202714. Native improves AR
24.37% and context 1.93%, H720 versus AR -12.85%; the registered PRIMARY power
engineering gate PASS. It is not unseen-city transfer, causal action utility or
multi-site population qualification. Power/MAE and all 8-channel scores are shown
separately. All Tetouan test periods are now exposed; no retuning on them.

At the 5s training deadline (H96/720), native power MSE.151090 versus context.146461,
GRU.151773 and AR.192950: context is better; separate gate FAIL. Exposure native
training averages 10.88 CPU seconds versus context 2.16. Legacy native macro.340467
is 0.15% worse than ME037.339944; new development target FAIL because context gain
0.80% misses 1%. **ME037 remains the preferred legacy benchmark candidate**; ME038
is a task-specific architectural research candidate, not a universal replacement.

All 1,008 parent/adaptation endpoints replay 37,310,976 validation scalars bit exactly,
8,448,000 initial AR values exact;48 matched optimization prefixes replay 23,103 steps.
Independent NumPy reduction audit recalculates all 52 Tetouan selected/reference
scores (1,479,493,632 forecast values) within a prospectively registered float64 bound.
The old 1,680 ME036/576 ME037 checkpoint hashes remain unchanged. 1,020 tests +12 subtests.
All original artifacts and old precision failures remain preserved, no merge.

Continue construction with native state interaction and efficiency as joint
engineering objectives. Reduce projection/loop overhead and improve how memory
supports forecast-specific corrections; do not require another mechanistic study
before building. Any new learned variant needs a separate registration and a new
reserved real cohort for qualification. Equivalent execution optimization may use
frozen models with exact prediction checks; it does not establish new accuracy.

## ME039 completion — measured native execution and repository delivery

[ME039](mathematical-engineering/me039/RESULTS.md) retains the new ME038 native
architecture and adds an explicit compiled-core training path. Eight real-data
cases, three paired repeats, 3,072 optimizer updates: 1.076x steady / 1.067x total
speed, below registered 1.20x/1.10x targets. Native-gradient/parameter numerical
agreement passes the registered engineering tolerance; retraining is not bit exact.
All 60 frozen models match 4,224,000 standardized/raw saved validation values exactly.

Serving is 1.338x faster than eager Python, but that is not the best pre-existing
serving baseline. The post hoc comparison with original ME038 full serialized
exports gives 0.963x: retain the ORIGINAL serving format. A second separately
registered freeze construction failed exact physical replay (1,574 different
values on the first failed case, max .015625 original units), was rejected, and did
not finish its remaining/timing gates. Do not loosen precision after measurement.
No new accuracy, architecture-novelty, controller or generalization claim.

Original 60-model ME038 bytes plus all 48 ME039 optimizer/RNG continuation states,
traces and gradient/parameter snapshots are now stored in checked repository
archives, alongside 60 clearly labelled experimental exports. The root README,
research index and evidence/claims/model cards link the latest studies. CI hashes
original payloads and provides a repository-owned Actions download. The older
full 1.05 GB evidence archive remains local after authenticated asset upload failed;
its exact location/SHA/status are recorded, with no false remote-availability claim.

Continue architecture-first development toward fewer sequential projections and
forecast-relevant native state interactions. Compiler wrappers alone have not
solved the measured CPU disadvantage. Preserve all negative attempts and the
preferred original serving model. Fresh learned variants require a new source/
cohort registration for qualification; no tuning on closed Tetouan/household/
KuaiRec tests, and no prerequisite standalone mechanistic study.
