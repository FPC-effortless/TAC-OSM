# TAC-OSM architecture research — 2026-10-21

This dated addendum preserves the historical roadmap and every failed study.
It directs native architecture improvement. Standing real-world/parity rules
remain in force; draft studies are not merged or commercially qualified.

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
