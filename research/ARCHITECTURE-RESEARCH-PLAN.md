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
