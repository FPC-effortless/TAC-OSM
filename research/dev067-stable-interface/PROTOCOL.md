# DEV067-FROZEN-SEMANTIC-INTERFACE — registered before fitting

Question: does a stable calibrated interface, or generic log-probability actor
coordinates, improve decision fidelity without changing the underlying encoder?
DEV065 exact-public substitutions did not repair the actor. Do not infer that a
STOP weight, hard support mask, posterior correction or new recurrent cell is the
answer. No handwritten action/STOP rule enters this experiment.

## Factorial and common training changes

Four TAC heads cross moving/frozen encoder exposure with probability/log-probability
coordinates. GRU/frozen/log and no-CASM/frozen/log are retrained controls, not
ripped-out ablations. Each complete candidate has the same19-dimensional actor
input and parameter count within5% of TAC. Log coordinates are log(max(q,1e-12))/16
for every class, without a health-specific branch. Output beliefs remain ordinary
probabilities. All actions, costs, outcomes and STOP still come from trained weights.

Three unique encoders per initialization (TAC, GRU, no-CASM) receive4096proper
posterior updates. All four TAC heads share EXACTLY the same final encoder.
They are six candidate checkpoints per seed, not six independently trained
encoders. Encoder equality must be verified from saved weights/hashes.

Encoder training uses float32, stable log-softmax KL against normalized float64
public targets, weight4. Encoder EMA is float64, decay.995. Encoder and actor
gradient clipping are separate (norm1), removing DEV065's global-clip coupling.
No policy/future/cost gradient can enter an encoder. There is no hidden latent
actor input. Actor input beliefs are computed in float64, matching replay precision.

Each head gets4096updates. Moving TAC heads start after1024encoder warmup updates,
see3072evolving-EMA batches, then1024final-EMA batches. Frozen heads see4096final-
EMA batches. The same precomputed actor batch-index and Bernoulli(.5) exact-public
target mask schedules are used across heads. Encoder and actor schedules have
separate deterministic draws. Frozen predictions may be cached; they must be
computed by the learned final EMA encoder, never replaced by teacher beliefs.
Half of actor training rows explicitly use exact PUBLIC posterior targets.

Actor loss: teacher-policy cross entropy1, futureKL.25(divided by5choices), cost
Huber.1(target/40). Future targets are normalized distributions. AdamW decay1e-4,
EMA.995; actor training is float64. No STOP reweighting, analytical VOI selector,
privileged actual-state policy target or scripted fallback. Common data, proper-loss,
precision and update-budget changes are not evidence for freezing/log coordinates;
only within-study contrasts isolate those factors.

## Data and integrity

8192new snapshots namespace6671001, public histories32/128, missing50% initial
readings, sensor errors.1/.2, inspection prices.05/.1/.2 and0..9action depth.
An additional4096snapshots are uniformly sampled with seed6671011 from DEV065's
TRAINING-only DAgger pools, source hashes verified against the completed manifest.
Allowed legacy world namespaces are6652000+initialization*4+round, initialization
96501..96520, round0..2. No old test, fixed-query or intervention histories may be
loaded. Preserve source file/row references. This is offline legacy student-visit
augmentation, NOT online DAgger by the new students. Prior collection cost remains
part of the earlier study; do not call labels free real operational experience.

Validation1024new snapshots6671002. Policy validation uses32starts at each ofH32/
H128, same namespace. Pilot seed96700, LRs.001/.003, all three encoders/six heads
at each LR, all retained. Select one LR per backend by minimum worst normalized
validation tail max(p95/.05,p99/.1), then mean KL, then smaller LR; all TAC heads
use that same LR. Actor LR equals encoder LR.
Temperature {.5,1,2,4} is selected by validation policy NLL; it does not change argmax.
Every model uses final EMA, no best-test or intermediate-checkpoint selection.

Before fitting fresh seeds, a pilot recipe must have validation p95KL≤.05 and
p99KL≤.1, zero validation final faults, and≥95%completion in BOTH validation
histories. If no selected-LR recipe qualifies, close VALIDATION_REJECTED, retain
all pilots and keep test worlds unopened. This is a preregistered engineering
entry gate, not a20seed claim or a conclusion that freezing/log coordinates fail
universally. Do not extend its budget or quietly open test anyway.

If any recipe qualifies, choose the deployment candidate BEFORE fresh fitting:
lowest validation cost rounded to1e-6, then fewer parameters, then configuration
insertion order. All six arms then receive the COMPLETE20fresh-seed matrix,
96701..96720, including poorly performing arms/seeds. Primary mechanism arm is
fixed independently as tac_frozen_log. A control may be the validation-selected
research candidate; no test-winner substitution is allowed.

Only after all registered fresh fits complete: write permanent TEST-EXPOSURE.json
BEFORE generating any test data, including failures. Fixed512queries6671005;
64primary starts6671003 at H32/128/512/1024; secondary64starts6671004 at
(error.2,price.1) and(.1,.4), H32. These are shared paired physical starts across
histories/heads/seeds, not thousands of independent worlds. Smoke adds10000000,
8updates/8snapshots/2worlds, no legacy pools; excluded from all claims.

## Gates and interpretation

For the primary AND separately the validation-selected candidate, all must pass:

1. ≥18/20seeds ≥95%completion in EACH primary and secondary stratum.
2. No per-world final-fault excess versus the public reference in ANY required
   stratum/seed. Report transient harm, unnecessary repair and healthy budget
   exhaustion separately; zero final faults is not zero harmful interventions.
3. ≥18/20individual seeds have BOTH fixed p95KL≤.05 and p99KL≤.1. Also report
   medians, every tail, Brier/TV, calibration and public/privileged reference gaps.
4. Paired-seed bootstrap95cost upper difference≤+.02 versus both GRU/frozen/log
   and no-CASM/frozen/log. This is noninferiority, not superiority/equivalence.
5. Complete source/checkpoint/state/trajectory/standalone replay, exact actions,
   float64/thread1 and numerical tolerance1e-8. Current reference epsilon1e-6
   deterministic candidate tie-break remains unchanged. Relative event clocks
   and semantic-only actor must pass structural invariants.

Mechanism contrasts: moving/frozen main effect, probability/log main effect and
interaction on completion and cost; paired-seed intervals/exact sign flips, Holm
over all six tests. A primary-versus-moving/probability completion gain requires
at least5percentage points and paired lower95bound>0 before calling it a useful
recipe gain. If the task is saturated, no benefit claim follows from parity.
The identical TAC encoders mean posterior-quality improvements cannot be credited
to the decoder treatment. Any improved common belief recipe is a separate finding.

Post-fit interventions: fresh6671006,64starts,H128/H1024,errors.1/.2,price.1,
closed-loop predicted/exact-public/blurred modes. Each mode follows its OWN public
history; no future truth enters a learned input. Candidate must also reach≥95%
completion in predicted and exact modes for≥18/20seeds with zero final faults;
blurred outcomes are damaged-agent diagnostics, not original-model failures.
Fixed-query same-public-graph shuffled-belief choices are reported as sensitivity
diagnostics, not a maximize-flips objective. Exact-action changes alone are not
regret. Holdouts/weights cannot be tuned after these probes.

## Budgets, boundaries and stopping

Two single-thread CPU workers; fit/preparation/pilots/selection≤9600CPU seconds,
≤600CPU seconds per three-encoder/six-head group. No paid compute. If any required
group fails/caps or the entire matrix cannot fit, preserve INCOMPLETE and keep
test closed. Counterfactual cap1200CPU seconds; replay/accounting separate. All
raw/final EMA weights, cached-source provenance, curves, failed outcomes, source
commits and pilots are retained. First fresh seed96701 exports every arm; never
the best seed. No fitting during CI.

If validation rejects or the complete fresh matrix fails, suspend this registered
synthetic actor recipe; no extra epochs, probability threshold, STOP penalty or
second test selection in this ID. The commercial goal stays active, with real
sensor baseline/repair-data dependencies explicitly unmet. A new question needs
a new source-frozen protocol and untouched evaluation.

Structural priors remain two entities/five actions, typed46fields, public16-vector
keys/cosine binding, a supervised six-type gate, FIFO64,12health×graph classes,
dense neural CASM and an event-clock prior for time-independent physics. Public
posterior/known-law supervision is not autonomous discovery or real maintenance.
No MTSK revival, CASM superiority, general symbolic programs or C5 scaling is
claimed. Prior PR225/222 evidence and MetroPT July B stay intact. Concurrent
SEQ-TRANSFER-002/003 reports are bounded frozen-probe findings, not an independently
replicated commercial/general-purpose capability claim for these controllers.
