# DEV-040 — Source-reconciled comparison draft

**FULL DEV-040 DRAFT. NOT REGISTERED. ITS TUNING/CONVERGENCE STUDY HAS NOT BEEN DISPATCHED.**

A separate bounded [DEV-040A pilot](DEV-040A-PILOT.md) was subsequently run.
Its [results](DEV-040A-RESULTS.md) do not complete or register this full study.

Reviewed 2026-10-17 against DEV-039 PR #194, head
`967c0e40420c76ad67c77d48e96a83b67d4af63c`.
DEV-040 appears unused on the inspected lineage and branch search; reserve it
only after another repository-wide numbering check. This document revises the
proposed design, not any historical contract. It does not claim raw-evidence
replay: the newest archive manifests identify workspace-only files unavailable
in this execution workspace.

## 1. Verified facts and corrections

| Item | Source-reconciled finding |
|---|---|
| DEV-039 mature primary | 312/384 original; 354/384 refitted MLP and linear; 42 additional correct worlds |
| DEV-039 cancellation | 120/192 to 162/192; 42 additional correct worlds |
| DEV-039 reissued cancellation | 48/96 to 66/96; 18 additional correct worlds |
| Pairing | 192 mature pairs, 96 cancellation pairs, 48 reissued pairs; these are correlated worlds within one trained-memory initialization |
| DEV-039 training | 576 worlds, all four roles equally represented; 500 full-batch head-only updates |
| DEV-039 loss | 576 x 8 entities x 3 query types = 13,824 labels; mature reissued target-wait labels are only 96, or 0.6944% |
| DEV-038 probe input | Target-local bank96 + encoded query12 = 108 features; existing readout3 + query12 = 15 features; no raw event sequence is supplied to either probe |
| DEV-038 split | 96 probe-training worlds and 96 held-out worlds; 48 held-out counterfactual pairs |
| DEV-027 budget U0 | 216 optimizer steps at B2/H512: 432 episodes and 221,184 pre-sensing public events; not 216 epochs or episodes |
| DEV-023 regimes | S: H128/B8; L: H512/B2 |
| DEV-027 A/B | Whole-model seed offsets23000/24000, with the same initializer; not different initialization algorithms |
| DEV-036 precision | The _fp64 suffix refers to gradient-norm clipping; the documented recurrent state is768 FP32 scalars |
| Current contract schema | Uses experiment_id, not id, and requires exactly one global primary endpoint |

The reported percentages do not establish denominators: 92.1875% can equal
59/64 or 354/384. The actual DEV-039 denominator is384. Its audited summary
reports a paired-world bootstrap gain interval of +8.07 to +14.06 points for
mature accuracy and +16.67 to +26.56 for cancellation. Those intervals are
conditional on one trained representation; they do not estimate fresh-training
seed reliability, and have not been independently recomputed here.

The DEV-039 runner explicitly asserts that its training/evaluation fingerprints
are disjoint. DEV-036 uses dev036-v1-holdout, DEV-038 uses dev038-v1-probe, and
DEV-039 uses dev039-v1-train/dev039-v1-eval, with different cohort seed bases.
There is no basis here to assert that DEV-039 reused the earlier evaluation
worlds. A complete cross-study fingerprint/template census is still absent.
Fresh holdout data and confirmatory research are different properties: DEV-039
can have a held-out within-study comparison while remaining adaptive development.

The DEV-038 result supports supervised held-out decodability from its frozen
state/readout. It does not establish that training the writer created that
information. A random recurrent model can itself retain information; a strong
probe on that control would weaken learned-writer attribution, not demonstrate
event-only decoding or absence of retained state. Preserve this distinction.

The manifests explicitly say the DEV-038/039 raw archives were not uploaded to
Actions. DEV-039's workflow explicitly performs integrity tests only. Source,
summaries and hashes are inspectable; full archived trials are not thereby
independently downloadable or reproduced.

## 2. Sequence and release gates

Stage A: publish/checksum the original DEV-036–039 evidence, reconstruct the
cohort/split census, profile capacity and runtime, and implement a config-aware
generator. Do not alter historical outcomes. A budget/preflight pilot may use
development cases only and must not access the eventual scientific test.

Stage B: finalize the finite model/recipe matrix and statistical analysis plan,
freeze them with source and configuration hashes, then conduct tuning on
development splits only. Fit/calibrate heads without test access.

Stage C: freeze selected recipes and every retained checkpoint for all final
draws and checkpoint times. Generate and seal the fresh test cohort afterwards;
evaluate all four saved checkpoint times only after training and selection end.
Retrospective test curves are allowed; live test-guided training is not.

Full PLM–CDL–CASM–OSM integration remains a separate prospective capability
study after candidate selection, with preserved original implementations,
component ablations, executed verification/repair and full decision costs.
This memory-comparison study must not label a common-head writer adaptation
as the unchanged complete TAC-OSM system.

## 3. Arm registry and references

Keep the eight proposed base candidates: native global DeltaMemory writer,
masked-local writer, bounded banked entity-local MTSK, learned schema41,
dense GRU, equation-level gated delta, full-history transformer and memoryless.
Bind each to an exact constructor, source hash, dtype and input/readout interface.
Preserve original CASM/CDL/six-expert MTSK as separately identified candidates;
the eight-bank/eight-expert adaptation is not the original six-expert kernel.

GRU and delta each have an M-state and an M-param variant. M-state targets768
persistent scalars including auxiliary responsibility state. Report bytes and
precision: FP64 and FP32 state have different storage at equal scalar counts.
M-param targets an active trainable-parameter ceiling of8,674, with closest
feasible natural architecture declared before tuning. Do not pad with unused
parameters to manufacture exact equality. M-param state size may differ from768;
publish both resource coordinates. Profile feasibility before registering widths.

The transformer retains full history and incurs its storage/attention costs.
The memoryless model receives current public observation and proposed-action
query, with no recurrent history. Supplied entity addressing is disclosed.
Use common public features and comparable heads where possible; document every
architecture-dependent adapter. Refit frozen memory/encoder/query/adapter with
both linear and MLP heads, preserving the untouched original prediction head.

Every cost table retains the exact public-history controller, majority class,
always-service, never-service, and a precisely specified per-entity last-event
table. The last-event table stores only each entity's most recent public event;
it must not silently become the full generation/schedule sufficient-statistic
tracker. Calibrate each arm's own probability-difference service threshold on
development/calibration data only; retain the fixed physical threshold too.

## 4. Probe and head controls

For the primary probe use the exact pre-head features and public query, with
train-only standardization, a fixed probe budget and paired held-out cases.
Architecture-specific raw-state probes are separately named diagnostics.
Compare against:

- Identical-architecture fully random initial memory/encoder, as a reservoir reference.
- The trained encoder/query/adapter with only writer weights reset to the matched
  initial snapshot, to narrow writer attribution; disclose distribution shift.
- Matched end-to-end retraining with writer frozen from initialization, if a
  causal claim about learning writer parameters is sought.
- Pair-preserving shuffled training labels, evaluated on true held-out labels.
- Query-only and current-observation-plus-query predictors; verify identical
  queries/current cues within opposite-future pairs.
- Runtime state reset and counterfactual state swap, preserving public queries.

Probe selectivity is an operational comparison, not the definition of retention.
All feature construction must accept only model state and public inputs, never
hidden simulator fields or a ground-truth generation channel. Label/target
selection may supervise training but must not become an inference feature.

DEV-039 already has balanced roles. Its focused DEV-038 probe and all-role head
use different objectives, features, initialization and optimization recipes, so
their disagreement alone does not identify head imbalance as the cause.
Retain uniform full-bit CE as the baseline. Add a separately registered
loss-balanced diagnostic, e.g. half the loss on mature target-wait labels equally
by role and half on the remaining labels. The exact weights must be frozen
before outcomes; this is a proposed recipe, not an implemented result.
Report role x deadline x query x target/other loss contributions. Balanced case
sampling without changing full-batch loss weights is not a new intervention.
Tune linear and MLP heads fairly on the same frozen features. Compare against
threshold calibration before attributing a benefit to head complexity.

## 5. Revised hypotheses and permissible conclusions

H1: a development-selected unstructured comparator attains the prespecified
H512 target after extended training. Keep median>=.70 as the drafter's prediction
if adopted, and <=.55 as a poor-performance region. Scores between them are
inconclusive. Even every comparator remaining <=.55 establishes failure of
these implementations/recipes/budgets, not that a schema is necessary.
Select the comparator on development data, not by maximizing test medians.

H2: quantify fresh-draw dispersion and, separately, persistence of the originally
sampled A/B initialization contrast under a common optimized recipe. Seed
offsets sample the same distribution. A new A/B offset comparison is not a test
of two initialization algorithms. If a genuine initialization-method comparison
is wanted, specify two different initialization distributions as a new factor.
Equivalence requires its entire interval within +/-5 points; failing to detect
a difference does not establish equivalence. Do not tune family assignment.

H3: test masked-local and bounded-MTSK distance from schema41 on the same sealed
cohort. Prespecify superiority/noninferiority/equivalence separately. For
equivalence use two one-sided tests and the registered +/-5-point margin,
including the multiplicity treatment below. An outside-margin interval concerns
this task and resource recipe, not a universal architecture ordering.

H4: measure head improvement at fixed representation. Use the paired contrast
(refit-original) minus0.5*(schema-original) for a 50%-gap-closure hypothesis.
Report raw point improvements. Report the ratio only when the original gap is
positive and at least5 points, as declared in advance; otherwise label it
undefined/uninformative. Do not divide by a tiny or negative gap or clip ratios.
A modest closure does not prove information absence or identify a memory limit.

H5: compare trained versus matched-control recurrent features on reissued
cancellation using identical probe procedures. +20 points and <10 points can
remain prospective selectivity regions, with the middle region inconclusive.
Low selectivity means no established advantage from training under this probe;
it does not mean raw events bypassed recurrent state. High selectivity alone
does not establish deployed decision reliability.

Never erase DEV-017/018/027 fixed-budget results. If later optimization succeeds,
update any overly broad interpretation while retaining the earlier outcomes,
budgets and cohorts as development evidence.

## 6. Training and finite compute

U0=216 optimizer updates. Main checkpoint times:216,864,3,456,13,824.
Define a common13,824-step scheduler horizon for all prefix comparisons, with
saved optimizer/RNG/scheduler states for continuation. A cosine schedule ending
at each shorter budget would change the recipe as well as training length.
Declare warmup and curriculum switches against this common horizon.

S means fixed H128; L means fixed H512. Curriculum switch times and batch sizes
must be explicit. A possible draft is H32 for updates0–215, H128 for216–863,
then H512, with batches32/8/2 to match1,024 pre-sensing events per update.
This still does not match episodes, supervision or receipt exposure; report them.
No claim of a horizon-only intervention follows.

Equal trials, optimizer updates and wall time within10% are incompatible for
architectures with very different per-update work. Choose equal tuning
opportunities as the primary allocation, report actual FLOPs/time/exposure,
and place any equal-compute comparison in a separately budgeted analysis.
Use FLOP accounting/profiling appropriate to each kernel, not scalar count.

The draft's eight-arm/three-regime design already involves11,943,936 updates
if each top-two continuation/restart also uses two seeds. Including both
GRU/delta matching modes makes ten candidates and14,929,920 updates, excluding
init-factor expansions, head fits, probes, evaluation and n=40 extensions.
This assumes fresh full-budget top-two reruns; resuming their16x checkpoints
reduces those counts. Final training alone for ten arms in L sees about2.83
billion pre-sensing events at B2/H512. A compute preflight and finite resource
allocation must precede a full launch; a small pilot is not the completed study.

Less than1-point dev-accuracy movement is an accuracy plateau, not proof of
optimization convergence. Check finite loss/gradients, CE trends, per-stratum
learning curves, constant predictions and learning-rate state. Chance can be
a stable plateau. Report non-converged or degenerate runs without dropping them.

## 7. Cohorts, configurations and sealing

Implement configuration identity before claiming config-disjoint splits.
A namespace or hash-disjoint history is not by itself a new entity
configuration. Define the template, identity mapping, initial-state constraints,
transition rules, nuisance generator and counterfactual intervention explicitly.
Counterfactual partners remain in the same configuration and resampling group.
Permuting eight supplied IDs is not learned entity discovery.

Use disjoint training/development/calibration/test configuration manifests and
disjoint public-history fingerprints. Retire previously inspected cases and
templates from confirmatory selection. Archive a ledger of every known prior
cohort, with unknown entries labeled unknown rather than certified absent.
Require at least32 test configurations; one pair per configuration per full
grid cell gives64 worlds/cell. With H32/64/128/256/512/1024, three delays, four
roles and three deadlines this is13,824 test worlds/checkpoint. Primary H512
mature all-role target-wait scoring then has1,536 worlds; cancellation768 and
reissued384. Freeze the exact census rather than leave stratum ambiguous.
H1024 is extrapolation. Varying entity count beyond8 requires changing and
validating the current fixed-eight interfaces; do not merely relabel input IDs.

Freeze every predictor, head, calibrator and probe before generating the sealed
test cohort. Record its hash before evaluation. Archive checkpoint-time models
first, then evaluate their curves together after the test unlock.

## 8. Endpoints, uncertainty and reliability

Keep one global primary: H512 pre-sensing mature target-wait accuracy across
all four roles, if compatibility with DEV-039 is intended. Record exact name,
weights and denominator. Report mature cancellation-only accuracy separately
to retain DEV-018/023 comparability; these metrics are not interchangeable.
Require pending and reissued role gates as well before claiming reliable causal
prediction, because an all-role average can hide cancellation failures.

Keep the proposed20 fresh (init,data-order) draws, with identical paired streams
across candidates. A five-draw development pilot may choose20 versus40 using
a fully prespecified rule; fix that choice before test access. Do not increase
n after observing test effects or call an optional test-driven extension a
routine deviation. Pilot draws are not silently reused as fresh confirmation.

Report median, IQR, minimum, training failures and draw success fraction.
18/20 passing is an observed90% gate: its one-sided95% Clopper–Pearson lower
bound is **71.74%**, not90%. Even20/20 has a lower bound86.09%. If a population
success rate>=90% with that confidence is the objective, the sample/gate must
change; at least29/29 is needed for an all-success exact-binomial rule.
Define a successful draw to include finite completed training, registered
prediction/cancellation gates and costs. Report accuracy conditional on valid
runs with its denominator; do not turn missing numerical failures into invented
accuracy scores or drop them from the reliability denominator.

Use paired, crossed resampling over initialization draws and configuration IDs,
preserving shared test configurations across draws/arms, all paired worlds,
and within-configuration correlated histories. Report performance conditional
on this finite test as well as any inference over new configurations.

Holm across H1–H5 requires actual prespecified valid p-values. Several hypotheses
contain multiple arms, regimes and two-sided/equivalence claims. Specify their
component tests and composite family p-values, the primary L-regime comparisons,
and the treatment of S/curriculum results before registration. A possible
conjunctive family requires every component to pass; merely applying Holm to
five labels while choosing favorable arms/tests afterwards is invalid.
Ordinary90% equivalence intervals correspond to unadjusted alpha=.05 TOST;
they do not automatically satisfy Holm-adjusted thresholds. Implement and
validate the statistical plan before labeling this confirmatory.

Retain physical cost4*failures+services+.1*inspections, normalized per entity,
and all its components. Report the proposed >=10% saving versus always-service
as an economic gate, separately from prediction reliability; equal/lower cost
alone does not establish fewer failures. Keep calibrated and fixed-threshold
frontiers plus never/always-service in every cost comparison.

## 9. Archive and execution gates

Archive raw tuning records, prediction/action rows, all completed/failed trial
checkpoints, source/environment/recipe hashes, split/config manifests and
seed mappings in durable storage outside the workspace. Publish a downloadable
index, retention policy and SHA-256 checks; a hash with no retrievable bytes is
not sufficient. Verify a fresh download before the results document is written.
Record planned archive locations before running so evidence is not stranded.
Evaluation replay and independent training replication remain different claims.

Maintain a historical comparison/cohort ledger from DEV-016 onward. This makes
adaptivity visible; it cannot retroactively correct unknown prior multiplicity.
Label the prior studies development evidence, not invalid data.

The repository loader requires experiment_id,title,question,hypothesis,arms,
endpoints,decision_rule,interpretation_order,h_levels,seeds,steps,eval_steps.
Arm entries require name/intervention/acts_on. Endpoint entries require
name/role/primary. It rejects multiple global primaries. Extra tuning/split/
statistical fields need their own validation; schema acceptance alone does not
enforce the full protocol. Do not put a placeholder in contracts/ or call it
registered until the runner, statistical plan, capacity and archive gates exist.

## Source links

- [DEV-039 results](https://github.com/FPC-effortless/TAC-OSM/blob/967c0e40420c76ad67c77d48e96a83b67d4af63c/docs/DEV-039-RESULTS.md)
- [DEV-039 audited counts/intervals](https://github.com/FPC-effortless/TAC-OSM/blob/967c0e40420c76ad67c77d48e96a83b67d4af63c/docs/DEV-039-AUDITED-SUMMARY.json)
- [DEV-039 runner and split assertion](https://github.com/FPC-effortless/TAC-OSM/blob/967c0e40420c76ad67c77d48e96a83b67d4af63c/scripts/run_predictive_head_dev039.py)
- [DEV-038 probe protocol](https://github.com/FPC-effortless/TAC-OSM/blob/967c0e40420c76ad67c77d48e96a83b67d4af63c/research/dev038/protocol.json)
- [DEV-039 evidence manifest](https://github.com/FPC-effortless/TAC-OSM/blob/967c0e40420c76ad67c77d48e96a83b67d4af63c/docs/DEV-039-EVIDENCE-MANIFEST.json)
- [DEV-027 initialization protocol](https://github.com/FPC-effortless/TAC-OSM/blob/967c0e40420c76ad67c77d48e96a83b67d4af63c/development_protocols/DEV-027-INIT-STREAM.json)
- [Existing DEV-029 optimization program](https://github.com/FPC-effortless/TAC-OSM/blob/967c0e40420c76ad67c77d48e96a83b67d4af63c/docs/DEV-029-OPTIMIZATION-AND-INTEGRATION-PROGRAM.md)

