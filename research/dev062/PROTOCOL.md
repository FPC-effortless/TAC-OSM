# DEV-062A — learned public beliefs and paid-information control

Pre-fit registration. Independent fixed-history uncertainty study, followed by
learned control evaluation; it does not promote an integrated DEV061.5 checkpoint
if the prerequisite gates fail. Preserve those failures. No advanced architecture
or commercial claim is inherited from either study.

Question: can a learned recurrent estimator reconstruct an exact public joint
posterior, and can a learned policy use it to purchase useful inspections and
finish without a scripted inference-time planner?

Public stream: two opaque episode identities, directed relationship records,
50% missing initial statuses, 10% sensor errors, explicit action-start events,
noisy repair receipts for BOTH entities regardless of hidden changes, and paid
exact inspection receipts. Identity manifests and typed vectors remain declared
structural priors. Training/reference physics uses exact four-state enumeration;
deployment never imports that filter, teacher or simulator. Action-start tokens
are necessary provenance; receipts alone cannot identify the repair transition.
No privileged teacher action is included in a history labelled an exact public
posterior. Dataset collection uses only public beliefs or random actions.

Four retrained arms: entity-CDL/neural-CASM, no-CASM, global GRU, ordered flattened
MLP. Each receives identical retained raw public events and cosine binding to the
same public identity manifest. No latest-health parser. All share a frozen
source-type gate trained to omit noise, with no memory-efficiency claim. The MLP
is order-sensitive, not an intentionally order-blind bag of events. Total parameter
counts matched within5%, actual CPU and retained state reported separately. Local
GRU content, soft CDL, dense operator mixtures and trained joint posterior/policy
are bounded neural computations, not symbolic program discovery or learned physics
from scratch. Fixed capacity64; no learned episodic retention.

Twenty training seeds 92601–92620; pilot92600, AdamW rates.001/.003, validation-only
selection by posterior KL then faults then cost; all pilot curves preserved.
Initial2048 public histories (inspection prices .05/.1/.2), validation512; namespaces6261001/2; 1024updates,
batch64, clip1, wd1e-4, EMA.995. First128steps posterior KL only, then posterior KL
+ action cross-entropy +.25 action-conditioned next-posterior KL +.1 cost Huber.
At256/512/768 each of four EMA students collects64 fresh training worlds, bounded
horizons2/5/10. Teacher-relabel union shared across all arms, half initial/half
union batches. Teacher is an information-matched depth2 expected-cost reference
with STOP excluded unless its public belief establishes zero physical faults
(within1e-12); this is a TRAINING-ONLY reliability-constrained teacher. Log all
collection costs. Do not describe it as globally optimal. All targets depend on
public histories, not privileged action selection. Probability loss is proper KL;
cost/risk asymmetry belongs in teacher/control, not weighted probability labels.

Float32 fit, float64 single-thread deterministic evaluation, generic first-valid
choice within1e-6 of maximum. Policy temperature0.5/1/2/4 selected on validation
only. No posterior calibration based on held-out results. Firstseed92601 exported,
not best seed. Two CPU workers; new fitting cap90CPUmin including collection and
validation; group240CPU seconds; retain capped/failed outcomes. Incomplete matrix
forbids acceptance/test opening. No extra tuning after tests. Source frozen before
fit. Engineering smoke is disjoint and cannot count as evidence.

Holdout6261003: 256 fixed public histories for dense KL/multiclass Brier/TV,
plus128 worlds per H32/H128 (256/model), budget10. Report each training seed,
posterior errors, empirical completion calibration, inspection count, stop errors,
physical faults, cost and actions. Same worlds/information for exact public
reference, latest-observation shortcut and privileged reference. Separate
representation gap (neural posterior vs public posterior), information gap
(public vs privileged outcomes), and decision gap (learned vs public control).
Secondary6261004: sensor error20%, inspection price.4, H512,64worlds each.
Sensor/price metadata are public; test generalization, not an undisclosed law.

Functional gate:18/20seeds >=95% primary completion; median fixed-history KL<=.02
and multiclass Brier excess over exact posterior<=.02. Physical gate:zero primary
residual faults. Competitive gate:paired20seed cost confidence interval upper
bound<=0 against matched GRU and MLP, Holm2; retain no-CASM comparison without
claiming unregistered CASM necessity. Report real CPU latency and memory; cross-host
replay is UNKNOWN until measured. Promotion requires ALL gates, never a cheap
faulty policy. No claimed perfect calibration, universal safety or commercial ROI.

This is posterior-supervised StageA. StageB (learning from public next-observation
likelihoods without posterior labels), general symbolic CASM, new sensor laws,
Transformer/BDH/SSM arms and commercial deployment remain separate future studies.
