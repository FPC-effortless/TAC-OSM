# ME105 — coherent point-preserving distributional TAC-Temporal

Fixed capability-development panel:15warm-started models, three complete objective
recipes ×five new run seeds105001–105005. ALLfive ME103 strong_reconstruction
parents102001–102005 are used in every recipe. Parent recipe is explicitly selected
from exposed ME103results because its mean point predictions are strongest and
its intervals overly broad. This is disclosed model development, not fresh
confirmation, favorable individual-seed selection or independent superiority.

New architecture and full input-to-output formulation: see MATHEMATICS.md.
Zero new heads preserve all ME103strong parent forecasts at initialization.
A zero-initialized median innovation Linear102→6 adds618params; the inherited
width head adds1,236. Total100,349 trainable/deployed33tensors. Scales retain the
FULL FIT-derived source floor and >=.8inheritedscale for missing channels. The
.8constant is an exposed-development-motivated design bound, not a coverage
promise. Bounded median correction±.5standardizedlogunits. Original parallel
source/evidence/associative scan and fixed streaming payload unchanged.

Exactly15fresh fits:three identical-workgraph complete loss recipes ×five
new sampling seeds105001..5, ALLfive ME103strong parents102001..5. No ME104weights
or optimizer are selected. Each gets1,536newupdates, batch16 contexts8/64/256,
freshAdamW wd.001, gradientclip1. Inherited29tensors lr3e-6, fournewheadtensors
lr3e-4; all100,349parameters trainable. No reset/retry/earlystop/selection, no more
fits after results. Everymodel/optimizer/RNG/receipt frozeninGitBEFOREREPORT.
New head1,536weightupdates; inheritedweights cumulative4,608. Two1-threadCPU
workers maximum. Runtime CPU is an efficiencyoutcome, no compulsory parity.

Same objectivegraphallrecipes:splitproperNLL +point_weight*rawmedianMAE/FITstd
+.1standardizedlogMSE+.3maskedcurrentreconstructionNLL
+interval_weight*raw90intervalscore/FITstd. Fixed recipeslikelihood(.1,0),
balanced(.5,.25),point_guard(1,.25). Zero coefficients still compute samegraph.
No new surrogate or fabricated target. FIT-only original firstbatch perparent
flow diagnostics record perloss/parameterblock gradients and sharedbackbone
cosines. This tracing performs NOoptimizer updates. It cannot prove general
interference or identify a unique causeofME104failure.

FIT only: original UCI5012013–2014, CC-BY-4.0 source SHA256
 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.
Real measured current values supervise training masks; no naturally missing value
gets an invented label. Per-seed initialization, real sampling plan/order,
augmentation RNG and measured operator graph/dense forward-backward estimates
must match across recipes. Unsupported hardware/pointwise/optimizer FLOPs remain
unquantified; CPU is an efficiency outcome, not a fairness gate. Relative parameter
matching is exact among new recipes. Old/new comparison changes parameters and
training exposure; it cannot establish controlled architectural superiority.

Score unchanged exposed2015cohorts only:143natural-gap queries/21calendar-week
resampling clusters and4,288ordinary queries/53overlapping clusters, six pollutant
channels, six-hour horizon,256hour input history. Original eligibility/targets/
source missing masks and scoring aggregation remain unchanged. Protected2016–2017
numeric rows remain unparsed/unscored; seeds do not multiply independent evidence.

EVERYseed in a recipe must meet ALL fixed gates:
- GapPM2.5 nominal90%coverage in[.90,.98].
- GapPM2.5 MAE <=min(1.05mappedME101full256reference,1.05ownME103strongparent).
- OrdinaryPM2.5 MAE <=min(1.05LOCF,1.05ownME103strongparent).
- Gap six-channel NLL <=min(6.234504261016846,1.05ownME103strongparent).
- GapPM2.5 interval score <=min(1.05mappedME101reference,.90ownME103strongparent).
The final interval requirement is at least10%material improvement per seed while
preserving all earlier guardrails. All5seeds required; all15published, including
negative outcomes. No promotion from exposed-development gates alone. Publish
all per-channel/seed metrics, proper NLL/interval score, coverage/width tails,
conditional paired week-cluster intervals and original independent replay.

Before fitting: source-pinned reciprocal PNDS Lean and independent Python graph,
passing scalar proofs and source FP32/FP64 forward/backward/state/mask/chunk checks
at original tolerances(32rtol2e-5/atol2e-6;64rtol1e-9/atol1e-10). Whole-neural,
Gaussian-measure normalization Lean and IEEE/calibration proofs remain OPEN.
Repeat actual-source checks on ALL15fitted networks before claiming verification.
Exactly six active families remain. No autonomy, safe control, customer value or
commercial readiness claim. Next steps depend on real outcomes, not attribution.
