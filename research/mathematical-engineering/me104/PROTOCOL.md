# ME104 — coherent point-preserving distributional TAC-Temporal

Fixed capability-development panel:15warm-started models, three complete objective
recipes ×five new run seeds104001–104005. ALLfive ME103 strong_reconstruction
parents102001–102005 are used in every recipe. Parent recipe is explicitly selected
from exposed ME103results because its mean point predictions are strongest and
its intervals overly broad. This is disclosed model development, not fresh
confirmation, favorable individual-seed selection or independent superiority.

New architecture: all98,495parent parameters remain trainable; add a zero-initialized
Linear(102,12) source/channel-conditioned width head (1,236params), total99,731.
Reuse original encoder/evidence/fast/slow/8×8scan and inclusive memory read.
No extra persistent state; old source, checkpoints, failed gates and scores stay
byte-unchanged. Old mean/log-location and every raw quantile are exactly preserved
at initialization. Changing only the new head leaves the raw median unchanged;
this is NOT arithmetic-mean preservation or a guarantee during end-to-end updates.

For standardized log1p location m and old positive scale s, left/right scales are
s*exp(log4*tanh(r_left/right)). A common standard-normal quantile z maps to
m+s_left*z below its median and m+s_right*z above. Each half retains mass1/2;
CDF is Phi((z-m)/appropriate_side_scale), continuous at m, with a possible density
jump there. Raw outcomes use max(0,exp(target_mean+target_std*z)-1), including a
zero-censoring atom with the side chosen at the zero threshold. No arbitrarily
clipped72hour interval; age remains capped256hours by the inherited input contract.
Scale multipliers are bounded[1/4,4], so the inherited variance floor becomes a
side-variance lower bound of old_floor/16, not preservation of its old amplitude
or a coverage/safety guarantee. All moments remain finite for finite parameters.

Training is one fixed1,536update stage per model, batch16, contexts8/64/256,
AdamW .0003/weight_decay.001, gradient clip1, fresh optimizer (NOT exact optimizer
continuation). Old weights have3,072updates; parent-owned weights reach4,608weight
updates and new width weights1,536. No new fit under ME103; no retries, tuning,
early stopping, extra stages or selection after scoring. Two one-thread CPU
workers maximum. All15original model/optimizer/RNG/source/order/mask receipts
must be committed BEFORE REPORT. Original data and augmentation are unchanged.

All recipes evaluate the exact same work graph. Loss is split-distribution proper
NLL +point_weight*FIT-raw-std-normalized physical median MAE +.1standardized-log
location MSE +.3authentic hidden-current Gaussian reconstruction NLL
+interval_weight*FIT-raw-std-normalized proper90%interval score. Fixed weights:
likelihood(.1,0); balanced(.5,.25); point_guard(1,.25). Zero coefficients still
execute the same graph. All31parameter tensors must be trainable, finite and
active in a genuine FIT batch before optimization. This joint loss can move the
shared backbone/median; the point guardrails are empirical requirements.

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
