# ME-030 — Query-conditioned native-state readout

Registered 2026-10-21, BEFORE this study's nine fits. This is native readout
engineering on already inspected KuaiRec historical development data. ME029's
negative chronological-state findings remain intact. No conditional-information
estimate, new cohort, independent qualification, or architecture promotion.

## Falsifiable mechanism and mathematics

The ME028/029 logit is `logit(prior32)+w'h+v'tanh(Ax+a)+b`.
Its mixed logit derivative with respect to latent state h and observable query
x is zero. Probability derivatives can still interact through the sigmoid;
therefore this restriction is specifically about the logit, not total statistical
independence. Replace only the hidden input by `[h,x]`:

`logit(p)=logit(prior32)+w'h+v'tanh(Bh+Ax+a)+b`.

For coordinates i,j, the mixed logit derivative is
`sum_k -2 v_k tanh(z_k) sech²(z_k) B_ki A_kj`.
This lets current item and duration alter state sensitivity. It does not ensure
that useful chronological information exists, or guarantee stable learning.
The unchanged fast/slow convex updates retain their coordinate bounds; no new
whole-network Lipschitz or gradient-retention guarantee follows.

## Complete frozen matrix

Three arms, seeds 14001/14002/14003, no search/retries/selection:
- additive_native: exact ME028, 135377 active parameters (7377 nonembedding).
- joint_native: unchanged native recurrence, joint 90->12 tanh head plus direct
  64->1 state branch, 136145 parameters (8145 nonembedding).
- joint_pool: ME029 order-free 11->56->56->64 mean-event encoder with the SAME
  joint readout, 136681 parameters (8681 nonembedding).

Largest total gap 0.964% (<1%); nonembedding differences disclosed. All have
16000x8 learned item embeddings and the same 26 available observable channels.
Final output weights and bias zero: identical initial prior prediction. Native
body initialization identical within seed. New hidden matrices use PyTorch's
ordinary seeded initialization. Pool order invariance applies to its LATENT,
not the shared order-sensitive EWMAs/prior. No inactive padding.

## Real source, exposure and compute

Official KuaiRec CC BY4.0, Zenodo18164998, Gao et al CIKM2022.
ZIP SHA256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c.
Reuse authenticated ME022 snapshot SHA256
fea9675380d501e81b740a0c1e0cd8c99893863ba6bed9a7e6619a572cc21177;
verify all original parent checkpoints and canonical fingerprints before use.
10776 FIT windows before Aug01 2020, 5952 historical DEV Aug01–13,
user%16==0, 128 genuine events, binary watch_ratio>=1. No new future/cohort
access. FIT-only duration standardization and smoothed item popularity.
Metadata duration is already log1p(raw duration)/12, feedback, gap.
Original source exclusions/missingness/window identity remain ME022's.

Exactly 300 updates x128 histories, identical seeded samples (`seed+300`), four
causal real targets at 32/64/96/128, weights 1/6,1/6,1/6,1/2; 38400 histories,
153600 response presentations per fit. AdamW body .001, all head layers .006,
weight_decay1e-4, grad clip1. CPU single thread, max180 charged CPU seconds per
arm; stop/preserve partial on failure, no altered cap or retry. Charge shared
source verification/preprocessing to EACH arm plus model/optimizer setup,
sampling, labels, forward/backward/update. DEV monitoring/serialization excluded
from charged training but whole-process CPU/wall and lifetime RSS reported.
Record every update CPU and FIT loss, dense DEV at 0/1/25/50/100/150/200/250/300.
ONLY fixed300 is primary; no best-checkpoint selection. Rotate arm execution by
seed. FLOPs and actual compute spend are NOT matched: exploratory exposure lane,
mandatory compute-parity FAIL, no architecture-superiority claim. Same allocated
caps are not equal consumed compute. No second compute lane is claimed.

## Locked analysis and decision

For BOTH joint_native minus additive_native AND joint_native minus joint_pool,
require all three paired Brier differences negative and mean <= -0.0005.
Mean ECE10 may worsen by <=0.003 against either control. Additional diagnostic:
replace h with FIT-first1024 mean, require >=+0.0002 Brier deterioration in all
joint-native seeds. Only all four gates permits further ENGINEERING candidacy;
even then no promotion without independent evaluation and parity qualification.
Mean substitution is out-of-distribution and cannot establish conditional value.
Report all Brier/logloss/AUC/ECE, CPU, inference B128 latency (3 warmups/9 repeats),
parameters, RSS, all seeds and curves regardless of gate. Paired user-clustered
bootstrap 1000 draws seed30030 on historical DEV; interval conditional on three
seeds, does not correct repeated DEV use. No genuine decision-cost data here.

Save original optimizer/RNG/checkpoint, TorchScript exports, predictions, mean
substitution forecasts, curves and per-update clocks. Audit all hashes BEFORE
source loading, replay nine checkpoint and nine export paths and diagnostic
forecasts. Freeze source commit on origin before first fit. Keep all negatives.

## Research roadmap

If joint head helps only common-feature fitting, chronology remains unqualified.
If it fails, inspect FIT-user cross-fitted conditional residual prediction with
identical nonlinear query interfaces and frozen representations; do not infer
mutual information from finite model scores. Sep-Latent-style compression,
NCM hierarchy and adaptive internal reasoning stay gated on learned-state value.
Real action consequence/cost signals remain necessary for decision objectives.
