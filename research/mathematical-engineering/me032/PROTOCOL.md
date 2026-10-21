# ME032 — Centered conditional state training

Registered2026-10-21 BEFORE fitting. This changes representation training after
ME031's frozen-state probe failed. Native fast/slow equations and32+32state size
remain unchanged. No compression/hierarchy/reasoning mechanism is introduced.

## Hypothesis and readout invariant

Train memory end to end on real S-user responses beyond a common predictor fitted
only on disjoint R users. Use the SAME centered state/query correction in all arms:

`logit(p)=logit(prior32)+b(C)+w'h+v'[tanh(Bh+AC+a)-tanh(AC+a)]`.

With h=0 the entire correction is exactly zero for ANY parameters and C. Remove
output bias because centering would cancel it (no inactive parameter padding).
This prevents a separate pure-context correction at zero state, but a learned
constant nonzero state could still encode a common-feature correction. It is NOT
a proof of conditional information, useful chronology or decision utility.

## Source, roles and prior exposure

Official KuaiRec CC BY4.0, Gao et al CIKM2022 DOI10.1145/3511808.3557220,
Zenodo18164998. ZIP SHA256
c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c.
ME022 snapshot SHA256
fea9675380d501e81b740a0c1e0cd8c99893863ba6bed9a7e6619a572cc21177;
validate original checkpoints/source fingerprints before fitting. Only10776
original FIT windows,376users, beforeAug01 2020,user%16==0,128genuine events,
binary watch_ratio>=1. Inherit original source exclusions/missingness/censoring,
fixed row identities/timestamps and four available-at-prediction targets. No new
labels, reassigned final test, fabricated outcomes or additional modality.

Same ME031 group=(user_ID//16)%3. Each rotation r=0,1,2 has R:r, S:(r+1)%3,
T:(r+2)%3. All windows from each user share a role. Original FIT was already used
for architecture design; within-study isolation does not make it independent.
Cache validator reads known DEV only to verify source fingerprints; no DEV fit,
score or selection, no future dates or reserved user cohorts opened.

## R-only common teacher and initialization (shared by all arms)

Verify EVERY ME031 frozen/final payload digest before source loading. Use ONLY
its nine original R-encoder checkpoints,3rotations x seeds14001/2/3, with known
300updatesx128/four-target training exposure. Reuse its R-trained embedding and
R-only duration mean/std/popularity to build the26common features; discard its
recurrent body and original readout in the deployed context extractor. Never
inherit ME031 S-trained base/probe weights. Copy original parent file digests.
All arms receive identical R pretraining exposure and pay full recorded R cost.

Normalize26common coordinates by R feature mean/std, floor.05; prior logit is
unstandardized. Common features are next offered item embedding8, normalized
stored duration1,EWMAs5,R-only item popularity1,pooled event embedding/meta11.
Metadata is log1p(raw duration)/12, observed feedback,gap. Preserve causal prefix
boundaries32/64/96/128; at early targets, known next item/duration are allowed but
next response and later events are excluded. S/T moments never fit normalization.
Save context features for R/S/T and stripped frozen context export (128000 frozen
embedding parameters plus preprocessing buffers, no unused recurrent modules).

Fit new common head26->43 tanh->1 (1205parameters) on R:500updatesx128histories,
four target weights1/6,1/6,1/6,1/2,AdamW.003,decay1e-4,clip1. Seed=original_seed+
1000*rotation+32000; sampler=seed+300. Final output weights/bias zero initialization.
Save original500endpoint; no best-checkpoint selection. Teacher-only and prior32
are unmatched practical references, not separately tuned competitors.

## Complete locked S-stage matrix

All students start their memory FROM SCRATCH, never inheriting R-native recurrence
or a previously S-fitted representation. This prevents a native-only R body
pretraining advantage over pooled control. Same initial student embedding seed,
source S histories, current context, R teacher, target exposure and supervision.
Seed=original_seed+1000*rotation+64000; correction separately seeded seed+1 so
its matrices match exactly across arms despite different body RNG consumption.
Initial direct-state/output weights zero: all arms initially predict the R teacher.

- conditional_native: native32fast+32slow with frozen R common head.
  Active S parameters136236 =134976memory+1260centered correction.
- joint_native: IDENTICAL initial native body/correction/common head, but common
  head also adapts on S. Active S parameters137441 (additional1205common weights).
- conditional_pool: order-free mean-event11->56->56->64 body with its own
  trained16000x8 embedding, same RMS and centered correction, frozen R common.
  Active S parameters136772 =135512memory+1260correction.

Maximum active S gap1205/136236=.8845% (<1%); nonembedding active counts
8236/9441/8772 are disclosed. Stored deployed totals including frozen R embedding
and common head are265441(native arms)/265977(pool). Historic parent pretraining
135377parameters is charged/reported separately; its7377nonembedding parameters
are not hidden in the context export. All active coefficients serve scoring.
Joint-native has more trainable common weights by design; this is the gradient
allocation mechanism, not a perfectly identical nonembedding-capacity comparison.
Pool learned latent is order invariant; shared context EWMAs/prior retain order.

Exactly300updatesx128 S histories per arm, four real targets/weights above.
AdamW memory.001,centered correction.003, joint common.003,decay1e-4,clip1 across
all active weights; frozen common/context receive no gradients or decay. Matching
sampler=student_seed+300; rotate arm execution order byseed/rotation. Record every
FIT loss,step CPU and post-clip recurrent-transition/pool-layer gradient. First
step memory gradients may be zero due zero output initialization; later steps
must have finite nonzero memory credit. Verify frozen common is bit-identical at
end, while joint common changes. Save100/200/300latest partial checkpoints,
optimizer/RNG,final exports. No tuning/retries/early selection/extra auxiliary loss.

## Resources and reporting

Local single CPU thread; hardware/software recorded. Teacher max30charged CPU sec,
student max180, shared+student pipeline max300 per arm/cell. Charge source/parent
verification and role construction, original R-pretraining cost from original
receipt, new R/S/T context construction and target preparation, R teacher setup/
sampling/training, S student setup/sampling/training. Exclude serialization and
receipt-writing from charged stages; report new physical study CPU/wall and
lifetime peakRSS separately. Original parent training is historical, not newly
executed CPU. Shared stages computed once, charged fully to EACH arm; neither
cost reuse nor unspent cap establishes equal consumption.

All arms: inherited R300x128/4targets, new R500x128/4, S300x128/4. Full resource
manifest per cell includes actual stage costs, histories and parameters. Primary
lane is exposure-matched mechanism diagnosis. Actual FLOPs/CPU spend across
native/pool differ: actual-compute parity FAIL/unqualified, no fully matched
architecture superiority. Measure full pipeline latency and profiler operator
subtotals after fits; never call partial coverage a complete FLOP count.

## Scoring and fixed gates

Freeze all9teachers+27students+9contexts, all originals and payload hashes BEFORE
ANY outer score. Evaluate final128-event targets once per user in their T role,
pool scalar predictions into original FIT row order per seed. No cross-encoder
latent pooling. Report Brier,logloss,AUC,ECE10 for prior,teacher and all3students,
per rotation/seed and pooled. Do not compare these scores directly with ME031:
common teacher is now trained on R instead of S and student objective changed.

Primary candidate conditional_native must beat EACH of joint_native,
conditional_pool AND initial R teacher: all3 pooled-seed Brier differences<0,
mean<=-.0005 for each contrast. Additional gates: >=2/3 fold wins per seed
against BOTH trained controls; mean ECE worsening<=.003 against EACH control and
teacher; all3 paired95%user-cluster bootstrap upper bounds<0 (1000draws,seed32032,
controls processed joint_native/conditional_pool/teacher in that order).
Only all gates yields further engineering candidacy, never promotion. Bootstrap
conditions on fitted models, fixed3seeds, overlapping training rotations and
previously exposed FIT; no refitting uncertainty or independent validation.
No conditional-MI, chronological-utility, causal action or commercial claim.

Audit all original hashes before data, exact source/user roles, context rebuild
from parent R embedding, all Adam/RNG/clock receipts, actual state gradients,
frozen/common parameter behavior, checkpoint/export forecast/metric replay,
fresh Torch-only full pipeline and corruption rejection before source loading.
Preserve all negatives and partials. Source/protocol pushed before first fit.
