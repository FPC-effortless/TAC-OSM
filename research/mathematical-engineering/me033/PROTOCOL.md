# ME033 — History/query semantics x embedding adaptation

Registered2026-10-21 BEFORE fitting. This isolates ME032's pooled correction gain
without introducing memory gates, compression hierarchies or internal reasoning.
Real-source mechanism diagnosis, not independent qualification or superiority.

## Source, roles and inherited components

Official KuaiRec CC BY4.0,Gao et al CIKM2022 DOI10.1145/3511808.3557220,
Zenodo18164998. ZIP SHA256
c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c.
ME022 source snapshot SHA256
fea9675380d501e81b740a0c1e0cd8c99893863ba6bed9a7e6619a572cc21177.
Verify ME032 frozen/final payload hashes and authenticated ME022 parent source
checkpoint/fingerprints before fitting. Only10776 original FIT windows,376users,
user%16==0,target beforeAug01 2020,128genuine observed events. Original binary
watch_ratio>=1 targets, exclusions,missingness,censoring,timestamps/row IDs unchanged.
Metadata remains log1p(duration)/12,observed feedback,gap. No fabricated outcomes.

Same groups=(user_ID//16)%3 and3rotations R:r,S:(r+1)%3,T:(r+2)%3. R/S/T user sets
disjoint for each model; all windows of one user share a role. Inherit only ME032's
R-only common head,context encoder/preprocessing and raw S/T context values. Never
reuse S-trained students or outcome-tuned choices. Parent R training includes
original native300steps and new common500steps, both128histories/four targets,
fully charged to all new arms. Copy parent file/result hashes into manifests.
R-only context26channels and its normalization remain fixed, as does prior32.

All54new models frozen before ANY outer metric. Only final128-event targets on
T users are scored, once in each user's T rotation per seed14001/2/3. This is reused
FIT in coeval user-crossvalidation, not a prospective temporal split, independent
dataset, or new held-out cohort. Cache validation reads known DEV for source
fingerprints only; no DEV fit/score/selection or reserved cohort access.

## Complete locked six-arm matrix,9cells

Common R predictor stays frozen in the four factorial arms and native control.
Their centered head is exactly ME032:
`delta(h,C)=w'h+v'[tanh(Bh+AC+a)-tanh(AC+a)]`,zero correction at zero latent.
All begin with zero direct/output weights and predict the exact inherited R base.

Four factorial arms use the SAME11->56->56->64 MLP and split RMS epsilon.01.
The11-vector is8item-embedding coordinates plus3mean historical metadata fields:
- history_adapt: mean embedding of genuine past items; train embedding on S.
- query_adapt: embedding of the known NEXT offered item; train embedding on S.
- history_frozen: mean past embedding; freeze embedding at inherited R weights.
- query_frozen: next-item embedding; freeze at identical inherited R weights.

All six new arms initialize any embedding from the SAME R-trained embedding,
not ME032's random S embedding. All bodies/correction matrices within the four
factorial arms otherwise initialize identically. Query arms still receive past
metadata means and the full common context, including its R-history summaries;
query does NOT mean history-free. Only the extra learned item-identity pathway
changes. Frozen-arm inputs are deterministic functions of existing common features,
so a gain there can be extra readout fitting, not new ideal information.

- native_adapt: unchanged original fast/slow body32+32, same R-initialized item
  embedding adapted on S, same common head/centered correction. No inherited R
  recurrent weights; native body otherwise from scratch. This is a native control
  at the new shared initialization, not a matched before/after ME032 result.
- common_adapt: fine-tune ONLY inherited common head on S. No extra state,body or
  student embedding; an intentionally much smaller practical reference.

Active S parameters:136772history_adapt/query_adapt,8772history_frozen/query_frozen,
136236native_adapt,1205common_adapt. History/query capacity is EXACTLY matched within
each adaptation setting. Native/adaptive-pool gap<.4%. Across frozen/adaptive and
vs common reference trainable counts FAIL1%: freezing128000coefficients is the
intentional treatment; do not pad or claim fully matched architecture ranking.
Stored pipeline totals incl shared frozen128000context embedding:265977factorial,
265441native,129205common. Frozen buffers/pretraining/storage all disclosed.

## Training, seeds and resources

All54fits: exactly300updatesx128 S histories, four actual next-event targets at
32/64/96/128,weights1/6,1/6,1/6,1/2;38400histories/153600response presentations.
AdamW memory.001,centered/common head.003,decay1e-4,clip1 over active weights.
Frozen common/embeddings absent from optimizer (no decay); common_adapt updates
common head only. Seed=original_seed+1000*rotation+66000; sampler=seed+300;
correction separatelyseed+1 for identical correction matrices across memory arms.
Arm order rotates by seed/rotation. No tuning,retry,best-checkpoint or early stopping.

Single CPU thread, hardware/software recorded. Same180charged S CPU sec ceiling,
same300charged full pipeline sec ceiling for all arms; shared inherited stages
fully charged to each arm, computed once originally. Record new physical study
CPU/wall separately from duplicated historical/component charges. Exclude original
historical computation from new physical cost. Charge source verification,role
construction,parent extraction and S setup/target preparation,sampling/steps;
serialization/receipt writing excluded. Save100/200/300latest partial checkpoints,
Adam/RNG,all step losses/CPU and embedding gradient/final weight behavior. Preserve
all partial/failure receipts; no different cap or retraining after evaluation.
Exposure matched; actual-compute parity FAIL/unqualified. Equal caps are not equal
spend. Profiler subcounts/CPU/RSS and inference measured, not full FLOP parity.

## Predeclared analyses and gates

Report Brier,logloss,AUC,ECE10 for inherited teacher and six new arms, all seeds/
rotations, pooled in original row order. Evaluate ONLY300endpoint, no outer curves.
Eight paired Brier contrasts in exactly this order:
1.history_adapt-query_adapt
2.history_adapt-history_frozen
3.history_adapt-common_adapt
4.query_adapt-query_frozen
5.history_frozen-query_frozen
6.interaction=(history_adapt-query_adapt)-(history_frozen-query_frozen)
7.native_adapt-history_adapt
8.native_adapt-common_adapt.

Bootstrap each contrast1000user-cluster draws, RNGseed33033 sequentially in that
order, average event losses over fixed3seeds; conditional95%intervals. Models are
not refit; overlapping rotated training roles and historical FIT design exposure
remain. Individual intervals not familywise adjusted; no causal/MI estimate.

History engineering gate: contrasts1,2,3 each improve in ALL3pooled seeds, mean
<=-.0005 and interval upper<0; mean ECE deterioration<=.003 against all three
controls. All conditions required. Adaptation-specific interaction flag separately
requires contrast6 negative all3seeds, mean<=-.0002 and interval upper<0; it is not
substituted for predictive gain. Native engineering gate: contrasts7,8 each meet
sameall3/-.0005/upper<0 rule and ECE<=+.003 against history_adapt/common_adapt.
Even a gate pass is further engineering candidacy, not promotion or parity-qualified
architecture superiority. No real causal action or decision-cost evidence here.

## Mathematical/runtime adjunct and audit

For fixed checkpoint embeddings, history input is a sum of8embedded coordinates
and3metadata coordinates plus count:12float32=48bytes. Query path accumulates
3metadata sums+count:4float32=16bytes, adding current offered embedding at readout.
Both feed the unchanged MLP/RMS to produce64ephemeral latent features. Sufficient
state merge is addition, associative over reals, approximate under floating-point
roundoff. No such claim for native nonlinear recurrence.

These are component states, NOT total forecast state; shared context/EWMAs/prior
have separate storage. Only the studied128-event windows qualify prediction.
Embedding updates invalidate cached sums: streaming equivalence is checkpoint-fixed
inference, not free online learning. Query-based metadata accumulation and offered
identity are causally available; no unknown next response or future event enters.
Bulk training remains prefix means; incremental API is separately verified.

Audit exact parent/role/source provenance, all54Adam/sampler/clock and embedding/
common-freeze behavior, checkpoint/export probabilities/metrics/contrasts, fresh
Torch-only full pipeline, source corruption rejection before observations, and
bulk/incremental/segment-merge latent+forecast equivalence on genuine T histories.
Test numerical gradients on tensor fixtures only; never promote them as real-data
performance. Archive original weights,derived contexts,predictions and logs;
source/protocol must be pushed before first fit. All negatives retained.
