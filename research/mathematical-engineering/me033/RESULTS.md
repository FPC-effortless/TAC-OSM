# ME033 completed — preserve item representations; common adaptation is strongest

**54registered real-data fits complete. Both history/native candidate gates FAIL.**
Source/protocol frozen/pushed atbf297d9 before first fit. No failed fit,retry,
hyperparameter change or endpoint selection. Post-fit immutable-table sharing is
a separate exact-execution/storage audit, not an added fitted/scored architecture.

## Design and source boundary

The four factorial arms vary only the extra item-identity pathway (past-item mean
or current offered item) and whether its R-initialized embedding adapts on S.
All four use the same11->56->56->64 MLP and centered state/context correction.
A native fast/slow control uses the same R-initialized embedding, adapted on S.
A common-only reference adapts the original1205parameter head on S without extra
memory. All54models train exactly300updates on identical sampled S histories and
four real targets; all are frozen/hash-indexed before any outer score.

Source=10776 original KuaiRec FIT windows,376users,128genuine observed events,
target beforeAug01 2020. Three user-disjoint R/S/T rotations and3seeds14001/2/3;
R-only context/head inherited from authenticated ME032, no S-trained parent weights.
Query controls still receive past metadata means and all common history features.
They are not history-free predictors. Current known offered identity is legitimate;
unknown current response and later events cannot enter an earlier prediction.

These are coeval entity-separated folds within previously used FIT, not an
independent or prospective temporal qualification. Prior design exposure remains;
DEV is integrity-checked only, never scored. Reserved cohorts stay closed.

## Fixed300 endpoints, three-seed pooled means

| Predictor | Brier ↓ | Log loss ↓ | AUC ↑ | ECE10 ↓ |
|---|---:|---:|---:|---:|
| Inherited R-only common teacher |.170203796|.513921993|.806423730|.026479599|
| Common-only S adaptation |**.161558311**|**.490315366**|**.824966956**|.013120492|
| Past identity, frozen embedding |.162330508|.492579803|.822897755|**.012508929**|
| Past identity, adapted embedding |.163599877|.496066229|.820807442|.019005165|
| Offered identity, frozen embedding |.163604463|.495721805|.820127784|.013195641|
| Offered identity, adapted embedding |.164592733|.498153782|.818037757|.013055921|
| Native, adapted R-initialized embedding |.166925030|.504184464|.813704821|.020747616|

Common-only adaptation is strongest on Brier in this matrix, with much fewer
trainable coefficients. This is a practical reference finding, not a parameter-
or compute-matched architecture win. The native control is not improved over
history adaptation or common adaptation. Do not compare ME032/ME033 as a controlled
before/after gain: student embedding initialization and matrix changed.

## Predeclared contrasts and gates

| Paired Brier contrast | Mean | Conditional user-cluster95%interval |
|---|---:|---|
| History adapted - query adapted |-.000992856|[-.001764852,-.000211135]|
| History adapted - history frozen |+.001269369|[+.000809943,+.001738045]|
| History adapted - common adapted |+.002041566|[+.001376222,+.002780061]|
| Query adapted - query frozen |+.000988270|[+.000522762,+.001446187]|
| History frozen - query frozen |-.001273955|[-.001658665,-.000864370]|
| History/query adaptation interaction |+.000281099|[-.000347176,+.000966699]|
| Native adapted - history adapted |+.003325154|[+.002564281,+.004059518]|
| Native adapted - common adapted |+.005366719|[+.004413657,+.006289766]|

Updating embeddings worsens history and query paths in ALL3pooled seeds:
history differences+.002327356,+.000687237,+.000793514;
query+.000592941,+.001110032,+.001261836. This supports preserving pretrained
representations under this recipe; it is not a universal rule against adaptation.

History beats query on average, but adapted-history minus adapted-query per seed
is+.000410825,-.001989398,-.001399995, so registered all-seed gate FAILS despite
its negative mean/interval. Frozen history beats frozen query in all3seeds, but
its input is already contained in common context (see below). The interaction
interval crosses zero; adaptation-specific-history evidence flag FAILS.

Native loses both controls in all3seeds. History candidate and native candidate
FAIL all their registered conditions, including calibration guards. No threshold,
checkpoint or favorable subgroup is substituted. No model promoted. All8contrasts
were preregistered; individual bootstrap intervals are not familywise adjusted.
They condition on3fitted seeds and reused FIT, not training/refitting uncertainty.

## Information diagnosis

Frozen histories use mean E_R(item)+mean metadata; frozen queries use E_R(offered)
+mean metadata. Both inputs can be reconstructed from the common predictor's
existing26features. The audit verifies this on real records, maximum latent
reconstruction error7.15e-7 from floating-point normalization. Their extra latent
contains no new ideal information beyond that context; different finite nonlinear
fitting can still improve forecasts. An apparent memory gain is not automatically
new learned historical information.

The next unexecuted native optimization hypothesis is **freeze the pretrained
input embedding while training fast/slow dynamics and readout**. The pooled/query
factorial result motivates that test, but native_frozen was NOT in this matrix.
It needs a separate fixed protocol and strong common-adaptation reference before
claiming improved native memory. No new gates,hierarchy or compression are earned.
No causal action,conditional-MI,independent generalization or commercial claim.

## Composable component-state engineering

For a fixed checkpoint, past-identity moments retain11sums+count (48float32bytes).
Query moments retain3metadata sums+count (16bytes), reading offered embedding at
prediction time. Additive segment merge and sequential updates reproduce bulk
prefix means/readouts within rounding tolerance. The64dimensional MLP latent is
recomputed, not persistent. Native recurrence does not gain this associativity.

Verified36trained factorial components on16real histories each and4prefixes:
576histories/2304prefix predictions. Maximum bulk/stream forecast error1.19e-7.
These bytes count ONLY the extra component; common-context/EWMA/prior buffers,
weights and activations remain separate. No full48byte forecast or unlimited
stream accuracy claim. Embedding updates invalidate cached sums; this is fixed-
checkpoint inference, not cost-free online learning. See MATHEMATICS.md.

## Exact frozen-table storage improvement (post-fit engineering)

Frozen context and frozen extra pathway hold IDENTICAL immutable128000coefficient
item tables. Optional SharedFrozenForecaster aliases those tables after validating
frozen flags and exact equality, leaving original artifacts untouched. No trained
weights or operations change; eligible models are selected by structural equality,
not accuracy. Original training/scientific gates stay unchanged.

All18eligible checkpoints were exported/reloaded with alias/storage identity
preserved and all64656real final forecasts BIT EXACT. Unique float32 tensor-weight
parameters fall265977->137977: **512000bytes saved per model,48.1245% of tensor
weights**. Buffers,activations,serialization overhead and total forecast state are
excluded; this is not a48.1% total-system-memory claim. It does not apply to adapted
tables or native recurrence. Original unshared checkpoints are retained.

This is a verified equivalence/storage success alongside a negative native
predictive result. It introduces zero fits and no new performance-selected arm.
See shared-storage.json and scripts/me033_shared_audit.py.

## Resources and parity

| Arm | Active S params | Mean new S CPU sec | Mean charged pipeline CPU sec* | B128 pipeline ms** |
|---|---:|---:|---:|---:|
| history_adapt |136772|2.0751|24.3609|1.5722|
| query_adapt |136772|1.1743|23.4601|1.3065|
| history_frozen |8772|1.0209|23.3068|1.4845|
| query_frozen |8772|.9647|23.2505|1.3375|
| native_adapt |136236|19.7389|42.0247|12.2973|
| common_adapt |1205|.2587|22.5446|.6623|

*Fully charges shared historical R pretraining/teacher and inherited/source setup
per arm. Those stages were not physically retrained54times. New physical study
CPU233.2849sec,wall233.3178sec; no extra fitting outside the registered54.
All stages remain below180S/300pipeline CPU caps. Exact row/target exposures are
38400histories/153600responses per fit; total16200new updates,2073600histories,
8294400supervised response occurrences. These are repeated exposure counts, not
new independent observations. Step receipts and lifetimeRSS/hardware preserved.

**Raw128events+context+student,4prefixes,seed14001rotation0,thread1,3warmups/9samples.
Implementation-specific timing, not a universal throughput claim. S-step operator
FLOP subtotals B8/four targets:1889047adaptive moments,1849623frozen moments,
42843391native,151392common-only. Nonlinear/fused/elementwise coverage,Adam and
context precomputation incomplete; no full FLOP certificate.

Exact parameter parity PASS for history/query within each adaptation setting.
Across freeze/adapt and common reference it FAILS intentionally (128000weights
become frozen; no padding). Exposure/selection/identical historical evaluation
PASS under the information-ablation design. Actual-compute parity FAIL/unqualified.
No fully matched architecture superiority regardless of score direction.

## Original-artifact verification

- 54Adam/model/sampler RNG records and16200step receipts pass. Frozen/adapted
  embedding behavior and common-head changes match the registered treatments.
- 6,982,848S/Tcontext values reconstructed from R-only source and preprocessing.
- 54new checkpoint paths/193968forecasts exact;63export/reference paths/
  226296forecasts exact.54fresh Torch-only pipelines/216forecasts exact.
- Incremental/segment-merge state checks and frozen-context reconstruction pass.
- Corrupt checkpoint rejected before source loading.953tests+12subtests pass.
- 18 optional shared-table exports/64656forecasts exact, independent from fitted
  matrix scoring. No new reserved cohort, merge, deployment or promotion.

See source manifests, original weights/Adam/RNG, replay/shared-storage JSON,
[figure](semantic-factorial-results.png), logs and archive restoration records.
