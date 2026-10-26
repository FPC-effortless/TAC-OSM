# Research Ideas Registry — TAC-OSM and related programs

## 2026-10-26 — ME109 trained residual roles: full gate FAIL; average protection misses gap objectives

[Draft TAC PR419](https://github.com/FPC-effortless/TAC-OSM/pull/419) and
[PNDS PR27](https://github.com/FPC-effortless/pnds-lean/pull/27) remain unmerged.
All five104,663-parameter native models completed4,096updates each. The actual
parent's learned affine current map is retained; three zero-output108→12→12/12/6
residual FFNs condition on real availability masks and source age. Initial parent
forecasts/state are bit-exact in tested original FIT cases; native recurrence,
608-byte primary FP32 payload plus metadata, and six active families unchanged.
No Transformer/SSM conversion, separately trained experts or model promotion.

All original models/optimizer/RNG/receipts frozen65e2f817 BEFORE original REPORT;
source452a1a977275823d21b3f04265a26ff836518ce6; prefit6cbcd6c3;
original scoresa68121d8. Authentic UCI5012013–2014FIT; exposed2015REPORT143gap/
4288general queries,6hours ahead with256hours context. Protected2016–2017 numerical
rows remain closed. Same original losses and thresholds; no extraME109fits.

| Five-seed metric | ME103 parent | ME108 | ME109 |
| --- | ---: | ---: | ---: |
| Gap PM2.5 MAE ↓ |37.524168|41.132003|44.437162|
| General PM2.5 MAE ↓ |33.403120|31.572055|32.014751|
| Gap PM90 interval score ↓ |379.623199|241.566815|222.612326|
| Gap PM90 coverage |95.6643%|96.6434%|93.7063%|
| General PM90 coverage |89.8647%|88.6707%|89.1931%|

ME109 general-point error improves4.1564% and gap interval score41.3597% versus
parents, but gap-point error worsens18.4228%. GapPMcoveragepasses5/5; gapPMpoint
passes0/5; generalpoint/NLL/interval gatespass5/5 each. **Full gate FAIL0/5**.
Only1/5gap point scores beats frozen ridge (43.915070), though all5beatLOCF
(56.076923). All5general scoresbeatLOCF/ridge. These historical comparisons are
UNMATCHED development references, not architectural superiority. GapNO2coverage
75.3623% exposes continuing all-channel uncertainty failure despite PM-only gate
success. No independent capability, useful associative retrieval, control or
commercial readiness claim.

All266,240actual blockwise computed average-point constraints pass,maxdot0;
94,810projections and24,443actual-rounding rejected blocks. Aggregate new-update
CPU1396.580641s excludes preprocessing and parent training; FLOPs are partial.
Ten original fitted FP32/64 forward/state/45parameter/input-gradient and read-only
optimizer-proposal checks pass. Clean restoration at5b2f5fb2 replays all originals,
398,790saved quantile scalars, scores/gates, summary and diagnostics without fit.
23payload hashes preserved. Seven limited Lean identity/role/count/bound theorems
and independent Python reference pass; full neural/AD/IEEE/source-law refinement
remains OPEN. Original-score GitHub replay40280842012 and science40280842069 PASS.

**Measured local optimization weakness, not causal attribution:**15posthoc FIT-only
candidate updates (5frozen models×3B32batches) preserve the six-channel average
point direction, but gap-PM direction is positive10/15overall and75/195blocks;
general-PM positive0/15overall. NLL versus interval gradients oppose13/15batches;
point versus interval also13/15; gap versus generalPM11/15. No weight/moment was
applied, counters stay4096; this is a local gradient/proposal observation, not a
realized finite-step or held-out causal explanation. Parent-preserving
initialization alone did not fix the trained gap-point trade-off.

**New master tools adopted honestly:** merged PR417/420anti-cheat/Observatory
source ada32393651614eeb60cf4f3a69f1aad44449e55 restored byte-for-byte for15files;
64new tool/automation tests, full1151local tests PASS. Adoption is AFTER fitting
and original scoring, not claimed as retrospective preflight. The native CLI is
incompatible with this Temporal model. A genuine FIT-specific adapter applies
generic inventory, activation, weighted-gradient and poison/noninterference APIs
to all5original checkpoints. Scoped local findings PASS and independently replay;
full anti-cheat ledger **INCONCLUSIVE**5/5 for uninstrumented CSV/joins,
partition/preprocessing provenance, nuisance/shuffle controls and native-state
adapters. The raw-future-target probe holds normalizer caches fixed. Fixtures
and call order are not model intelligence or a complete computational DAG.
Automatic tool workflows are preserved; branch protection is not asserted.

**Next capability target:** source-availability-conditioned, per-channel/per-regime
optimization rather than another larger shared FFN or scalar loss-weight sweep.
The next protocol should retain native parallel state and identity-preserving
roles while tracking multiple relevant task directions explicitly. A protected
average cannot be promoted as protection of every task. This is a PROPOSED new
training contract, not an implemented/fittedME110 or a guaranteed fix. Preserve
all negative results, source-boundaries, real-data gates, checkpoint/independent
replay and source-pinned PNDS Python/Lean obligations; no moreME109fits.

### Diagnostic replay V2 — original failures preserved; cross-host drift measured

V1diagnostic input-digest replay failed in40283071768/40284177624 while original
model/forecast/score replay passed. Normalized batch0 values differ across hosts
up to1.430511474609375e-6; current normalized labels up to1.1920928955078125e-7;
other8fields exact. Both changed fields satisfy the existing combined relative
1e-5/absolute1e-6 contract; primitive cause remainsOPEN, not presumedlibm.

Authenticated original FIT tensor exports preserve exact previously committed
input identities. V2replays model/atlas on original tensors, separately validates
39fresh producer fields and actual fresh/frozen model outputs, and retains exact
within-run poison invariance. Both producer digests/differences are disclosed;
original inputs, weights, JSON, scores, scientific gates and forecast tolerances
are unchanged. Full anti-cheat ledger remainsINCONCLUSIVE. CI40284891806 on
fa186595 and independent clean V2replay PASS;1151local testsPASS. Extended bundle
now26payload hashes. This reproduction repair is not a cause or cure of the
18.42%gap-point regression. Source-pinned Python/Lean/full-IEEE obligations remain.

Default-master merged-tool post-push runs40281108276anti-cheat,
40281108278Observatory and40281108326fullCI are verifiedPASS; branch-protection
required-status enforcement is not asserted. Future capability training should
freeze exact FIT preprocessing provenance and test current-source reconstruction
as well as typed per-task/per-regime optimization boundaries.

## 2026-10-26 — ME108 trained role models: ordinary/interval gains, gap-point failure

[Draft TAC PR411](https://github.com/FPC-effortless/TAC-OSM/pull/411),
[PNDS PR22](https://github.com/FPC-effortless/pnds-lean/pull/22),
[complete results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me108-role-trained-2026-10-26/research/mathematical-engineering/me108-role-trained-2026-10-26/RESULTS.md),
[all5 original checkpoints and replay](https://github.com/FPC-effortless/TAC-OSM/tree/research/me108-role-trained-2026-10-26/research/replay-bundles/ME108).

The ME107 specialized FFN prototype is now genuinely trained as an integrated
native TAC model: five fixed lineages108001–108005,102713active trainable/deployed
parameters/39tensors,4096new optimizer updates each. Allfive original ME103 strong
backbones102001–102005 retained; their current linear head discarded and three
nonlinear FFNs initialized freshly. This is not exact parent-function warmstart.
No Transformer/SSM replacement, new family or extra primary recurrent payload.

Native encoder/fast/slow/8x8 memory, input-conditioned parallel scan and source
floors remain unchanged. BlockProtectedAdamW protects the complete proposed
rate/momentum/preconditioned/decay displacement, rechecks after displacement and
parameter rounding, and rejects residual opposing protected-coordinate moves.
Raw clipped combined gradients advance the moments even if weight moves reject.
The point snapshot is the same-batch unclipped .5six-channel normalizedMAE
VJP; other loss roles retain ME106's explicit uncertainty-gradient view.

Eight limited exact-real Lean declarations and source-pinned independent tensor
optimizer equations are committed. Full-neural/AD/normal-law/IEEE refinements
remainOPEN; a nonpositive local point dot is not finite-step/population safety.
An initial launch aborted BEFORE any optimizer update because its plan validator
expected2D instead of4096x16x2 station/hour pairs. Original registration/log and
zero-step incident preserved; v2sourceeb40cd1c corrects only that check before
real training. Initial optimizer source95cfd29e remains byte-identical. No fit
retry, held-out feedback or budget change was hidden in this correction.

All5 original model/optimizer/RNG/plan/runtime receipts frozen5e4b0ad9 BEFORE
REPORT. Fitted graph/proposal verification e8fe84f1 preceded scores674682f1.
Source/reference fwd/state/parameter gradients, read-only next proposals and
moments match FP32/64; additional normalized value/age/calendar-input gradients
checked. No audit optimizer step. Independent clean checkout25917e9d restored
all5weights, scores/gates, input/parameter/proposal assertions and summary; PASS.
1059full Python testsPASS,14dedicated optimizer/runner testsPASS. Original-model
GitHub replay40257702125PASS; code/integrity verification is not predictive gain.

Authentic UCI501 CC-BY-4.0 archiveSHA256
`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.
Weights/statistics only2013–2014FIT; exposed2015 evaluation143natural-gap and
4288general queries,6channels,6h targets,256h histories. Future PM availability
conditions the scored cohort; unobserved future values are not invented labels.
Protected2016–2017 numerical rows remain closed. Each model65536sampled examples
and7163648repeated history events, not that many independent observations.
Retained weights3072parent+4096new=7168lineage steps; fresh FFNs4096.

| Five-seed means | ME103 strong parents | ME108 role models |
| --- | ---: | ---: |
| Gap PM2.5MAE ↓ |37.524168|41.132003|
| Ordinary PM2.5MAE ↓ |33.403120|31.572055|
| Gap mean marginal NLL ↓ |5.151004|5.138193|
| Gap PM90interval score ↓ |379.623199|241.566815|
| Gap PM90mean width ↓ |369.90933|229.90420|
| Gap PM90coverage |95.66%|96.64%|
| Ordinary PM90coverage |89.86%|88.67%|

Ordinary point error improves5.48%, gap interval score36.37% and width37.85%,
but gap point error worsens9.61%. **Full predictive gate FAIL,0/5full seedpasses:**
allfive fail gap-point; coverage4/5 (10800198.60%exceeds98%maximum), ordinary-point,
gap-density and interval-score gates5/5. Ordinary coverage is below nominal90%
and worsened. Same original seed-specific thresholds retained, no rounding into
pass or favorable-seed selection. Parent98495parameters versusnew102713(+4.28%)
and extra training make this a systems-development trade-off, not a matched
architectural advantage or independently confirmed gain.

Frozen practical point references: gapLOCF56.076923/ridge43.915070;
ordinaryLOCF34.515625/ridge32.767415. Allfive models beat both references on both
point cohorts. That descriptive exposed-data capability does not erase the
stronger parent-gap failure or establish efficiency/control/customer utility.

20480actual updates×13blocks=266240computed applied-point checks, maximumdot0;
79632projections and20645final-rounding rejections. New-update CPU1487.231s
aggregate with atmost2one-thread workers; earlier pretraining/preprocessing and
unsupported optimizer FLOPs are not silently included. Protection of a shared
minibatch mean can coexist with worse per-channel/per-regime generalization.
The invariant ledger now records that limitation instead of declaring conflict
or uncertainty mathematically solved. Calibration/useful memory remainOPEN.

**Next architectural hypothesis, not another ME108fit:** identity-preserving
residual FFNs around validated affine predictors, with corrections conditioned
on causally known availability/age. Preserve the parent's function at
initialization while learning more expressive corrections, rather than making a
narrow nonlinear head relearn it. This is motivated by the trade-off, not a
proved explanation of it. Require a new source/math/prospective training record,
live parameter budget and alloriginal real-data gates; do not tune against the
protected cohort. No extra ME108fits, parameter/loss/gate changes or promotion.

Verification/indexing addendum: initial repository-wide scanner did not discover
the dated registration directory. Canonical me108/FAIR_COMPARISON.json and
POSTFIT.json are now byte-identical aliases of the preserved originals, with
post-scoring INDEX_PROVENANCE. The original direct preflight and live model
pre/post guards passed before training; this was a discovery/wiring correction,
not a new backdated registration or relaxed gate. Science preflight40260231748
and original-model replay40260231715PASS on04befd7d. Future studies should use
canonical meNNN/registration paths and verify --check-changed as well as the
direct --manifest preflight before launch. The real4096x16x2 station/hour plan
geometry regression is retained. No additional ME108 fits or promotion.

## 2026-10-26 — ME107: typed role FFNs and a mathematical failure ledger

[Draft TAC PR410](https://github.com/FPC-effortless/TAC-OSM/pull/410),
[PNDS PR21](https://github.com/FPC-effortless/pnds-lean/pull/21),
[invariant ledger](https://github.com/FPC-effortless/TAC-OSM/blob/research/me107-role-specialized-2026-10-26/research/ARCHITECTURE-INVARIANTS.json),
[full formulation](https://github.com/FPC-effortless/TAC-OSM/blob/research/me107-role-specialized-2026-10-26/research/mathematical-engineering/me107/MATHEMATICS.md).
Owner requested mathematical prevention and role-specialized FFN computation,
retaining TAC rather than replacing it with a Transformer/SSM. A bounded untrained
prototype is implemented, not merely proposed and not a new predictive result.

The existing native masked encoder, source/evidence state, fast/slow proposals,
8x8 associative schedule/read, input-only parallel scan and streaming memory stay
intact. Three independently learned102→16→output tanh FFNs replace linear heads:
current-state reconstruction12, uncertainty shape12, bounded point innovation6.
Roles hard-assign allowed outputs/computation, NOT correct answers or oracle
regimes. The shared102-feature state supplies these decoder branches. This is
standard task-specific nonlinear-head engineering, not a novel FFN invention or
proof that specialization has been learned. Shared ancestors still can conflict.

The actual model has102713trainable/deployed values/39tensors, within95k–105k;
no padding, frozen encoder or new recurrent payload. Primary state remains152FP32
floats=608B/sample plus existing integer/provenance metadata, excluding full model
and runtime RAM. Fresh head initialization is not an identical parent warm start.
The size differs from ME106 by more than1percent, so an old/new matched
architectural-superiority claim would require new matched controls.

A pure proposed-displacement guard checks each exhaustive parameter block AFTER
all optimizer effects: if <g_point,delta> is positive, remove that component;
FP64 accumulation, rounded-output recheck and fail-closed fallback retain free
zero-gradient coordinates. A one-shot iterable-validation bypass was caught and
fixed before any fit; regression tests cover it. **The guard is not integrated
into a trained optimizer.** Its exact-real first-order bound does not prove
finite-step loss descent, behavior at MAE kinks, uncertainty quality or capability.

The versioned ledger tracks10 failure classes: gradient/optimizer conflict,
role coupling, missing-data coverage, scale-floor/location coupling, unused memory,
scan legality, numerical/preprocessing drift, state-count/age limits, leakage and
parameter/resource fairness. Each has a contract/status/evidence or explicit open
obligation. Fixed-size storage does not prove bounded cumulative sums/counts or
unlimited age resolution. No statistical failure is labeled mathematically solved
just because a related kernel theorem compiles.

Source ddf04323, reciprocal row18295b69, PNDS5e50d297. Full typed Lean role-forward
formulation, independent Python chronological reference and source-pinned tests
are linked in both repositories. Fifteen limited Lean declarations compile with
standard axioms only, including exact left/right source and missing-scale bounds,
parameter arithmetic and generic finite-vector displacement properties. Whole
neural/AD/IEEE and normal-law implementation refinement remainOPEN; median bound
is analytically specified/tested, not separately Lean-proved here.

Validation:1045full Python testsPASS;9dedicated tests;6pinned PNDS source tests;
actual-source/reference outputs/state/parameter gradients match FP32/64 under
unchanged2e-5/2e-6 and1e-9/1e-10. One registered authentic2013–2014FIT minibatch has
finite nonzero composite-objective gradients in all39 tensors; maximum gradient
errors9.54e-7/7.78e-16. Streaming and structural role-gradient tests pass. Dedicated
GitHub TAC source40251245492 and initial PNDS source/math40251245108/generalLean
40251243841 pass; final source/math40251495439 and generalLean40251495515 also pass on
PNDS5e50d297, including both new left/right scale-floor lower-bound proofs.

**No new models fitted or optimizer steps. No REPORT/protected2016–2017 numerical
access.** This no-fit registration cannot authorize future training. ME106 failed
qualification, original checkpoints and DIAGGRAD001 FP32 exceptions stay intact.
Six families unchanged; no merge, deployment, Mamba-speed parity or intelligence
claim. The next capability study must register and train the integrated candidate
on real data, wire the proposed-update semantics explicitly, preserve original
checkpoints/replay and test actual predictive gates. Attribution stays secondary.

## 2026-10-26 — DIAGGRAD001: measured interval/point conflict and optimizer-direction limits

[Draft TAC PR409](https://github.com/FPC-effortless/TAC-OSM/pull/409),
[PNDS PR20](https://github.com/FPC-effortless/pnds-lean/pull/20),
[full trace/results](https://github.com/FPC-effortless/TAC-OSM/blob/research/diaggrad001-loss-flow-2026-10-26/research/mathematical-engineering/diaggrad001/RESULTS.md).
User requested direct input-to-output and per-objective tracing before selecting
another remedy. This is a bounded no-fit diagnostic, not another trained-model
success. ME104's static weighted sum is historical; ME106 already implemented
selective median-head routing and global point-priority gradient projection.

Registered code2d5edb8c, originals d9099acc. All15 frozen ME106 checkpoints,
100,349parameters unchanged, 180 FIT-only probes: contexts8/64/256h, natural-gap
and general origins, two batches16 each. Original causal masking and allfive
objective terms;13 exhaustive parameter blocks. No optimizer steps, REPORT or
protected2016–2017 numerical access. No modified models, scores, thresholds,
persistent state, native parallel scan, loss weights or neural family.

**Measured, rather than inferred from performance:**

- NLL/point whole-model gradients oppose in0/180 probes; median cosine+0.629.
  Stratifying supervised coordinates by corresponding last-input channel
  observation status also yields0/180 negatives for both observed/missing groups.
- Point/interval gradients oppose in139/180, including88/90 natural-gap-origin
  probes; global median cosine-0.330. NLL/interval oppose88/180. Median weighted
  norms interval7.110, NLL2.192, point2.124, reconstruction1.071, location0.232.
- Global projection retains alignment within rounding: minimum after-cosine
  -6.59e-9, not a substantive theorem failure. It still leaves local block
  opposition (content66/180, slow54, reconstruction76, belief81).
- Read-only next-step algebra using saved AdamW moments/rates/decay predicts a
  locally point-uphill displacement in34/180 (30/90 gap-origin,4/90 general).
  Group-rate SGD alone reverses globally in0/180 on these actual probes.
  This is a hypothetical FP64 direction, NOT an executed update, historical
  training causal attribution or finite-step loss increase.
- Missing-channel scale floors bind7549/11175 left outputs and10270/11175 right
  outputs across repeated slots/models. When scale shrinkage is blocked, raw
  interval width can still decrease by lowering median location. This is a
  specific decoder-coupling hypothesis; the floors remain required and intact.
  Reconstruction has no path through the separate future query; angle conflicts
  with very small norms must not be overinterpreted.

Independent local clean detached-checkout180-probe replayPASS;1036full Python
testsPASS, including9diagnostic/replay/precision unit tests. Read-only independent cheaper-agent review found no
blocking defect. PNDS adds4compiled exact-real counterexample declarations and an
exact-rational Python test: positive Euclidean dot can reverse under unequal
coordinate rates. This possibility is not claimed as observed group-rate failure
here. Standard Lean axioms only; whole-neural/Gaussian/IEEE refinement remainsOPEN.
PNDS GitHub math and general Lean CI pass on de75865. Initial FP32 GitHub
replay fails: one rounding-sign histogram differs47vs48; a later runner also
exceeds the original raw-dot tolerance (-0.932906 vs -0.932928). These failures
are preserved and FP32 cross-host replay remains unqualified; no original
tolerance is widened. A separately registered FP64 supplement (358860dc before
measurement) repeats all180 frozen FIT probes with its own1e-9/1e-10 contract.
It reproduces0/180 NLL/point conflicts,139/180 point/interval conflicts and34/180
AdamW locally-uphill directions; projected residual cosine is about-4.25e-16.
FP64 is a separate numerical diagnostic, not a replacement of original FP32
artifacts, a new fit, proof of FP32 identity, or improved predictive capability.
Independent local and pinned-environment GitHub FP64 replay40249198749PASS
at1e-9/1e-10 (source68c48886). Initial FP64 remote run40248912057 stopped on an
input checksum mismatch with NumPy2.5.4; matching original NumPy2.3.5 resolves
that environment mismatch. Both failed attempts remain recorded. No FP32
tolerance, predictive gate, original input or model artifact was changed.

**Next training candidate, not yet trained:** protect point directions blockwise
in the proposed optimizer displacement, including preconditioning/momentum/decay,
rather than relying on global Euclidean projection. Preserve dedicated width
learning and shared reconstruction, the protected distribution, source floors,
allfive lineages and fixed-size native scan. A proposed block correction
`delta -= max(0,<g_point,delta>)/||g_point||²*g_point` needs explicit zero-norm,
clipping/decay/moment semantics, reference/Lean engineering and a NEW prospective
matched-budget protocol before real training. It supplies only a first-order
bound, not calibration, capability or gate-pass guarantees. A source-conditioned
multi-regime predictive law remains another untested architectural direction.

No new capability claim or model promotion. ME106's full qualification remains
FAIL. Do not retune weights, repeat ME106 fits, weaken floors, select favorable
seeds or open protected data to turn this diagnostic into a success story.

## 2026-10-25 — ME106: point-priority training verified; full predictive gate fails

[Draft TAC PR408](https://github.com/FPC-effortless/TAC-OSM/pull/408),
[PNDS PR19](https://github.com/FPC-effortless/pnds-lean/pull/19),
[original full results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me106-point-priority-training-2026-10-25/research/mathematical-engineering/me106/RESULTS.md).
Completed15 authentic-data fits, not a diagnostic-only proposal. No promotion,
merge or deployment; this result does not establish greater general intelligence.

The complete100,349-parameter/33-tensor ME105 forward graph, native parallel
source/evidence/8x8 memory and uncertainty floors are unchanged. ME106 adds a
training-role interface: NLL/interval prediction uses
`mu-(correction-correction.detach())`, preserving primal values while removing
the new median-head contribution to that gradient. Point and log-location losses
retain the live correction. ADtangent(c-detach(c)) is dc, notzero; cancellation
occurs because dmu=dbase+dc. Independent Python and corrected Lean semantics agree.

For P=weighted raw-medianMAE/FITstd and
U=NLL+.1log-locationMSE+.3masked-currentGaussianNLL+.25proper90intervalscore/FITstd,
compute both parameter gradients. If <gP,gU><0, remove the opposing component
`gU'=gU-<gP,gU>/||gP||²*gP`, then applygP+gU'. Exact-real directional bounds are
proved, inspired by established PCGrad (Yu etal.,NeurIPS2020), not new theory.
AdamW preconditioning, group rates and finite steps do NOT inherit a loss-descent
or coverage theorem. FP64 reductions/correction precede projected-array rounding.

Three fixed complete recipes with point weights .5/1/2 ×allfive ME103 strong
parents102001..5, sampling indices106001..5. These are warm-started lineages, not15
new independent root initializations. Everymodel gets1,536 freshAdamW updates,
batch16, contexts8/64/256; inherited weights lr3e-6, four new-head tensors3e-4.
All33 are active. Same plans/masks/initializations/profiled graphs and dense FLOP
estimates per seed; unsupported pointwise/reduction/optimizer/hardware work stays
partial. Training CPU1450.273s, two one-thread workers; no cloud training. Extra
backward work is training-only. No ME105 fit/replay was changed or repeated.

Source/registration/math freeze preceded optimization: TACadb71873, PNDS72388e9d.
All15 model/optimizer/RNG/gradient/resource artifacts were Git-frozen at6ae491e7
BEFORE REPORT; original scores ate812f655. No retries, favorable-seed selection,
threshold changes or additional ME106 fits. Cross-study inputs/orders and gradient
work differ; no old/new controlled architectural-superiority claim is eligible.

| Five-seed means, primary PM2.5 | Gap MAE | Gap90%coverage | Gap six-channel NLL | Gap interval score | Ordinary MAE | Full seed gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ME103 strong parents |37.5242|95.66%|5.1510|379.6232|33.4031|1/5 old gates|
| ME105 balanced |40.0881|94.69%|5.1491|260.6768|31.0666|2/5|
| ME106 balanced |38.6675|95.10%|5.1548|304.6523|33.4309|2/5|
| ME106 point_guard |38.0435|95.38%|5.1504|315.5035|33.5522|2/5|
| ME106 point_strong |37.2829|95.52%|5.1441|332.0939|33.6224|1/5|

**Full qualification FAIL: no recipe passes every gate in allfive seeds.** Five
individual models pass allgates; that cannot override the allfive rule. Primary
coverage remains15/15. Stronger point weighting trades interval quality for gap
point accuracy, while ME105's ordinary/sharpness gains are not retained. Strongest
point recipe improves gap MAE only0.64% vs parent mean; ordinary MAE worsens0.66%,
interval score improves12.52%. Its gap change -0.24128 has conditional posthoc95%
week-bootstrap interval[-3.19990,+2.34032], includingzero. This is not independent
confirmation. Other pollutant coverage remains below90% for several channels.
Allper-channel/seed errors, width tails and bootstrap estimates are preserved.

Unmatched practical references, not padded neural competitors: LOCF gap56.0769,
ordinary34.5156; FIT-ridge gap43.9151, ordinary32.7674. New models improve over LOCF
and gap ridge, but allthree have worse ordinary meanMAE than ridge. Benefits over
simple gap references already existed in this parent sequence; no new controller,
customer utility or general intelligence follows from them.

Authentic UCI501 CC-BY-4.0 archive SHA256
`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.
FIT2013–2014, already exposed2015:143 gap queries/21 calendar-week clusters,
4,288 ordinary queries/53 overlapping clusters; six-hour targets/256-hour histories.
Only recorded FIT labels supervise artificial hiding. Protected2016–2017 numerical
outcomes remain unparsed/unscored; repeated studies/seeds do not add independent
observations. Original ME105 and EWA physical-cache findings remain preserved.

**Engineering/reproducibility PASS:**1,027 local tests,13 pinned actual-source
PNDS tests,13 NEW limited Lean role/projection theorems and all30 fittedFP32/64
forward/state/input+weight-gradient/chunk/distribution AND new loss/projected-
training-gradient checks. Maximum projected-gradient discrepancy1.1920929e-6;
raw median0.0008544922 original units, within original relative/absolute limits.
Whole-neural/IEEE/Gaussian-law and optimizer-descent proof remain OPEN. The
inherited full forward graph is explicitly defined, not wholly Lean-refined.
Clean detached checkoute812f655 replays all15 scores/gates, all30 fitted checks and
publisher-label summaries without retraining;67 payload hashes checked.
[Original GitHub replay40220362323](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40220362323)
PASS; PNDSbuild/source-math at72388e9d PASS. These do not overturn gate failure.

**Next substantial architecture hypothesis:** richer source-conditioned forecast
regimes/conditional predictive distributions, keeping native state and source-gap
bounds. Do not continue point-weight sweeps as a claimed intelligence solution.
A multi-regime predictive head would be a NEW protocol and forward graph with full
parameter/source/math checks, not a trained result here. Attribution stays
secondary; authentic capability, independent qualification and six-family task
expansion remain outstanding. No new active family or authority is introduced.

---

## 2026-10-25 — ME105: explicit neural graph, primary gap coverage recovered; full gate still fails

[Draft TAC PR406](https://github.com/FPC-effortless/TAC-OSM/pull/406),
[PNDS PR18](https://github.com/FPC-effortless/pnds-lean/pull/18),
[original complete results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me105-protected-belief-2026-10-25/research/mathematical-engineering/me105/RESULTS.md).
This is completed capability-first architecture development and real training,
not another diagnostic-only proposal. No merge, promotion or deployment.

The input-to-output trace exposed two concrete issues: ME104 could reduce the
inherited variance-floor amplitude sixteenfold, and median-preserving head
initialization did not protect points during joint training. FIT-only autograd
confirmed point loss DOES reach the inherited location network, but has zero
direct width-head gradient; likelihood and interval losses share upstream paths,
with batch-dependent conflicting directions. It did not establish a unique cause.

New complete Temporal model: **100,349 trainable/deployed parameters,33 tensors**.
A bounded median innovation, asymmetric widths that preserve the full FIT-derived
source floor, and missing-channel scale >=80% of inherited scale. Old weights
update at3e-6, four new head tensors at3e-4. All remain trainable and active.
Parallel source/evidence/8x8 memory is retained, six active families unchanged.
This does not train the other families or establish additional memory usefulness.

Python independent serial equations reconstruct the entire network. New Lean
`SensorGraph105` explicitly defines finite affine parameters, encoder, station
lookup, chronological state, fast/slow proposals, controlled rank-one memory,
pre-event/inclusive read paths, belief/reconstruction, protected forecast and
parameterized probability-transform interfaces. It proves the100,349 scalar
shape count. Together with ProtectedSensor,26 limited theorems build with only
standard Lean axioms. Full PyTorch/gradient/IEEE refinement and Gaussian-law
instantiation/normalization proofs remain OPEN; typed definitions are not those
proofs. The full source is pinned at573f81cb, PNDS at52d65e76.

**Fifteen original fits completed:** three fixed integrated objectives ×allfive
ME103 strong parent lineages,1,536 freshAdamW updates each. These are warm-started
lineages, not15 independent root initializations. Source/registration/math preceded
training; allmodels/optimizer/RNG/receipts committed at640cafca BEFORE REPORT,
original scores atd2a8bc96. No additional ME105 fits or threshold changes allowed.
Same plans, augmentation masks, active size, profiled operator graph/dense FLOP
estimates per seed; unsupported scalar/optimizer/hardware FLOPs remain partial.
955.686 aggregate training CPU seconds, two one-thread workers; no cloud training.
Old/new differ in size, objective and exposure, so no controlled architecture win.

| Five-seed mean, original units | Gap PM2.5 MAE | Gap PM2.5 90% coverage | Gap six-channel NLL | Gap PM2.5 interval score | Ordinary PM2.5 MAE | Full seed gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ME103 strong parents |37.5242|95.66%|5.1510|379.6232|33.4031|1/5 old gates|
| likelihood |36.2703|95.24%|5.1371|403.5915|34.9623|0/5|
| balanced |40.0881|94.69%|5.1491|260.6768|31.0666|2/5|
| point_guard |39.0857|95.52%|5.1428|273.1647|31.1216|2/5|

**All15 pass the primary PM2.5 coverage gate, versus0/15 in ME104. Full qualification
still FAILS: no recipe passes allfive seeds.** Balanced improves ordinary point
MAE7.00% and interval score31.33% vs parents, but gap point MAE worsens6.83%.
Likelihood improves gap point MAE3.34%, but ordinary point error worsens4.67% and
allfive miss the interval-score requirement. Four individual seeds pass allgates;
favorable seeds do not override the allfive rule. Other pollutant coverage remains
uneven/below90% for several channels. This is not a general calibration solution.
All six-channel metrics, width tails, conditional week-bootstrap intervals and
per-seed failures are preserved. No isolated attribution claim from joint changes.

Authentic UCI501 CC-BY-4.0 archive SHA256
`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.
FIT2013–2014; already exposed2015:143 gap queries/21 calendar-week clusters and
4,288 ordinary queries/53 overlapping clusters. Protected2016–2017 numerical
outcomes remain closed. These repeated development studies are not independent
prospective evidence; no safety, policy benefit or general intelligence qualified.

**Verification:**1,015 local Python tests,8 PNDS source tests,26 limited Lean
proofs and all30 fitted FP32/64 source-vs-independent-reference graph/gradient/
state/chunk/distribution checks PASS. Clean detached checkoutd2a8bc96 reproduces
all15 scores/gates, all30 graph checks and publisher-row summaries without
retraining.67 bundle payload SHA256 checks pass.
[Original GitHub replay40216493230](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40216493230)
PASS; PNDS fullbuild/source-math at52d65e76 PASS. Original raw median numerical
max discrepancy0.000732422 units, within unchanged relative/absolute tolerances;
not an absolute2e-6 whole-output guarantee. State audit:632B/sample state tensors
(608 FP32+24 integer),72B model buffers separately, plus runtime metadata/weights/
activations. Registered736B is an inherited allowance, not total runtime RAM.

The bounded autoresearch cycle is now executable: trace FIT→implement full
candidate/reference→register/freeze→parallel fixed fits→freeze originals→score→
keep/reject→clean/GitHub replay. It adapts the method, not a claim to run external
Karpathy software. All failed candidates remain visible; no endless DEV search.

**Next substantial capability target:** separate median-learning and uncertainty-
shaping gradient roles, then improve channel-specific calibration without losing
source-gap coverage. These are prospective hypotheses, not implemented successes.
EWA001/002's negative usefulness reranking and strong physical-velocity references
remain preserved in the latest master plan; no unqualified cache is silently
added to the sensor scan or represented as608B. Attribution remains secondary.

---

## 2026-10-25 — EWA002 new real ALOHA task: cosine memory marginally helps velocity, usefulness weighting worsens, hash speed gate fails

[PR405](https://github.com/FPC-effortless/TAC-OSM/pull/405) (draft research); [prescored protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/ewa002-velocity-residual-external-2026-10-25/research/mathematical-engineering/ewa002/PROTOCOL.md); [full actual results](https://github.com/FPC-effortless/TAC-OSM/blob/research/ewa002-velocity-residual-external-2026-10-25/research/mathematical-engineering/ewa002/RESULTS.md); [first successful external scoring CI40211321061](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40211321061), original scored artifact12605927072. Software/source/checkpoint PASS, scientific **full registered gate FAIL 0/5**.

Built on EWA001 negative utility reinforcement without rewriting EWA001: five original checksum-verified, FROZEN ME097 95,874-parameter trained checkpoints provide source-causal 8D memory matching keys. The main physical prediction is **measured constant velocity**, not the worse out-of-domain neural forecast. Identical 256-slot outside-memory FIFO records are compared with (a) exhaustive cosine, (b) 11-bit LSH cosine, (c) the SAME indexed memory with strictly later-verified outcome ensemble leave-one-out utility reranking. Record retention identical across arms, thereby isolating usefulness-ranking effects. Original input-only exact parallel affine scan and 608-byte native primary FP32 state unchanged. The extension needs minimum23,552 MORE FP32 external memory payload bytes plus hash/identity/source/runtime overhead; conditional feedback cache itself is sequential, not a globally associative scan.

**New authentic source/task:** pinned MIT LeRobot ALOHA static candy source commit6728831ef9674b74f3ce20bf517890e9ee20b0d9, actual publisher consolidated v3 path data/chunk-000/file-000.parquet. The publisher card gave stale v2 per-episode paths, which was found and source-parser corrected before any scoring, preserving original failed preflight logs and exactly preregistered arms and gates. 25 real FIT experience episodes; later 25 new-task episodes,256 actual next-state transitions each, 6,400 genuine new-task records per source-model checkpoint; all five checkpoint seeds use the SAME 25 episodes. No original ALOHA-cabinet protected TEST numerical outcomes were opened. New task was never used for original ME097 fit, but causal prequential cache may learn from each earlier test episode after scoring; new-task transfer diagnostic, NOT independent safety or robotics policy qualification.

| Real heldout new-task standardized14D one-step MSE (5 model seed mean) | Result |
| --- | ---: |
| Measured constant velocity | **0.000017434** |
| Velocity + full similarity cache | 0.000017314 |
| Velocity + LSH similarity cache | **0.000017295** |
| Velocity + LSH utility reranking | 0.000017346 |
| Frozen pretrained model direct output | 0.000210494 |
| Persistence | 0.000183784 |

Full similarity improves velocity mean error only0.687%, below >=1% per-seed H1 requirement. Indexed similarity mean improvement0.795% vs velocity, not enough for gate. Utility adds a **0.298% ERROR REGRESSION against matched hashed cosine, in ALL five original checkpoints**, H2 fails. Raw hash candidate recall@8 **97.96%**, but average number of actual neural-key cosine candidates scored was **157.93 / 256 per query**, vs prereg <32 budget; H3 fails. Original all gates **FAIL 0/5**. Output does not show trained architecture superiority, usefulness-learning success, true actuator counterfactuals, safety or general autonomous intelligence. Current measured real velocity remains the strong physical reference.

Next work must preregister on a FRESH source rather than retune EWA002 on exposed candy; test motion-regime-specific memory novelty, bounded verified ensemble damage and correct source identity/provenance. Separate index candidate hit/recall from cosine+utility reranking, compare physical velocity/servo/Kalman residual models, and count CPU/RAM; no extrapolation of simple unit tests to learned physical intelligence. Preserve all original checkpoints, all six proposed research families, original negative results and compulsory equal trained parameters/data/training-compute budgets for actual new architecture comparisons.

---


## EWA001: tested verified-outcome memory utility + 8-bit LSH retrieval (2026-10-25)

**IMPLEMENTED AND SCORED, NOT QUALIFIED.** [PR404](https://github.com/FPC-effortless/TAC-OSM/pull/404) trains no new neural model, preserving five original trained ME097 weights. It adds 256-slot external learned-key/residual memory, real source-lagged event comparison by 8-bit/9-bucket-probe LSH, and updates utility only after the genuine future outcome is observed; utility optionally governs future retrieval ordering/eviction. Same-size similarity-only memory is the matched control. This is 23,552 extra float payload bytes minimum PLUS dynamic metadata/index, outside primary 608B memory.

[Executed real-source EWA001 five-seed results](https://github.com/FPC-effortless/TAC-OSM/blob/research/ewa001-usefulness-feedback-2026-10-25/research/mathematical-engineering/ewa001/RESULTS.md): exposed authentic 13 DEV episodes, 3,328 predictions/seed, original FIT-trained checkpoint and original TEST untouched. Mean standardized MSE frozen model0.000512171; neutral similarity cache0.000339658 (33.68% lower); learned-usefulness cache0.000371861 (**9.48% worse than neutral, all5seeds**); strong velocity0.000031119 (**10.9× better than neutral**). LSH 102.67 or87.31 candidate scores/query vs <32 registered budget. Full EWA registered capability gate **FAIL 0/5**, original JSON in [CI40209655856](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40209655856). Utility ranking changes cosine top8 overlap; that is NOT an isolated LSH collision metric. Early failed pre-result runs retained. Do not retune or claim independent qualification, safety, whole-neural Lean proof, additional trained family, or that external cache fits inside native 608B.

**Required next:** separate source-authorized true retrieval contribution versus false reinforcement, sparse lookup recall before utility reranking, and velocity/servo-residual comparisons on new untouched genuine data. Preserve all original results, six families, global input-only scan and matched parameter/source/training budget rules.

---


## 2026-10-25 — ME104 trained asymmetric distributions: sharper, severely undercovered

[Draft TAC PR403](https://github.com/FPC-effortless/TAC-OSM/pull/403) and
[PNDS PR17](https://github.com/FPC-effortless/pnds-lean/pull/17) implement and train
a new native Temporal candidate with99,731trainable/deployed parameters. A
zero-initialized1,236parameter102→12head predicts separate lower/upper scales
around the inherited median, reusing the source/evidence/memory read. Native
fast/slow/8×8parallel scan and persistent state are unchanged. Exactly six active
families remain; this does not train the other domain heads or establish useful
associative-memory attribution. No attribution campaign blocked development.

Fifteen original warm-started fits: three fixed complete objectives ×five sampling
seed indices104001–104005. ALLfive ME103strong_reconstruction parents are used
in every recipe; they are not fifteen new independent root initializations. Each
receives1,536fresh-AdamW updates (.0003/.001), batch16, contexts8/64/256. Old weights
reach4,608lineage updates; new head1,536. Source89d756d5, reciprocalmanifest6021a05f
and PNDSbindingd81bad75preceded optimization. All15model/optimizer/RNG/receipts
were committed ata163708eBEFORE REPORT, original scores/forecasts at6d684f07.
No old source/weights/results were replaced, no retries or post-score extra fits.

| Five-seed means | Gap PM2.5 MAE | Gap90%coverage | Gap six-channel NLL | Gap PM2.5 interval score | Ordinary PM2.5 MAE | Full gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ME103strong parents |37.5242|95.66%|5.1510|379.6232|33.4031|1/5oldgates|
| likelihood |36.9680|76.36%|5.1439|214.7926|35.4530|0/5|
| balanced |36.2121|74.41%|5.1359|222.5394|35.4438|0/5|
| point_guard |36.1055|74.27%|5.1253|223.8236|34.6475|0/5|

**Qualification FAIL: every one of15models misses the unchanged90%coverage floor.**
Gap point means improve1.48–3.78%, interval scores41.04–43.42%, and mean gap NLL
improves slightly. But ordinary point error worsens3.73–6.14%; only2of5seeds per
recipe meet its daily point guardrail. No recipe passes allfive, and0of15pass
every gate. New per-seed interval requirements additionally require10%material
improvement versus ownparent, but the failure is not merely a stricter new gate:
all15also fail the original coverage criterion. Mathematical coherence, sharper
intervals and lower average proper scores do not establish adequate calibration.

The output law is an equal-half Gaussian quantile transform in standardizedlog1p
coordinates, followed by the original zero-censored positive transform. One CDF,
proper density/zero atom and quantiles agree, including either side of the zero
threshold. It is not the standard continuous-density two-piece normal. At fixed
backbone weights the head preserves the median, not arithmeticmean; initialization
reproduces parent forecasts exactly, but end-to-end updates may move the backbone
and points. Scales may shrink to onequarterofoldscale, reducing the inherited
variance-floor amplitude to a conditional old_floor/16lower bound. No preserved
old floor amplitude,90%coverage guarantee or absolute72hour cap was claimed.
Source age remains capped256hours. These are conditional per-channel laws at one
horizon, not identified latent posteriors or coherent action trajectories.

Same authentic UCI501 FIT2013–2014, CC-BY-4.0 archive SHA
d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.
2015was already exposed:143natural-gap queries/21calendar-week clusters and
4,288ordinary queries/53overlapping clusters. Artificial training masks use actual
recorded labels only. Protected2016–2017 numerical outcomes remain closed; repeated
models do not add independent cases. All raw per-channel/seed scores, width tails
and conditional posthoc paired intervals are preserved in
[original evidence](https://github.com/FPC-effortless/TAC-OSM/tree/research/me104-median-split-belief-2026-10-25/research/replay-bundles/ME104).

Per model24,576sampled examples/2,686,976history exposures; records repeat. New
recipe parameters/initializations/dataorders/masks/profiled graphs/dense FLOPs
match.897.529aggregate CPU seconds, max2one-thread workers; CPU is an outcome.
Unsupported pointwise/optimizer/hardware FLOPs are unquantified. Old/new differ
in parameters, optimizer/objective and exposure, so no controlled architectural
superiority is eligible. All31parameter tensors have finite nonzero FIT gradients.
All997local tests,7PNDS source tests and six core-axiom-only Lean scalar proofs
pass; PNDSbuild/source-mathCIpasses. Every fitted network passes30combinedFP32/64
source/reference state/output/input+weight-gradient/CDF/NLL/chunk checks at the
original tolerances. Whole-neural/IEEE, Gaussian integration Lean, calibration,
hardware latency, customer value and safe control remain OPEN.
Clean detached checkout6d684f07replayed all15original models/forecasts/scores/gates,
all30fitted dtype audits and the summary/publisher-row audit without retraining.
All67bundle payload SHA256checks pass.
[Independent GitHub replay40195130742](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40195130742)
passed all model/score/graph/summary/source checks. Latest metadata-only commitCI
is separate from original-source replay and from predictive qualification.

**Next integrated capability target:** separate point-learning and uncertainty
updates, while retaining and learning adequate source-gap uncertainty floors.
Differential optimization/gradient roles and stronger source-floor preservation
are hypotheses for a new source-pinned real-data protocol, not proven causes or
implemented fixes. Do not run more ME104fits, select favorable seeds, widen gates
or open protected data to tune. Preserve all negative evidence; no model promoted,
merged, deployed or represented as generally intelligent or commercially ready.


## 2026-10-25 — ME103 completed: more training improves points, full gate still fails

[Draft TAC PR402](https://github.com/FPC-effortless/TAC-OSM/pull/402) and
[PNDS continuation PR16](https://github.com/FPC-effortless/pnds-lean/pull/16)
complete the registered one-time continuation of EVERY15original ME102 model.
Three fixed complete recipes ×five shared seed indices; these are continued
lineages, not15new independently initialized seeds. Exact model/AdamW moments,
augmentation RNG and sampling-stream suffixes were preserved. Each model receives
1,536new guarded updates, reaching3,072cumulative, with the unchanged98,495parameter
graph. Source/registrationfb50159b and mathematical-link330526b6 were frozen before
updates. All15child checkpoints/optimizer/RNG/receipts were committed at1a728fdb
BEFORE any REPORT; original scores/forecasts were committed ate85268a0.

| Five-seed mean | ME102→ME103 gap PM2.5 MAE | ME103 gap90%coverage | ME102→ME103 gap interval score | Full-gate seeds old→new |
| --- | ---: | ---: | ---: | ---: |
| light_floor |38.7814→38.2721|82.80%|229.2109→224.0540|2/5→0/5|
| strong_reconstruction |39.7432→37.5242|95.66%|316.7059→379.6232|2/5→1/5|
| physical_score |44.1222→41.8294|93.29%|221.6744→225.6172|1/5→2/5|

Mean gap point error improves1.31%,5.58%,5.20%; ordinary PM2.5MAE improves from
34.3075/33.8963/33.3507 to33.1687/33.4031/32.5165. Every model now passes the
ordinary-point guardrail. But mean gapNLL worsens in all three recipes, light-floor
coverage fails every seed, strong reconstruction's interval-score gate fails4/5,
and physical-score's gap-point gate fails3/5. **No recipe passes all five seeds;
only3of15individual models pass every gate**, compared with5of15ME102parents.
This is a real same-task point-fitting improvement with substantial uncertainty
trade-offs, not overall qualification, architectural superiority or intelligence
readiness. Keep every model/failure; no favorable-seed selection or promotion.

Authentic UCI501 FIT2013–2014, original CC-BY-4.0 archive SHA d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.
2015was already exposed:143natural-gap queries/21calendar-week clusters and
4,288ordinary queries/53overlapping clusters. Repeated models do not multiply
independent observations. Protected2016–2017 numerical outcomes remain closed.
The posthoc paired bootstrap conditions on fitted models and is not multiplicity
adjusted or independent confirmation. All original per-channel/seed metrics,
width/tail summaries, natural missing masks and publisher-row replay checks are
[repository-owned](https://github.com/FPC-effortless/TAC-OSM/tree/research/me103-equal-continuation-2026-10-25/research/replay-bundles/ME103).

New exposure24,576sampled examples/model and2,686,976history-event exposures/model;
records repeat. Same-seed augmented-input/order checksums and profiled ATen graphs
match across recipes. Dense forward/backward estimates match, but optimizer/
pointwise/hardware FLOPs are not fully accounted for; CPU is an efficiency outcome.
The new stage used857.674aggregate CPU seconds with two one-thread workers. The
before/after doubles training exposure, so it is not an equal-compute architecture
comparison. All984local tests and all30fitted FP32/FP64 graph/state/chunk/gradient
checks pass. The existing ME102Lean/Python formulation is inherited byte-unchanged;
PNDSPR16build passes. Full-neural/IEEE and calibration guarantees remain OPEN.
Clean detached checkoute85268a0replayed all15models/forecasts/scores/gates, all30
fitted dtype checks and the summary/publisher-row audit with no retraining.
All52bundle payload checksums pass.
[GitHub original replay40191654367](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40191654367)
passed all original model/score/graph/summary/source-label checks. Source replay
is separate from latest metadata-only commit CI and from predictive qualification.

**Next integrated capability direction:** preserve strong point predictions while
learning a coherent source/channel-conditioned predictive distribution with useful
sharpness under missing observations. A fixed floor alone does not guarantee
coverage. A mean-preserving distributional decoder/warm start is a hypothesis for
a new mathematical/source-pinned protocol, not an implemented or verified fix.
No further ME103continuation is authorized by this fixed protocol; no default
attribution campaign, protected-cohort tuning, unsafe variance cap or favorable
seed selection. Exactly six active families remain; only the Temporal candidate
was trained here. No robot-controller, customer ROI, embedded-hardware or
commercial-readiness claim follows. Historical failures/weights remain unchanged.


## 2026-10-25 — ME102: fifteen trained belief-reconstruction models, mixed capability result

[Draft TAC PR398](https://github.com/FPC-effortless/TAC-OSM/pull/398) and
[PNDS PR14](https://github.com/FPC-effortless/pnds-lean/pull/14) extend the existing
Temporal shared architecture, keeping exactly six active model families. Three
complete training recipes, five shared seeds each, test masked current-state
reconstruction, source-aware associative retrieval and non-cancellable FIT-derived
staleness variance floors. These are actual integrated training experiments, not
a component-ablation campaign. Each original model has98,495learned parameters
and1,536updates. Same-seed initial weights, real training orders and actual
augmented-input checksums match across recipes; profiled operator graphs and
dense FLOP estimates match, with unsupported hardware/pointwise accounting
explicitly excluded. CPU runtime remains an outcome, not a parity gate.

| Five-seed mean | Gap PM2.5 MAE | Gap90%coverage | Gap six-channel NLL | PM2.5 interval score | Full-gate seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| ME101 historical reference |37.7470|68.67%|6.2345|301.2206|historical|
| ME102 light_floor |38.7814|88.53%|5.1446|229.2109|2/5|
| ME102 strong_reconstruction |39.7432|96.08%|5.1327|316.7059|2/5|
| ME102 physical_score |44.1222|93.43%|5.1779|221.6744|1/5|

**No recipe passes the registered full gate in every seed.** Coverage and gap
likelihood improve; average point accuracy regresses. Strong reconstruction also
worsens interval score. No favorable-seed selection, blanket safety/calibration
claim, model promotion or independent architectural superiority is justified.
The full15original model/optimizer/RNG/budget receipts were committed at4a0e75a9
before REPORT; original scores and all failures were committed at93ecebd4.
Actual-source FP32/FP64 fitted graph, state, chunk and gradient checks pass;
Lean supplies partial scalar kernels, not full-neural/IEEE or safety proof.

Authentic UCI501 FIT2013–2014; REPORT2015 already exposed:143natural-gap queries
in21calendar-week clusters and4,288ordinary queries in53clusters. Repeated seeds
do not add independent observations. Protected2016–2017 numerical outcomes stay
closed. Training masks are augmentation, not new real evidence. No embedded
hardware benchmark, action intelligence, customer utility or commercial-readiness
percentage is established. Full original reports and per-seed distributions are
[repository-owned](https://github.com/FPC-effortless/TAC-OSM/tree/research/me102-reconstruction-panel-2026-10-25/research/replay-bundles/ME102).

Original15-model score, graph, publisher-row and summary replay passed from a
clean checkout of93ecebd4. GitHub original-score replay
[40189256081](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40189256081)
also passed. Expanded original-model/graph/summary/publisher-row GitHub replay
[40189770334](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40189770334)
passed after correcting a cross-CPU diagnostic-metadata equality bug, without
changing tensor tolerances or saved evidence. All982local tests pass.
Checkpoint replay establishes reproduction, not qualification.

**Next fixed hypothesis:** ME103 is a proposal to continue every15original model
equally for1,536additionalupdates (3,072lineage), preserving exact optimizer/RNG,
with a new source-frozen preflight and unchanged gates. This would test budget
sufficiency after ME102 point regressions. It is not registered or trained yet;
no continuation is claimed. The recipe/seed search ledger remains fully visible.

**Owner-supplied ALOHA update:** the owner reports5models,49FIT training episodes,
10FIT-only internal evaluation episodes,1,840next-state predictions and clean
replay, with neural MSE below persistence but every TACseed failing the strongest constant-velocity reference
(about one-tenth neural MSE). No oldDEV/protectedTEST was reportedly scored. The
exact original study/checkpoint/score/replay references have not yet been located
in the inspected records: retain this as attributed evidence pending verification,
not a new independently audited result or an overwrite of earlier ALOHA studies.
Physical development must account for velocity/dynamics references, without
interpreting command-conditioned prediction as a trained robot controller.


## ME081 — learned horizon-query forecasting: stronger averages, quiet-state trade-off

[Draft PR #375](https://github.com/FPC-effortless/TAC-OSM/pull/375) trains **five complete 100,165-parameter native models**. Fast/slow/8×8 associative dynamics remain intact. New training uses 32 actual observation frames with full gradient credit, a shared horizon-query distribution decoder, and properly scored CRPS plus a small log score and factual motion supervision. This is actual end-to-end learning, not an execution-only result or attribution study.

Primary evaluation preserves the exact original eight-frame task: 40 already-exposed real REPORT episodes, 960/956/928 factual pairs at 1/4/16 frames. All figures below are means of individual models, not an ensemble deployment.

| Physical MAE, lower is better | Frozen ME080 | ME081 | Persistence |
| --- | ---: | ---: | ---: |
| 1 frame | 0.059393 | **0.055968** | 0.068668 |
| 4 frames | 0.219619 | **0.166113** | 0.302400 |
| 16 frames | 0.883994 | **0.737952** | 0.975966 |

**Improved:** mean errors fall **5.77%, 24.36% and 16.52%** versus ME080, and 18.49%, 45.07% and 24.39% versus persistence. Four-frame MAE is also 10.67% below the stronger ME079 native reference. Actual-moving one-frame MAE falls from 0.533420 to **0.408457**, below ME078's 0.421182 and clipped velocity's 0.482509 on the same 84 moving queries. Proper interval scores fall **51.57%, 38.10% and 50.50%**, with nominal 90% coordinate coverage 92.92%, 92.88% and 91.88%.

**Regressed:** stationary MAE rises **59.04%, 55.65% and 162.28%** versus ME080 at the three horizons. One-frame stationary MAE is 0.022168, about 13 times ME077's 0.001701. Physical marginal NLL also worsens at every horizon. One-frame risk AP/AUC (0.54992/0.91466) improves over ME080 but remains below the stronger historical ME078/079 rankings. Better interval scores do not establish uniformly better calibration.

All five fits improve primary one-frame mean error over their ME080 counterparts, but the fixed-model paired 40-episode bootstrap 95% interval for new-minus-old error is **[−0.012166, +0.005730]**, including zero. Quiet preservation and this confidence gate fail. **Full gate FAIL / MIXED_TRADEOFF; no further ME081 fits.** There is no independently confirmed intelligence or architecture-superiority claim.

**Actual longer-context readout:** 904 registered genuine 32-frame query histories provide 904/900/872 future pairs. Mean errors are 0.055557/0.157857/0.740098. On exactly those IDs, the same model with eight frames scores 0.055775/0.159718/0.748874. These small descriptive differences do not demonstrate a material memory advantage or arbitrary-history retention. Both information conditions and all adverse cases are preserved.

**Resource/source scope:** unchanged licensed BodenAI/RW-RL-HIL archive, 120 FIT / 40 exposed REPORT, source revision 9290f5a96dc91ea58256e81d9ce2dd9621890b51. Five cold starts, 2,048 updates and batch 24, final-only, no sweep/restart/warm start. Each model sees 1,572,864 observed prefix presentations, **four times ME080**; no matched-compute, sample-efficiency or architectural-winner claim. Protected tissue-box, source CAL and TEST remain closed. Factual action inputs are not identified causal control.

**Original proof:** prefit `b5698c86`; all five whole-model/optimizer/RNG/live receipts frozen at `dd71fee2` before either readout; original scores `bf23f52b`. Fresh remote sparse checkout restores and replays **27,600 original model forecast records** and recomputes the full summary exactly locally. These remain repeated forecasts over 40 episode clusters. The initial disposable full clone hit disk capacity; original worktrees/checkpoints were untouched.

**Strict external replay FAIL:** [run 40021655456](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40021655456) completed the all-model/both-history census. Out-of-tolerance record counts: point prediction 8, MAE 5, lower quantile 8, upper quantile 6, width 12, interval score 11; counts overlap. Max absolute point drift 1.50443e-5 and MAE drift 4.18854e-5. Targets, coverage, risk probabilities/Brier and NLL meet the frozen contract, but full replay remains failed. No old/new tolerance or original output changed. Local full suite: **1,169 tests PASS**.

[Full results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me081-horizon-query-world-2026-10-23/research/mathematical-engineering/me081/RESULTS.md) · [Original checkpoints and forecasts](https://github.com/FPC-effortless/TAC-OSM/tree/research/me081-horizon-query-world-2026-10-23/research/replay-bundles/ME081).

**Next capability requirement:** retain these real moving-state/mid-horizon skills while recovering quiet precision and density quality, then independently qualify the complete model. New fitting requires a new substantive prospective registration. No default attribution campaign, hidden retry, protected TEST access, universal-intelligence claim, merge or deployment. Enterprise/Temporal/Edge/Counterfactual, causal policy and customer/operator value remain separately unqualified.

## Native computation reuse — formal architecture and measured execution improvement

[Draft PR #372](https://github.com/FPC-effortless/TAC-OSM/pull/372) implements the owner's request for mathematical notation and an optimized computational workflow. The [specific native specification](https://github.com/FPC-effortless/TAC-OSM/blob/research/native-computation-reuse-2026-10-23/research/computation-reuse/MATHEMATICAL_ARCHITECTURE.md) defines a shared encoder tensor DAG, explicit fast/slow/associative transition, conditional mixture decoder and versioned caller-owned inference effects. It preserves the trained ME080 architecture and all 99,816 parameters; **no model was fitted or predictive-intelligence improvement claimed**.

The execution-only implementation batches independent decoder positions, decodes only the final prefix when requested, shares three quantile inversions, and rejects stale/out-of-order/conflicting cached events. Ordinary parameter/buffer/mode changes invalidate the inference handler. Cache state never substitutes for a training graph. No SSM conversion, associative nonlinear scan, Triton/GPU/compiler fusion or category-theory novelty is claimed.

Original registered CPU timing, five frozen models, batch 24/history 8/one thread: median paired speedups across fits **1.457× all-prefix forward, 1.369× forward/backward, 1.845× final-prefix inference and 1.958× three quantiles**. Five warmups and 31 randomized paired measurements per workflow/model, raw times/IQRs/bootstrap intervals preserved. The fixed timing batch repeats one eligible FIT transition; this is a shape-specific local execution result, not workload-wide latency, independent hardware or energy qualification.

All five models matched the reference bit for bit locally on all 960 original exposed query windows each, including final optimized quantiles. Float64 software tests also check every model parameter gradient. **1,166 local tests passed.** External optimized/reference agreement is recorded separately; this never changes ME080's original three lower-quantile and one interval-score cross-platform exceptions. Original checkpoints, scores, tolerance and failed full capability gate stay intact.

The [results and boundaries](https://github.com/FPC-effortless/TAC-OSM/blob/research/native-computation-reuse-2026-10-23/research/computation-reuse/RESULTS.md) distinguish reusable execution from useful memory. Eight-frame training still does not qualify arbitrary long-history prediction. The next capability-first training registration must address real motion, mid-horizon and quiet-state trade-offs; execution efficiency does not satisfy an intelligence or customer-value gate. No protected source, new fit, main merge or deployment.

Independent [GitHub verification 40012640343](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40012640343) **PASS**: software values/gradients/cache tests, original model/receipt integrity, all five models and every original real query, including final quantiles. This compares optimized and reference execution on the external host; external timing was not measured, and historical ME080 cross-CPU replay exceptions remain FAIL and unchanged. Engineering tier only; no new fit or learned-intelligence gain.

## ME080 — native mixtures and reusable state: mixed real capability

[Draft PR #371](https://github.com/FPC-effortless/TAC-OSM/pull/371) trains **five original models with 99,816 parameters each**, retaining native 32D fast state, 32D slow state and an 8×8 associative matrix. The integrated model learns quiet/movement Laplace mixtures at 1, 4 and 16 frames; medians and marginal intervals come from the same CDF. Observation-only projections are shared within chunks, and explicit caller-owned state reuses prior computation. TAC's nonlinear dynamics remain intact. No SSM conversion, associative scan, fused GPU kernel or category-theoretic novelty is claimed.

The same licensed source supplies 120 FIT and 40 **already exposed** REPORT episodes: 960, 956 and 928 actual query–target pairs at the three horizons per seed. Each cold start receives 2,048 updates, batches of 24 windows and eight supervised causal prefixes. One fixed likelihood/BCE recipe, no search, restart or warm start. This is single-architecture capability development, **not** a matched-compute architecture comparison or a five-model deployment. No protected tissue-box, source CAL or TEST access.

| Physical MAE, lower is better | Frozen ME079 native | ME080 native | Persistence |
| --- | ---: | ---: | ---: |
| 1 frame | 0.060595 | 0.059393 | 0.068668 |
| 4 frames | 0.185953 | 0.219619 | 0.302400 |
| 16 frames | 0.947424 | 0.883994 | 0.975966 |

**Improved:** relative to ME079, one-frame mean MAE falls 1.98%, 16-frame MAE falls 6.69%, stationary one-frame MAE falls 43.79% (0.024795 → 0.013938), and 16-frame proper interval score falls 12.62%. Mean errors are 13.51%, 27.37% and 9.42% below persistence at the three horizons.

**Regressed:** four-frame error rises 18.10%; actual-moving one-frame error rises 22.93% (0.433934 → 0.533420); risk AP/AUC falls to 0.53443/0.89689 from 0.60775/0.91971; one/four-frame interval scores worsen. Stationary error remains 8.19 times ME077 and 3.50 times persistence. Overall one-frame error also remains slightly worse than clipped velocity. Nominal 90% coordinate coverage of 93.46%, 91.96% and 91.91% does not establish sharp calibration.

The paired 40-episode bootstrap 95% interval for one-frame new-minus-ME079 MAE is **[−0.009942, +0.006327]**, including zero. Four of five seeds improve, but the registered 5% mean threshold and several guardrails fail. **Full gate: FAIL. Tier: MIXED_TRADEOFF. No further fits under ME080.**

**Reuse scope:** the trained models' raw outputs agree for full eight-frame, split 3+5 and one-observation execution on all original 960 queries per seed on the local host. State tensor payload is 628 bytes per entity, excluding weights, activations and container/allocator overhead. Actual training and accuracy scoring reset after eight frames; 40-step software equivalence is **not** long-history prediction qualification. Callers supply identity/known commands and invalidate caches when weights or preprocessing change. Policy, causal action, safety, enterprise transfer and customer value remain unqualified.

**Preservation:** prefit `4c7309ff`; all five original model/optimizer/RNG and live receipts frozen at `e37a4b88` **before** scoring; original scores at `ea6cdd5e`. A fresh remote clone restored all five models, replayed 14,220 raw forecast records and recomputed the entire summary exactly locally. These are repeated forecasts over 40 episode clusters. Strict external replay `40005926076` **fails at seed 80002's lower quantile**; the full read-only census is recorded separately without changing tolerance or original outputs. No all-platform-green claim. A separate helper repairs the missing original-registration reference in final delivery metadata; original model bytes, registration, runner and forecasts remain unchanged, and the initial failure remains documented.

[Full results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me080-streaming-mixture-world-2026-10-23/research/mathematical-engineering/me080/RESULTS.md) · [Original checkpoints](https://github.com/FPC-effortless/TAC-OSM/tree/research/me080-streaming-mixture-world-2026-10-23/research/replay-bundles/ME080).


Full external census [40006593903](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40006593903) restored all five models and 14,220 forecasts. **Three lower-quantile records and one interval score exceed the original tolerance; every other recorded field passes**, including point predictions, actual targets, physical MAE, NLL, risk probabilities/Brier, upper quantiles, widths and coverage. Max lower drift is 8.3226e-6; max interval-score drift is 1.09736e-4. Counts can overlap. The audit process completed, but strict full replay remains FAIL. Original outputs and tolerances stay unchanged. The final local suite passes 1,161 tests.

**Next requirement:** develop a complete model that preserves quiet precision **and** changing-state/mid-horizon skill. Any additional fitting needs a new substantive prospective registration; no hidden retry, default attribution campaign or protected TEST tuning. Independent-source qualification and operator usefulness remain outstanding. Enterprise, Temporal, Edge and Counterfactual evidence stays separate.

## ME078–ME079 — trained real motion capability advances; no native architecture winner

Two complete integrated physical-model recipes are now trained and preserved:
[ME078 draft PR#368](https://github.com/FPC-effortless/TAC-OSM/pull/368) and
[ME079 draft PR#369](https://github.com/FPC-effortless/TAC-OSM/pull/369).
**30original fits**: each recipe has native fast32/slow32/8×8associative TAC,
GRU56 and compact causal RMSNorm Transformer, five paired seeds per family.
Each family received both recipe opportunities, identical actual source data,
sampling/importance weights,8causal-prefix supervision and1/4/16frame targets,
2048updates/batch24, final-only checkpoints, no sweeps or restarts. All actual
trainable/deployed model counts remain95k–105k; pairwise gaps<0.78%.
Audited forward/backward/clip plus analytical AdamW scalar-operation estimates
match within0.66%, with zero unsupported floating ops and independent matrix
profiler confirmation. This is **estimated algorithmic work**, not hardware
FLOP/instruction/time/energy or foundation-scale equivalence.

Same original licensed robot source:120FIT/40already-exposedREPORT episodes;
960H1queries,956H4pairs,928H16pairs. No source CAL/TEST/protected tissue-box.
Scores are means of individual model errors, not deployment of five-model
ensembles. Source-ending/takeover censoring is explicit.85,320raw forecast
records across the two recipes are repeated model/seed evaluations, **not**
85,320independent real observations or newly gathered trajectories.

| Mean physical MAE | Frozen ME077 | ME078 native | ME079 native | ME079 GRU | ME079 Transformer | Persistence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1,960queries | 0.066476 | 0.061221 | 0.060595 | **0.058553** | 0.059470 | 0.068668 |
| H4,956pairs | not trained | 0.195231 | 0.185953 | **0.181330** | 0.188688 | 0.302400 |
| H16,928pairs | not trained | 0.950255 | 0.947424 | 0.937354 | **0.923996** | 0.975966 |

**Real skill gains:** ME078 native actual-moving H1MAE0.421182 versus ME077
0.741981, persistence0.743241 and strongest tested fixed physical reference
(clipped velocity)0.482509: **43.24%below ME077 and12.71%below velocity**.
ME079moving0.433934 is slightly worse than ME078 but still substantially better
than those physical references. ME079 overall improves persistence by
**11.76%/38.51%/2.92% at1/4/16frames**. Mean individual native-fit motion AP/AUC
rises from ME077**0.16222/0.63493** to ME078**0.61717/0.92603** and ME079
**0.60775/0.91971** on the same84positive/876negative factual cases. This is
actual developmental movement discrimination, not safety, policy or customer
utility. Separate mean-probability ensemble diagnostics are not100kmodel claims.

**Material failures:** stationary H1MAE regresses from ME0770.001701 to ME078
0.026704 and ME0790.024795. ME079's added within10%stationary-preservation gate
FAILS. Both registered dual learned-control superiority gates FAIL. ME079's
aggregate horizon-relative error is0.822702 vs GRU0.804255 and Transformer
0.812258; adjusted97.5%episode-cluster intervals favor controls on this exposed
cohort. Full gates remain FAIL and evidence tier **MIXED_TRADEOFF**, not a
superior native architecture, independent intelligence or general world model.
ME077 remains the better stationary specialist; do not silently stack multiple
100kmodels and advertise a single100kdeployment. No further fit is authorized
by either completed registration; all attempts and adverse results stay visible.

**Preservation:** ME078 prefitsdd824ffe, all15original checkpoints frozen0352faba
before scoring, scores8b1dfb8c. ME079 prefits5fe54ce4, all15originals frozenaabb05b4
before scoring, scoresd1c8e2ed. Original model/optimizer/RNG/live receipts and raw
forecasts/targets/probabilities/intervals are Git-owned. Independent fresh remote
clone replays PASS for all30originals and original metric summaries, without
refits.1156local tests PASS in the final branch. **Strict external replay FAILS**
on54ME078 and29ME079 derived interval scores. Full external audits restore all
models and confirm every other forecast/score field meets the frozen tolerance;
no original model/score or tolerance changed. No all-platform-green claim.

Read the full[ME078 report](https://github.com/FPC-effortless/TAC-OSM/blob/research/me078-native-kinematic-world-2026-10-23/research/mathematical-engineering/me078/RESULTS.md)
and[ME079 report](https://github.com/FPC-effortless/TAC-OSM/blob/research/me079-risk-conditioned-physical-2026-10-23/research/mathematical-engineering/me079/RESULTS.md).

**Next capability priority:** a stronger integrated native predictive-state model
that retains stationary fidelity while forecasting genuine motion and longer
future trajectories, using authentic available context and established methods.
No attribution campaign is a prerequisite, but no extra small-gate retry may hide
these failures. Independent source/condition qualification remains necessary
before broader claims; the protected target is not automatically authorized.
Physical learning does not qualify the separate Enterprise/Temporal/Edge/
Counterfactual lanes. All outputs remain advisory; action/control authority,
operator benefit, investment/customer value and deployment are NOT_MEASURED.


## ME077 — native selective physical prediction, mixed authentic development result (2026-10-23)

[Draft PR #364](https://github.com/FPC-effortless/TAC-OSM/pull/364) trains the next
**whole native System1 model**, not another attribution study. Five original
102,503-active/deployed-parameter models combine bounded fast recurrence,
already-observed velocity context, a persistence/motion Laplace mixture,
mixture-median prediction and coherent uncertainty. Natural within-task outcome
frequency replaces ME076's enriched motion sampling. Larger training budget is
explicit: 1,024×24 transition presentations per seed versus ME076's 160×12.
This is a **practical integrated-system development comparison**, not matched
architecture, sample-efficiency, new-source transfer or control superiority.

Same exact authentic 40 exposed REPORT episodes / 960 genuine next-state queries:

| Cohort | Frozen ME076 MAE | New ME077 MAE | Persistence MAE | FIT servo MAE |
| --- | ---: | ---: | ---: | ---: |
| All 960 | 0.080158 | **0.066476** | 0.068668 | 0.097457 |
| Actual moving 84 | 0.748786 | 0.741981 | 0.743241 | **0.704517** |
| Near-stationary 876 | 0.016043 | **0.001701** | 0.003983 | 0.039245 |

Five-seed means. Overall error decreased **17.07% vs ME076**, **3.19% vs
persistence**; stationary error decreased **89.39% vs ME076**. All five seed and
task means improved overall. Paired episode-cluster 95% CI for new-minus-
persistence MAE [-0.002446,-0.001945], conditional on the fixed fitted models and
this exposed population. **Full registered gain gate FAILED**: persistence gain
missed 5%; moving-state gain was only 0.17%, and servo remains stronger there.
Tier **MIXED_TRADEOFF**, not independent intelligence, reliable motion anticipation,
causal action utility, safety or investor/customer qualification. Original
ME076 negative evidence and protected tissue-box holdout remain unchanged.

Original prefit/source freeze `1396564b`; all five original weights+optimizer+RNG
and live receipts frozen at `1b358218` **before** REPORT scoring; raw forecasts
and scores at `3cabf74c`. Fresh remote Git clone restored and replayed all five
models and raw real scores within the unchanged numerical contract, with no
refit. Separate [GitHub original-model/source replay39992505986](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39992505986) and normal CI passed at score commit3cabf74c; the final local suite passed1,142tests. No universal platform guarantee.
[Full results and limits](https://github.com/FPC-effortless/TAC-OSM/blob/research/me077-selective-physical-state-2026-10-23/research/mathematical-engineering/me077/RESULTS.md).

**Next physical capability priority:** improve genuine changing-state prediction
without losing stationary gains, through an integrated model trained on authentic
multistep outcomes and strictly available action context. No automatic new fit,
sealed-task opening, attribution campaign or policy deployment. Enterprise,
Temporal, Edge and Counterfactual lanes retain their separate evidence status.
Operational/customer value is UNKNOWN/NOT_MEASURED.


This is a **starter registry in TAC-OSM**, pending creation of a separate cross-project ideas repository. The registry does not claim that proposals were implemented or evaluated.


## DIAGKEY004 native memory interface repair — synthetic-only

[Draft #362](https://github.com/FPC-effortless/TAC-OSM/pull/362): original guarded
101,527-parameter native model, shared write/query addresses and tied value
codes, decoupled gates and protected queries. Five final synthetic accuracies
99.80/100/99.41/100/100%; all-five90% immediate-recall gate passes. Same-task
640-update mean97.93% versus unchanged native30.31%; at4096 unchanged native
improves to94.02% but one seed77.15%. All25 original fits and50 milestone
checkpoints/logits preserved. Full FLOP parity is unverified. Fresh local clone
replay passed; GitHub predicted classes/NLL match, with2/819,200 logit elements
failing the original tolerance. That strict external exception remains visible.
**ENGINEERING_VERIFIED synthetic mechanism**, not a real skill/architecture winner.
Next interference rung is not trained or automatically authorized. Preserve
independent authentic System1/Physical/Enterprise work and ME076 negatives.

## ME076 — Actual authentic System 1 native predictive failure (2026-10-23)

**Real original FIT training and replay completed, strict skill gain FAILED.** [Draft PR #360](https://github.com/FPC-effortless/TAC-OSM/pull/360) now includes **five fitted 100,476-parameter** native fast-only whole models, 160 real source optimizer updates per seed, original source FIT 120 episodes / source-exposed episode-disjoint 40 report, five original checkpoint+optimizer+RNG files and 5×960 original real per-case predictions. Independent clean-runner checkpoint+score replay **PASS all five**; independently audited original training budgets 100,476 learned parameters and 160 updates/seed **PASS**, no untrained-head camouflage. Original source 160-episode SHA `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`; original CAL, original TEST and the reserved protected task **never opened**.

| Source-exposed REAL next-state evaluation | Original persistence MAE | FIT-identified servo lag1 | Five-seed native mean MAE | Verdict |
| --- | ---: | ---: | ---: | --- |
| 960 observations/seed on 40 original report episodes | **0.068668** | 0.097457 | 0.080158 | **FAIL**: native +16.7% error versus persistence, all five seeds |
| Actual 84 motion-rich observed cases | 0.743241 | **0.704517** | 0.748786 | **FAIL** versus both practical controls |
| 876 near-stationary observed cases | **0.003983** | 0.039245 | 0.016043 | **FAIL**: excessive corrections |

See [full original real negative report](https://github.com/FPC-effortless/TAC-OSM/blob/research/me075-system1-native-gated-delta-2026-10-23/research/mathematical-engineering/me076/RESULTS.md) and its [posthoc per-real-episode uncertainty](https://github.com/FPC-effortless/TAC-OSM/blob/research/me075-system1-native-gated-delta-2026-10-23/research/mathematical-engineering/me076/POSTHOC_REAL_SCORE_DIAGNOSTIC.json). The 40-case episode-cluster posthoc 95% interval for **native minus persistence** physical MAE is **+0.011116 to +0.011862** on the **already exposed DEV** population. Worse performance was also observed for all five genuine source tasks separately. This is real negative evidence, **not** independent heldout transfer, module attribution, causal action intelligence or an investor-worthy System1 product advantage.

**Next capability lane `S1-002` (not fitted or approved as a gain):** jointly learned, uncertainty-aware movement gate + sparse conditional bounded physical residual, keeping factual `x_t` persistence as an explicit candidate and accounting for realistic ∼9% genuine movement prevalence. Study early identification of actually changing states, stationary false corrections, physical proper NLL and tail/abstention risks; no target-derived movement decision feature. Five fitted original ME076 models and same-case persistence/servo are now **frozen comparisons on development data**; a truly untouched real task/source must be preregistered for `INDEPENDENT_GAIN`. Preserve owner 95–105k live model gate, genuine real source, original Git checkpoint+independent replay, and no default component-ablation cycle.

**Investor policy:** distinguish authentic training/replay integrity (achieved) from physical predictive advantage (failed), device latency (unmeasured), safe action control (unverified) and willingness-to-pay (unknown). No 25k System1 extraction, hardware production or ALLOW/ABORT deployment is justified by this record.

## ME075 actual System 1 engineering status — 2026-10-23

**Current owner priority:** focus the next capable-model design cycle on truthful real one-step System1 performance and a **separate** mathematically grounded Gated Delta associative memory candidate. DIAGKEY001 PR #357 guarded original synthetic recall fits (TAC 3.75/4.49/3.48% versus Transformer 99.06/98.79/98.87% at16/64/256) reveal *this fitted mechanism/recipe* did not learn the synthetic key/value task, but their ~18–21% partial FLOP budget disparity and artificial sources preclude universal architectural inferiority or real-world prediction conclusions. Do **not** block proven ability in five other domain lanes merely because random-key recall failed.

| Registration | Implemented now | Measured evidence / exact tier | Biggest missing capability |
| --- | --- | --- | --- |
| `ME075-S1` | [Draft PR #360](https://github.com/FPC-effortless/TAC-OSM/pull/360) `System1FastModel`: 96D fast-only recurrence, exactly **100,476** active trainable/deployed parameters on physical registered 14D streams, bounded residual one-step next-state distribution, no ungrounded risk head, no authority or rollout | Engineering **only**, [test workflow](https://github.com/FPC-effortless/TAC-OSM/actions/workflows/me075-system1-engineering.yml); **zero original new ME075 fitted native checkpoints or real scored cases** | Actual training and honest real one-step prediction on source verified physical motion/action windows; strong persistence, motion-rich cases, hardware p95/p99, authorization boundary |
| `ME075-GDM` | `gated_delta_memory.py`: optional corrective 8×8 `alpha M + beta(v - alpha M k)⊗k` with separate retention/write gates and no-write query | Known published mathematics, analytical unit invariants only; **not** a new trained keyed-recall pass or 90%/Transformer-matching model | New explicit lawful diagnostic preregistration if synthetic fitting; genuine real long-context evidence for any memory-intelligence claim |

Historical genuine ME068 FIT-only robotic audit scores **257,971** autonomous original transitions, near-stationary diagnostic fraction **91.23%**, one-step persistence MAE **0.070195** versus FIT-only servo best tested causal lag1 MAE **0.119142**. These are previously measured original FIT-only reference statistics **not** a new independent held-out System1 evaluation, physical policy success or causal actuator alignment. They show why a naive one-step accuracy/MAE could appear high on static trajectories; select and report **movement-rich** original factual outcomes as well as all-case results.

**Hard constraints preserved:** Future trained models across all variants still must have **95k–105k** actual learned parameters (25k System1 would require an explicit prospective owner amendment); real original outcomes and observed-at causal fields; original Git checkpoint/optimizer/RNG and independent replay; same-task ability improvement and appropriate heldout confidence; no trained memory or System1 intelligence claim from synthetic/CI alone. `ME075` branch stacks on six-variant draft #358; master plan is separately updated, neither branch merged.

## New six-lane TAC-OSM capability research registry — 2026-10-23

The integrated six-regime architecture is now the prospective research portfolio; it is **not** evidence of six trained neural models. Common native recurrent and bounded residual mathematics can be shared while event, physical, temporal and policy targets remain separate. Read the binding [six-lane master plan section](MASTER_RESEARCH_PLAN.md) and [isolated six-variant branch](https://github.com/FPC-effortless/TAC-OSM/tree/research/six-variant-core-lanes-2026-10-23/src/tac_osm/variants).

| Lane ID | Research focus | Historical original evidence | Next real capability gate | Current tier |
| --- | --- | --- | --- | --- |
| `S1-001` | One-step native reactive risk/action advice | ME054/061 fast dynamics; ME074 exposed fast readout lesion, not policy or sub-ms evidence | Authentic reactive/action outcomes; guarded human decisions; measured hardware p95/p99 | `ENGINEERING_VERIFIED` adapter only |
| `PHY-001` | Action-conditioned actual physical dynamics and robust fallback | ME054/056, ME061–068; real transfer to persistence failed ME064/066 | Genuine heldout robot motion/contact, proper joint dynamics vs strongest physical prior | `DEV_CAPABILITY_OBSERVED` in preserved historical native; new lane adapter `ENGINEERING_VERIFIED` |
| `ENT-001` | Enterprise early branch, action/time path and entity forecast | ME069–074, five original ME073 trajectory models; exposed easy/hard negatives | Early rare transition + calibrated time/END-aware future on genuinely new lawful real cases | `DEV_CAPABILITY_OBSERVED`; original unmerged weights remain preserved |
| `TMP-001` | Bitemporal known-at/valid-at predictions and late corrections | Woyengi adapter proposals; no actual bitemporal correction-trained model | Real authorized audit revisions with as-of nonleakage and independently tested forecast correction | `ENGINEERING_VERIFIED` time-cut adapter only |
| `EDG-001` | INT8/frozen model on actual embedded hardware | ME055/056 CPU & operation profiles; no original embedded speed run | Exact quantized score fidelity and device memory/energy and p95/p99 latency, policy-compliant learned parameters | `DESIGN_ONLY` trained edge skill; export audit software |
| `CF-001` | Conditional observed action consequences and separate identifiable policy value | Historical CASM, ME049/061/073 and Veritas simulations; not real counterfactual benefit | Real actions/propensities, observed outcomes and randomized/identifiable policy utility with authorized pilot | `ENGINEERING_VERIFIED` scenario adapter; no causal capability |

**Parameter governance conflict:** user-suggested System1 20k–35k, Physical 50k–80k and Edge-Lite 10k–15k learned architectures are **not authorized** by the current future-live-model 95k–105k requirement. Preserve them only as potential future *design alternatives*. No new tiny model fit or hidden distillation until an explicit policy amendment; quantization by itself does not change learned parameter count. Architecture comparisons additionally require ≤1% parameter and equal authentic data/exposure/training/tuning compute parity; CPU wall time is not a matching constraint.

**ME074 interpretation:** exposed original associative readout lesion small/inconsistent, not universally useless or safe to delete from frozen ME073; fast zero improves earliest predictions and harms later ones, not direct proof of System1 execution benefits. Physical router must not assume servo is strongest, since real persistence won previous transfer. EOR in BPI is a published suffix proxy, **not** verified business completion. Temporal replay is not a new neural trained skill and Veritas simulated branches are diagnostics, not true causal evidence.

**Work order:** safe six-lane interface refactor (NO NEW FIT) → Enterprise/Pysical integrated full-native capability training in separate genuine datasets → a real reaction task for System1 → authorized temporal and hardware qualifications → verified action-rich Counterfactual model. Maintain the compulsory Capability Progress Scorecard and investor-value ledger independently for every trained study. Do not promote different task accuracies into an invented overall intelligence percentage.

## Compulsory evidence tier for every capability claim — 2026-10-23

All future CAP/ME/DEV research records must link to the canonical [Capability Progress and Intelligence Evidence Standard](CAPABILITY_PROGRESS_TRUTH_STANDARD.md) and name the fixed `tier`. Merely integrating, training and replaying a model is **not** a measured improvement. Development tasks and previously exposed TEST are not fresh confirmation. For a before/after gain, compare frozen complete systems on the exact same real cases, targets, prediction ages and horizons, retaining primary proper scores, original per-case outputs, statistical uncertainty and all worse outcomes. Independently confirmed improvement additionally needs a genuine unseen real cohort and predefined meaningful effect, no severe risk/calibration regression and independent replay. Mark any tradeoff honestly; never invent a scalar “intelligence increase”. Comparative architecture winners require separate matched parameter/data/exposure/compute evidence, while single-architecture capability development may proceed without default ablations.

Every new registry entry records `what_improved`, `what_regressed`, `what_is_unknown`, case counts, frozen old/new model identities, result tier, investor/customer meaning or `NOT_MEASURED`. Historical ME071–ME074 interpretation is specified in the canonical policy without modifying their actual checkpoints or score receipts.

## Compulsory runtime size-policy integration — 2026-10-23

Before the next capability fit, integrate the universal live-model guard from
[draft #354](https://github.com/FPC-effortless/TAC-OSM/pull/354) or its
[ME073 overlay #355](https://github.com/FPC-effortless/TAC-OSM/pull/355).
Both remain unmerged; an old study ID or declared manifest count cannot authorize
new fitting. Capability development and deferred-source studies require the same
95k–105k live trainable/deployed counts, pre-update runtime receipt and original
checkpoint verification. Fair comparisons retain the independent 1% and real
data/compute conditions; CPU is efficiency only. Preserve original models and
registrations through explicit immutable replay, never retroactive attestations.
This closes software-governance gaps; it adds no predictive capability claim.

## Completed native trajectory capability: ME073

**ME073 trained and independently replayed, 2026-10-23:** five original99,781-parameter models, six authentic FIT epochs, shared eight-step native imagined-state rollout and activity-conditional bounded incremental clocks. At six observations, conditional1/2/4/8-event accuracy81.85/68.83/62.11/53.44% over367/326/275/189 actual future records; one-step NLL.372472, proper interval score18.97h, nominal90% coverage85.99%. **Early goal failed:** two-observation accuracy40% in every seed and NLL1.416298; distant8-step probability scoring regresses relative to the historical ME072 recipe. Age16/h8 mean width636.94h remains broad. Different budgets/objectives/inference prevent an architecture-winner claim. Original checkpoints frozen100671c7 before scores190c2bc8; [independent replay39948180370](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39948180370) restored all five and recomputed8,495 case/query vectors,33,980 horizon marginals and67,960 representative weighted paths plus actual scores. These are repeated forecasts over925 exposed DEV cases. See [ME073 results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me073-autoregressive-native-trajectories-2026-10-23/research/mathematical-engineering/me073/RESULTS.md). Original TEST remains unscored by ME073; no external source/customer, long-memory, causal action or production qualification.

**Next integrated skill:** stronger early and distant probability forecasts with useful timing tails; original FIT-only changes and a newly registered readout before independent qualification. No additional attribution matrix is required. Reuse existing persistent carry rather than call it new unlimited memory. A selective linear scan would need a separate core fork; a lowest-variance selector and Student-t/hard72h cap are unproven fixes and were not substituted for measured predictive quality.

## Commercially guided capability-to-value program — investor-readiness companion (2026-10-23)

**Owner's actual objective:** TAC-OSM should get *more capable* as a predictive world model **while** building evidence for credible deep-tech investment and a specific independent customer use case. Do not turn this into an endless component-ablation, architecture-attribution, marketing, or fundraising-only project. The canonical, binding [investability and capability master plan](MASTER_RESEARCH_PLAN.md) defines evaluation and stage gates. The current code may be sufficient for **exploratory research pre-seed conversations**, but enterprise-scale product value, validated paying customers, commercial seed readiness and competitive IP rights are **NOT VERIFIED** from research repository artifacts.

| ID | Rank / capability-to-customer milestone | What would genuinely count as completion | Status / boundaries |
| --- | --- | --- | --- |
| CAP-022 | **P0 model:** useful multi-horizon future workflow-state forecasting | Integrate and train native next 1/2/4/8 observed-event forecasts and genuine timing from real online prefixes; record eligible cases at each query age, difficult event classes, proper NLL/Brier/coverage/calibration and end-to-end real checkpoint replay | **ME072 AND ME073 TRAINED/REPLAYED**: ME073 adds dependent weighted paths; six-prefix4-event accuracy62.11%, but early accuracy40% and distant probability losses remain failures. Historical ME072: five original ME072 models; six-event-prefix conditional accuracy82.02/68.47/59.64/54.18% at1/2/4/8 future events. Exposed DEV only; early branches and timing remain weak; no independent operational qualification |
| CAP-023 | **P0 model:** useful event-time and early-warning confidence | Actually measured sharp, trustworthy forecast intervals and decision-time warning lead, error/abstention costs when real outcomes/SLAs exist; calibrated proper scores, p50/p90/p99 widths and missed/false risks | **Open**: ME073 one-step coverage85.99%, proper interval score18.97h; age16/h8 width636.94h, still broad. Historical ME072 coherent clock trained, but one-step nominal90 coverage84.74%; age16/horizon8 mean width3,536h. ME071 postprocessing failures preserved; external warning cost unknown |
| CAP-024 | **P0 product:** minimal safe enterprise workflow prediction interface | Authorized case/event streaming, real-before-outcome forecasts, entity state updates, provenance and uncertainty in an advisory dashboard/API; access/privacy, no automatic authoritative writes | **Not verified in TAC repo**; Woyengi/other portfolio code can be proposed integration, not proof of deployment |
| CAP-025 | **P0 customer:** discover one buyer and validate a pilot | Document authentic buyer pain, price/budget and baseline, lawful case stream, explicit consent, ideally 2–3 design partners plus >=1 structured external pilot as a *suggested* discovery target | **UNKNOWN**, no claimed signed customer/pilot/revenue; an internal prototype does not satisfy |
| CAP-026 | **P0 commercial:** observed operator value | Precommitted prospectively observed time-to-warning, true false alerts/missed deadlines, actual costs and net workflow benefit against customer existing process; human approval | **Not established**; process-label accuracy is not cost saving or causal action value |
| CAP-027 | **P1 investment diligence:** ethical/licensed business and reproducible research | Technical claims/data/source licenses, IP and contributor assignments, planned legal entity/cap table, commercial usage rights, privacy/security, financial model and milestone use of capital verified with appropriate advisors | **Incomplete/unknown**, not evidenced merely by public code; never infer exclusive IP rights |
| CAP-028 | **P1 transferred intelligence:** generalization beyond single organization/domain | Untouched authentic external case/cohort / time/site shift, persistent state, missing events, verified feedback, real action-outcome if available, bounded error and meaningful end-user utility | **Not qualified**; long-horizon dataset access and causal-action evidence remain separate gates |

**Every new ME/DEV whole-model study should link to the right CAP ID and include a research-first Capability-to-Value Card** (model skill, original legal source/temporal availability, factual metrics, customer decision hypothesis, observed pilot/customer evidence or UNKNOWN, genuine checkpoint/replay, stage R0/R1/P0/C0/C1/I0). Allow exploratory foundational model research with customer impact marked unknown. Do not claim completion of product/customer stages from original FIT/DEV trainability alone. Research comparison parity and memory-specific rejection gates remain intact *for claims of superiority*; they are not prerequisites for integrated capability development.

**Commercial sequencing:** valid native model and real scored evidence → investor-legible full-model skill → permission-bound advisory product → authorized independent pilot → measured operational benefit → seed-quality customer diligence. These are evidence stages, not calendar, valuation, revenue or fundraising guarantees. Founder's investor deck may convey honest early research potential now; never fabricate commercial traction.

## Investor-readiness capability-to-value registry (owner direction 2026-10-23)

**Two north stars:** (A) develop an intelligent native world model that predicts and updates authentic states; (B) demonstrate customer-understandable, measured operational usefulness capable of supporting a credible deep-tech investment thesis. **Do not replace (A) with superficial dashboard/product work** or conflate synthetic demo scenarios with experimental evidence. Follow the full [investor-readiness roadmap](INVESTOR_READINESS_AND_CAPABILITY_ROADMAP.md) and [master plan](MASTER_RESEARCH_PLAN.md). No disclosed paying customers, real external pilot/ROI, completed investor round or independently verified commercial product has been established in this research record.

| ID | Capability-to-value objective | Required honest evidence / minimum deliverable | Status as of registry addition |
| --- | --- | --- | --- |
| **CAP-INV-001** | Integrated **multi-horizon world-state forecasting** for operators | Genuine frozen original native model scores for supported 1/2/4/8 event horizons, query-age coverage and proper evaluation with actual event-time available labels. [ME072 draft #351](https://github.com/FPC-effortless/TAC-OSM/pull/351) has five original trained models and independently replayed1/2/4/8-event scores; retain proper timing failures and conditional-case denominators. | **ME072/ME073 TRAINED/REPLAYED ON EXPOSED DEV; early/distant forecast failures and independent operator value unqualified** |
| **CAP-INV-002** | Coherent event/exception/delay confidence that supports decisions | Conditional empirical real-data logs and horizon delays, Brier/ECE/NLL, quantile pinball and proper interval score, p50/p90/p99 width, alarm lead time / false alert rate and abstention on all eligible real observational cases | **RESEARCH GAP**: ME071 sharp NLL but worsened calibration and broad interval tails |
| **CAP-INV-003** | Reliable current persistent entity state and verified new-outcome corrections | Factual read-only streaming state by verified ID and timestamp, no future information, source-eligible long histories and recovery/missingness probes; later score observed correction benefits, not code-path existence alone | **PARTIAL MECHANICS; capability benefit unproven** |
| **CAP-INV-004** | Evidence-backed **advisory** enterprise workflow demo/API | Predictions from actual checkpoint-backed model, real or licensed/anonymized demo stream, uncertainty and provenance, no automated approvals, no real-customer performance claims from synthetic UI fixture | **NOT VERIFIED PRODUCT READY** |
| **CAP-INV-005** | Independent validation of an actual operator problem | Real stakeholder discovery and documented frequency/cost of operational exception, lawful prospectively agreed pilot protocol and proposed 2–3 design partners / >=1 pilot; observed warning lead time, false alarm, net decision/time/cost savings; pricing validated rather than invented | **NOT VERIFIED; targets only** |
| **CAP-INV-006** | Investor diligence readiness | Entity/ownership and collaborator assignments, data rights and license strategy, security/privacy/infrastructure, source/weight/data room, team/company context, realistic financing use-of-funds and honest commercial traction | **NOT VERIFIED; legal/human authority required** |
| **CAP-INV-007** | Independent repeatable real-world capability and product value | New legally available untouched source from a different process/site/task, pre-registered learned ability and prospective pilot validation. Preserve original checkpoints/splits/negative results and distinguish robust evidence from research DEV | **NOT QUALIFIED** |

Every future completed CAP/ME/DEV cycle records **both ledgers**: `model_capability: actual metric / source / sample / frozen Git model / independent replay / exposure status`, and `commercial_relevance: exact buyer problem / candidate operational action / observed or conjectured lead time / actual pilot-evidence status / investor-readiness gap`. Treat unknown revenue, company formation, customer evidence, market size and claimed IP exclusivity as **UNKNOWN** rather than zero or success. Preserve the user's hard real-data and matched-architecture-comparison constraints; unrelated attribution studies are optional, not the research objective.

## Owner-ordered next capability work — 2026-10-23

The research owner is **more interested in model capabilities than ablation/attribution studies**. Prioritize integrated trained model progress over proving every TAC submodule novel or better than GRU. A comparator remains mandatory **if claiming architectural superiority**, but is not a prerequisite for a real single-architecture capability-development cycle. Existing held-out data, fair-budget gates for comparisons, authentic-only success claims, and full original Git checkpoint/replay rules are unchanged. ME070's 64-event scientific rejection applies only to a *memory-superiority claim*, not to whether TAC-OSM should continue learning useful skills.

| Capability ID | Whole-model capability to develop | Measured development evidence required (not an untrained proposal) |
| --- | --- | --- |
| CAP-017 | Native multi-horizon workflow forecasting from observed prefixes | Actual original real-case queries at eligible horizons; case coverage, next-event/next-several-event metrics and proper time scores; no hindsight midpoint; record rare branch and status-change errors |
| CAP-018 | Informative reliable waiting-time, event/stop uncertainty | Jointly trained zero/positive event and temporal distributions, proper log loss/Brier/ECE/pinball/interval score, coverage with median/p90/p99 width, real censoring and terminal-case boundary |
| CAP-019 | Persistent, updateable, missingness-robust factual belief | Real observational entity continuity, state carry/read/revision over genuinely supported histories, verified-outcome feedback, time gaps and missing observation behavior; execution mechanism alone is not a predictive gain |
| CAP-020 | Action-conditioned future state and event outcome representation | **Only when authentic records include actions available at prediction time and observed consequences**; multi-horizon action-conditioned factual outcomes, error/decision costs, safe uncertainty; no causal-treatment promise without identification |
| CAP-021 | Native hierarchical / iterative state correction and generalization | Integrated mathematical engineering only where errors warrant it, train/checkpoint full original variants with real outcomes, and qualify on independently sourced real domain/site/task before external-product claims |

**Execution:** extend frozen ME071 into an integrated next candidate and actually train it on legally available real data. Keep original checkpoints and source-replay provenance available inside TAC-OSM's repository. If novel long-history source acquisition remains blocked, improve genuinely supported enterprise forecasting and outcome/time calibration rather than staging more model-comparison diagnostics or manufacturing artificial long cases. **Update:** CAP-017 multi-horizon event forecasting and CAP-018 coherent time modeling now have real trained ME072 developmental evidence, with undercoverage/wide-tail failures; CAP-019–021 broader benefits remain unqualified. [ME072 results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me072-marked-multihorizon-world-model-2026-10-23/research/mathematical-engineering/me072/RESULTS.md); [capability-to-value card](https://github.com/FPC-effortless/TAC-OSM/blob/research/me072-marked-multihorizon-world-model-2026-10-23/research/mathematical-engineering/me072/CAPABILITY_TO_VALUE.md).

## Completed integrated capability development: ME072

**ME072 completed, 2026-10-23:** five original 123,377-parameter native models trained six authentic BPI2012 FIT epochs and independently replayed from Git. At six observed events, 555/925 exposed DEV REPORT cases reach the query; actual future-record denominators at horizons1/2/4/8 are367/326/275/189. Conditional activity accuracy is **82.02/68.47/59.64/54.18%**, NLL **.383476/.712665/1.067113/1.378439**. Nominal90% one-step clock coverage is **84.74%** (undercoverage); distant tails remain broad (age16/horizon8 mean width3,536h). Original weights were frozen before scoring; independent GitHub run39941010981 restored all five and recomputed all registered real-case outputs. See [ME072 results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me072-marked-multihorizon-world-model-2026-10-23/research/mathematical-engineering/me072/RESULTS.md) and [capability-to-value card](https://github.com/FPC-effortless/TAC-OSM/blob/research/me072-marked-multihorizon-world-model-2026-10-23/research/mathematical-engineering/me072/CAPABILITY_TO_VALUE.md). This demonstrates multi-horizon developmental forecasting, not an architecture winner, business completion, long-memory qualification or external operational value.

## Current integrated capability work: ME071

| ID | Capability | Delivered evidence | Remaining objective |
| --- | --- | --- | --- |
| CAP-016 | Persistent, probability-aware native enterprise event model | [ME071 PR #350](https://github.com/FPC-effortless/TAC-OSM/pull/350): five original 93,770-parameter native fits; six real FIT epochs/715,134 targets each. Full-case detached fast/slow/associative carry, history log pooling, smoothing, ordered waiting quantiles, disjoint temperature/CQR, independent entity streams. Fixed-six-event DEV REPORT accuracy81.58%, fitted NLL .401741 vs raw .420459; Brier/ECE worsen and calibrated mean intervals51.69h. Original Git checkpoints, forecast arrays and independent five-model real-score replay preserved; later preprocessing mismatch preserved and loader repaired to restore exact frozen values. | Improve complete model's probability quality and interval usefulness, then qualify on an untouched authentic source. No new baseline matrix, attribution fit, fresh TEST, 256-memory, action-policy or commercial claim. State carry is not longer gradient credit. Preserve ME070's long-memory rejection rule. |

The owner's current emphasis is integrated architectural power before detailed
mechanism attribution. This single-architecture development lane cannot be
used to bypass the matched-resource contract for future architecture winners.
Synthetic fixtures/interleaving are supporting engineering checks only.

| ID | Idea | Origin | Status | Evidence or next test |
|---|---|---|---|---|
| TAC-001 | Direct associative retrieval for physical forecast | ME-054 review | Proposed | Ablate on authentic robot trajectories |
| TAC-002 | Multiscale learned retention | MTSK transfer | Proposed | 128-state matched comparison |
| TAC-003 | Prediction-error responsibility writes | MTSK transfer | Proposed | Key collisions and long-horizon error |
| TAC-004 | Entity-local memory updates | MTSK transfer | Data-dependent | Requires reliable persistent IDs |
| TAC-005 | Bounded norm projection versus tanh compression | ME-054 mathematics | Proposed | Memory stability and forecast MAE |
| TAC-006 | Concept routing diversity and specialization | ME-056 | Diagnostic proposed | Effective rank, entropy, error attribution |
| TAC-007 | Cross-context concept attention | LLM routing analogy | Proposed | Equal-budget routing ablation |
| TAC-008 | Adaptive recurrent decoder depth | LLM latent scratchpad / CED | Proposed | Per-pass held-out forecast and cost |
| TAC-009 | Hierarchical latent-first dynamics | JEPA/world models | Proposed | Action-conditioned multihorizon real forecasts |
| TAC-010 | Elapsed-time liquid-inspired recurrence | LTC/CfC | Conditional proposal | Irregular timing analysis first |
| TAC-011 | Memory/working-state role separation | KV-cache analogy | Proposed | Retrieval and interference tests |
| TAC-012 | Counterfactual decision-preserving retention | TAC-OSM research | Research program | Logged actions, costs, identification limits |
| TAC-013 | Bounded verification and correction | ME-054; DEV-035 | Existing principle | Compare real correction errors |
| TAC-014 | Hardware-efficient recurrent execution | LLM/SSM engineering | Proposed | FLOPs and wall time, secondary to capability |
| TAC-015 | Structured entity-factorized state | DEV-017/020 | Historical baseline evidence | Verify protocols and replicate real-data relevance |
| TAC-016 | Fully learned neural action/stop control | DEV-058/061 | Synthetic evidence only | Real decision-cost validation |
| TAC-017 | Native and Concept-CED real-world benchmarking | ME-054/056 | Developmental results | Persistence beats both reported MAEs |
| TAC-018 | Retrieval versus retained-state diagnostic | Berkeley stale binding analogy | Proposed | Probe memory contents vs actual readouts |
| TAC-019 | Unproductive refinement loop detection | RADAR analogy | Proposed | Pass-wise prediction gain |
| TAC-020 | Long-horizon execution reliability | NVIDIA long transduction analogy | Proposed | Real long-history error and action robustness |
| CROSS-001 | CDL/Hebbian-style relevance learning | Cross-project idea | Unverified | Formalize update and benchmark |
| CROSS-002 | CASM decision mechanism outside ML | Cross-project idea | Unverified | Define domain-specific prototypes |
| VER-001 | Absorb MiMo RL-environment design into Veritas | Veritas research | Proposal | Verify repo design and dependencies |
| CROSS-003 | Unified reusable GitHub skills | Cross-repo workflow | Proposal | Repo-wide compatibility audit |

## 2026-10-22 portfolio-aligned capability work items

Canonical sequence and acceptance gates: [MASTER_RESEARCH_PLAN.md](MASTER_RESEARCH_PLAN.md#ordered-integrated-capability-program--supersedes-the-previous-component-first-sequence). These items supersede any implicit demand to run TAC-001 through TAC-020 sequentially. Existing TAC research ideas remain optional and **only** activate for a diagnosed integrated capability blocker.

| ID | Rank / target | Candidate external integration or data lane | State | Falsifiable proof required / do not claim |
| --- | --- | --- | --- | --- |
| CAP-001 | P0 / real persistent state + factual action-conditioned forecasting | Authentic independent robot time-series and ME059 registered comparison | Partially implemented, not yet qualified | Sealed real future-state performance exceeds persistence/servo reference and a fully matched GRU; correct pre-feedback state, parity receipts |
| CAP-002 | P0 / uncertainty, inspect, abstain and cost-sensitive intervention | Real supported action/outcome/cost logs, possibly approved Woyengi operational pilot | Not established in real operation | Lower net observed cost than best fixed/no-action policy, justified causal support, calibrated uncertainty, safe escalation |
| CAP-003 | P0/P1 / decision-relevant long-term entity memory | Verified identity-linked temporal history (not MovieLens coordinate recursion) | Partial synthetic / mixed real negatives | Future heldout prediction or decision benefit robust across unseen IDs, long histories and seeds |
| CAP-004 | P0 / verified feedback-correction loop | Woyengi authorized timestamped event/receipt read; Veritas engineering replay | Mechanism partially implemented; real benefit not established | Measured later real error/cost reduction, strict no-leakage and preservation of independently verified facts |
| CAP-005 | P1 / hierarchical abstractions and extra thinking depth | Concept-CED, TAC-Transformer, TAC-Prime, ATLAS-CASSA, Neural-Field-Dynamics | Research ideas / candidate modules | Adaptive computation or learned hierarchy causes statistically supported downstream real gains under matched budgets; no System-2 assertion by recurrence alone |
| CAP-006 | P1 / transfer and adaptation | Nsat regional/site forecasting; other legally sourced operations | Not established | Independent unseen-site performance advantage and forgetting/calibration controls; distinguish frozen transfer from online learning |
| CAP-007 | P2 / native multimodal temporal grounding | Actual time-aligned observation streams from authorized systems | Not established | Additional modality's heldout predictive/decision lift with matched sources and rights; external encoders are separately credited |
| CAP-008 | P2 / genuinely selective compute at scale | Native CDL/routing/CASM research; real executable service traces | Mechanisms exist; C5 not established | End-to-end retrieval/index, routing, execution and verification cost scales with relevant subset while meeting utility gates |
| CAP-009 | Product / governed autonomous operations | Woyengi + KoreFrame + SafeEdge/Aegis + Veritas, after real data qualification | Integration proposal only | Traceable prediction→authorization→actual action→verified outcome; independent measurable customer benefit and strict permission boundaries |
| CAP-010 | Vertical options / logistics, workforce, Nigeria environment | Shipping, Scrappy/HCSE, Nsat, BusinessData | Proposed evaluation pathways | Confirm legally usable genuine entities, histories and targets first; prediction-only datasets cannot validate intervention |
| CAP-015 | P0 / native categorical enterprise state and event-time prediction | [ME069 draft PR #348](https://github.com/FPC-effortless/TAC-OSM/pull/348), authentic BPI2012 loan-process log; five paired seeds, original Git-owned checkpoints and complete clean-checkout real score replay | **TRAINED/REAL-SCORED; local REPO_REPLAY PASS; cross-CPU numerical exception retained; joint superiority gate FAIL** | Native TEST next-event log loss .122626 vs empirical .348946 and matched GRU .119182; accuracy 94.77%, TAC 1/5 seed wins vs GRU. Waiting-time MAE 6.3561h vs empirical 6.4953h; wide 230.12h nominal90 intervals. One GitHub replay failed fixed timing tolerance; a subsequent run passed unchanged. Improve integrated history/timing model on FIT/DEV, qualify on a new authentic cohort; no actual SLA, intervention, lifetime memory or commercial claim. |
| CAP-011 | P0 / competitive action-conditioned physical world model | ME060/ME061 native trajectory/uncertainty models, PRs #339/#340 | Implemented and trained; source improvement, both final gates failed | ME061 source .603894 vs persistence .702952; new task .493206 vs GRU .509535 but persistence .455992. Next: learn residual utility under shift while preserving strong fallback forecasts; qualify uncertainty and exceed strongest prior on a newly registered task. CPU reported independently |
| CAP-012 | P0 / selective reliable physical forecast and governed Woyengi handoff | ME062 draft PR #341, original FIT preflight/trained checkpoint and frozen real transfer receipts | **TRAINED/REAL-EVALUATED; both superiority gates FAILED**; original 10 checkpoints in Git; independent model-load and real score replay PASS; science superiority FAIL | Source DEV TAC .609373 vs GRU .605164 vs persistence .702952; new task TAC 1.256390 vs GRU 1.223530, servo 1.031359, persistence 1.295032; 0/5 TAC seed wins in each; transfer nominal90 episode coverage 30%. No decision/online learning/independent-site claim. Next newly untouched task and task-shift robust strongest-prior selection, calibrated risk, plus matched no-memory/long-history tests |
| CAP-013 | P0 / causal task-shift regime arbitration with native recurrent memory | ME063 draft PR #342; original FIT 208 episodes, 10 repo checkpoints, 80 new real clothes-folding transfer | TRAINED/REAL-EVALUATED; **both superiority gates FAILED**; independent original model checkpoint audit PASS and frozen real score SHA-256 bitwise replay PASS | Source TAC .572882 vs GRU .573576 vs persistence .702952, source TAC 4/5 seed wins but block CI spans 0. Transfer TAC .828290 vs GRU .829905 vs stronger servo .793413, TAC 3/5 wins but block CI spans 0. Router raised servo weight from 40.9% to 55.1% on transfer but failed strongest-prior test; source-cal 90% interval coverage 72.25% target. No unique memory, independent-site, policy or causal claims. Next isolate genuine memory retention and better shifts with new untouched cohorts. |
| CAP-014 | P0 / multi-task real physical world-model generalization and Woyengi uncalibrated forecast guard | ME064 draft PR #343, real publisher five-task FIT 160 original episodes / source CAL 40 / untouched 80-episode spray-nozzle transfer, ten original matched Git checkpoints; Woyengi guard PR #53 | **TRAINED AND REAL-SCORED; strongest-persistence gate FAILED**. Final parameter/data/supported-op FLOP parity PASS, ten Git checkpoint SHA/load audit PASS; first independent score JSON byte match FAIL on tiny CPU float variance, strict numerical-tolerance replay pending. Woyengi CI PASS but runtime wiring not yet complete | Untouched target physical MAE: TAC .135153, GRU .148456, servo .918424, **persistence .012371** (10.9x better than TAC). TAC 2/5 paired seed wins; episode-cluster ensemble TAC-GRU CI [-.01809,-.00862] but persistence CI strongly positive. Nominal90 source-CAL target episode coverage 100% with overly wide interval mean 17.07; not qualified. Requires independent narrow-float replay, real change/intervention detection, matched memory-free/long-history controls and true policy value before any production action. |

**Audit caution:** ME-054 original recovered DEVELOPMENTAL physical MAE (TAC ~0.07881; GRU ~0.09232) loses to persistence ~0.06695 and is neither sealed independent confirmation nor fully true-FLOP-certified architecture superiority. Concept-CED ME-056 PR #332 originally failed prefit FLOP parity; its score claims require exact artifact reconciliation. Distinct anchored ME-056 PR #334, fused ME-057 PR #338 and independent-task ME-059 PR #337 must never be conflated. PR #338 is now scored: TAC 0.663284, GRU 0.658526 and persistence 0.672681 TEST MAE, with the registered TAC superiority gate failed. PR #337 still requires its own exact artifact result. No synthetic task, code-level test, audit receipt or README can promote a real-world or commercial ability.

### Hard stop and handoff to execution

For every CAP item, preserve external source licences, real identifiers and time cuts, source hashes, entity-linked original data, missing/censored outcomes, scientific preregistration and frozen tests; enforce [REAL-WORLD-EVIDENCE-RULES](REAL-WORLD-EVIDENCE-RULES.md) and [CAPABILITY-FIRST-RESEARCH-RESET](CAPABILITY-FIRST-RESEARCH-RESET-2026-10-22.md). Match **trainable parameters, real data/exposure, training and tuning compute budgets, and evaluation** before claiming architecture superiority. Report CPU speed independently (no equal CPU-time requirement). Record PASS/FAIL/UNKNOWN, negative results and independent artifact links. Data/project code availability is not proof of outcome coverage. This registry neither starts training nor authorizes writes outside TAC-OSM.

## Historical experiment index (verify source evidence before interpreting)
| Experiment | Reported result | Limitations |
|---|---|---|
| DEV-016 | Joint 73.64%, frozen writer 59.32% | Predictive reliability gate failed |
| DEV-017 | Structured 91.84%, TAC 57.90% future prediction | Experimental scope must be retained |
| DEV-020 | Structured probes 100%, TAC ~50-51% | Probe failure does not identify unique cause |
| DEV-043 | Pairwise ranking 100%, hard accuracy 50% | Calibration/representation concern |
| DEV-058 | 576/576 default, 560/576 repeat | Synthetic, no real-world claim |
| DEV-061/R1 | Reported 100% primary completion | Synthetic; persistence wins real forecasting |
| ME-054 | MAE 0.07881 vs GRU 0.09232 | Persistence 0.06695; parity needs audit |
| ME-056 | MAE 0.08037 vs GRU 0.09082 | Persistence 0.06695; parity needs audit |

## Mandatory artifact availability for any implemented CAP/TAC study

A trained research idea is **not completed** until original TAC/GRU/other model checkpoints for each seed and each recoverable failed run, plus all lawful data/metadata, config, code, dependencies, outcome predictions, score scripts, hashes and clean-checkout replay receipts are **available from this repository**. See [repository checkpoint/replay rule](REPOSITORY-REPLAY-AND-CHECKPOINT-RULE.md). Temporary GitHub Actions links or locally ignored `results/` do not count; external-provider runs have exactly the same requirement. Track `REPO_REPLAY=PASS/FAIL/BLOCKED` separately from real-world scientific and parity gates.

## Record schema for future ideas
Each idea should include: ID; title; problem; proposed mathematics; origin; relevant repositories; relation to existing ideas; hypotheses and failure modes; status (proposed/diagnostic/implemented/trained/verified/rejected); commit/PR; datasets and splits; parameter count; training budget; baselines; seeds; metrics; confidence and limitations; next experiment.

No synthetic result may be labeled real-world success. Never erase negative findings. Source code, checkpoints, reports and runs should be linked rather than duplicated without provenance.


## 2026-10-23 real robotic world-model diagnostic evidence: ME067 / ME068

These are **real-world mechanistic and source-data diagnostics only**, not new independently heldout capability successes. Both are implemented and independently GitHub Actions verified, with zero new model fits; successor evaluation must preserve a sealed physical holdout.

| ID | Evidence | Honest research status | Next decisive test |
| --- | --- | --- | --- |
| TAC-021 | [ME067 native frozen readout lesions](https://github.com/FPC-effortless/TAC-OSM/pull/346): 80 already exposed original charger episodes; full native physical MAE 0.41059; zero fast 0.60016; zero associative 0.44286; zero velocity+innovation reliability 0.29521. 20 existing original checkpoint bytes intact. | **Exposed-only mechanistic sensitivity:** native fast/associative state influences prediction, while motion reliability conditioning transfers poorly. Posthoc out-of-training-distribution readout lesions are not independent model improvement or proof of memory necessity. | New original source FIT-only, equal-resource trained causal motion/no-motion and persistence-preserving native TAC plus equally informed GRU, genuine NEW heldout original robot task. |
| TAC-022 | [ME068 real original robot action-time audit](https://github.com/FPC-effortless/TAC-OSM/pull/347): 160 original FIT episodes, 257,971 genuine autonomous transitions, about 91.23% with posthoc mean per-joint movement <0.025; one-step last-state MAE 0.07020 versus original FIT-identified servo action at t 0.12046 and t−1 0.11914. | **In-FIT source measurement, not independently heldout forecasting.** Every tested servo lag loses to persistence; one-frame past action modestly scores better across all five original FIT tasks. It does NOT establish a causal actuator delay or authorize an online future logged action feature. | Verify physical command/ack semantics; model conditional motion from actually available causal action and verified outcomes; score false-positive motion, mean physical MAE and interval sharpness on new real tasks. |
| CAP-001b | ME064/066 independent original robot transfer failed strongest persistence baseline; ME067/068 indicate potentially useful native state but fragile reliability/movement prediction. | No physical intervention-policy utility or independently calibrated safe production autonomy. Woyengi PR53 still advisory only. | Shared physical conditional dynamics head and training objective for native TAC and strong GRU, same authentic data/parameter/optimizer budget and <1% supported-operator profiled FLOPs, full original Git weight/optimizer/RNG checkpoint and independent numeric score replay, untouched physical target. |

Do not re-run historical control proofs or retroactively promote exposed-task lesions. The integrated successor must **beat the simplest strong physical predictor**, not merely match another recurrent neural network.

## ME070 new evidence and falsifier (2026-10-23)

| ID | Status | Non-negotiable next evidence |
| --- | --- | --- |
| CAP-011 — authentic 64/256-event memory over GRU | **Source blocked / no fits**: original BPI2012 TEST longest case 109 events; BPIC2018 publisher source timed out in GitHub Actions; all 256-event cohort counts UNKNOWN | Qualify legally accessible real XES with >=50 FIT and >=25 independent TEST cases with 256 observed events before original next label; register new matched `FAIR_COMPARISON.json`; kill broad memory claim if at 64 events TAC does not beat matched GRU by >=1% NLL, clustered upper CI <0, >=4/5 seeds, and learned associative memory does not add value |
| CAP-012 — original enterprise score/error overlap | **Real retrospective diagnostic measured; not new TEST**: 11–36 paired correct/wrong disagreements / 2618 cases; McNemar p .227–1.000, no accuracy edge | Preserve [original scored diagnostic](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/EXPOSED_ME069_DIAGNOSTICS.md); independent untouched score needed for promotion |
| CAP-013 — calibrated useful waiting time | **Real retrospective tail failure**: native median 90% width 4.8–10.6 sec but p99 89–192 days and maximum 2.4–3.8 years | On new qualified source record proper interval score, median/p90/p99 width, pinball loss and conditional empirical-quantile baseline; demand sharpness AND calibrated coverage |
| CAP-014 — lifecycle difficulty and true process history | **Real source audited**: TEST 38.43% same base activity, 13.06% same-activity START→COMPLETE; TEST longest original case 109 events | Use real decision-time prefixes, lifecycle-collapsed separate task, higher-order n-gram and process-specific specialist, rare transitions; no hindsight-midpoint promotion |

See [ME070 draft PR #349](https://github.com/FPC-effortless/TAC-OSM/pull/349), [source access blocker](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/SOURCE_ACCESS_BLOCKER.md), and [master-plan corrected evidence](MASTER_RESEARCH_PLAN.md). Zero new ME070 trained models or new memory advantage is claimed.


| CAP-015 — observed-prior attribution control, actual original real fits | **COMPLETED (retrospective, not scientific promotion)**: original FIT-only 1/2/3/5-event count-reference state and original exposed score file are Git-owned, independently clean-clone replayed; 5-event NLL 0.159743 vs first-order 0.348946, TAC 0.122626, GRU 0.119182 on the historically exposed ME069 TEST. Demonstrates much of improvement over first-order is generic history context, not TAC memory. | [ME070 executed ngram results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/PRAGMATIC_NGRAM_RESULTS.md); next unique-memory claim requires authentic new 64/256-event supported source, matched associative ablation, strong GRU and preregistered untouched TEST. |

**Publisher source access updated:** BPIC2018 original record Figshare article 12688355, file 24025820, size 158270992, publisher MD5 `66e54ddaa6b50d81a0166766d6f7a5d4`; multiple official delivery endpoints HTTP 504. This is metadata verification only, NOT model input source verification or actual 256-history counts. No new ME070 neural fits, no capability promotion. Historical BPI2012 after-64 histories: FIT174/DEV15/exposed TEST61; after-128: FIT4, DEV0, TEST0. Original source receipt preserved in PR #349.


| CAP-024 — factual all-case END-aware future scoring and state attribution | **ME074 EXECUTED, EXPOSED DEV ONLY; no new neural fit.** FIT-anchored original 1st/3rd/5th order non-neural Markov/survival controls scored at query ages2,6,16 and horizons1/2/4/8 with all eligible-case categorical-or-END NLL/Brier. At age6 on 555 original real cases, native five-seed mean h1/h8 NLL **0.24668 / 0.88572** versus fifth-order **0.56627 / 1.06720**, but at age2 on 925 real cases fifth-order h1/h8 **1.14613 / 1.40970** beats TAC **1.41638 / 1.67668**; 3gram h1/h8 **1.14613 / 1.38203** also beats TAC. This is an activity-only unmatched practical baseline, not a neural architecture winner. | [ME074 original real baseline](https://github.com/FPC-effortless/TAC-OSM/blob/research/me074-real-baseline-memory-diagnostics-2026-10-23/research/replay-bundles/ME074/REAL_DEV_END_AWARE_BASELINE.json), [draft PR #356](https://github.com/FPC-effortless/TAC-OSM/pull/356). Require equal 100k GRU autoregressive and direct multi-horizon heads with trained original checkpoints and genuinely unexposed real evaluation. |
| CAP-025 — native fast/slow/associative contribution on genuine workflow histories | **ME074 5 frozen source-trained ME073 seeds, posthoc original exposed cases; zero fitted models.** One-step joint NLL query age2/6/16 full **1.41638/0.24668/0.23178**; fast-zero **1.19197/1.16326/0.58114** (helps early but harms mature cases, all five trained seeds in each direction); association-zero **1.42889/0.24727/0.23817** (small mixed effect). Full original Git-owned probabilities replayed exactly before each lesion. **No causal physical action, long-memory ability, learned associative necessity or superior recurrent capacity proven by OOD state ablation.** | [Original real memory diagnosis](https://github.com/FPC-effortless/TAC-OSM/blob/research/me074-real-baseline-memory-diagnostics-2026-10-23/research/replay-bundles/ME074/EXPOSED_MEMORY_LESIONS.json). Next use prospective trained parameter/compute-matched memory ablations and independent 64+ and 256+ source entity records, not synthetic-success promotion. |


| CAP-MOTION-001 — true real movement discrimination separate from mean-state error | **ME077A EXECUTED NO-FIT ON ACTUAL EXPOSED ORIGINAL ROBOT DATA:** original five ME077 frozen 102,503-param models, same 960 real next states / 40 authentic recording episodes (84 actual moving). Source-case exact SHA and all original model prediction bytes preserved. Auxiliary motion risk had ROC-AUC **0.64412**, episode-bootstrap 95% CI **[0.59382,0.69674]**, AP **0.17056** vs real source prevalence **0.0875**, Brier **0.07870** versus hindsight REPORT prevalence-oracle **0.07984** (NOT deployable). Mean p_moving **0.1084**, p_stationary **0.0982**; threshold0.25 **missed all 84 actual movements**. Top 10% highest scores captured20/84 genuine movements, 401 false alarms at threshold0.10. Mixture 14D component probability is NOT the global event head. No trained new model, no untouched external test or policy value; ME077 originally improved exposed all-state MAE to0.066476 but failed its own >=5% gate and this auxiliary head is not a reliable anticipatory detector. | [Original fully paired real data score](https://github.com/FPC-effortless/TAC-OSM/blob/research/me077a-real-motion-risk-latest-base-2026-10-23/research/replay-bundles/ME077A/ORIGINAL_REAL_MOTION_RANKING.json), [no-fit reproducible PR #367](https://github.com/FPC-effortless/TAC-OSM/pull/367). Next register 100k integrated change-risk and conditional residual **jointly evaluated** on authentic independent robot data; keep real motion false alarms and missed movements separate from physical MAE and do not force DIAGKEY004 memory into System1 without same-task empirical gain. |

| CAP-GRAPH-082 — native parallel local dynamics plus quiet precision | ME082/083/084:15authentic guarded fits completed, original Git checkpoints and paired scores preserved; all full gatesFAIL/MIXED_TRADEOFF. Mean errors improve but quiet/H4/clusterCI fail. | Preserve complete attempt ledger in draftPR377. No extra fits under those registrations. |
| CAP-CLOCK-VALUE-085 — time-aware native state with precision-preserving value retrieval | Separately preregistered99731-param integrated model; first5-seed4096-update recipe training, second conditional recipe frozen. Exported pure graph2579nodes, all20parameter tensors connected; learned gain NOT YET SCORED. | Keep original full gate, explicit exposedDEV/clock feature disclosure, all original weights beforeREPORT and independent strict replay. Do not present engineering as intelligence, scan-parallel recurrence or verified product value. |

| CAP-CLOCK-VALUE-085 outcome | Five real fits completed; H1/H4/H16MAE0.053390/0.157590/0.681905, stronger mid/long forecasts but quiet0.022057 and episodeCI fail. Full gateFAIL; independent original27600forecast replay and compiled-profile restorationPASS. | Preserve originals and exposedDEV status; ME086's separately frozen second recipe runs unchanged. |
| CAP-BOUNDED-VALUE-087 — coherent median learning and robust quiet value retrieval | Separately frozen five-seed complete-model training started:99731params, FIT-noise-bounded quiet borrowing plus correct implicit CDF gradient of the actual mixture median. No REPORT score yet. Existing gates unchanged; all6attempts visible. | Whole-model improvement first, no attribution prerequisite, no future-conditioned inference, no original-checkpoint rewrite or protected cohort opening. |

| CAP-BOUNDED-VALUE-087 outcome | Five guarded originals completed, H1/H4/H16MAE0.050888/0.148123/0.682106; movingH1 0.363536. Original95%pairedCIcleared but quiet0.020909 and stronger repeated-attemptCI failed. FullFAIL; independent original and compiled restorationPASS. | No extra fits under ME087. Preserve all original fields and cohort exposure; no memory attribution required to continue native capability development. |
| CAP-PRECISION-088 — coordinated precision/movement fine tuning | Fixed ALL5ME087 warm starts, all99731weights trainable;2048new+4096priorupdates per lineage, two independent seed processes. Predictive noise units, near-step residual trust, factual coordinate routing and explicit quiet precision objective; no REPORT score yet. | Frozenbbf30a57; all prior gates unchanged, all7attempts visible, all5new originals before scoring. Explicitly cost-sensitive development, not automatic probability calibration or independent general intelligence. |

### Native visual predictive-state capability — source qualification, 2026-10-24

ME088–ME090 full joint gates FAIL, originals and all attempts preserved. Quiet
precision and moving accuracy remain in conflict. Qualify synchronized current/past
RGB observations from the existing licensed FIT/REPORT episodes before building
an integrated <=105k native visual world model. Preserve native fast/slow/delta
memory, causal cutoff, fixed physical gate and original replay. No future images,
new sealed episodes, causal policy, architecture-winner or customer-value claims.
This is a capability direction, not evidence of a trained visual ability.

### ME091 — integrated observed vision and native predictive location (training)

One whole-model recipe,104225active weights; shared overhead-image CNN, native
fast/slow/association and parallel horizon readout. Shared bounded translation
moves both conditional mixture means together without new state rollout. All
licensed same-role input pixels are Git-owned; source/frame SHA verification
passed for160episodes. All29parameter tensors active in a real-FIT graph. Five
fixed-parent fits running, no REPORT result or gate pass yet. Existing quiet/
moving/multi-horizon gate and protected sources unchanged; no model restarts,
matched winner, policy control, site transfer or customer-value claim.

### ME092 — train the integrated model against compatible task gradients

ME091 full gateFAIL, all five originals/replays retained. New104225parameter
whole-model recipe uses fixed paired native/visual initializers, shared proper
forecast terms, factual quietH1/movingH1/long-horizon loss partition and batched
VJP/PCGrad. No extra task models/forward passes, no model reset or winner claim.
One registered2048update recipe per five seeds, no REPORT yet. All50prior trained
versions remain visible; same protected sources and joint gate unchanged. This
is an optimization/capability attempt, not proof of a gradient-conflict cause.

### ME093 — integrated local-noise and transition distribution (2026-10-24)

Motivation: ME092 improved movement/long forecasts but quiet accuracy and paired
intervals failed. Frozen output trace suggests a narrowhold/broadtransition mixture
may conflate small observation noise with actual movement; descriptive hypothesis,
not established cause. Build one three-Laplace native visual predictive distribution
with causal robust local center, ordered scales, exact proper scoring and shared
quantile inversion.96,595active parameters, ALL5indexedME092parents,3,072newsteps,
whole-model training, unchanged gate and source cutoffs. Registered51120067;
training underway, no skill-success claim. Attribution secondary; no extra fits
under exhaustedME092. Commercial/customer evidence remains NOT_MEASURED.

### ME094 — explicit median and coherent multimodal shape (2026-10-24)

ME093's extra noise regime did not meet quietH1/stricterCI gates. New whole model
separates a learned predictive median from shape mixing/scales, translates the full
multimodal CDF with correct implicit derivative, and adds causal eight-value time/
content relevance.101,173parameters,4,096newupdates, ALL5fixedME093parents,
unchangedgate; registered c87dc63c, fits underway, no result yet. Standard quantile/
mixture/retrieval mathematics reused, no new theorem, extra ablation campaign or
architectural-superiority claim. Genuine customer/decision value NOT_MEASURED.

### ME094 query-selective computation reuse — ENGINEERING only

Implemented and first-original real-FIT checked: request point/risk/state from the
same complete median-anchored model, prune unused probability roots by compilation.
FixedB2incremental0.74ms vs3.43msfulloutput, sharedCPU; no new learned arm or added
skill/superiority claim. Full probabilistic query remains available and model still
training. Exact original score replay and joint predictive gate remain compulsory.

### ME095 — bounded selective state and known context

Whole model registered00295abc,104,689params,5fixedME094parents,2,048newupdates.
Adds input-selective affine level/velocity memory with a verified parallel prefix
scan alongside unchanged nonlinear native fast/slow/M, known operation priors and
current clock phases. Real-source causality/gradient/stream/cache checks pass;
training underway, no predictive result. Same fixed numerical gate, stricterfamily
interval, no protected data or future commands. Extra known context/cost disclosed;
not a controlled architecture win or proof that a particular memory block helps.


### SHARED-EVIDENCE-ME097 — trained contextual retrieval candidate (2026-10-25)

Status: EXPOSED_DEV_GATE_PASS; INDEPENDENT_QUALIFICATION_OPEN. TACPR391/PNDSPR9.
One95,874parameter core shared by six family configurations; one physical lane
trained. Separate context-conditioned query + direct retrieved correction;
input-only affine schedule preserves scan/stream equivalence. Five original
replication seeds clear5%whole/late persistence criteria and moving no-regression:
meanMSE0.000301391vs0.000467679, moving0.001496158vs0.002507034. Original archive
freeze1901d395 before scoresaaa58ad5; independent replay40163334508PASS.
No component attribution, compute-matched architecture winner, independent-site,
long-memory, control or customer utility claim. All10ME096/097 originals remain;
ME096 archive-order exceptionBLOCKED, ME097 unchanged new-seed replication.
ME0954completed+1interrupted/noREPORT retainedb30b5aaf; no hidden retries.
Next capability: coherent multi-step outcomes/uncertainty and independently
qualified source-supported delayed/identity-aware forecasting on this shared
engine. Do not simply lengthen retention again. Full neural Lean refinement OPEN.


### ME098 / ME099 — composed trajectory forecasts and latent uncertainty regimes (2026-10-25)

Implemented/trained: one shared TAC conditioning graph, fixed-coefficient future
mean semigroup, response-Gram joint scale, shared uncertainty regime and proper
interval objective.103,322/103,380parameters;10original real-data fits preserved.
TACPR392/393, PNDS10/11. ME099 MSE39.36/40.67/41.95%below persistence at1/4/16frames
on13exposedDEVepisodes; interval scores21–25%better thanME098, widths45–53%wider.
Four/fiveseeds fullpass; seed99001H4coverage84.192562<85 =>FULLGATEFAIL.
No morefits/changedgates in this correction protocol. Independent originalreplay
and actual-source values/gradientsPASS. FullNNLean/IEEE, independent cohorts,
otherdomainheads, usefullongmemory and action/customerutility remainOPEN.

Proposed NEXT-SHARED-NOISE: nonnegative state-conditioned future-noise envelope
within the response Gram; test middle-horizon reliability on new real data.
Proposed NEXT-SHARED-CLOCK: actual elapsed-time/entity corrections and typed
workflow time/lifecycle heads, no false completion or missingness claims.
Proposed NEXT-SHARED-QUERY: reusable coherent path sampling, square-root density
kernels and measured chunk/hardware processing. No new implemented skill claimed.


### ME100 / NEXT-SHARED-SELECTIVE — implemented sensor admission pilot

[TAC draft PR395](https://github.com/FPC-effortless/TAC-OSM/pull/395) and
[PNDS draft PR12](https://github.com/FPC-effortless/pnds-lean/pull/12). Five
97,235-parameter original models trained on UCI501 FIT 2013–2014 and were saved
before 2015 REPORT scoring. Native Temporal masked source sum/mass traces,
affine associative memory and six-pollutant coherent forecasts are implemented.
Long-context gap PM2.5 MAE is 38.986917 versus 46.073106 with recent history.
All five directional history gates pass, but the clustered interval includes zero.
Full gate FAIL: 0/5 pass, including gap coverage 67.69% below 85%. General PM2.5
MAE is 35.689 versus LOCF 34.516. The 967 local tests and original clean/GitHub
replay pass. Protected 2016–2017 outcomes remain unscored. No seventh family,
action intelligence, broad gain or Phase2 qualification is claimed.

### NEXT-SHARED-BELIEF — proposed uncertainty over missing world state

ME100 source retention improves gap point forecasts but fails uncertainty on the
cases needing retained observations. Learn a coherent latent belief over unobserved
current state and future change, conditioned on causal source age, evidence mass
and innovations. Consider bounded second-moment traces and a jointly trained
latent-regime emission mixture. Count state/covariance bytes and preserve
input-conditioned scan scheduling. Require actual 95k–105k parameters and a new
registration before training; retain all original models and ME100 failed gates.
Do not use the protected cohort for model selection. This is an unimplemented,
untrained hypothesis with no gain promised. Geometry, CED and sparse lookup remain
draft PR394 proposals, not prerequisites for this step.


### ME101 / NEXT-SHARED-BELIEF — implemented and trained; MIXED_TRADEOFF

TAC draft PR397 / PNDS draft PR13. Five fresh 97,259-parameter models; source-age
belief variance marginalized through a conditional affine emission; unchanged
native scan and 608-byte FP32 state. Ordinary PM2.5 MAE35.689→32.771 (8.18% lower),
with lower ordinary mean error for all six pollutants. Gap PM2.5 MAE38.987→37.747,
but gap density and interval scores worsen; coverage68.67% below85%. Full gate
FAIL0/5. All originals frozen3752a98e before scores5430d4b1; clean/GitHub replay
PASS40178388335; PNDS source/math PASS40177277628. 970 local tests. Protected
2016–2017 unscored. No further ME101 fits or changed gates.

### Capability idea queue — implemented advisory research workflow

PR397 adds scripts/research_idea_queue.py, CAPABILITY_QUEUE.json and a fast-loop
guide. It lists all preflight-valid registrations, excludes completed trials from
next-fit candidates, recommends bounded concurrent compute and keeps missing future
registrations distinct. It executes no fit or arbitrary shell. Live source, math,
resource and data-access gates are separate; a queue entry is not authorization.
Two lower-cost delegated tasks and two CPU fit workers were used in ME101; no
speedup or dollar-saving claim. Attribution remains deferred.

### NEXT-SHARED-BELIEF-LEARNING — proposed complete-model development

ME101 did not establish calibrated missing-state beliefs. Consider jointly learning
current-state representations and future outcomes, with authentic observed-label
reconstruction and coherent uncertainty regimes. Any generated observation masking
must be explicitly registered training augmentation over genuine labels, never
sold as independent natural-gap evidence. Use a bounded FIT-only selection budget
before the frozen complete-model study. Keep source-specific channel selection,
multi-horizon sensor paths, typed enterprise event heads and optional useful CED
in the shared queue. These are proposals, not new trained capabilities. Do not
consume protected TEST in selection or rerun the failed ME101 protocol.

### DIAGTRAIN001 — observed objective drift and unresolved training sufficiency (2026-10-26)

No new fits. [Draft research PR #425](https://github.com/FPC-effortless/TAC-OSM/pull/425)
records the five original ME109 models, all4,096saved losses/seed and three
authenticated original FIT batches. Source/results live on `research/diagtrain001-training-sufficiency-2026-10-26`.
No REPORT/protected rows, weights or optimizer states changed; ME109FAIL0/5 stays.

Within each8/64/256hcontext, early/late matched256record windows show point loss
rising7.7364%/11.6322%/9.7917%, while interval score falls34.1899%/38.0027%/36.8682%
(five-seed means). All15seed/contextpairs improve fullobjective but worsenpoint;
last-halfobjective slopes improve11/15, pointslopes worsen12/15. StochasticFIT
curves cannot establish validation convergence or causality.65,536neworigin draws
per model cover30,284–30,388unique station/hourorigins, not65,536independentlabels.
Onlyfinalcheckpointswere saved; insufficienttraining remainsUNRESOLVED.

Fifteen read-only original-rate proposals evaluated functionally: average-point
guard improves totalandaveragepoint on15/15originatingbatches but worsensgapPM
on10/15; onnextfixedFITbatch, gapPMworsens12/15. Unguardedoriginatinggap worsens11/15;
removingguardisunsupported. No optimizerstep orcandidatecheckpoint waswritten.
Theseactualfinite effects extend priorgradientdots but are local, correlated,
posthoc observations. Nextsubstantivetraining must protectavailability/channel
regimes and savefixedvalidationcurves/intermediateweights atregisteredbudgets;
merelytrainingtheoldrecipe longer is not an evidenced remedy.

ActualME109Observatory/anti-cheat replay passes scopedprobes; completeintegrity
remainsINCONCLUSIVE.81relevanttool/optimizer/contracttestsPASS. Separate nofitCI
replaysoriginalcurves andfiniteproposals. Noarchitecturechange, newLeanproof,
newcapability/superiorityclaim, deployment, merge or protectedqualification.

### DIAGTRAIN001 numerical correction — original replay FAIL preserved (2026-10-26)

Update to the preceding diagnosis: original-host clean replay was not portable.
GitHub40287673335FAIL; expanded40289295662shows144out-of-contract fields,27changed
fallbackflags and12.86%guardednorm discrepancy. Original exact-boundary projection
can discard entire protected-coordinate blocks after tiny rounding sign changes.
The previous numerical proposal tables are original-host observations, not
portable exact results; original same-gap10/15 and next-gap12/15 counts persist
in this observed second host, but some other counts/magnitudes differ.

[Draft TAC PR#425](https://github.com/FPC-effortless/TAC-OSM/pull/425) implements
an isolated prospective inward-margin candidate utility; historical optimizer,
all weights, original scores/gates and protected cohorts remain unchanged.
FP64 correction, precision-derived FP32 parameter-rounding margin and conditional
realized-dot certificate avoid exact-zero boundary projection.195/195blockchecks
pass with0fallbacks locally. Independent GitHub inward job40289990676PASS under
unchanged relative1e-5/absolute1e-6; historicalguard job and overallworkflowFAIL.
This is an observed numerical repair, not a global platform/IEEE proof.

[Draft PNDS PR#30](https://github.com/FPC-effortless/pnds-lean/pull/30) contains
source-pinned executable Python and Lean exact-real projection plus conditional
rounding-certificate proofs.3source-reference tests and Leanaxiom auditPASS.
Reciprocal manifests match and preserve17legacyrows. Full1156local regression
suitePASS. No optimizersteps, fits, modelarchitecture/parameterchanges or merges.
The candidate is not installed in any historical training runner.

The corrected candidate still worsensgapPM11/15originating and12/15nextFITbatch
probes. Numerical stability does not resolve objective/regime tradeoffs. Training
sufficiency remainsUNRESOLVED: no intermediatevalidationmodels were saved. Next
substantivestudy must jointly evaluate regime-specific optimization and fixed
learning curves atregisteredbudgets, retaining native TAC parallel state and role
FFNs. Do not claim predictive improvement or silently repeat ME109.

### DIAGTRAIN001 source-diversity finding (2026-10-26)

Original FIT-only pool/plan audit finds371natural-gap origins and186,071general
origins. Each original model drew32,768fromeachpool: all371gap origins repeated
88.3235timesonaverage; generaldraws cover29,983–30,079origins(16.11–16.17%pool).
Pools overlap; they are samplingstrata, not disjoint partitions. Different
augmentations/historylengths are not identicalexamples or independentoutcomes.
This is a concrete diversity limitation, not overfitting/insufficienttrainingproof.
Nextbudget study should improve source-diverse actual-availability sampling as
well as regime-specific learning and fixed intermediate validation. No further
ME109fits, newtrainedmodel or protectedcohort access. Reproducible script and
EXPOSURE_COVERAGE.json live in draftTACPR425. PNDS inward source/math CI40290225913
and Lean40290225790PASS; original guard replay failure stays preserved.

### ME110 — coordinate-typed learning: real point gains, uncertainty qualification failed (2026-10-26)

[Draft TAC PR#431](https://github.com/FPC-effortless/TAC-OSM/pull/431) completed
five original104663parameter models,4096updates each (20480total). The native
ME109forward graph/primary152FP32state and six families are unchanged.100253
point/state coordinates and4410uncertainty coordinates receive separately clipped,
coordinate-routed gradients. No Transformer/SSM, new parameters or component
attribution claim. Diverse2gap+14general sampling visits57344distinct general
update origins/model, with ancestor-exposed fixed320caseFITmonitoring.

Original exposed2015 REPORT, same143natural-gap+4288general queries:

| Metric | ME103 parents | ME109 | ME110 |
|---|---:|---:|---:|
| Gap PMMAE↓ |37.524168|44.437162|34.815926|
| General PMMAE↓ |33.403120|32.014751|30.972970|
| Gap six-channel NLL↓ |5.151004|5.127208|5.075791|
| General six-channel NLL↓ |4.944393|4.891830|4.865349|
| Gap PM90 interval score↓ |379.623199|222.612326|401.850403|
| Gap PM90 interval width↓ |369.909332|197.248291|396.649622|

All5seeds improve gap and general point scores versus their ownparents; mean
7.2173%/7.2752%lower. All5beat the frozen practicalLOCF/ridge point references.
These are system-development comparisons with additional/different training,
not parameter/data/compute-matched architecture superiority or independent proof.

**Complete registered gateFAIL0/5.** Passcounts coverage4, gappoint4, ordinarypoint5,
gapNLL5, intervalscore0. Seed110003gapMAE35.709831 exceeds35.684479limit;
coverage98.601399%exceeds98%ceiling. No rounding/waiver. GapPMmean96.6434%coverage
cannot override per-seed failure or NO2gap75.0725%coverage. No promotion or more
ME110fits. Protected2016–2017numeric rows remainclosed.

Saved0/1024/2048/4096monitor gapPMMAE28.717729/24.888126/24.825922/24.145534;
general35.599698/30.866501/30.659743/30.585923. Late2048→4096point gains2.7406%gap,
0.2408%general (4/5each), while gapintervalscore worsens3.8191%. Moretimealone is
unsupported as a fix; long-runconvergence remainsunmeasured. Monitors exclude new
update targetorigins but ancestors/stats sawFIT and historiesoverlap: notfresh.
No earlystopping/checkpointselection. Retain improved point learning; next work
must engineer conditional uncertainty/current-belief versus future-residual
semantics while preserving safeguards and testing intervalscore/calibration.

All15trained stagecheckpoints/optimizer/RNG and monitor arrays frozen6389521before
REPORT; originalscores/arrays/postfit/summary c8e25cc. Detached clean restoration
reproduced398790REPORTquantile scalars,115200FITmonitor scalars across20views
(15trained+5initial), all original gates/summary and10fittedFP32/FP64forward/backward
checks, no fits/updates. Full1171postfitlocaltestsPASS; GitHuboriginalfinalreplay
40377943112PASS.266240conditionalinwardblockcertificates,0fallbacks.

[Draft PNDS PR#35](https://github.com/FPC-effortless/pnds-lean/pull/35) preserves
source-pinned Python and limited Lean routing/mask/conditional-congruence proofs;
40374187754PASS includes3sourceparitytests/build/axiomaudit. Fullneural/AD/IEEE,
all-channelcalibration, independentmemory/decision/customerqualification remainOPEN.
Scopedanti-cheat/activation/poison/gradientchecksPASS; completeintegrityINCONCLUSIVE.
Originalnegative results, all17legacy mappings and setupincidents remainvisible.
No merge/deployment. Modeloriginals are repository-owned, not a local-only archive.

### DIAGBELIEF001 — no-fit mathematical and leakage audit of ME110 (2026-10-26)

Original5ME110models/fixed320FITorigins, no parameter edits, optimizer or new
REPORT/protected numeric scoring. MissingPM2.5count66: fixed source variance
floor dominates87.8788–92.4242%of cases/seed; upper width is floored96.9697–100%.
At fixed recorded locations, minimum/actual meanwidth99.6807–99.8673%. This is a
measured sharpness constraint, not a causal decomposition of all gate failures.
A pooled24hour observed-pair squared difference linearly extrapolated with age
is not a universal conditional-variance lower bound (stationaryAR(1)counterexample).
Numerical positivity must be distinguished from statistical coverage.

Reconstruction current-mean gradients are nonzero (.272151–.321097norm), but
ME110typedrouting drops their P-coordinate update: reconstruction can tune
variance but cannot directly supervise its mean. RetainedU NLL/intervalcosines
+.598815to+.762008on the fixed16originbatch/all5models; apparent fullshared
reconstruction conflicts mostly occur on discarded coordinates. No conclusion
of globally conflict-free learning or proof of finite-step calibration. Prioritize
coherent current-belief/future-residual semantics, controlled reconstructionmean
learning and conditional-tail/availability training over genericPCGrad/largerstate.

ActualFITloader instrumented4,400,584numericfield accesses before2015,12cutoffrows
without protectednumeric parsing, and independently reproduced FITmean/std bitwise.
ActualCSVfixture REPORTpoison leavesFITstats/earlierinputs unchanged; direct+6target
poison preserves augmentedforwardinputs. No forbiddenleakage detected at named
boundaries; fullsource/cache/ancestor/REPORTtruthjoins/nuisancecontrols remain
INCONCLUSIVE. FITmonitorancestor/history exposure and repeatedlyused2015REPORT
remain retrospective. Phase strings alone do not authenticate rawarrayprovenance.

Tenprimarypapers reviewed, with fulltextassumptionchecks forCQR/ACI/PCGrad/NeuralCDE.
CQR needsexchangeability and offersmarginal, notconditional, coverage; ACIlongrun
coverage can requireinfiniteintervals; bidirectional/cubicspline methods needcausal
cutoffs; likelihoods/SSMs/MoEs supplyno automatic uncertainty/retention/isolation
proof. NoTransformer/SSMconversion, newactivefamily, newtraining or gatewaiver.
ME110fullqualificationFAIL0/5remains. Alloriginalsourceclosurehashes unchanged.

[Draft audit PR#433](https://github.com/FPC-effortless/TAC-OSM/pull/433). Full1178localtests and76targetedchecksPASS; original76ME110payloadhashes/sourceclosurePASS. DedicatedremoteCIstatuspending; no newmodelqualification.

### ME111 — conditional belief/state reconstruction: sharper forecasts, outage calibration still fails (2026-10-26)

[Draft TACPR442](https://github.com/FPC-effortless/TAC-OSM/pull/442) completed five
new104663parameter native Temporal models,1536additional updates each. The source
fixes DIAGBELIEF001's two measured restrictions: no unsupported pooled-difference
hard width floor in the new law, and explicit hidden-current mean supervision on
696existing coordinates. Native fast/slow/8x8affine scan,152FP32primary payload and
six model families retained; noTransformer/SSMconversion or newfamily.

| Original exposed2015 metric | ME110 | ME111 |
|---|---:|---:|
| GapPMMAE↓ |34.815926|34.762480|
| GeneralPMMAE↓ |30.972970|30.637195|
| Gap six-channel NLL↓ |5.075791|5.000335|
| GapPM90intervalscore↓ |401.850403|196.633594|
| GapPM90width↓ |396.649622|115.168202|
| GapPM90coverage |96.6434%|86.4336%|

Intervalscore51.0680%lower and width70.9648%lower; intervalgates5/5PASS. Gap point
mean essentially unchanged, only1/5pairedseed improves; ordinarypoint4/5improves.
**Full legacy gateFAIL0/5; stricter all-channel jointgateFAIL0/5.** Gapcoverage
fails all5, NO2gapcoverage mean71.1594%; seed111003alsofails oldgappointlimit.
No rounding/waiver or promotion. No moreME111fits. Targets143natural-gap+4288general
queries, sameoutcomes acrossseeds, retrospective2015;2016–2017numericrowsclosed.
Additional parent+newtraining means no matched architectural-superiority claim.

A posthoc descriptive check on identical320ancestor-exposedFITorigins/recorded
artificiallyhidden currentvalues yields meanstandardized-log reconstructionMAE
0.525913to0.507204 (3.5573%lower), sixchannelmeans improve. Not independent,
physical-unit recovery, natural-gaptruth or attribution to onecomponent. Fixed
monitor0/512/1024/1536gapPMMAE24.145534/23.960646/23.852308/23.519529. FinalFITgap
coverage96.5625%versusREPORT86.4336% exposes a calibration/generalization gap;
moreepochs alone is not an established solution.

All15originaltrainedmodel/optimizer/RNG checkpoints frozen26ccea9beforeREPORT;
originalscores/arrays823a60b. Detached clean823a60b restored398790REPORT+115200FIT
forecastscalars,20views,10finalFP32/64sourcegradientchecks and originalgates/summary.
GitHuboriginal finalreplay40418385349PASS. Full1181localpre/posttestsPASS; live
countsall104663;7680updates/122880exampledraws;99840conditionalblockcertificates,
0fallbacks. No universalfinite-stepcalibration or IEEEproof. AlloriginalsGitowned.

[Draft PNDS PR37](https://github.com/FPC-effortless/pnds-lean/pull/37): actual-real
scalar positivity/order and limited role algebra, Python/source masks/VJPs all
coordinates. Source-pinned Leanbuild/axiomaudit+11Python tests40415435779PASS;
initialnoncomputable compileerror preserved/fixed beforetraining. Full neural,
causal, calibration and independent realcapability qualification remainOPEN.

This implements master P0statecorrection/coherentuncertainty and uses the existing
Observatory, consistent with [companyintake#441](https://github.com/FPC-effortless/TAC-OSM/issues/441)
and [INT-001#437](https://github.com/FPC-effortless/TAC-OSM/issues/437); noissueclosed.
Nextcapabilitytarget: transferable conditional tail calibration under authentic
outage/site/age regimes while retaining state/point gains, with a new prospective
protocol. Do not reinstate an unjustified floor, tuneagainstprotecteddata or
expandvision/mobile/control complexity as a substitute for this failure.
Otherfivefamilies and product/decision utility remain separately unqualified.
