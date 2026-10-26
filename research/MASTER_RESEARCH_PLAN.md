# TAC-OSM — Unified Master Research Plan

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

## 2026-10-26 — ME108 source head complete joint-adjoint proof + physical CDF/atom event bridge: formally checked PARTIAL refinement

Following [PNDS draft PR23](https://github.com/FPC-effortless/pnds-lean/pull/23) and [TAC draft PR412](https://github.com/FPC-effortless/TAC-OSM/pull/412), NEW incremental [PNDS draft PR25](https://github.com/FPC-effortless/pnds-lean/pull/25) and [TAC draft PR414](https://github.com/FPC-effortless/TAC-OSM/pull/414) formalize and verify a larger same-model boundary, without altering six proposed model-family contracts, trained model weights or any historical negative evidence.

**Proved actual exact-real source-shaped mathematical properties:** One 102-feature→16-tanh-hidden→12/6-output role FFN now has a SINGLE joint JVP/VJP adjoint for input and both learned matrices and both biases, with exact mathematical tanh derivative (the earlier theorem covered input and only individual dense layer parameters); 5,454 total explicit FFN-head learned weight/bias coordinates across three roles. A new conditional finite-arithmetic error estimate propagates local source input/GEMM/tanh/last-layer errors through each full head, but **assumes** local per-kernel IEEE error envelopes and a tanh Lipschitz inequality, NOT proving CPU/GPU FP32/64 kernels, reduction order, overflows, or an end-to-end trained-network error bound. New probability theorems transform source raw physical CDF events y≥0 into exactly the source log1p latent threshold for targetStd>0, and identify the physical zero-censoring atom with its latent event. The prior normalized mathematical standard-Gaussian pushforward is retained; matching source CDF to numerical \`ndtr\` / \`log_ndtr\` / \`ndtri\`, inverse quantile, positive censored density/NLL Jacobian and actual kernel accuracy are **not** yet proved.

**Source numerical evidence:** [Original training-FIT-only source CI40272003389](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40272003389) PASS, actual result artifact12628282555. Five original checkpoint SHA-verified ME108 4096-step models×3 role heads=15 source double-precision fully JOINT input+four parameter-group directional audits, independent NumPy directional Jacobians, Torch autograd and central finite differences; genuine source UCI501 2013–14 FIT only, no original checkpoint changes, no optimizer updates, protected2016–17 numerical rows not opened. This is numerical evidence separate from verified Lean equations. [Original signed report](https://github.com/FPC-effortless/TAC-OSM/blob/research/me108-full-vjp-refinement-2026-10-26/research/mathematical-engineering/neural-refinement/REPORT.md); [PNDS explicit proof ledger](https://github.com/FPC-effortless/pnds-lean/blob/research/me108-full-vjp-refinement-2026-10-26/foundation/v1/ME108_JOINT_AD_CDF_PROGRESS.md).

**Global verification remains OPEN:** complete exact neural source mapping, every recurrent state/mask/13 optimizer block's AD including nondifferentiable branches, actual IEEE CPU/GPU/exp/tanh/special function refinement and whole-network error budgets, concrete standard Normal-law equality to Torch CDF/quantile/NLL, and a compositional full forward+backward implementation theorem. Passing Lean mathematical lemmas cannot satisfy a source-level theorem, statistical prediction calibration or robotics capability. ME108 original registered full real-data predictive gate remains **FAIL 0/5**. **No retraining, merge, posthoc evaluation or full formal correctness promotion.**

---


## 2026-10-26 — ME108 exact-real neural/AD and censored Gaussian refinement: PARTIAL proof, full source correctness OPEN

[Native TAC draft PR412](https://github.com/FPC-effortless/TAC-OSM/pull/412), [PNDS proof draft PR23](https://github.com/FPC-effortless/pnds-lean/pull/23), [source-scope report](https://github.com/FPC-effortless/TAC-OSM/blob/research/neural-refinement-2026-10-26/research/mathematical-engineering/neural-refinement/REPORT.md), [formal proof obligations](https://github.com/FPC-effortless/pnds-lean/blob/research/neural-refinement-2026-10-26/foundation/v1/ME108_NEURAL_REFINEMENT_PROOF_BOUNDARY.md).

User's requested final target is **complete, compositional source-to-Lean neural correctness**, including exact finite-tensor forward/backward, honest IEEE FP32/64 and automatic-differentiation semantics for every PyTorch operator, and actual Gaussian measure/CDF/log-density/quantile correctness after censoring. This target is NOT currently achieved; no unproved primitive is declared solved. Existing ME108 typed whole-graph specification, original pretrained five SHA-frozen 102,713-parameter models and FAIL0/5 real-data qualification are unchanged.

This research increment proves in Lean exact-real two-layer 102→16→12/12/6 FFN input VJP/JVP adjoint laws and linear parameter-gradient algebra; adds scalar tanh derivative proof, conditional two-operation numerical roundoff bound; constructs the actual censored source-shaped probability measure as a pushforward and proves normalization and zero negative-support mass, including a genuine standard Gaussian measure instantiation. Native model source verification extends with a separate read-only original trained checkpoint audit: five original ME108 seeds ×three role heads×FP32/FP64=30 full input/weight/bias Torch-vs-NumPy derivative checks on genuine UCI501 2013–14 FIT samples. [Original audit GitHub CI40268747906](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40268747906) PASS, original source receipts artifact12626708917. No optimizer/training step or sealed2016–2017 target opened. All original research gates retained.

**Scientific/formal limits:** Full-input masked scan/decoder source proof, all five objective AD including stop-gradient/max/clamp subgradient conventions, bit-level IEEE arithmetic/transcendental/reduction order and error propagation to outputs/gradients, equality of the mathematical Gaussian pushforward to actual torch.special.ndtr/log_ndtr/ndtri CDF/quantile/NLL kernels, and optimizer/model composition theorem remain **OPEN**. Local scalar error bound assumes roundoff premises; pushforward normalization alone is not an NN probability-law implementation refinement and cannot prove calibration. Full IEEE source equivalence or deployed control is not claimed. Dedicated PNDS proof build/axiom audit must remain passing under pinned Lean4.14/mathlib before any next promotion; preserve historical failed preflights and failed predictive gates. Six-family shared parallel architecture and CPU/parameter/data/compute fairness rules unchanged.

---


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


## 2026-10-25 — EWA001 implemented useful-history feedback and fast event lookup; full registered gate FAIL

[Draft implementation/research PR404](https://github.com/FPC-effortless/TAC-OSM/pull/404); [preregistered full real-source protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/ewa001-usefulness-feedback-2026-10-25/research/mathematical-engineering/ewa001/PROTOCOL.md); [verified full negative results](https://github.com/FPC-effortless/TAC-OSM/blob/research/ewa001-usefulness-feedback-2026-10-25/research/mathematical-engineering/ewa001/RESULTS.md); original first complete [CI40209655856](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40209655856) **PASS software / FAIL capability**.

Following owner direction to reward *previously useful* memories and rapidly compare new events with existing memories, EWA001 implemented a separate bounded 256-slot 8D learned-source-query LSH memory with 14D verified-real-residual values and priority. The utility arm updates an item's reward ONLY after real next-observation truth arrives: clipped relative squared-loss advantage of a candidate correction versus same native prediction, EMA0.05; utility changes retrieval ordering and capacity eviction. A same-size memory arm uses only similarity, with identical source episodes, hash planes, 8-item topK, residual cap and output correction. The **original ME097 weights/checkpoints (five source-authenticated,95,874 learned parameters), original 8x8 input-only associative scan and original forecast/score gates are UNCHANGED**. This is online nonparametric memory adaptation, **NOT new neural training**. The 23,552B additional cache float payload plus index, identity and runtime costs is OUTSIDE the 608B original FP32 primary recurrent core.

Authentic original SHA-pinned LeRobot ALOHA mobile-cabinet recordings: 59 FIT episodes/15,104 chronological cache-learning transitions per seed, 13 previously EXPOSED DEV episodes/3,328 prequential predictions per seed, five pretrained model seeds; source original TEST outcomes not opened. Real observed labels always arrive AFTER every relevant prediction, with lagged recorded actions rather than a do(action) counterfactual. Preserved original experiment compares no external memory, similarity-only and delayed-utility memories plus strong measured velocity and persistence references.

| Standardized 14D exposed DEV one-step MSE (five-original mean) | Result |
| --- | ---: |
| Real constant velocity | **0.000031119** |
| Frozen ME097 + similarity-only external memory | **0.000339658** |
| Frozen ME097 + verified-outcome utility memory | 0.000371861 |
| Original frozen ME097 unchanged | 0.000512171 |
| Persistence | 0.000616392 |

Neutral similarity-only memory lowers ME097 model-only error33.68% on *already exposed DEV*, but still loses to velocity by10.9x in error. Adding utility **worsens error9.48% relative to similarity-only in ALL FIVE original seeds**, despite improving vs the frozen no-cache model. Hash lookup scored102.67candidate slots/query on average (neutral) and87.31 (utility) versus registered <32 efficiency gate; 9 bucket probes/query. Original complete real-source utility+speed gate **FAIL 0/5**. The "exact top8 overlap" audit is measured AFTER utility reranking, so a low utility-vs-cosine overlap must NOT be called a hash retrieval miss rate. No claimed speed win, general causal learned utility, independent test qualification or useful robotic control follows.

Integrity audit: original pretrained SHA256 checks and outcome-delayed software tests pass; original result in CI artifact12605049012. Early unit test and raw hit-metric reporting failures [40209426834](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40209426834), [40209527278](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40209527278) are preserved as initial PRE-COMPLETE-score failures, not hidden. All final scores negative and original ME097/ME104 historical results retained unchanged. The utility cache is a causally sequential **post-output** stage and does not make the original scan nonparallel; its stored key/utility is not an identified real-world causal effect. Mathematical source-equivalence/Lean update proofs for this extension remain OPEN.

**Next falsifiable direction:** a newly registered experiment should isolate hash candidate recall before score reranking, more faithful attribution to verified future reduction in loss, nonstationary utility/expiry and robust physically structured residual controls. Test with a truly untouched real source/time and matched strong mechanical models. Never tune EWA001 after scoring, choose favorable seeds, expand 608B claims to external cache, or promote from already exposed DEV. Retain six family names, parameter/training/data-matching comparisons and sealed evaluation rules.

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


## 2026-10-25 — ME101 integrated belief memory: forecasting gain, uncertainty failure

[TAC draft PR397](https://github.com/FPC-effortless/TAC-OSM/pull/397) and
[PNDS draft PR13](https://github.com/FPC-effortless/pnds-lean/pull/13) implement
NEXT-SHARED-BELIEF within the existing Temporal family. Five fresh 97,259-parameter
models train the complete native graph: masked source traces, associative memory,
forecast decoder and source-aware conditional uncertainty. No attribution campaign
or extra trained competitor was required for this capability-development cycle.
Exactly six active model families remain; PNDS remains the mathematical foundation.

The model declares a latent current-state variance conditioned on real observation
age and admitted evidence mass. A diagonal conditional affine emission propagates
that uncertainty into the same forecast distribution used for likelihood and
quantiles. Strict positive future noise supplies positive marginal variance; a sum
of PSD terms alone would not guarantee it. No extra persistent tensors are added:
608 FP32 bytes plus separately counted integer/provenance/staleness/runtime memory.
This is a model assumption, not identified Bayesian epistemic uncertainty or a
proved posterior over truly missing current observations.

| Exposed-2015 result, five-model mean | ME100 | ME101 |
|---|---:|---:|
| Ordinary PM2.5 MAE |35.689034|32.770872|
| Natural-gap PM2.5 MAE |38.986917|37.747018|
| Ordinary six-channel density NLL |5.010164|4.963346|
| Natural-gap six-channel density NLL |5.769106|6.234504|
| Ordinary PM2.5 proper interval score |228.490656|188.780493|
| Natural-gap PM2.5 proper interval score |279.347305|301.220560|
| Gap nominal90% coverage |67.69%|68.67%|

Ordinary PM2.5 error improves 8.18%; ordinary mean errors improve for all six
pollutants. All five new models beat ordinary LOCF, with mean error 5.06% lower;
mean performance is approximately tied with FIT-only ridge. Gap PM2.5 error improves
only 3.18% in the mean and three of five mapped parent comparisons regress. Gap
coverage remains below the unchanged 85% gate; density and proper interval scores
worsen. **Full capability gate FAIL: 0/5. Evidence tier MIXED_TRADEOFF.** The belief
formula did not fix uncertainty. There is no overall intelligence, architecture,
independent-site or company-deployment qualification.

There are 4,288 daily queries in 53 calendar-week clusters and 143 gap queries in
21 clusters. The within-model long-minus-short gap MAE interval is
[-17.187783,-0.438010], conditional on the fitted models and already exposed cohort,
without selection/multiplicity or retraining uncertainty. It is not independent
confirmation. ME101 uses twice ME100's updates and a changed loss; no matched
architecture/data-efficiency claim. FIT 2013–2014; protected 2016–2017 numerical
outcomes remain closed.

Original models/optimizers/RNGs/schedules/live receipts freeze3752a98e before
scores5430d4b1. All five have 27 finite nonzero parameter-tensor gradients in the
recorded check. 970 local tests pass, along with FP32/FP64 complete serial/parallel
outputs/state/gradients and actual fitted-weight chunk checks. Clean original
forecast, score and width replays PASS; [GitHub replay40178388335](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40178388335) PASS.
[PNDS source/math40177277628](https://github.com/FPC-effortless/pnds-lean/actions/runs/40177277628)
and build40177277596 PASS. Full-network IEEE refinement remains OPEN. Original
artifacts and failed gates are directly repository-owned; both PRs remain drafts.

### Faster capability engineering, without bypassing evidence

The advisory capability queue and fast experiment loop are implemented in PR397.
Two lower-cost delegated tasks handled mathematical review and the queue while the
owner integrated and verified the model. At most two single-threaded CPU fit workers
ran; no wall-clock speedup or billing claim is inferred. Independently registered
studies may proceed in parallel; missing design registrations do not block others,
and the queue itself never authorizes a fit or protected-data access.

Prioritize complete-model bundles and bounded, registered FIT-only selection over
repeated diagnostic studies. Next ideas include better conditional current-state
representations, authentic observed-label reconstruction and coherent uncertainty
regimes; source-specific channel representations and multi-horizon sensor paths;
and typed event heads on qualified enterprise data. Geometry/CED remain possible
later components when they serve an explicit useful capability, not mandatory
attribution milestones. Selection ceilings, actual active parameters, source/math
checks, original checkpoint freezing and independent replay remain compulsory.
No further fits, interval inflation or changed gates are allowed under ME101.



## 2026-10-25 — ME100 source-aligned sensor world-state capability pilot

[ME100 draft PR395](https://github.com/FPC-effortless/TAC-OSM/pull/395) and
[PNDS draft PR12](https://github.com/FPC-effortless/pnds-lean/pull/12) implement a
first source/admission step inspired by [draft direction PR394](https://github.com/FPC-effortless/TAC-OSM/pull/394).
PR394 remains unmerged; its geometry/CED/sparse-lookup/Muon proposals are not
claimed as implemented or qualified by this pilot. No ALOHA refit, historical
checkpoint replacement, new active family or architecture promotion occurred.

Five new 97,235-parameter TAC-Temporal models forecast six authentic UCI501
station pollutant measurements six hours ahead. Learned observed-only per-channel
sum/mass traces feed the native causal affine associative scan; a coherent
censored-lognormal decoder shares location/scale across likelihood and quantiles.
The complete graph is documented from masked observation to final distribution.
The source has 12 station identities and genuine missing measurements; it does NOT
verify late arrivals, source authority, action consequences or entity discovery.
FIT 2013–2014; all five originals Git-frozen 1d721082 before first 2015 REPORT scoring;
2016–2017 numerical outcomes remain closed for later matched qualification.

| PM2.5 raw MAE, lower is better | Natural sensor gaps | Ordinary daily queries |
|---|---:|---:|
| LOCF within256h |56.076923|34.515625|
| FIT-only ridge |43.915070|32.767415|
| Trained model, recent8h |46.073106|35.477731|
| Same trained model, long256h |38.986917|35.689034|

Long access improves gap MAE 15.38% over short access in the five-seed mean, with
all five directional history gates passing. However the global-week-clustered
95% long-minus-short interval [-15.369833,+1.868555] includes zero. There are 143 gap
queries in 21 calendar-week clusters, not 715 independent seed-query cases. Gap 90%
coverage averages 67.69%, below the 85% gate, and daily performance does not beat
LOCF/ridge on average. **Full registered capability gate FAIL: 0/5 pass.** This is trained,
replay-verified new-source development evidence, not an overall intelligence gain,
a matched neural superiority result or NEXT-SHARED-SELECTIVE Phase2 qualification.

967 local tests pass. FP32/FP64 full graph/state/all 25 parameter-tensor gradient
refinement passes on all five fitted models using real FIT queries; full/chunk
predictions agree. Lean source-trace bounds, missing identity and affine action
compile; full-network IEEE proof remains OPEN. Independent clean original-score replay and
[GitHub replay40173945086](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40173945086) PASS.
Original models/optimizers/RNG/schedules, forecasts, results, live parameter receipts
and checksums are repository-owned in ME100. No further fits or gate changes are
allowed under this completed protocol.

**Next integrated hypothesis, NEXT-SHARED-BELIEF:** represent uncertainty about the
unobserved current state jointly with future change, rather than treating a retained
source anchor as known. In mathematical terms, learn a coherent predictive mixture
`p(y_future|history) = integral p(y_future|x_current,z) q(x_current|history) dx_current`.
Source age, admitted evidence mass and causal innovations can condition that belief;
none alone guarantees calibrated uncertainty. A bounded second-moment trace or
coherent latent-regime mixture is a proposal, not a new result. Retain native
parallel memory and genuine fixed-state streaming. Enforce actual 95k–105k parameter
counts and count every additional covariance/codebook/state byte. Train a complete
new model under a newly frozen protocol; do not posthoc inflate these intervals or
spend protected outcomes on iterative DEV selection. Detailed attribution remains
secondary to better useful end-to-end predictions with trustworthy uncertainty.





## Latest: structured future trajectories and uncertainty regimes — ME098/ME099 (2026-10-25)

**Status: learned capability observed / MIXED_TRADEOFF; full gate FAILED.**
The next shared-core development phase follows the owner's request for mathematical
engineering comparable in discipline to SSM/Mamba: define the actual state and
operators, derive legal composition and parallel execution, analyze stability and
precision, verify independent equations/gradients, then train real complete models.
This is a standard of engineering, **not demonstrated Mamba-level performance or
whole-network formal verification**.

### Implemented, trained and preserved

[ME098 draft PR392](https://github.com/FPC-effortless/TAC-OSM/pull/392) adds a
103,322-parameter coherent trajectory decoder to the shared TAC engine. One
conditioned stable-velocity law predicts1/4/16recorded frames, with one response-
Gram uncertainty structure and proper joint Student-t likelihood. Five originals
improve mean forecasts but all fail the coverage guard. This negative result stays.

[ME099 draft PR393](https://github.com/FPC-effortless/TAC-OSM/pull/393) is the
single preregistered correction:103,380parameters,58new uncertainty-regime values,
all inherited weights still trained. A latent narrow/broad regime is shared across
the complete future trajectory; one Cholesky is reused. Proper interval-score
training supplements the likelihood; analytic marginal CDF/root derivatives
avoid empirical DEV inflation. Both iterations retain one shared engine and
exactly six active families; only the physical lane is trained here.

| Recorded future horizon | Persistence MSE | ME099 mean MSE | Reduction | Nominal90%coverage |
| --- | ---: | ---: | ---: | ---: |
| 1frame | 0.000493550 | 0.000299312 | 39.36% | 89.42% |
| 4frames | 0.007526495 | 0.004465788 | 40.67% | 85.77% |
| 16frames | 0.094477579 | 0.054845821 | 41.95% | 87.44% |

All five ME099 models pass forecasting, same-case H1parent preservation and proper
interval-score criteria. **Four of five pass every gate. Seed99001 H4coverage is
84.192562%, below the unchanged85%minimum: FULLGATEFAIL.** No rounding, favorable-
seed selection, threshold change or further fits under this bounded protocol.

Against complete frozen ME098parents on the exact same cases, mean point MSE
improves1.14/1.52/2.08%; interval scores improve20.75/25.32/21.71%. However,
interval widths increase44.97/47.41/52.60%. This repairs some underdispersion,
not sharper intervals. JointNLL improves(-6.61269to-6.78332percoordinate) but
still loses to the FIT-moment reference(-6.89786). No uniformly improved
calibration, architectural superiority or independent intelligence claim.

### What is mathematically engineered

Explicit observed state:88prefix values,8x8associative memory and count. The
input-conditioned write schedule remains independent of its memory read, allowing
ordered affine scan; fast/slow maps refine the retrieved representation. These
are the consolidated parallel graph's actual semantics, not a claim of lossless
conversion of the original nonlinear DIAGKEYrecurrence.

Future mean: x(h)=x0+g_h(r)v0+(h-g_h(r))u, g_h=sum(r^j). Its fixed-coefficient
flow composes exactly; bounded retention gives a velocity-deviation bound. A
shared innovation-response Gram yields a valid exact-real scale matrix. A latent
uncertainty regime preserves horizon marginal consistency. Future queries do
not update observed-world evidence. No future neural rollout is needed; observed
conditioning can be reused for multiple horizon requests.

[PNDS PR10](https://github.com/FPC-effortless/pnds-lean/pull/10) and
[PR11](https://github.com/FPC-effortless/pnds-lean/pull/11) contain compiled Lean
semigroup/stability/Gram and conditional CDF-mixture statements, typed interfaces,
source-pinned Python equations and independent actual-source FP32/FP64/mask/
gradient checks. StandardLean axioms are audited; no sorry. All5ME099fitted models
also pass actual-FIT reference checks. Full concrete NN/IEEE refinement,
probabilistic calibration and hardware throughput remain OPEN. Kernel proofs
and code tests do not establish empirical capability.

### Evidence scope and integrity

Same source revision7a752b39f7e69de7e38aee485a6bab07528a061a, publisherApache-2.0,
parquetSHAa3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a.
59FITepisodes;13previously exposedDEVepisodes/104windows, prefixes1..64,
19,968window-target positions permodel. Overlap and seeds do not create additional
independent episodes. No TEST scoring in these studies. Currentunits are recorded
frames and standardized state, not qualified elapsed-time/SI forecasting.

Five ME098originals freeze36e8371b before scoresde167f43; five ME099originals
freeze12d72fde before scoresa2e58687. All weights/optimizer/RNG/schedules/live
parameter receipts/raw probabilistic forecasts and episode scores remain inGit.
Independent original replay40165990316(ME098) and40166817604(ME099)PASS.
962local tests PASS on the ME099 branch. The two new studies total10original fits.
ME099oldcore lineage1,280updates versus768forME098; no equal-training-compute
architecture win. Validated honest scorecards report MIXED_TRADEOFF, not
independent gain. Earlier ME096archive-order and ME095interruption records remain.

### Prioritized next ideas — proposals, not implemented gains

1. **Learned nonnegative future-noise envelopes.** Permit middle-horizon process
   uncertainty to vary through a response factor while preserving PSD and horizon
   consistency. The remaining H4miss motivates the capability question, not a
   proven mechanism. Freeze the next model and qualify on a genuinely new cohort;
   do not continue tuning these13episodes until a threshold passes.
2. **Authentic elapsed time, entities and corrections.** Extend the transition
   law to real irregular observation clocks and stable entity histories. For
   Enterprise/Temporal, build typed event-time/lifecycle heads on qualified real
   logs; neither physical frames nor log endings prove workflow completion.
3. **Reusable future-path queries and square-root kernels.** Implement coherent
   trajectory-noise sampling that never overwrites observed evidence; investigate
   factor-space/QR likelihoods and hardware-aware chunk kernels under the same
   numerical/gradient contract. Current graph work/depth is analyzed, but custom
   GPU fusion, edge latency and actual action selection are not established.

Useful independent predictive state, real action consequences, authority-safe
Woyengi integration, customer benefit and commercial qualification remain separate
open goals. All original17historical mappings and the six-family identity remain.
No draft was merged or model promoted.



## Current shared-core capability update — ME097 (2026-10-25)

**One shared parallel/streaming architecture; exactly six active families.**
System1, Physical, Enterprise, Temporal, Edge-Lite and Counterfactual use the same
prospective core by configuration. PNDS is their mathematical authority, not a
seventh model. [TAC #390](https://github.com/FPC-effortless/TAC-OSM/pull/390) and
[PNDS #8](https://github.com/FPC-effortless/pnds-lean/pull/8) preserve the17 indexed
historical mappings. Older "separate lanes" sections remain historical; they do
not authorize duplicated memory engines or imply six independently trained heads.

[Draft TAC #391](https://github.com/FPC-effortless/TAC-OSM/pull/391) develops one
**95,874-parameter physical candidate** within that shared architecture: a causal
context-conditioned retrieval query, original bounded input-only affine write
schedule, and a direct retrieved-evidence prediction path. Valid masks preserve
stored state; batch scan, carried chunks and streaming share equations. This is
not another retention-only gate change, an SSM replacement or a seventh model.
The new candidate meets95k–105k; the prior93,324parameter pilots are preserved
as reported and are not retroactively treated as policy-compliant new fits.

### Measured learned development capability

ME097:5new original seeds,256updates each,59 authentic ALOHA FIT episodes;
104previously exposed DEV windows from13episodes,64observations per window.
Next-state standardized MSE **0.000301391 vs persistence0.000467679 (35.56% lower)**.
Late-half MSE0.000261363vs0.000438049(40.33%lower); movement-rich MSE
0.001496158vs0.002507034(40.32%lower), threshold from FIT90thpercentile.
**Every seed passed** the preregistered5%whole/late improvement and no movement
regression criteria. Originalweights/optimizer/RNG/schedules were committed at
1901d395 BEFORE original scoringaaa58ad5. Clean-checkout local and independent
[GitHub original replay40163334508](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40163334508)
PASS. Full local suite947testsPASS. This is **DEV_GAIN**, not independent-site,
long-memory, learned-rival superiority, causal control or customer intelligence.
Window positions/seed repeats are not independent new episodes.

Exact data revision7a752b39f7e69de7e38aee485a6bab07528a061a; publisher Apache-2.0,
parquetSHAa3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a.
Observed x[t] plus recorded action[t-1]; FIT-only normalization/scales. Original
TEST periods were not scored. No controlled before/after architectural win is
claimed: old93krecipes used different action alignment, windows and4×fewer updates.

### Full audit trail and formal boundary

ME096's5original models also passed numerical thresholds, but a sparse-checkout
Git-add failure let scoring run before archive commit. **ME096 promotion BLOCKED**;
all originals and exception preserved. ME097 was an openly registered unchanged
replication with5NEWseeds and a hard committed-byte pre-score guard, not a hidden
retry or changed criterion. Both5-fit sets count. ME097's initial self-hash loader
path failed before any model/update; corrected and committed before first fit.
ME095 remains **4completed originals+1interrupted after last1,024-step log**, no
recoverable fifth model and no REPORT scoring; archived b30b5aaf inPR377. Earlier
"ME095 running" entries below are historical; no silent retry or passed old gate.

[PNDS #9](https://github.com/FPC-effortless/pnds-lean/pull/9) includes typed whole
pipeline and independently executable serial neural equations. FP32/FP64 actual
source outputs/state/input+weight gradients pass; all5fitted models on real FIT
also pass (maxoutputerror1.1921e-7,maxgradienterror4.6567e-9). Lean padding,
read-before-write and query-nonmutation statements compile without axioms.
**Full concrete neural/finite-precision Lean refinement remains OPEN**. These
checks do not attribute the measured gain to memory or prove general intelligence.

**Next development priority:** build on the successful shared trained candidate
for useful multi-step outcomes and coherent uncertainty, source-supported delayed
observations and stable entity state; qualify on a genuinely new real cohort
under a separate frozen protocol. Preserve originals and failed gates; do not
replace capability training with another retention-only diagnostic campaign.
Other domain-specific encoders/heads, action selection, independent useful memory,
external pilots, authorized production integration and commercial value stay OPEN.
No PR merged or model promoted.

Version: 2026-10-23 (integrated capability and investor-readiness guiding objectives; ME071–ME073 trained and independently replayed; drafts unmerged) | Status: research proposal/evidence register; not a commercially qualified product

## Hard proof gate: when TAC-OSM may be called more capable or intelligent (2026-10-23)

**Compulsory canonical policy:** [Capability Progress and Intelligence Evidence Standard](CAPABILITY_PROGRESS_TRUTH_STANDARD.md). This rule is **above** general development progress descriptions and investor enthusiasm. It applies prospectively to **all** new model-study claims, research PRs, master-plan “improved” assertions and investor-evidence claims; old studies keep immutable receipts. A successful fit, different architecture, additional head, bigger model, successful CI, independent checkpoint replay or exposed-DEV score by itself **does not** prove model improvement or more intelligence.

**What earns an improvement claim:** (1) pre-registered factual skill, decision-time inputs, old frozen full model and new complete model, exact identical real task/source/cases/targets/horizons/metrics and source exposure; (2) report a truthful material result and all relevant calibration/delay/long-tail/rare-branch/abstention regressions; (3) for **independently confirmed gain**, score on a **genuinely untouched authenticated real cohort** with original raw paired case-level evidence, source/model identity, prespecified nonzero practical gain and conservative paired-case confidence/seed decision, appropriate practical reference and no critical guardrail violation; (4) retain all original Git checkpoint/optimizer/RNG/predictions/scores and independent clean-checkout replay. Score **every** pre-registered horizon and eligible case, not the most flattering one. Cross-domain intelligence requires further independent-source transfer, and policy/customer utility requires separate verified actions, outcomes and economics. If a new skill was absent before, show authentic task performance relative to a real appropriate reference without claiming the old architecture was beaten.

**Claim vocabulary is mandatory:** `DESIGN_ONLY`, `ENGINEERING_VERIFIED`, `TRAINED_REPLAY_VERIFIED`, `DEV_CAPABILITY_OBSERVED`, `DEV_GAIN`, `MIXED_TRADEOFF`, `INDEPENDENT_GAIN`, `TRANSFER_REPLICATED`, `OPERATIONAL_VALUE_VERIFIED`, `REGRESSION`, `INCONCLUSIVE`, `BLOCKED`. No invented intelligence percentage. Every study must include the policy's Capability Progress Scorecard and explicit **what improved / what regressed / what remains unknown**. An improvement claim cannot be based on a higher accuracy computed on a different prediction question.

**Research priority remains capability-first:** these proof rules do **not** mandate default ablations or GRU comparisons before one integrated native candidate is trained and tested. Architecture-winner claims still require the exact matched parameter/data/exposure/training/tuning-compute contract; future live fits additionally face the repository's **95,000–105,000** trainable/deployed parameter target. CPU runtime is not an equality gate. This new standard is an evidence/promotion control, not proof of an additional trained-model gain.

### Implementation and current enforcement boundary

The prospective claim checker is at [`scripts/validate_capability_progress.py`](../scripts/validate_capability_progress.py), with [a deliberately invalid claim template](CAPABILITY_PROGRESS_SCORECARD_TEMPLATE.json), eight adverse/positive software fixtures and [GitHub Actions run 39969267686](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39969267686): **8/8 tests PASS** and the incomplete template was correctly rejected. These are **policy-software tests**, not new TAC-OSM model results. The script rejects self-declared independent gain on exposed DEV/TEST, insufficient paired CI/seed evidence, absent same-task old model, missing genuine observed-case counts, failed guardrails and unsupported transfer/customer promotion. Its PASS means **declaration format only**; source authenticity, original per-case score calculations and funding/pilot evidence still require independent audit.

**Not yet globally enforced:** the implementation is on `research/master-research-plan`, currently **not integrated into each open model-development draft branch or demonstrated as a mandatory protected-branch check**. Every new study must bring this validator/standard into its exact working branch and publish its honest scorecard; do not call the new policy universally installed, do not edit old study/checkpoint records and do not merge research drafts without a separately authorized decision.

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

## DIAGKEY004 — trained native write/read repair, synthetic-only (2026-10-23)

[Draft PR #362](https://github.com/FPC-effortless/TAC-OSM/pull/362) completes one
prospectively registered immediate-recall repair with25 guarded original fits,
five seeds and50 original model/optimizer/RNG milestone checkpoints. The new
**101,527-parameter** native candidate retains88D fast/88D slow and ONE8x8 matrix;
shares learned key addresses between writing/querying, ties value-code decoding,
separates retention/error-write gates, protects QUERY from memory modification
and uses a radius-four projection. Published gated-delta math is credited and
checked against a pinned independent Flash Linear Attention reference. The old
model already normalized keys/queries; normalization alone was not the repair.

On the same new512 synthetic TEST queries/seed and identical original training
examples/order, revised native exact recall is **97.93% at640 updates and99.84%
at4096**, final seeds99.80/100/99.41/100/100%. The original **>=90% in EVERY seed**
gate passes without relaxation. Unchanged native reaches30.31%/94.02% at640/4096,
with one final seed77.15%; budget was an important limitation too. No-association
TAC and GRU reach24.26/24.18% final; the Transformer reaches99.92% (100% at640).
The task has four stored pairs,16 possible initial keys,64 total key vocabulary,
32 values; unconditional chance3.125%, but ignoring the query and uniformly
choosing a stored value gives25%. The old one-hot oracle had8 values, not32;
its capacity conclusion must remain restricted to protected orthogonal keys.

**Evidence boundaries:** all25 models and both milestone sets were Git-frozen at
04002ef6 BEFORE first TEST scoring; actual original guard/source/data digests
verified. All arms have<0.12% parameter discrepancy and the same optimizer
exposure, but partial FLOP profiles differ and full-compute parity is unverified:
**no fair architecture-winner or real-world intelligence claim**. A clean fresh
local Git clone reproduced all50 original score sets under the registered1e-5
contract. A local AVX2 check also passed. The separate GitHub runner reproduced ALL25,600 classifications exactly and
all50 NLL scores within the original limit, but2/819,200 raw logits exceeded
abs/rel1e-5 (overall max abs difference3.62e-5). The strict external replay gate
stays FAIL; no tolerance widened. The complete external audit is Git-preserved. See
[results](https://github.com/FPC-effortless/TAC-OSM/blob/research/diagkey004-aligned-gated-retrieval-2026-10-23/research/mathematical-engineering/diagkey004/RESULTS.md)
and the adjacent numerical-replay status. Canonical real-world scorecard remains
**ENGINEERING_VERIFIED**, authentic_real_data=false. Neither delay/interference,
long memory, authentic forecast improvement nor action utility is established.

**Next decision:** this qualifies the simplest synthetic mechanism rung only.
No automatic1/4/16/256-delay training. Any interference curriculum needs a new
prospective registration and preserved original evidence; authentic System1,
Enterprise and Physical development priorities continue independently. Do not
use this result to overwrite ME076's real negative result, promote the model,
or make memory-capacity/causal-attribution claims from cosine alone. The cheap
frozen DIAGKEY003 read check finds strongly negative learned key/query cosine
in several higher-accuracy seeds; decoder coding/sign and OOD substitutions
prevent treating cosine as a standalone retention/retrieval diagnosis.

## ME076 first authentic System 1 original model training — NEGATIVE CAPABILITY RESULT (2026-10-23)

**Preserved original evidence:** five genuinely fitted 100,476-active-parameter native fast-only System1 models, 160 AdamW updates/seed, on real original BodenAI RW-RL-HIL robot FIT episodes, registered in [ME076 FAIR_COMPARISON](https://github.com/FPC-effortless/TAC-OSM/blob/research/me075-system1-native-gated-delta-2026-10-23/research/mathematical-engineering/me076/FAIR_COMPARISON.json). Original 160-episode SHA `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776` matched frozen source; source FIT redivided by deterministic episode key into 120 FIT and 40 **already exposed original FIT-derived episode-disjoint REPORT**. FIT-only normalizer/servo. 960 genuine eligible observed next-state queries on 40 REPORT episodes **per seed**; 84 genuinely moving under preregistered 0.025-unit mean 14-joint change threshold, 876 near-stationary. This source's previous-exposure status means **DEV evidence only**.

**Source-backed mean physical MAE across five original seeds:**

| Same exact real query cohort | Persistence `x_t` | FIT servo causal lag1 | Native System1 (mean over five real-trained seeds) | Scientific interpretation |
| --- | ---: | ---: | ---: | --- |
| All 960 physical next states/seed | **0.068668** | 0.097457 | **0.080158** | **Native 16.7% worse than persistence**; all five seeds lose |
| 84 motion-rich factual next states/seed | 0.743241 | **0.704517** | **0.748786** | Native worse than both; movement-rich capability target fails |
| 876 near-stationary actual states/seed | **0.003983** | 0.039245 | **0.016043** | Native issues excessive corrections on mostly stationary trajectories; ≈4× persistence error |

ME076 trained the entire real native architecture, and original fitted model/optimizer/RNG bytes plus 5×960 raw real forecasts and score arrays were Git-preserved. An independent fresh-runner checkout [original replay](https://github.com/FPC-effortless/TAC-OSM/blob/research/me075-system1-native-gated-delta-2026-10-23/research/replay-bundles/ME076/INDEPENDENT_ORIGINAL_REAL_REPLAY.json) **passed all five checkpoints and real scoring without any refit**. The replay is *reproducibility* of the old real exposed development task, **not independent predictive generalization**. The strict [ME076 protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/me075-system1-native-gated-delta-2026-10-23/research/mathematical-engineering/me076/PROTOCOL.md) preregistered beating same-case persistence by ≥5% on both all/movement-rich; **failed all required predictive capability gain gates**. Do not claim the model became more intelligent, competent as an execution gate, risk-aware, or business-deployable.

**Scientific cause *hypotheses*, NOT verified module attributions:** original sampling deliberately boosted motion-rich training windows to half of optimizer examples, although only ≈8.9% of naturally eligible training windows were motion-rich. The observed native/last-state MAE gap is largest on 876 near-stationary development queries, **consistent with** over-predicting transitions and over-service; but this does not isolate what part of the architecture learned incorrectly. Lag1 logged action is a real prior-known command but prior ME068 source timing audit did not identify true physical actuator effect. A proper probabilistic next-state likelihood objective alone does not guarantee optimal physical MAE. The original frozen 100k System1 architecture with a bounded residual and fast-only state was not demonstrated better than persistence.

**Binding next integrated capability investigation (not fitted):** design a **jointly trained selective residual native System 1** with explicit last-observation/persistence anchor, learned *probability of meaningful next-state movement* and a movement-conditioned bounded delta/uncertainty. The motion occurrence target may use authentic future observations for **FIT supervision only**; the gate at inference may see **only already observed** states and known-at-time commands, never future movement or human takeover. Calibrate movement probability and residual amplitude using only a newly registered FIT/TEMP schedule and measure *real* stationary false corrections, changing-state errors, proper event/motion uncertainty, all-case physical MAE, near-stationary and movement-rich MAE, plus task-level robustness. Compare the **new whole model** against original frozen ME076 predictions on **identical original real queries**, and strongest same-task FIT-selected persistence/servo. Any improvement on that exposed REPORT stays `DEV_GAIN` at most. For an **INDEPENDENT_GAIN**, preregister a **genuinely untouched authorized real task/site** before tuning and assess actual full task availability, action alignment, safety and case-cluster uncertainty; original older protected tissue-box task must stay unopened unless specifically preauthorized. Require 95–105k actual trainable/deployed parameters, genuine Git original whole optimizer/RNG checkpoint, independent replay and all negative scores. No automatic ALLOW/ABORT or 25k sub-ms deployment claim.

**Investor effect:** ME076 is *credible negative experimental diligence* and a hard stop on selling System1 as an immediate physical prediction/product winner. Authentic repeated model-training/replay is achieved; an actual **predictive practical advantage or operator value is still absent**. Keep the six-lane registry distinct and real-world capability-first, not synthetic keyed-recall-only. Do not require attribution studies merely to develop a more capable full model.

## ME075 priority correction: System 1 versus random-key associative recall (2026-10-23)

**Binding evidence correction from actual DIAGKEY001 [draft PR #357](https://github.com/FPC-effortless/TAC-OSM/pull/357):** five original **guarded** synthetic-mechanism fits each for TAC 101,554, no-association 101,560, GRU 101,500 and causal Transformer 101,616 (20 original models) were independently restored/replayed in the `DIAGKEY001_GUARDED` preserved audit. Native exact recall was **3.75% / 4.492% / 3.4766%** at 16/64/256 distractors, Transformer **99.0625% / 98.7891% / 98.8672%**, and chance3.125%. The first unguarded originals remain quarantined as failed policy evidence; newer guarded rerun does not retroactively repair the earlier attempt. *Partial* supported-operator per-update FLOP profiles differed roughly18–21% among models and complete compute parity was not established. These striking **synthetic-specific** outcomes show the tested native key/value representation/training recipe failed this probe, not a theorem that all TAC memory or real-world prediction skills fail; do not promote Transformer architecture-superiority or a new memory success claim. ME074 fast-zero sensitivity was early-beneficial/later-harmful and the 8×8 association lesion small/mixed on exposed BPI2012; no universal license to delete association from frozen ME073.

**Owner focus: System1 first, memory repair as a separable core candidate.** Current native [draft PR #360](https://github.com/FPC-effortless/TAC-OSM/pull/360) stacks on six-lane [draft PR #358](https://github.com/FPC-effortless/TAC-OSM/pull/358) and implements: **(A)** `system1_native.py`, a **100,476-active-parameter** 96D fast-only bounded one-step continuous state predictor (14D state +14D at-time issued action + elapsed known time), with **no risk head** absent independently verified event labels, and no slow/association memory, policy authority, auto-approval or rollout; **(B)** separate `gated_delta_memory.py`, published 8×8 gated corrective fast-weight update with distinct retention/write gates; query events never write. Dedicated honest tests and [no-fit protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/me075-system1-native-gated-delta-2026-10-23/research/mathematical-engineering/me075/PROTOCOL.md). The branch adds **engineering functionality, not a new learned model capability**. ME075 has **no original new fitted native model checkpoint, real observed forecast gain, qualified health/risk head, actual hardware latency measurement or causal action utility**.

**System1 next scientific gate:** requalify the authentic ME068 14D action/state source and time-alignment from recorded FIT episodes; preregister exact valid observed `x_t, a_t, delta_since_last -> x_(t+1)` horizon, no future commands, no post-takeover labels and an untouched actual external real cohort. Compare native complete model with persistence, original appropriate causal servo baseline and the frozen prior native task where comparable. Measure proper one-step NLL, movement-conditional MAE, nonmovement prevalence, calibration/abstention, actual per-episode paired uncertainty, all 5 original registered seeds and hardware p50/p95/p99. Commit original fitted 100k checkpoint+optimizer/RNG/model-source preprocessing/raw forecasts/case scores inside Git and pass independent clean-clone replay before a `DEV_CAPABILITY_OBSERVED` or `INDEPENDENT_GAIN` claim. Label risk/ALLOW as **UNQUALIFIED** until authentic outcomes plus independent authorized safety/permission requirements exist; no user-facing sub-ms or 25k claim without actual real hardware or amended size policy.

**Prospective budget contradiction:** owner's suggested “extract 25k params for System1” conflicts with existing compulsory **95,000–105,000 parameters for ALL new trained/deployed learned models**. This branch adopts the compliant **100,476** instead. A future explicit owner-approved, dated **prospective policy amendment** and revised live guard must be adopted before trying a 25k new training/deployment lane; no bypass by names, quantization or hidden frozen weights.

**ME075 optional content recall:** retain 8x8 gated delta mathematics as a reusable candidate; a correct orthogonal-key unit test is not evidence of general 16/64/256-content-addressed learning. The existing owner-approved synthetic exception in DIAGKEY001 does not automatically permit a new series of unregistered synthetic training studies. Only run a new narrow, explicitly authorized scientific diagnostic with the actual guarded source/compute and Git original evidence and keep the result labeled synthetic. In no case does >90% synthetic retrieval imply a real industrial world model or prove fewer FLOPs than attention.

**Order:** (1) no-fit correctness engineering complete on the System1 candidate; (2) qualify a truthful real immediate physical observation/outcome source; (3) actually train/checkpoint independently replay an integrated 100k native one-step model with proper baseline and hard real outcomes; (4) separately assess the corrected memory math and long-context only where the real domain requires it. Do not halt unrelated authentic Enterprise and Physical forecasting development solely over DIAGKEY001; investable claim follows measured operator benefit, not synthetic associations.

## Binding six-regime TAC-OSM model family and evidence roadmap (2026-10-23)

**Owner direction (prospective research architecture, not six claimed trained systems):** Stop forcing one native decoder/output and one held-out task to represent fundamentally different regimes. Organize future TAC-OSM into **six distinct, interoperable research lanes** around proven shared fast/slow recurrence, selective associative/relevance and bounded residual mathematical concepts, while preserving all original historical checkpoints and equations. Each lane uses its own authentic real-data source and metrics; there is **no meaningful cross-regime accuracy leaderboard or invented aggregate intelligence score**. The strategic goal stays **stronger learned model intelligence and useful capabilities**, guided by a plausible, independently validated enterprise product wedge and the investor-readiness evidence ladder.

**Implemented engineering foundation (not trained capability):** isolated branch [six-variant source refactor](https://github.com/FPC-effortless/TAC-OSM/tree/research/six-variant-core-lanes-2026-10-23/src/tac_osm/variants), with [historical evidence and phase map](https://github.com/FPC-effortless/TAC-OSM/blob/research/six-variant-core-lanes-2026-10-23/research/SIX_VARIANT_ARCHITECTURE_AND_EVIDENCE.md). It starts from the ME073 branch with live parameter safeguards and preserves `workflow_me069/071/072/073.py`, original trained tensors, robotic models and replay bundles without alteration. Six new **advisory and audit adapters** are software construction, not six fitted models, Woyengi integration or authorized real action. The refactor is **not merged** into the canonical default branch; source and master-plan branches are currently separate.

### Variant ownership, research question and existing authentic evidence

| Lane | Responsibility and proposed mathematical specialization | Historical research that genuinely informs it | New honest empirical capability gate |
| --- | --- | --- | --- |
| **1 / System 1 — reactive** | Fast observation/risk readout; bounded immediate recommendation; optional fast-only recurrent candidate; no mandated multistep planner | ME078/079 authentic exposed next-frame movement-discrimination gains (per-fit AP0.617/0.608), no control/safety qualification; ME054/061 short-horizon dynamics; **ME074 fast-state readout lesion** made early-age forecasts *better* when fast was zeroed, and late-age worse. This is sensitivity on exposed data, **not** evidence of sub-ms action policy | Authentic contemporaneous reactive sensor/action outcome task, baseline & missed/false action cost, confirmed real hardware p50/p95/p99, zero unauthorized policy authority |
| **2 / Physical — robot** | Continuous original **14D** recorded state/action adapter first (not assumed universally 6D/12D); observed action-conditioned deltas, physics/persistence references, possible reliability-routed bounded learned residual | ME054/056, **ME061–ME068**; ME064/066 original real transfer **lost to persistence**; servo is **not guaranteed** strongest reference; ME078/079 beat persistence on exposed1/4/16frame tasks, but stationary preservation and native-over-control gates FAIL; ME067 lesions and ME068 source timing diagnostics are not contact/force oracle | Untouched authentic physical sensor/motor trajectory, real motion/contact labels when present, accurate movement under true causal actions, strongest FIT-selected physical reference and no dangerous false positives |
| **3 / Enterprise — Woyengi advisory hypothesis** | Authentic categorical events/resources, timestamped fast/slow/optional associative state, direct and autoregressive multi-event/time likelihood with explicitly recorded source boundary | **ME069–ME074**: ME073 five 99,781-parameter original fitted trajectory models; ME074 exposed baseline/native-lesion results; early 2-event accuracy ≈40%, distant uncertainty unresolved | Full native candidate with earlier difficult-branch benefit, properly coupled activity/time, truthful censoring and source completion semantics, real independent heldout domain and operational warning utility |
| **4 / Temporal — bitemporal** | Valid time and record/knowledge time, late/backdated revisions, as-known-at time travel and deterministic replay into native predictor state; Woyengi remains authoritative | Existing Woyengi request contracts, ME069/071/073 event-time context **only**; no independently qualified bitemporal neural experiment | Authorized *actual* corrective/versioned event histories, tests of as-of non-leakage and reconstructability, quantified trained forecasting improvement under late facts |
| **5 / Edge-Lite — embedded** | Explicit frozen model export, quantization, runtime memory/energy/latency and error audit, no silent policy bypass; full scan/INT8 compute only where deployment supported | ME055/056 resource profiles and ongoing live model budget guard are computational history, **not** edge deployment qualification | Actual ARM/microcontroller/GPU evidence, verified INT8 numerical score preservation, resource power/latency distributions, lawful model export and no trained capability degradation |
| **6 / Counterfactual — Veritas scenario** | Conditional real action/outcome representation with sandbox alternative branches, uncertainty and safe refusal; true causal `do(a)` and MCTS require extra evidence | CASM historical control, ME049 proxy, ME061 factual observed actions, ME073 passive imagined trajectories, DEV016/017/020 state research; none proves causal action-policy utility | Registered action-availability/propensity, verified alternate-action evidence or randomized policy, actual cost/value, off-policy/causal identification and permission-bound prospective test before live recommendations |

**Shared architecture, separate scientific responsibility:** the six variants are **research lanes first**, not six equally capable checkpoints. Shared typed observation/forecast/evidence contracts, source-backed manifest identity, original Git checkpoint/replay rules and capability-value scorecards can be reused. Distinct event/physics/temporal/edge/action heads must not share incompatible target labels; a future decoder can be specialized while core state memory equations remain reusable where supported. Never destroy historical checkpoint compatibility by deleting dimensions/heads in-place.

### Correct the proposed fast-state / associative-memory / model-size assumptions

1. **ME074 is not a deletion order.** The original frozen 8×8 associative readout lesion has small age-dependent effects; it neither proves memory is universally dead nor that removing a trained tensor will preserve the original model. An Enterprise native candidate *may test* no-association and early gated fast state as **integrated newly trained designs**, but cannot retroactively mutate ME073 weights. Fast-state removal helped at age2 on exposed task but harmed at ages6/16; nothing about this establishes System 1 reaction latency or usefulness.
2. **Do not hard-switch physical regimes by assumption.** Historical real target tasks repeatedly favored last-observation persistence over servo. Start from the **best appropriately FIT-chosen factual physical reference**; test optional risk/reliability routing and neural residual only on authentic observed contact/movement settings. There is no proven unlabeled contact-force detector or actuator torque improvement.
3. **The 100k size policy is binding for ALL future fitted lanes.** The user-proposed nominal **System1 20–35k, Physical 50–80k, Edge-Lite 10–15k** are **unapproved capacity aspirations** conflicting with the current prospective **95,000–105,000 actual trainable and deployed learned parameters per new model**. Do **not** fit these tiny architectures or smuggle pretrained encoders/routers or frozen weights around the gate. Any revised lane-specific scale needs an explicit prospective owner policy amendment, maintained original prior evidence and new updated runtime guards. Quantization may change memory/precision without lowering parameter count; it does not authorize a 15k trained model. Within architecture comparisons preserve separate **≤1% actual parameter difference plus same real data, exposure, compute and tuning**; CPU execution time is not a fair-budget equality requirement.
4. **System 1 advisory is not policy authorization.** TAC may estimate risk or suggest bounded action only. SafeEdge/Aegis/Woyengi/execution controllers independently validate permission, uncertainty and irreversible actions. Never make a neural `ALLOW/ABORT` flag authoritative based on sub-ms target rhetoric.
5. **Temporal replay is a data/governance invariant before a neural claim.** Reconstructed as-of state must not use a revision recorded later than the requested knowledge cutoff. A deterministic reconstruction algorithm is software functionality, not proof a neural model understands history, nor a complete Woyengi product integration.
6. **Counterfactual branches are not `do(a)` causal facts.** Imagined trajectories, Veritas synthetic worlds and observational conditional estimates can support *diagnosis and development* only. Policy effect or financial benefit demands real intervention/propensity support and separately authorized outcome evaluation.

### Prioritized implementation roadmap — build six interfaces, train selectively

**Phase A (code refactor — foundation / engineering only):** implement shared versioned input/output/evidence and no-fit registry; physical baseline fallback, Enterprise predictor envelope, safe one-step System1 advisory, bitemporal reconstruct-as-of replay, edge export audit, and non-executing scenario assessments. Preserve every historical checkpoint/source hash and add isolated invariant tests and CI receipts. This is **not** a new trained predictive breakthrough. **This is the limited scope of the current six-lane refactor branch.**

**Phase B (first qualified native model work — core trinity):** **Enterprise first** because its authentic case/event records and investable workflow hypothesis are strongest: improve early-stage branch prediction, real event/time coupling, calibrated heavy tails, coherent longer autoregressive forecasts and factual state correction in one new live-budget-compliant candidate; use truly untouched externally qualified real data for independent progress claims. **Physical separately** tackles genuine action-conditioned motion/contact shifts and persistence-beating benefit on new real equipment. **System1 third**, only when real reaction/effect data and a safe executor permission boundary exist. Do not delay good Enterprise/Physical capability work on an unqualified edge or causal lane.

**Phase C (specialized evidence-gated execution):** Temporal gets authentic dual-clock corrected operational histories and measured readjustment forecasts; Edge-Lite gets actual export/hardware and numerically audited model fidelity, while respecting live size policy; Counterfactual gets real recorded interventions, identification design, and independently verified action values. Veritas simulated diagnostics never count as real-world proof.

### Uniform prospective success conditions for each lane

All new model-study claims obey the canonical [Capability Progress Truth Standard](CAPABILITY_PROGRESS_TRUTH_STANDARD.md) and [authentic-real-data promotion gate](REAL-WORLD-EVIDENCE-RULES.md): original licensed real records, exact online-available fields, fixed IDs/time split/censoring, honest sample counts, pre-registered task and minimum effect, valid practical baselines, full original model/optimizer/RNG and all raw original scores committed to Git, independent clean-clone score replay. If claiming a native improvement, compare to frozen full-native prior **on identical actual task and cases**, report negative hard strata, critical time/calibration guardrails and uncertainty; `DEV_GAIN` is not `INDEPENDENT_GAIN`. If a new lane has no previous model skill, score actual outcomes and appropriate real reference without inventing a before/after gain. For architecture-winner claims, enforce independent strict parity and source verification; for commercial value, require externally authorized prospective customer pilot and actual utility outcomes. Every lane's new research registration must name **what got better, what got worse and what is unknown**; no fake six-variant intelligence rating.

**Portfolios remain clean:** TAC code is a predictive/advisory research dependency only. Woyengi owns authoritative and bitemporal source-of-truth, Veritas owns sandbox/world tests, and external permission engines own action execution. No cross-repository integration is claimed from this plan, nor is the six-lane proposal sufficient for independent funding or commercial seed readiness.

### Six-lane literature and technical evidence attachment (verified 2026-10-23)

The corresponding [external mathematical prior-art map and next authentic-data experiments](https://github.com/FPC-effortless/TAC-OSM/blob/research/six-variant-core-lanes-2026-10-23/research/SIX_VARIANT_LITERATURE_TO_EXPERIMENTS.md) connects established research to each lane: Hasani et al. **CfC continuous-time networks** (System1 time irregularity, not permission/latency proof); physics-guided **residual dynamics and real robot residual modeling** (Physical, not proof servo dominates persistence); Shchur et al. **intensity-free temporal point processes** (Enterprise, not novel TAC mathematical invention); established **valid-time vs transaction-time bitemporality** (Temporal, not a finished governed neural model); Lin et al. **MCUNet** (Edge-Lite hardware co-design, not measured TAC edge speed); and Dudík et al. plus Jiang–Li **doubly robust off-policy evaluation** (Counterfactual, subject to genuine real interventions/overlap and costs). Source papers and exact task-specific experiments are linked in that map.

**Rule:** reuse existing mathematics; never claim the literature's demonstrated results for TAC. Attaching a paper, proposed mathematical head or simulation cannot promote a real-data capability. Every lane's future full trained model must pass the owner’s existing source/authenticity, 95–105k learned parameter, preregistered task, checkpoint and independent replay rules; comparative architecture winners separately require matched neural resources. These six lanes are a product/research **portfolio organization**, not a commitment to fit six variants immediately. Enterprise and Physical have the first authentic-scored capability work, while System1/Temporal/Edge/Counterfactual require their own real qualification records.

## Mission and constraints
Build a persistent action-conditioned predictive world model that learns what to remember, where to store it, how to retrieve it, how dynamics evolve, when to refine predictions, and when to correct beliefs. Retain ME-054 native and ME-056 Concept-Swarm as independently testable branches. Do not replace TAC-OSM with an LLM, Transformer, MTSK, or SSM.

**Compulsory:** real-world longitudinal data and authentic observed outcomes for performance claims; synthetic data only for labeled diagnostics. Architecture comparisons match trainable parameters, data/splits, training budget/compute and evaluation. Report CPU wall time, do not force equality. Include persistence, GRU and relevant stronger baselines. Preserve untouched holdouts, seeds, checkpoints, code, provenance, logs, and results. Do not call untrained designs improvements.

## Binding model scale for all future trained TAC-OSM variants — ~100k parameters

**Effective October 23, 2026 for NEW training registrations. Target exactly 100,000 trainable parameters per independently fitted neural model, with inclusive ±5% allowed (95,000–105,000).** This is a compulsory training authorization gate, not a proposal, an average over runs, or a requirement that only TAC itself satisfy. It applies to every future native fast/slow/associative TAC, Concept-CED, GRU, RSSM, SSM, Transformer/attention, event/process world model, physical world model, memory/control ablation, trainable baseline, fine-tuned model and every seed. Count the actual trained encoder, memory/state, associative/relevance components, all forecast/action/uncertainty/decision heads, adapters and routers. Shared weights count once. **All deployed learned weights**, including frozen learned backbones, must also be within 95k–105k. No dummy padding, untrained projection inflation, hidden externally trained modules or selectively excluded heads.

**Two independent numerical conditions apply:**
1. **Absolute target:** every NEW trained model must have 95,000–105,000 trainable AND deployed learned parameters, measured from the actual instantiated model BEFORE its first optimizer step and verified against the actual model/checkpoint AFTER training.
2. **Fair comparator parity:** any TAC-vs-GRU/RSSM/SSM/CED/other trained comparison must STILL have at most **1% trainable AND deployed parameter discrepancy between arms**, matching authentic data and splits, original sample exposure, training/tuning compute/FLOPs, and evaluation. Two 40k models or two 123k models are not permitted for NEW training even if closely matched; 95k vs 105k is within the absolute band but fails the separate 1% comparison parity rule. CPU wall time remains a reported efficiency outcome, NOT a matched-budget requirement.

**Implementation and failure behavior:** Before new training, preregister the exact model/arm-by-seed parameter budget and source in `FAIR_COMPARISON.json`. Use `scripts/validate_research_parity.py --manifest ... --phase preflight`; for a solitary future trained model use `study_type: single_model_training` with each model count, and use `--phase final` to audit exact postfit real counts and checkpoint receipts. The standalone implementation counter `--model-parameters COUNT --deployed-parameters COUNT` also fails outside 95k–105k. Genuine source-only diagnostics are explicitly no-fit. **STOP if the target count, independent 1% match, real data, budget, or provenance gate fails; do NOT start fitting as an exploratory workaround and do not promote trained results on a claimed disclosure alone.** The agent-level guidance and active CI are in `AGENTS.md`, `research/REAL-WORLD-EVIDENCE-RULES.md` (binding rule 12) and `.github/workflows/research-scientific-gate.yml`.

**Preserve the historical research:** ME001–ME072 and DEV001–DEV061 remain exactly as previously executed, including ME-063/066 roughly 37k micro-world-model controls and the previously completed ME-072 ~123k event model. Do not resize their original checkpoints, overwrite evidence, or call those completed results violations. Any NEW fitting—including an extra seed or new model using a historical architecture—falls under this prospective scale rule. Reusing a historical study ID to bypass the scale check is prohibited. Different parameter scales require a new explicit owner-approved research policy amendment before training, not an undocumented exception.

**Scientific purpose:** The 100k budget is a controlled research scale, NOT evidence that growing a model from 37k to 100k by itself improves intelligence. Continue to prioritize real-state prediction, calibrated uncertainty, action-conditioned physical forecasting, hierarchical persistent memory, verified outcome correction and realistic decision value. Interpret performance only against strong, appropriately matched 100k neural controls and *unmodified* persistence/servo/analytical baselines on genuine external outcomes with full original Git-owned model bytes and independent replay. The upcoming predictive-motion model must start at this budget and demonstrate a benefit over persistence and other matched controls before consuming the final unopened robot task.

## Live parameter-enforcement update — 2026-10-23

The review found declaration-only gaps in capability-development and deferred-source
registration, plus historical-ID exemptions. [Draft PR #354](https://github.com/FPC-effortless/TAC-OSM/pull/354)
closes these in the default-branch implementation; [draft PR #355](https://github.com/FPC-effortless/TAC-OSM/pull/355)
applies the same enforcement to ME073's newer research entrypoints. **Both are
unmerged proposals: enforcement is not claimed to be integrated everywhere or
protected by mandatory branch settings.** Do not start another fit until its
actual branch has the guard and its complete preregistration passes.

The shared `src/tac_osm/parameter_budget.py` checks all registration routes and
the actual whole model immediately before each optimizer update. It records
trainable/deployed counts, optimizer coverage, arm/seed identity, named parameter
inventory and the frozen registration digest before update zero. Final validation
reads the original checkpoint tensors and their runtime receipt. A standalone
`--model-parameters` invocation is declaration checking, **not** evidence that a
live model was checked. Both counts must be 95k–105k; comparisons retain the
independent 1% requirement and authentic data/exposure/training-compute parity.
CPU time remains efficiency metadata, including for an inherited v2 schema label.

No new fit is exempt because its study number is old. Immutable historical
replay requires explicit pinned registration/checkpoint provenance, authorizes
no optimizer update, and never retroactively qualifies old training. Original
ME071/ME072/ME073 registrations, scores and checkpoint bytes remain unchanged.
The ME073 overlay restores original code for replay and blocks all optimizer
updates in that process. Existing unsupported legacy fitting lanes are manual
opt-in and still face the live gate; ordinary software/policy CI remains automatic.

Local verification: **981 tests** on the default-branch proposal (924 before the
change), **1,105 tests** on the ME073 overlay (1,043 before); guard coverage audits
passed across 88/164 Python files respectively. Different counts reflect different
branch suites. Six immutable registrations and 15 distinct original checkpoints
passed the Git-byte provenance audit. GitHub's latest root and scientific gates passed at PR354 commit `5175574c`
and PR355 commit `485e026b`. All applicable checks passed at both exact code heads, including the
ME071/ME072/ME073 pinned scientific replays with optimizer updates blocked. The initial Torch-only hashing portability failure and its
fix remain recorded, rather than erasing the failed workflow.
These are software-enforcement results, **not new trained-model capability evidence**.
See [the enforcement contract](https://github.com/FPC-effortless/TAC-OSM/blob/governance/universal-live-parameter-budget-2026-10-23/research/governance/LIVE_PARAMETER_ENFORCEMENT.md).

## Completed integrated architecture: ME073 native autoregressive trajectories

[ME073 draft PR #353](https://github.com/FPC-effortless/TAC-OSM/pull/353) retains native fast64/slow64/8x8 associative observed state and generates successive activity/time proposals using separate imagined copies. First-record context initializes the state; untimed requested-amount/registration metadata remains excluded. No SSM replacement, fixed72h cap or lowest-predicted-variance hard router.

**ME073 trained and independently replayed, 2026-10-23:** five original99,781-parameter models, six authentic FIT epochs, shared eight-step native imagined-state rollout and activity-conditional bounded incremental clocks. At six observations, conditional1/2/4/8-event accuracy81.85/68.83/62.11/53.44% over367/326/275/189 actual future records; one-step NLL.372472, proper interval score18.97h, nominal90% coverage85.99%. **Early goal failed:** two-observation accuracy40% in every seed and NLL1.416298; distant8-step probability scoring regresses relative to the historical ME072 recipe. Age16/h8 mean width636.94h remains broad. Different budgets/objectives/inference prevent an architecture-winner claim. Original checkpoints frozen100671c7 before scores190c2bc8; [independent replay39948180370](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39948180370) restored all five and recomputed8,495 case/query vectors,33,980 horizon marginals and67,960 representative weighted paths plus actual scores. These are repeated forecasts over925 exposed DEV cases. See [ME073 results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me073-autoregressive-native-trajectories-2026-10-23/research/mathematical-engineering/me073/RESULTS.md). Original TEST remains unscored by ME073; no external source/customer, long-memory, causal action or production qualification.

**Next capability priority:** improve early probability quality and jointly train useful free-running distant event/time forecasts, then qualify a frozen whole model on a new lawful authentic cohort. Do not re-fit the original models, retune on REPORT/TEST or default to a component-attribution campaign. Weighted imagined paths do not guarantee process legality or successful execution. The Woyengi integration document remains an advisory draft; no automatic SLA rerouting, customer value or investment milestone is established.

## Two parallel north stars: stronger native predictive intelligence AND investor-relevant usefulness (owner direction, 2026-10-23)

**Governing choice:** prioritize **actual integrated TAC-OSM model capabilities** and design those capabilities around high-value real-world questions that could ultimately make TAC an **investable deep-tech product**, without confusing an unvalidated commercial thesis with technical proof. Investor readiness **guides choice of problem, honest capability evaluation, pilot readiness and documentation**; it does **not** mean interrupting model development for promotional materials, replacing core architecture research with feature checklists, or requiring customers before training.

**Investor-readiness assessment:** TAC has meaningful independently replayed original trained research artifacts and enough evidence for possible **research-focused pre-seed/grant conversations**, contingent on founder/team, market and entity evidence; it is **not yet a demonstrated commercial enterprise AI product** or conventionally seed-ready on observed traction. Do **not** claim verified paying customers, live partner pilots, outside deployment ROI, production safety/service guarantees, novel exclusive IP or financing until documented. Reference: [Investor Readiness and Capability Roadmap](INVESTOR_READINESS_AND_CAPABILITY_ROADMAP.md) is the binding detailed stage/gate and evidence-tracking document.

**Candidate first market wedge (hypothesis only): enterprise workflow intelligence.** Train a persistent event/world-state model to answer: *What is the actual current state? What events are likely in the next 1/2/4/8 observed steps? When could a transition/exception/delay occur? How uncertain is that warning, and when should a human intervene or abstain?* Link each output to available-at-time evidence, state updates and subsequent authenticated outcomes. A recorded log ending is not proof that a business process was actually resolved or successful. Pilot with legally sourced customer workflows when authorization exists. Continue physical world/action-conditional research on its own merits and data, but never promise general cross-domain capability on the strength of one process log.

### Investor-relevant capability engineering priorities (integrate the whole trained model, not default attribution)

| Rank | Native world-model capability | Honest demonstrated-progress readout | Operational value hypothesis |
| --- | --- | --- | --- |
| **P0** | Persistent time-aware **online state tracking and verified corrections** | Real stream entity state under restarts/interleaving, irregular/missing observations, recorded subsequent factual outcome and online latency | Operators see a current reliable process belief, not a stale classification |
| **P0** | **Multi-horizon next-state/event forecasting** from truly known prefixes | Next 1/2/4/8-event distributions, proper scores at each horizon, input/prefix eligibility and censored/source-ended cases, hard transitions and error costs on authentic data | Earlier visibility into emerging problems and likely paths |
| **P0** | **Useful coherent event/no-event/delay uncertainty** | Log loss, Brier/ECE, time quantile pinball, proper interval score, actual coverage/width median/p90/p99, lead times, false alerts and risk-selective abstention | Predictive warnings whose confidence a human can act on; no unjustified SLA promises |
| **P1** | **Long history, selective memory and robust observation correction** | Measured benefit on authentic supported long histories, verified state read/write across time gaps; unknown future facts stay unknown | Understand long-running operations without forgetting relevant past facts |
| **P1** | **Real observed action/outcome conditioning and downstream decision utility** | Authenticated contemporaneous action/outcome/cost data, factual conditional predictions and later verified pilot decisions; identify causal effects separately if data permits | Safer prioritization/triage, not autonomous authority from observational correlations |
| **P2** | **Transfer, depth, hierarchy and mathematical scaling** | Independent new domain/source evaluation and resource receipts from real trained candidates; use methods from literature without rediscovery | Repeatable differentiated product rather than a single historical-log demo |

**Current factual anchor:** ME073 now adds independently replayed native autoregressive path forecasting with mixed development outcomes; see the completed section above. ME071/ME072 originals and all earlier negatives remain preserved. Independent business usefulness and architectural superiority are unqualified.

### Commercial evidence-gated path, running ALONGSIDE research

| Stage | Required genuine evidence | Current classification |
| --- | --- | --- |
| **R0 — research/pre-seed narrative** | Original reproducible native research, ethical source/data rights, clear customer/problem/market thesis, truthful gaps, team/IP ownership verified separately | **Technical component partially supported**; business and ownership facts unverified |
| **R1 — demonstrable specific prediction capability** | One complete frozen native model scoring useful online-available event/time outcomes on an independent untouched authentic population; hard-case and calibration reporting; versioned read-only demo/API | **NOT ESTABLISHED** |
| **R2 — partner-ready advisory product** | Model-to-service interface; authorized observations; state/timestamp integrity; monitoring, alert/abstain boundaries, access controls, privacy/security, predictable inference cost and reliability | **NOT VERIFIED** |
| **R3 — external demand and measured pilot value** | Customer discovery and documented willingness to pay; proposed goal **2–3 real design partners and >=1 structured external pilot, preferably paid**, with advance-warning lead time, false alerts, operator effort, costs and ROI measured before making claims | **NOT VERIFIED; targets, not customer counts** |
| **R4 — company and investor diligence** | Verified legal entity/cap table, founder/contributor IP assignments, license/open-source/patent strategy, dataset/contract rights, security/incident plan, credible market and unit economics, financial model, evidence-backed pitch/data room | **NOT VERIFIED** |

**Practical work order:** (1) preserve completed **ME072** original training/evaluation/replay evidence without duplicating its runs; (2) address substantive whole-model forecasting/time/retention shortcomings using real source-supported ability targets, then qualify a frozen candidate on a new untouched authentic source; (3) in parallel identify exact operations buyer/job-to-be-done through user discovery, design a small read-only operator interface over *actual* predictions; (4) seek an independent lawful prospective pilot before discussing measured commercial impact; (5) prepare investor materials only from the evidence actually collected. Source acquisition, model development, user research and demo plumbing may proceed in parallel. Do not turn customer status or blocked 256-history source into reasons to stop genuine shorter-horizon capability improvements.

**Measurement policy for every ME/DEV result:** maintain a *dual ledger*: (A) **trained model capability:** real original source/time/label availability, frozen version and checkpoint/clean replay, qualified cohort, proper accuracy/uncertainty/cost/error measurements; and (B) **relevance and commercialization:** specific operational decision, warning lead time, true/candidate customer and ROI evidence, pilot/demo readiness, license/security/IP status and investor-evidence change. Status tags: `SOFTWARE_ONLY`, `HISTORICALLY_EXPOSED_DEV`, `NEW_UNTOUCHED_REAL_QUALIFIED`, `EXTERNAL_PILOT_VERIFIED`; commercial evidence `UNKNOWN`, `HYPOTHESIS`, `OBSERVED`, `PILOT_VERIFIED`. Do not roll these into an invented success percentage. Preserve negative evidence.

**Boundaries preserved:** no synthetic primary real-performance evidence, no hidden raw personal/client data in public Git, no invented revenue/traction, no model autonomy/causality claims from logged correlated actions, no sales/deployments/purchases/company filing without explicit authority, no automatic merges. A single full-native `capability_development` experiment needs **no default ablation/GRU matrix**, but claims of architecture superiority require matched trainable parameter/data/exposure/compute and evaluation. CPU wall time remains diagnostic only. Original trained checkpoints for *every* fitted model, legal source contracts and clean-checkout replay remain non-negotiable.

## Owner research priority: model capabilities before attribution (2026-10-23)

**Binding priority and experiment-selection rule:** maximize demonstrated **capabilities of the integrated trained TAC-OSM model**. The main research loop is (1) choose one high-value missing ability on authentic real-world data, (2) integrate mature published methods and TAC-specific mathematical ideas into the full native model, (3) train actual original model checkpoints, (4) evaluate the new ability using truthful, task-matched development metrics and real observations, (5) retain improvements, then advance the next ability. **Do not require another attribution study, component lesion, micro-ablation, architecture-comparison matrix, or GRU-superiority experiment before the next integrated capability-development fit.** Negative controls and attribution are targeted diagnostic tools if a capability stalls, not the deliverable.

**Ranked next capability work following trained ME071, rather than another small isolation study:**

1. **Genuine sequence understanding and foresight.** Train the complete native model to predict multiple future event horizons, elapsed time and observed status change from truly available prefixes. Evaluate variable query ages (e.g. 2/6/16/32/64 observed events *only in real cohorts where supported*), outcome-conditioned error and likelihood, and difficult/rare non-lifecycle transitions. Historical known-length midpoint and cherry-picked surviving cases must not masquerade as online prediction. Count cases with no verified next observation, censoring and optional stopping as explicit coverage limitations rather than invent terminal outcomes.
2. **Useful, coherent time uncertainty.** Improve the *jointly trained* delay / zero-event probability / quantile head and evidence-conditional update, not merely posthoc interval width. Evaluate log loss, Brier and calibration together with quantile pinball, proper interval score, median/p90/p99 width and coverage on appropriate real observational strata. Current ME071 NLL improvement alongside worse Brier/ECE, mean 51.69-hour post-CQR width and 52.28-hour interval score are gaps to fix, not new claims of calibrated intelligence.
3. **Persistent world-state learning.** Build on actual carried fast64/slow64/8x8 associative recurrence by learning when to retain, revise and forget factual state, including irregular elapsed time and partial/missing observations. Train and score with authentic available long entity trajectories; a 32-event gradient detach and six-event query do not prove long-term intelligence. Do not block shorter-prefix capability training on unavailable 256-event histories. To claim 256-event forecasting, require actual recorded 256+ observations and audited available future outcomes.
4. **Action- and outcome-aware forecasting.** When an original real dataset supplies *at-the-time* available actions and independently observed outcomes, learn (P(\text{future observed state, outcome} \mid \text{past observed state, observed action})) and make verified-observation correction part of training and evaluation. Later aim for useful action-choice estimates, uncertainty-aware abstention and measurable operational consequences. Do **not** interpret purely observational loan activity sequences as randomized interventions, infer causal effects or claim autonomous policy value without proper intervention data and identification.
5. **Integrated scale and generalization.** Depth, concept hierarchy, trained iterative readout/correction, and longer memory are candidate improvements when motivated by actual errors, not independent endless microprojects. Reuse existing literature and training technology; do not reinvent quantile calibration, recurrent memory, credit assignment or process baselines. Qualify transferable abilities on distinct untouched original institutions/sites/tasks once a lawful authentic dataset is available.

**What counts as progress:** (a) a new, reproducibly demonstrated *end-to-end ability* or a meaningful task-matched predictive gain from the prior frozen native full model, (b) accurate proper uncertainty/risk scores and reduced severe operational errors, (c) authentic longer context or observation-shift robustness, or (d) observed action/outcome decision usefulness. Record genuine failures, coverage, original data and domain limits. One genuinely trained integrated candidate can establish developmental progress without claiming it is scientifically superior to GRU or that an individual memory module caused the gain.

**Keep scientific safeguards without letting them set the agenda:** use authentic records only for claimed real-world performance; label existing DEV and previously exposed TEST honestly; no fabricated tasks or metrics; retain original Git-owned checkpoints and independent clean-checkout replay for every fit; register truthful resources/seeds/data access before fitting. **When and only when claiming an architecture winner**, do equal-parameter, data, exposure, compute and evaluation comparisons against appropriate rivals. ME070's registered 64-event kill rule applies to the specific **TAC-memory-superiority claim**, **not** to the entire TAC-OSM capability-development program. Report efficiency separately; optimization of CPU is secondary unless it prevents necessary training or deployment.

**ME072 completed, 2026-10-23:** five original 123,377-parameter native models trained six authentic BPI2012 FIT epochs and independently replayed from Git. At six observed events, 555/925 exposed DEV REPORT cases reach the query; actual future-record denominators at horizons1/2/4/8 are367/326/275/189. Conditional activity accuracy is **82.02/68.47/59.64/54.18%**, NLL **.383476/.712665/1.067113/1.378439**. Nominal90% one-step clock coverage is **84.74%** (undercoverage); distant tails remain broad (age16/horizon8 mean width3,536h). Original weights were frozen before scoring; independent GitHub run39941010981 restored all five and recomputed all registered real-case outputs. See [ME072 results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me072-marked-multihorizon-world-model-2026-10-23/research/mathematical-engineering/me072/RESULTS.md) and [capability-to-value card](https://github.com/FPC-effortless/TAC-OSM/blob/research/me072-marked-multihorizon-world-model-2026-10-23/research/mathematical-engineering/me072/CAPABILITY_TO_VALUE.md). This demonstrates multi-horizon developmental forecasting, not an architecture winner, business completion, long-memory qualification or external operational value. The next integrated change should address early branches and distant mark/time coupling and tails, followed by independent new-source qualification. Do not refit the original ME072 models or spend original TEST for this update.

## Dual north star: investability through actual model capabilities (2026-10-23)

**Binding owner direction:** build a demonstrably more capable native TAC-OSM world model **and** make that capability the foundation of an investable enterprise AI product. These goals reinforce one another but have **different evidence gates**. Scientific progress means learned predictive ability on authentic observations; product progress means an authorized usable system; commercial progress means external customer value and willingness to pay. Never substitute any one for another. **No new ablation/attribution campaign is required before strengthening the integrated model.**

### Investment readiness verdict — current evidence, not a fundraising guarantee

- **Research/deep-tech pre-seed:** credible to **begin investor discovery and exploratory pre-seed conversations**, subject to the actual founders, IP ownership and fundraising circumstances. Trainable native software, actual real-source fits, models/optimizer/RNG checkpoints and clean-clone replays exist. No claim that any particular investor will fund the program.
- **Conventional commercial seed:** **not yet proven ready**. No verified independent paying design partner, deployed customer production pilot, repeatable measured operational ROI, product-market fit or sustainable unit economics is established by the TAC-OSM repository. Absence of a record is **UNKNOWN**, not proof no external customer exists.
- **Enterprise-scale/growth:** not ready; production security/privacy, service reliability, regulatory and data-rights diligence, incident/rollback controls, operational costs and scaled customer evidence remain unqualified.
- **Technical honesty:** ME069 genuine same-task enterprise evaluation was competitive with empirical references but lost NLL to a matched GRU; ME070 practical FIT-trained five-event history reference closed much of the one-step prior gap; ME071 trained five 93,770-parameter native models on authentic BPI2012 and independently replayed fixed-six-event DEV results (**81.58% accuracy; post-temperature NLL 0.401741 vs raw 0.420459**, but Brier/ECE and interval score worsened). Fixed-six-event development scoring does **not** prove long-memory or commercial value. ME072 now has five original trained/replayed models and actual multi-horizon developmental scores; its timing undercoverage and long tails remain unresolved. See the completed ME072 evidence above. No independently qualified scored superiority or customer benefit follows. Original BPI2012 TEST was already exposed in earlier research; original BPIC2018 long-history file access remains unqualified. No native architecture or general intelligence superiority claim.

**Investor narrative to validate, not present as accomplished:** “TAC-OSM is being developed as a persistent predictive AI engine that follows evolving operational state, forecasts likely events and delays, revises its estimates with new observations, and helps humans make earlier, evidence-backed decisions.” Begin with **one authentic workflow problem**, not a generic AGI or all-industry claim. Do not imply causal interventions or autonomy from observational next-event predictions.

### First commercialization wedge and safe product boundary

**Primary hypothesis:** an **advisory workflow-risk and delay-forecasting engine** for organizations with repeated entity-linked, timestamped cases and measurable operator costs. Example customer task: flag a case likely to stall, miss an actual documented deadline, or need inspection **early enough for an authorized operator to benefit**.

- **Buyer / champion hypotheses (to validate):** operational leaders, process-improvement teams, service/claims/finance operations, and workflow-software providers; select **one** based on customer interviews and actual lawful source access. Explicitly document buyer, user, budget, decision frequency, stakes and current workaround. Never assert a market size or customer demand without evidence.
- **Product demonstration:** a read-only or advisory API/dashboard consuming **time-cut authorized observed events**, with per-case forecast and horizons, meaningful uncertainty, update-on-new-event, provenance, case status, abstention and monitoring. Include a human-readable example showing **forecast before verified later outcome**, without reconstructing answers from hindsight or disclosing confidential records.
- **Safe integration:** Woyengi/KoreFrame/Veritas and governed execution may be potential **internal integration testbeds**, **not** evidence of actual deployment or independent customers. Woyengi maintains authoritative facts and permissions; TAC only produces bounded predictions and proposals, never edits source-of-truth records or triggers unapproved actions. Product-specific adapters stay outside the generic TAC model.
- **Scope limits:** no automatic action selection, promised SLA effects, profit savings, causally interpreted alternative treatments, unsupported target labels or autonomous operations until independent observed-action / outcome / cost evidence and authorization justify them. Start in shadow mode on true streams with documented consent, privacy and access rights.

### Investor-oriented capability research priorities (technical effort should produce customer-facing abilities)

1. **Prediction that improves an operator's next decision (P0).** Advance the **trained complete model** beyond easy next-step lifecycle completion: multi-horizon future activities/outcomes, likely case delays, verified status change, early warnings and changing-facts correction. Train and evaluate from **actually available prefixes**, with no hindsight case-length choice, and report all eligible case/query counts and coverage.
2. **Trustworthy probability and time forecasting (P0).** Measure multiclass NLL, Brier, ECE (descriptive), time MAE, proper quantile/pinball/interval scores, calibration by cohort and sharpness (median/p90/p99 widths). The current NLL vs Brier/ECE trade-off and long upper-tail widths are blockers for customer-facing risk assurances; optimize joint utility, not one headline statistic.
3. **Persistence under real operational complexity (P0).** Carry and *usefully update* state over real entity histories, late/missing observations, irregular timestamps, changing context and case completion/censoring. Promote a long-context claim only when original qualified histories truly exist. Explicitly separate trained state updates from online weight learning and from external entity-ID routing.
4. **Operationally useful risk and abstention (P1).** Using **verified real outcomes and actual business labels**, detect rare costly branches and relevant delays at actionable lead times. Quantify precision/recall, missed alerts, false-alarm burden, abstention/coverage and performance drift under time/site shifts. If actual SLA labels or monetary costs are absent, label task **proxy-only** and do not claim financial value.
5. **Action-outcome forecasting / learning from feedback (P1, data gated).** When authentic observed actions, available alternatives and verified later outcomes exist, model executed-action-conditioned factual futures and use verified feedback to correct state. Counterfactual benefits and optimal interventions need separate identification, randomized pilots or credible observational design; **do not turn simple temporal associations into causal gains**.
6. **Efficient, deployable model (P2).** Only after capability adequacy, instrument resource cost, streaming latency, safe error handling and scaling; small parameter count or fast CPU execution alone is **not an investor-ready differentiation**.

These priorities **do not forbid** a sophisticated research program in robotics, physical world models, Concept-CED or long-horizon latent architecture. They sequence **initial commercial proof** around the available authentic workflow lane while preserving cross-domain ambitions as separately evidenced future opportunities. No forced 256-history prerequisite for training genuine shorter-horizon capabilities. No new needless micro-ablation as a progress substitute. Published mathematics may be reused rather than rediscovered.

### Stage gates: from research to investors and external customers

| Gate | Concrete output / owner | Evidence that qualifies | Present status |
| --- | --- | --- | --- |
| **R0 — reproducible actual learned model** | Research: source-qualified native original weights, source/legal acquisition, fixed training/reporting identities and original score arrays | Repository-owned checkpoint/optimizer/RNG for each fitted seed, separate clean-checkout prediction/score replay; exact DEV vs sealed test distinctions | **Partial PASS** in ME069/ME071 for specified original real-data forecasts; architecture/commercial superiority **NOT ESTABLISHED** |
| **R1 — customer-relevant forecasting capability** | Research + product: one complete native multi-horizon model linked to a named operational use case | Independent previously unexposed real case/time/site cohort; eligible-case coverage; meaningful lead time, risk/time metrics, rare event errors, reliability and strong appropriately informed practical reference | **NOT QUALIFIED**; known DEV model skills exist, no new independent commercial task success |
| **P0 — working governed product demonstration** | Product engineering: minimal observable-event API and dashboard or controlled Woyengi integration | Actual before-event forecast update and replay, per-entity state, abstain/safety, provenance, access/privacy and no unauthorized action; demo is honestly marked internal vs external | **NOT VERIFIED in TAC repo** |
| **C0 — external market validation** | Commercial: named design partners, one chosen buyer segment, documented interviews/problem economics | As a **suggested initial discovery target, not universal funding criterion**: 2–3 serious external design partners and >=1 authorized structured external pilot, preferably paid; explicit consent, source provenance and signed evaluation plan | **UNKNOWN / no repository-verified evidence** |
| **C1 — measurable customer value** | Joint: controlled external pilot retrospective or shadow/live evaluation | Pre-agreed customer KPIs with denominator and baseline: warning lead time, missed delay/failure, false alerts, operator effort, actual cost of intervention, cost of service and net observed benefit; report uncertainty and negative outcomes | **NOT VERIFIED** |
| **I0 — investment / legal data room** | Founders/legal/product: dated concise pitch, technical evidence pack, business model and plan for use of funds | Correct cap table and entity docs, founder/contributor IP assignments, open-source/repo license policy, source-data/processing rights, privacy/security controls, pilot terms, realistic unit economics and milestone-based budget; expert legal review | **Not established from repo** |

**Stage gates are non-monotonic:** if a pilot shows no value, fix use case/forecast/cost assumptions before scaling. A demo is not a pilot; an internal integration is not an external partner; paid usage is not by itself proven product-market fit. Treat commercial statuses as **UNKNOWN** absent customer-verified records, not automatic FAIL or claimed PASS. Avoid hardcoded fundraising valuation, market sizing, pricing, raise size or dates without independent evidence.

### Research experiment planning and reporting — mandatory business linkage

For every next **integrated capability-development study**, add a short **Capability-to-Value Card** to the same research registration/report, alongside existing research integrity controls:

| Field | Required content |
| --- | --- |
| **Customer hypothesis** | One named workflow role/problem and the eventual decision the forecast could improve; if exploratory world-model science instead, explicitly label *research-only* |
| **Model skill being trained** | Precisely what new factual forecast/state update/uncertainty/action-conditioned ability the full native architecture is expected to gain |
| **Real observability** | Original dataset/license/hash, event and outcome definitions, observed-at availability, missingness, source boundaries and untouched evaluation design |
| **Capability success / failure** | Prior frozen full-native result on a task-matched cohort when available, actual model forecast metrics by horizon, eligible denominator, reliability/width, difficult cases and registered failure conditions |
| **Operational transfer hypothesis** | Lead time, false positive, missed event, abstention and human-action relevance; mark **UNMEASURED** until a real pilot supplies actual labels, costs and decisions |
| **Preservation** | Original Git-held weights, seed/splits/budget receipts, complete legal replay and uncertainty about real-world transfer |
| **Stage unlocked** | R0 / R1 / P0 / C0 / C1 / I0, with **PASS / FAIL / PARTIAL / UNKNOWN**, grounding link and next blocker; no product milestone claimed from model-only success |

**Scheduling/priority rule:** If a proposed experiment cannot plausibly improve a missing predictive skill or answer an investor/customer-critical uncertainty, deprioritize it unless necessary for safety or scientific validity. Prefer *one trained integrated model plus end-to-end evidence* over many tiny attribution runs. A single-architecture \`capability_development\` lane remains permitted for model training; existing **parameter/data/exposure/compute equality is compulsory only when claiming architecture comparative superiority**. All users' existing original-data, sealed holdout, checkpoints, replay and permissions rules remain binding.

### Fundraising proof package — assemble as evidence, not promotional invention

Maintain, when genuinely available: a one-page technical claim/evidence/limitation sheet linked to original commits, a live or accurately recorded product demo with before-outcome predictions, customer interview/pilot summaries with authorization, an explicit buyer/problem and pricing hypothesis, a credible security/data/IP posture, a concise pitch deck, founder/company documents and a use-of-funds plan specifying which of the above **unmet** gates new capital will close. Any headline model number must identify the precise real dataset, split, prediction query, denominator, confidence/scoring quality and whether the observation was previously exposed. No unsupported claims of proprietary rights, superior general intelligence, commercial ROI, deployed customers or authorized autonomy.

**Research-to-product pivot criterion:** do **not** halt foundational TAC research indefinitely to chase pitch materials; initiate external design-partner discovery in parallel with trained native forecasting development. Once an independently verified end-to-end forecast solves a real operator problem, invest in the minimal safe product surface and prospective pilot rather than extending another large isolated model variant. If the use case has no customer value, re-evaluate the wedge without relabeling scientific capability as commercialization success.

---

## Mandatory repository-backed model preservation and replay (2026-10-23)

**Hard rule, all local and remote/GitHub training:** before any new or continued ME/DEV/model fit is treated as complete, every applicable arm/seed's original trained checkpoint and the complete lawful, auditable model-loading and experiment-scoring replay bundle **must be preserved in this TAC-OSM GitHub repository at pinned Git commit paths**. This includes failed/rejected runs with valid recoverable weights, all dependencies and data manifests, optimizer/RNG state when resumption is claimed, and independent restoration/scoring verification from a **clean Git checkout**. Temporary Actions artifacts, caches, local `results/`, remote notebooks and hash-only receipts **never** replace actual retrievable repository checkpoint bytes. Use tracked checksum-indexed shards or fully retrievable repo Git LFS if needed. If source data cannot legally be redistributed, store pinned legal source-acquisition instructions/hashes and label any inaccessible full-evaluation replay **BLOCKED**, not PASS. No secrets, unlicensed data or private user records may be uploaded.

**Final research gate: REPO_REPLAY=PASS**, separate from real-world validity and matched-comparison qualification. If actual checkpoint bytes, loader, exact evaluation dependencies or independent clean-checkout replay are missing, status is **NOT_PRESERVED/BLOCKED; no new completed-fit, verified-experiment, model-promotion or superiority claim**. Preserve historical studies as historical with disclosed incomplete archives, never retrofit claims. This rule covers local, GitHub Actions, Kaggle, GPUs and other external hosts equally. [Full mandatory policy and file checklist](REPOSITORY-REPLAY-AND-CHECKPOINT-RULE.md).

## Company-aligned north star and scope (2026-10-22 revision)

### Current architecture-first execution: ME071 persistent event model

The owner's latest direction prioritizes making the full native model more
powerful before detailed attribution. [Draft PR #350](https://github.com/FPC-effortless/TAC-OSM/pull/350)
implements and actually trains one integrated native architecture rather than
another baseline matrix. The explicit `capability_development` registration
enforces authentic data, original checkpoint preservation, seeds and exposure;
it rejects multiple architectures and confers **no architecture-superiority
eligibility**. Future comparative claims still require existing parity gates.

**Delivered:** all fast64/slow64/8x8 associative values carry across detached
32-event chunks; width128 SwiGLU decoder; .05 label smoothing; learned log-linear
pooling with the FIT-only five-event prior; ordered log1p-second .05/.50/.95
quantiles and positive-delay occurrence; disjoint DEV temperature and CQR;
entity-separated timestamped streaming; masked assimilation and interleaving
execution tests. These reuse established methods while retaining native TAC.
State carry does not extend backward credit beyond 32 events or prove useful
256-event memory. BPI2012 still lacks adequate long-history confirmation.

**Actual training:** five original native seeds, 93,770 trainable parameters,
six complete real FIT epochs and 715,134 supervised transitions per model;
mean optimizer CPU 88.87s per model. No new trained comparator, architecture
ablation, fresh TEST evaluation or BPI2018 retry was performed. The inherited
FIT count prior's deployed bytes/hash are recorded separately.

**Development readout:** predict after exactly six observed events, using
367 eligible cases from the later DEV REPORT partition. Distinct TEMP=240
and CQR=235 eligible cases fit postprocessing. Original weights froze in Git
at `7c503e95`; calibration froze at `080976ee` before REPORT scoring.

| Same native pipeline | Raw | Frozen temperature + CQR |
| --- | ---: | ---: |
| Next-event log loss | 0.420459 | **0.401741** |
| Next-event accuracy | 81.58% | 81.58% |
| Multiclass Brier | **0.225696** | 0.234267 |
| ECE, ten bins | **0.071837** | 0.073577 |
| Waiting median MAE, hours | 2.0573 | 2.0573 |
| Nominal90 interval coverage | 91.39% | 93.35% |
| Mean interval width, hours | **48.15** | 51.69 |

NLL improves 4.45% through fitted temperature, but Brier/ECE worsen. CQR
widens two seeds' intervals and improves coverage; useful sharpness is still
unresolved. No exact chronological coverage or 50–80% width reduction is
promised. The early fixed-prefix target differs from ME069's hindsight
midpoint: do not compare 81.58% and 94.77%, or new/old widths, as a controlled
model improvement/regression. This is previously exposed DEV development,
not independent qualification or a general intelligence milestone.

On 16 already exposed real cases per seed, entity interleaving reproduces
isolated states exactly; hiding four observations worsens next-event loss.
These support implementation, not unique retention or GRU failure. The
ME070 64-event rejection criterion and authentic-source qualification
requirements remain intact for later memory claims.

All five original model/optimizer/RNG files, calibrators, primary forecast
arrays, original scores and executable replay are Git-owned. Independent
[GitHub run 39929979913](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39929979913)
restored originals and recomputed all 1,835 case vectors. A later runner
reported regenerated-preprocessing mismatch; its log is preserved and the
loader now restores exact frozen FIT preprocessing instead of recalculating
numerical priors. No weights, calibration or tolerance was changed. The
post-repair replay receipt and current test count are maintained in
[ME071 results](https://github.com/FPC-effortless/TAC-OSM/blob/research/me071-persistent-calibrated-event-model-2026-10-23/research/mathematical-engineering/me071/RESULTS.md).
Post-repair [GitHub run 39934866901](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39934866901)
passed full five-model, 1,835-vector real scoring replay with unchanged original
weights/calibration/forecasts/tolerances. **REPO_REPLAY=PASS**; **1,024 repository
tests passed**. Earlier success and preprocessing failure remain recorded.

The initial calibrated pinball-reporting error is archived separately; the
corrected score uses adjusted quantiles and leaves forecasts unchanged. An
inherited old robot CI training auto-trigger was cancelled and made opt-in;
its partial unrecovered state is explicitly disclosed, not counted as a
completed fit. Woyengi authority, actual action/cost utility, online weight
learning and language capabilities are not implied by this event model.

**Continue architecture-first:** improve the jointly learned probability and
time representation on development data, then freeze a stronger integrated
candidate for a separately sourced authentic qualification cohort. Do not
insert another diagnostic-only milestone or claim component attribution as
a prerequisite for construction. Preserve the untouched physical holdout and
the established no-synthetic-primary-evidence boundary.

### ME069 enterprise foundation (historical, preserved)

The user's latest direction prioritizes making the integrated native model more
capable for enterprise operations. ME069 therefore adds an authentic enterprise
observation lane rather than another robot-memory diagnostic. Preserve all
physical negatives and the untouched robot tissue-box holdout. Synthetic
Woyengi/Veritas logs may test interfaces; they cannot qualify predictive or
commercial ability. Claims about current revenue, valuation, fundability or
being one pivot from commercial readiness remain unverified.

[ME069 draft PR #348](https://github.com/FPC-effortless/TAC-OSM/pull/348)
implements categorical activity/resource encoders, elapsed-time decay of native
64D fast/64D slow/8x8 associative state, a learned repeat/empirical-transition/
neural event mixture and a hurdle-lognormal waiting-time distribution. It
retains TAC's native dynamics. Standard predictive-process monitoring,
time-aware recurrence and probabilistic heads are established prior art, not
claimed inventions. State dimensions are configurable resource choices, not
proof that earlier 32D state was inherently inadequate.

The original BPI Challenge 2012 loan-process log is publisher-hash verified:
13,087 cases, 262,200 events; a chronological, outcome-availability split yields
7,096 FIT, 1,850 DEV and 2,618 TEST cases. FIT-only vocabulary and priors, no case
ID or future event features. Five paired seeds receive identical schedules,
800 updates and dense supervision over at most 32 observed events. Native has
80,296 parameters versus GRU 80,966; common supported-operator training FLOPs
differ by approximately 0.284%. Final registered parity PASS under that
profiler convention; full hardware FLOPs remain uncertified. Native mean
optimizer CPU is 35.251s versus GRU 13.777s; CPU is an efficiency measurement.

| Frozen temporal TEST | Native TAC | Matched GRU | Empirical transition/time reference |
| --- | ---: | ---: | ---: |
| Next-event log loss | 0.122626 | **0.119182** | 0.348946 |
| Next-event accuracy | 94.77% | 94.77% | 84.76% |
| Waiting-time MAE, hours | 6.3561 | 6.3618 | 6.4953 |
| FIT-threshold long-wait proxy Brier | 0.070188 | 0.070767 | 0.077039 |

The whole TAC predictor demonstrates useful next-event forecasting relative
to the empirical reference (64.86% lower log loss, 5/5 seed wins). It does
**not** establish native superiority: TAC log loss is 2.89% worse than GRU,
1/5 seed wins; paired case-bootstrap TAC-minus-GRU 95% interval
[+0.001552, +0.005243], conditional on fitted seeds. The registered joint
superiority gate **FAILS**. Nominal 90% interval coverage is 92.28%, with a
wide 230.12-hour mean width; useful operational uncertainty is not qualified.

**Original-environment REPO_REPLAY=PASS; cross-CPU numerical exception retained:** all ten original model/optimizer/RNG checkpoints were
Git-owned at `491f085b` before scoring, restored from a separate clean checkout,
and all 44,680 DEV/TEST case-score records were independently recomputed
bit-identically. Original score commit `7b458208`; no retraining replacement.
GitHub's different CPU runner subsequently failed the fixed waiting-time
per-record tolerance at DEV / GRU seed 69005 / row 520. Preserve that numerical
portability exception and the original tolerance; no all-environment replay
pass or model promotion is claimed.
Subsequent GitHub run `39905632298` passed the identical fixed tolerance without
a numerical/model change; this does not erase the original failing run
`39905268211` or establish uniform CPU portability.
Latest code-validation run `39905869711` printed a bit-identical DEV/TEST
replay PASS receipt. The second failing run `39905753892` reported a
1.221ms waiting-error discrepancy against a 0.226ms permitted difference.
Both failures and successful receipts are repository-owned; no tolerance was
relaxed and no original score was replaced.
The full repository regression passed 1,005 tests. See the
[results and limits](https://github.com/FPC-effortless/TAC-OSM/blob/research/me069-authentic-workflow-world-model-2026-10-23/research/mathematical-engineering/me069/RESULTS.md)
and [original checkpoints](https://github.com/FPC-effortless/TAC-OSM/tree/research/me069-authentic-workflow-world-model-2026-10-23/research/replay-bundles/ME069).

**Next architecture objective:** improve the integrated event model's history
representation and waiting-time sharpness using FIT/DEV only, then preregister
qualification on an untouched authentic cohort/source. This midpoint TEST is
now exposed and cannot supply another prospective claim. Keep a strong
empirical process prior and matched GRU in the results table; improve the
complete system rather than inserting a diagnostic-only stage after every
failure. Do not assume adding width, hierarchy or reasoning loops will help.

**Unchanged capability boundaries:** this bounded observational forecast does
not establish lifetime state, learned actions, actual SLA failure, verified
process completion, monetary benefit, causal effects, System 2 reasoning or
Woyengi deployment. Woyengi owns authoritative state and execution permission;
TAC emits advisory predictions. Actual operational actions/costs/outcome
verification remain required for subsequent decision-intelligence stages.

**North star:** build TAC-OSM into a reusable *learned predictive-state and decision-intelligence engine* for real operational systems and authorized autonomous software. It must maintain useful beliefs across change, forecast *observed* futures under available action contexts, recognize uncertainty and information gaps, choose or recommend cost-aware interventions, and improve after independently verified outcomes. This is a scientific target, **not a claim of current intelligence or product readiness**. The primary commercial proof is measured improvement over the best appropriate simpler solution in an independently held-out, authentic operational setting.

**Portfolio review scope:** the connected GitHub owner `FPC-effortless` was reviewed 2026-10-22. GitHub-confirmed forks and the owner's online-banking, crypto and Paws & Claws projects are excluded from this strategy. Repositories are possible components or test beds, not evidence that integrations or deployments exist. An app, codebase, mock, synthetic world or README cannot stand in for independently collected real outcomes.

### System boundary — do not duplicate the rest of the company

| Project / research family | Appropriate responsibility | TAC-OSM interface / restriction |
| --- | --- | --- |
| [Woyengi](https://github.com/FPC-effortless/Woyengi) | Identity-aware, authoritative, reconstructable organizational and agent records, provenance and permissions | Read *authorized time-cut snapshots/events*; write **proposals/predictions**, never silently overwrite canonical truth or bypass authority. Learned belief is distinct from authoritative state. |
| [KoreFrame](https://github.com/FPC-effortless/KoreFrame) | Agent runtime, tool scheduling and execution | TAC predicts action outcomes / recommends a bounded next action; runtime executes only if separately allowed. |
| [Veritas](https://github.com/FPC-effortless/veritas) | Executable operational worlds, independent agent evaluation, rollout and evidence packaging | Use for **reproducible engineering diagnostics and replay**. Generated worlds are synthetic; only separately collected real-world evaluations can promote TAC capabilities. |
| SafeEdge and AegisFlow | Permission-bound action enforcement, security, policy and audit trails | Safety/governance remains **external hard constraint**: confidence scores never grant authority. Log proposal, authorization, action and result. |
| BusinessData / knowledge-refinery | Versioned, policy-bound ingestion and retrieval | Supply legally available observation/context with timestamp and provenance; TAC learns relevance and state, does not certify source truth. |
| [Nsat](https://github.com/FPC-effortless/Nsat) | Nigerian market, economic, weather and Earth-observation data engineering | Candidate *authentic forecast/transfer* lane (time/site/market split); **not** action-effect or policy-value evidence without actual interventions. |
| Shipping / Scrappy-HCSE / other operational apps | Potential logistics and workforce application surfaces | Candidate measurement/pilot paths **only if** actual entity-linked temporal events, action logs, costs and verified results exist. No deployment assumption. |
| [TAC-Transformer](https://github.com/FPC-effortless/TAC-transformer), [TAC-Prime](https://github.com/FPC-effortless/TAC-Prime), [ATLAS-CASSA](https://github.com/FPC-effortless/ATLAS-CASSA), PSN2, Neural-Field-Dynamics, USEF-X, CDL, PNDS-Lean | Comparative architectural hypotheses: reusable structures, hierarchical latent reasoning, concept dynamics, emergent specialization, mathematical proofs | Import **specific falsifiable insights only**. No automatic fusion, architecture replacement or component novelty claim; keep every imported idea's provenance and matched control. |
| Product Bible / education / sports / consumer prototypes | Potential later product experiences | Not first-line evidence until legal data, entity continuity, measured outcomes and consent/permissions are independently established. |

**Proposed integration contract** (design only):
`authorized {entity_id, event_time, observed_at, provenance, observation, executed_action?, observed_outcome?, measured_cost?, available_actions, constraints}`
→ TAC `{belief_version, horizon, factual_forecasts, uncertainty, proposed_action_or_inspection, predicted_value?, abstain_reason?, evidence_refs}`
→ governed executor / human approval → signed actual execution receipt → independent outcome verification → delayed TAC update.
Explicitly model missing actions, delayed/censored feedback, abstention and unknown costs. No permission transfer via model output, and no model-invented verified result. For causal claims add randomized assignment/propensity and support/identification fields.

### Portfolio-specific first product hypothesis

1. **First-choice integration:** Woyengi + KoreFrame + SafeEdge/Aegis + Veritas for long-running *authorized workflow decision support*. Target actual delayed completions, failure/intervention cost and evidence-backed stopping; demand real operational logs or authorized prospective collection before claiming improvement.
2. **Second candidate:** logistics/shipment risk and next-action recommendations only where actual time-ordered shipments, interventions and verified delivery outcomes are available.
3. **Third candidate:** Nsat regional/market forecasting and across-site transfer as a separate **prediction-only** lane, not a surrogate for decision intelligence.
4. Workforce/other applications follow domain-data and legal qualification. Do **not** build bespoke product logic into the TAC core; use explicit domain adapters. Avoid implying a dataset exists just because a related app or data pipeline exists.

### Capability scorecard (evidence status, not a product-completion percentage)

`ME` studies and DEV variants often sit in **separate open draft PRs**, not a merged deployable checkpoint. Classifications below are evidence across research lineages, not an assertion that one model possesses the whole set.

| Priority | Target capability | Evidence as of branch review | Promotion gate |
| --- | --- | --- | --- |
| P0 | Accurate persistent world-state belief | **Partial:** recurrent/entity-aware learned state and long-history tests exist; identity transfer and robust retention fail some gates | heldout real entity/time continuity and better calibrated future/error performance versus matched recurrent and structured baselines |
| P0 | Real action-conditioned factual forecasting | **Partial, trained ME061:** source DEV TAC MAE .603894 vs persistence .702952 and GRU .602600; different-task TAC .493206 vs GRU .509535 but persistence .455992. Both superiority gates fail | sealed real factual multi-horizon outcomes, persistence/servo priors and fully resource-audited matched GRU |
| P0 | Cost-sensitive decision / stop / inspect / abstain | **Partial synthetic only** for learned multi-step action/stop; randomized ME049 chose the same fixed action | verified real intervention/action support, uncertainty-aware decisions, net cost or reward improvement versus best fixed/no-action policies |
| P0 | Uncertainty and value of information | **Partial, measured ME061:** positive forecast scales and source split-conformal intervals implemented; nominal 90% whole-episode coverage reaches 85.79% on DEV with broad intervals. Useful uncertainty/inspection value unresolved | out-of-time calibration, selective-risk/coverage, paid-inspection utility and safe abstention on genuine records |
| P0 | Verified closed-loop belief correction | **Partial engineering:** masked bounded corrections exist, and synthetic learned controllers replay; no qualified real improvement | independent real outcome receipt, strictly delayed pre-feedback update, error reduction without overwriting verified facts |
| P1 | Entity and relationship representation | **Partial synthetic:** entity-local DEV061 98.5% completion, below structured GRU 99% and public MLP 100%; unfamiliar IDs fragile | unseen-real-entity generalization, interference reduction, true identity continuity |
| P1 | Decision-preserving selective long-term memory | **Partial:** routing/consolidation mechanisms exist; ME051 old-event gains inconsistent, GRU stronger | controlled real long-history decision/forecast gain and verified forgetting/interference trade-offs |
| P1 | Multi-horizon latent and hierarchical abstraction | **Partial architectures:** fast/slow/associative and concept-CED prototypes; added depth does not imply thought | consistent real future/policy improvement across horizons over simpler matched controls |
| P1 | Continual adaptation under shift | **Not established:** recurrent *state* updates are not online *weight learning* or useful adaptation | authorized sequential real site/time adaptation without leakage, forgetting or unauthorized side effects |
| P1 | Transferable state intelligence | **Not established:** ME047 new-site inference executes but usefulness/persistence gates fail | sealed different site/task transfer with advantage over strong target-independent references |
| P2 | Multimodal grounding | **Not established as native understanding:** structured events, actions and sensors exist | properly time-aligned independent audio/image/document observations and measured downstream lift |
| P2 | Selective computation at scale | **Partial mechanics only:** actual routing/dispatch and timing exist; cost-law C5 untested | end-to-end relevant-subset scaling including index/addressing without loss of accuracy; real latency and resource receipts |

**No capability is yet qualified as a cross-domain, commercial TAC-OSM advantage.** DEV058/DEV061 synthetic success remains diagnostics. The honest real-data negatives remain prerequisites, not inconveniences to optimize away.

## Existing baseline (source review, verify at pinned commits)
ME-054 persistent fast32 + slow32 + associative 8x8 = 128 state elements. ME-056 adds 6x15 concept states = 218 elements; 4 shared-weight CED passes. State elements are not parameter counts. Both use gated recurrence, associative delta updates, bounded feedback. Neither explicitly models elapsed time. The 2026-10-22 reviews report real developmental forecasting losses to persistence; these are not independent confirmations of fused ideas.

Native associative update: M_next = rho*M + beta*(v-rho*M*k)*k^T, rho=0.95+0.0499*sigmoid(r), initial ~0.99. Unrefreshed contribution at 512 steps: 0.99^512 ~ 0.0058. Write interference into another key j: delta(M*k_j)=beta*e*(k_i^T*k_j), excluding retention. Native observation updates additionally tanh-compress M. The physical prediction head x_next=x+D*f(fast,slow,action) does not directly consume an action-conditioned associative read.

ME-056 concepts communicate by softmax((q_i dot k_j)/sqrt(d)+bias_ij), weighted value sums. Uniform routing and concept collapse are risks, not established failures. CED pass: q_next=q+0.1875*g*(candidate-q). Bounded interpolation does not imply contractive convergence or improving forecast at every pass. Initial preflight: CED 45,822 parameters and 143.60M profiled training-update FLOPs; initial GRU 45,995 parameters and 55.54M FLOPs (~2.59x). This is before registered GRU refinement and not ME-054 versus ME-056.

## Integrated architectural hypotheses (not implemented)
1. **Memory preservation:** learned multi-timescale retention within compact memory, responsibility-weighted delta writes, entity-local updates only where persistent identities are evidenced. Avoid MTSK unrestricted correction (DEV-035 reported state-difference amplification).
2. **Memory retrieval:** key collision diagnostics, direct action-conditioned associative read into physical predictor (r=M*normalize(a), fast_bar=fast+W*r), compare with indirect read. Candidate norm projection M_next=M*/max(1,||M*||_F/C) versus repeated tanh compression.
3. **Working versus persistent state:** fast transient dynamics; slow evolving context; associative retrievable knowledge; concept working representations; bounded verifier. These are learned roles, not imposed semantic labels.
4. **Concept specialization:** measure effective rank, diversity, routing entropy and causal forecast contribution before trying sparse/competitive routing or physical/action/memory-conditioned cross-attention. Softmax is not inherently incorrect; identical scores across nine entries give each 1/9 weight, not evidence of the same TAC failure.
5. **Useful iterative depth:** inspect factual physical errors at each of four CED passes; optionally supervise intermediate forecasts and learn bounded adaptive stopping against measured predictive benefit, not latent movement alone.
6. **Hierarchical latent-first world prediction:** global environment state, persistent relational/entity state and fine event residuals; predict future latents conditioned on *executed* actions then decode physical measurements. Prevent latent collapse and target leakage.
7. **Elapsed-time dynamics:** if real timestamp irregularity matters, compare decay exp(-lambda_t*delta_t) and input-dependent liquid-inspired dynamics to discrete recurrence. Borrow mathematics, not wholesale architecture replacement.
8. **Verification and decision intelligence:** action-conditioned multihorizon forecasting, grounded bounded correction, counterfactual decision-preserving memory retention when data supports identification; avoid claiming unobserved action effects are measured.
9. **Scalability:** batched operators, valid chunked recurrence, useful computation per CED pass; concept exchange approximately O(N²d+Nd²) per pass. Parameter sharing does not eliminate FLOPs.

## Candidate unified objective (research proposal, not current loss)
L = sum_h w_h[lambda_x*L_physical(h)+lambda_z*L_latent(h)] + lambda_o*L_outcome + lambda_v*L_verification + lambda_r*L_retrieval + lambda_c*L_concept + lambda_d*L_decoder + lambda_b*L_budget.
Introduce only justified terms in controlled ablations; prevent future-information leakage.

## Prior evidence and known limitations
- ME049 genuine three-arm randomized dataset: trained TAC's heldout action selection collapsed to the training-best fixed Mens email action, yielding no personalized policy benefit; context achieved better Brier (PR #314). Do not treat action-head existence as real decision intelligence.
- ME051 genuine MovieLens H256: oldest-history TAC MAE 0.787059/0.780177 across two seeds, recent TAC 0.791441/0.779113; GRU controls 0.775609/0.776760 best recent (PR #317). Old-history benefit inconsistent; cross-model capacity not parity-qualified.
- ME047 executed frozen cross-site ETTm2 inference but failed useful forecast-transfer gates (PR #312). 
- ME057 fused completed a five-seed real-robot comparison and failed its registered superiority gate (PR #338); ME059 remains a separately registered lineage whose result must be checked independently (PR #337).
- DEV-016 joint training reported 73.64% versus 59.32% frozen writer, but reliability gate failed.
- DEV-017 structured 41-value memory reported 91.84% versus TAC 57.90% future prediction and 38.3% unnecessary interventions; historical protocol details must be checked before cross-experiment comparison.
- DEV-020 audited structured state/read probes at 100%, TAC near chance; do not infer a unique causal defect.
- DEV-043 preserved pairwise ranking did not translate into hard cost-sensitive action accuracy.
- DEV-058 learned controller had strong synthetic completion but imperfect repeat seed; synthetic results do not establish real-world success.
- DEV-061/R1 reported strong synthetic completion; real-world forecasting still lost to persistence.
- ME-054 recovered developmental MAE 0.07881 versus matching GRU 0.09232; ME-056 0.08037 versus matching GRU 0.09082; persistence 0.06695. Check provenance, data splits, parameter and compute parity before claiming fair superiority. Neither beats persistence.
- ME-056 full fair GRU comparison not established by the initial preflight. Do not confuse early preflight with later experiments.

## Ordered integrated capability program — supersedes the previous component-first sequence

Run **a small number of complete preregistered capability experiments**, not an endless queue of memory-gate/decoder microvariants. Each stage begins by reusing the strongest valid prior result and relevant established literature (R2D2, DreamerV3, TD-MPC2, modern recurrent/SSM baselines, offline-policy identification). Component ablations are **conditional failure diagnosis**, not default milestones.

**Gate 0 — Evidence reconciliation and real-data authorization (no new architecture claim).**
- Reconcile **separate lineages**: ME054 native, ME056 *Concept-CED* PR332, ME056 *anchored-native* PR334, ME057 fused PR338, ME058 source preflight PR336 and independent-task ME059 PR337. Do not conflate the duplicate ME056 names, source/test exposure, FLOP receipts or scores. Read their latest branch result artifacts before updating this plan.
- Preserve original ME054 developmental robot MAE 0.07880869, GRU 0.09231612, persistence 0.06694929; the task's test was already exposed and full training-FLOP parity is not certified. **Do not call this a matched independent TAC win.** Independently verify any Concept-CED MAE attribution against actual training artifacts before promoting it.
- ME057 PR338 reports 38,268 TAC vs 38,295 GRU parameters and profiled-update 42,640,106 vs 42,784,699 FLOPs. Its completed 80-episode TEST means are TAC 0.663284, GRU 0.658526 and persistence 0.672681 original-unit MAE. TAC won one of five seeds; the episode-clustered TAC-minus-GRU interval was +0.000285 to +0.009266, so its registered gate failed in GRU's favor. ME059 PR337 remains separate and must not inherit this result. The ME058 PR336 source preflight is not ME059 final training; partial profiler FLOPs are not a complete audited training-compute certificate.
- Before a new fit: freeze dataset licence, authentic source hash/receipt, original chronological entity IDs, known action availability, missing/censored outcomes and exact source-specific time-cut. Mark **prediction-only** datasets separately from supported real-policy datasets. Confirm PR status and original artifact outcome rather than inferring from a workflow YAML.
- **Accept:** auditable qualification ledger and real observations with eligible target definition. **STOP:** identity/actions/costs missing for the proposed estimand; missing parity manifest; invalid/test-exposed confirmation.

**Gate 1 — Prove factual real-world prediction and stable state (P0).**
- Use independent authentic multistep physical-robot trajectories as the immediate first test, following the sealed ME059 training/validation contract only when every preregistration, original data and true compute receipt is qualified. Evaluate future physical state (and human takeover only with enough positive support), *pre-feedback only*. Compare against last-state persistence, servo-only physical prior, GRU and a contemporary appropriately budgeted recurrent predictor.
- Score MAE/MSE per physical channel and horizon, divergence/exploding state, calibration, long-history degradation, missingness and independent new-task generalization. Include factual action conditioning with the *executed recorded action*, never label potential outcomes as observed.
- **Accept:** predeclared independent holdout, repeatable improvement over strongest practical reference and fully qualified matched neural comparison where claimed. **STOP:** still worse than persistence/servo reference, post-feedback leakage or unequal fairness budgets; diagnose root failure before building another architecture.

**Gate 2 — Show real decision-relevant memory and learned entity structure (P0/P1).**
- On a new authentic longitudinal entity dataset with verified IDs/times, test whether earlier facts materially change later observed outcome/decision quality. Compare identical-input equally resourced structured GRU, memory-free and direct-history baselines; include hard interference, ID permutation, unseen IDs and delayed contradictions.
- Favor *future utility* over reconstructing an old summary. Only activate proposed direct associative reads, retention/responsibility writes, entity-local indexing, cross-concept messages or extra hierarchy if a prespecified failure demands them. Preserve separate native and Concept-CED arms.
- **Accept:** consistent sealed heldout improvement in factual prediction or supported decision utility, not merely probe accuracy or synthetic retention.

**Gate 3 — Establish action value, uncertainty and safe intervention (P0).**
- First acquire/qualify authentic supported **action → measured outcome → actual cost → attribution** chains with known available choices and propensities or justified identification. If such data are absent, continue **prediction-only** reporting without claiming policy intelligence. Historical ME049 RCT was valid randomized one-step context but native selected the same fixed Mens email arm for every test case (no personalized policy gain).
- Estimate uncertainty/calibration, expected costs/benefits, information value for inspect/wait, hard action support and abstention. Optimize over the **governed available-actions set**; compare best fixed action, no action, strongest bandit/GRU/policy references with appropriate off-policy variance checks. Distinguish predictive association, policy value and causal effect.
- **Accept:** out-of-time/entity or independently randomized real policy net utility gain, cost distribution, risk coverage, confidence bounds, no unauthorized actions. **STOP:** unsupported counterfactual extrapolation or fictional decision prices.

**Gate 4 — Verified closed-loop learning and useful deliberative depth (P0/P1).**
- With actual outcome receipts, verify that correction changes only beliefs that evidence contradicts, preserves authoritative facts, and improves *subsequent* registered real predictions/actions. Demonstrate delayed/out-of-order outcome handling and bounded failures.
- Then test fixed versus adaptive repeated latent refinements or hierarchical/global/entity/event representations **only if** they improve measured external predictions or net decisions under the same compute/data rules. A four-pass decoder, latent movement or teacher imitation is **not** System-2 intelligence.
- **Accept:** independent temporal improvement after feedback with bounded unsafe/costly actions and demonstrated incremental value from extra reasoning.

**Gate 5 — Transfer and scalable serving (P1/P2).**
- Evaluate independently unseen sites, customers and relevant data modalities. Separate zero-update transfer from few-shot adaptation and from continual weight updates; preregister unseen shifts, data rights, retraining allowance, baseline and forgetting checks. ME047's existing cross-site ability to execute is **not** transfer success.
- Only once capability thresholds pass, assess genuine relevant-subset compute scaling (**C5** including addressing/index build, reads, execution and verification), CPU/GPU memory, latency, energy and portability. Measure real throughput without demanding equal CPU wall time in scientific comparison.
- **Accept:** independent transferable utility, stable calibration and operational service-level qualification; otherwise preserve the failed gate.

**Gate 6 — Company pilot and product qualification (after scientific gates).**
- Select **one** legally authorized customer/workflow wedge from Woyengi/KoreFrame/Veritas, logistics or Nsat. Use a time-cut read-only integration first; then shadow recommendations; then small approved governed interventions with independent outcome collection. Maintain product-specific SLAs, rollback, human escalation, data minimization and signed audit receipts through the appropriate external platform.
- Measure improvement against the incumbent workflow **including full cost of collecting data, model execution, human inspections and failed actions**. A working demo or synthetic Veritas world is not commercial proof.
- **Accept:** multi-period customer-relevant improvement, lawful source rights, verified real actions/outcomes and secure rollout; only then discuss commercial readiness.

### Minimal common reporting card for EVERY stage

**Additionally enforce the mandatory repository checkpoint/replay gate**: every reported fit has tracked restorable checkpoint bytes in a pinned TAC-OSM commit plus environment, exact loader, score/replay script, SHA manifest, permitted data reacquisition and independent clean-checkout validation; no ephemeral-only assets. Missing pieces are BLOCKED, and a local directory or Actions artifact never makes a result complete.

Record: task question and owner; branch and exact commit; original external data licence/source/hash; entity/time cutoff and actual accessible features; preregistration; risk/support/censoring; strongest practical baselines; exact trainable/total parameters; real training data/order/tokens/updates; **actual audited training compute and tuning**; sample counts, >=2 seeds where the policy requires; sealed DEV/TEST access chronology; predictive metrics and interval; real decision cost/policy identification when applicable; calibration/abstention; full inference cost; raw checkpoints/logs; all failed attempts; reproducible CI; PASS/FAIL/UNKNOWN for each scientific gate. An unmeasured output is UNKNOWN, not PASS.

**No automatic model training, merge or deployment is authorized by this planning document.**

## Research provenance, interface decisions and ownership

The company-facing mapping above is a **research/product hypothesis**, not permission to modify those repositories. Product teams own canonical state, agent execution, security and customer-specific data. TAC research owns belief dynamics, action predictions, loss contracts, uncertainty and model scores. Veritas may audit deterministic replay, but only authentic operational data can qualify real performance. Keep model recommendations distinguishable from authorized execution in code and evidence.

## Provenance and research hygiene
Every idea needs ID, origin, repository paths, mathematical specification, claimed benefit, counter-hypotheses, implementation commit/PR, datasets and splits, controls, budgets, seeds, artifact links, quantitative outcomes, limitations, and status. Do not overwrite negative results. Keep exploratory tests separate from confirmatory holdouts.

Source entry points:
- ME-054: https://github.com/FPC-effortless/TAC-OSM/blob/research/me054-real-physical-residual-stable-feedback-2026-10-22/src/tac_osm/predictive_state_agent.py
- ME-056: https://github.com/FPC-effortless/TAC-OSM/pull/332
- Historical DEV PRs: #172, #173, #176, #217, #220, #221. Revalidate precise linkage and metrics before treating summaries as independently audited.


## 2026-10-22: Anchored-native ME-056 and successor ME-057 evidence correction

Two separate ME-056 study lineages exist: concept-CED ([PR #332](https://github.com/FPC-effortless/TAC-OSM/pull/332)) and 128-frame anchored-native ([PR #334](https://github.com/FPC-effortless/TAC-OSM/pull/334)). Never aggregate their parameter counts, source splits or outcomes.

Anchored-native PR #334 reports 38,266 TAC vs 38,295 GRU trainable parameters (0.076% difference), 42,257,505 vs 42,546,619 profiled FLOPs/update (0.680% difference), and an 8.007% paired CPU overhead (95% CI 6.353–9.999%) against the historical 5% CPU gate. This is an engineering result, **not a trained model improvement**.

The owner-confirmed CPU rule is diagnostic-only. Successor [draft PR #335](https://github.com/FPC-effortless/TAC-OSM/pull/335) implements prospective scientific parity-v3, preserving historical v2 failures. The successor audits all 228 authentic FIT robot episodes before new training registration. Its separate source-only diagnostic manifest explicitly forbids model fitting until original source hashes and a complete comparison plan have been verified.

**Critical discovered information-boundary issue:** PR #334's dense auxiliary snapshots were saved after the same transition's actual next-state feedback. This can leak the supervised target into the state representation. The ME-057 successor saves pre-feedback snapshots and tests that mutating an unseen future target cannot change its own input. Historical resource receipts remain valid as descriptions of the old execution, not validated measures of corrected candidate training. Reprofile and train only after source qualification and complete preregistration. No new predictive performance result follows automatically.

Primary capability gate remains authentic future-state MAE versus persistence and a genuinely equal-resource GRU under complete audited real data/compute exposure. After-takeover outcomes and uncontrolled action effect claims remain outside the supported causal estimand.


### ME-057 FIT source qualification completed — audited real data, not training

Original GitHub Actions qualification [run 39836744099](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39836744099) passed. Its downloaded artifact 12451242831 was parsed, and the canonical per-episode receipt hash independently recomputed.

Verified original FIT source: 228/228 parquet episodes, 329,306 measured frames, 210,192 control-eligible candidate origins, and 375 actual takeover onsets. The canonical original source FIT evidence SHA256 is `c4064651e91cced7aafc11337c32a27266f470ea10388bb0dd2562ae955f88fd`. DEV and TEST trajectory data stayed unopened; no model trained in this phase. Detailed evidence is preserved at [research/ME057_AUTHENTIC_FIT_SOURCE_AUDIT.md](https://github.com/FPC-effortless/TAC-OSM/blob/research/ideas/research/ME057_AUTHENTIC_FIT_SOURCE_AUDIT.md).

Also recovered and byte-verified all ten archived ME-054 original robot checkpoints from workflow 39781336855. Authentic developmental original TEST mean physical MAE: TAC 0.07880869, GRU 0.09231612, persistence 0.06694929; original TEST previously exposed, full training FLOP parity unverified, and takeover labels very sparse. See [ME054 artifact audit](https://github.com/FPC-effortless/TAC-OSM/blob/research/ideas/research/ME054_RECOVERED_ARTIFACT_AUDIT.md).

**Next genuine research gate:** reprofile the corrected pre-feedback dense objective on verified FIT; freeze a prospective matched comparison with identical real trajectories/target access, active parameters, profiled and budgeted training FLOPs, paired schedules, seeds and final evaluation. Do not promote model performance until actual original new-task FIT training and heldout evidence exist.

### ME-058 original FIT parity preflight passed; ME-059 real fit preregistered

Source-backed original FIT comparison [GitHub Actions run 39840368795](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39840368795) passed with 228 original physically measured FIT episodes, original per-episode source SHA `c4064651e91cced7aafc11337c32a27266f470ea10388bb0dd2562ae955f88fd`. It reprofiled the **corrected pre-feedback causal dense objective**; do not substitute the leaked PR #334 preflight.

| Original-source corrected one-update profile | Anchored TAC | GRU |
|---|---:|---:|
| Active trainable parameters | 38,266 | 38,295 |
| Reported operator FLOPs | 42,257,730 | 42,546,844 |
| Instrumented CPU time, efficiency only | 0.87256 s | 0.64799 s |

Gaps: parameters 0.076%, reported FLOPs 0.680%; the prospective numeric preflight passed both 1% thresholds. Whole true training FLOPs are not independently certified. The original FIT-only normalization/servo prior SHA is `3d66d00ba01f345d7fcf3c23149f808986da795b1dea07ead4520b8bafd8af96`. Five outcome-independent paired sampling schedules for seeds 58001–58005 are frozen in the preflight receipt.

A new genuine original task training [draft PR #337](https://github.com/FPC-effortless/TAC-OSM/pull/337) prospectively commits the source/feature/target/optimizer/evaluation contracts, identical real source windows and 600 updates per arm per seed, and a genuine ten-model FIT training workflow. Strong physical persistence and servo-only priors will be scored on factual new-task DEV/TEST **only after all ten checkpoints freeze**. The anchored TAC has a disclosed extra 128-FIT-transition auxiliary while the GRU uses the same observed original physical frames without that gradient loss; improvements cannot be attributed solely to native memory. This protocol is a machine-qualified developmental experiment, **not a certified architecture-superiority result** until complete true FLOP parity is independently measured.

As of this documentation commit, no ME-059 checkpoint result or new source TEST predictive MAE is yet available. Keep that distinction explicit and update only with original run artifacts.

### ME-057 fused predictive-anchor training completed — useful dynamics, no TAC advantage

[Draft PR #338](https://github.com/FPC-effortless/TAC-OSM/pull/338) completed five paired 600-update fits per architecture on the independently reserved `clean_stains_off_tabletop` task. All ten checkpoints froze before DEV/TEST opened. Both trained recurrent systems beat physical persistence on the 80-episode TEST cohort, showing that this task contains useful action-conditioned temporal structure.

| TEST original-unit physical MAE | Result |
|---|---:|
| Native fused TAC | 0.663284 |
| Matched GRU | 0.658526 |
| Persistence | 0.672681 |

TAC improved 1.40% over persistence but was 0.72% worse than GRU, won one of five paired seeds, and had a +0.000285 to +0.009266 episode-clustered TAC-minus-GRU 95% interval. Its preregistered requirement of at least 2% improvement over both GRU and persistence therefore failed. Stable eager TAC also required about 56% more optimizer CPU than GRU in the completed local fits.

The Actions run evaluated heldout data and then failed only while serializing a NumPy seed-win scalar. The committed result was recovered by rescoring hash-verified checkpoints without refitting; the workspace could not independently download the Azure-hosted Actions artifact, so local checkpoint hashes are not presented as a byte-for-byte audit of that archive. See the PR's completed-results receipt and retain this provenance distinction.

**Architecture consequence:** preserve the bounded, causal physical feedback and action-conditioned rollout that now beat persistence. Do not promote the fused anchor as the competitive core. The next complete candidate must improve the *forecasting computation itself*: make the strong servo/physical prior and current observation a direct base prediction, then learn a bounded horizon-conditioned mixture of fast-state, slow-state and associative residual experts with calibrated uncertainty/abstention. Train the entire system against factual multihorizon physical outcomes under matched parameters, authentic windows and profiled FLOPs. Do not spend another stage merely probing ME057's state; use ablations only if the stronger integrated candidate fails.

### 2026-10-23: ME060/ME061 implemented and trained capability tranche

[ME060 draft PR #339](https://github.com/FPC-effortless/TAC-OSM/pull/339) and
[ME061 draft PR #340](https://github.com/FPC-effortless/TAC-OSM/pull/340) implement
and train complete native fast/slow/associative physical world models. Twenty
original fits completed across two integrated recipes, with five paired seeds
per recipe. All failures remain preserved. This is a two-design development
program with historical source DEV exposure, not a single selection-free
independent source test.

ME060 introduced bounded 8x8 associative delta memory with norm projection,
direct horizon/action-conditioned reads, a SwiGLU residual trajectory decoder
and positive Laplace scales. Its source DEV TAC MAE .845300 lost to persistence
.702952. Frozen clothes-hanging TAC .373251 beat GRU .381716 but lost to
persistence .351410. Both gates failed; merely adding experts and a physical
prior did not make the system useful.

ME061 retained native state dynamics, added shared causal velocity/innovation
statistics, initialized forecasts exactly at persistence, learned signed
reliability-attenuated physical corrections and aligned its primary loss with
original-unit MAE. This is an integrated architecture/training improvement;
individual component effects and expert specialization have not been proven.

| ME061 factual physical MAE | Native TAC | GRU | Persistence | Servo |
|---|---:|---:|---:|---:|
| Historical source DEV,76 episodes | .603894 | .602600 | .702952 | 1.105522 |
| Frozen tea-pouring task,80 episodes | .493206 | .509535 | .455992 | 1.421244 |

Source TAC improves 14.09% over persistence and 28.56% over the preceding
ME060 recipe, with changed capacity/features/loss/seeds disclosed. It still
loses narrowly to the matched GRU, so the source gate fails. On the separately
registered different task, TAC beats GRU by 3.20%, wins all five paired seeds,
and has TAC-minus-GRU episode interval [-.021299,-.012083]. TAC remains
8.16% worse than persistence, so the transfer gate also fails. Target trajectories
opened only after models froze; no target fitting or preprocessing adaptation
occurred. Same robot interface does not establish different-site/operator
generalization. Intervals condition on fitted seeds and are not multiplicity
adjusted across this program.

ME061 uncertainty is implemented and measured, not qualified: 90% nominal
whole-episode intervals attain 85.79% source DEV coverage with mean physical
width 11.116. No observed inspection-cost utility, policy value, causal control,
iterative reasoning benefit or online weight adaptation was evaluated.

ME061 parameters 33,548 vs 33,802 and common operator-profile estimates 51.22M
vs 50.85M per update pass 1% gates. Mean optimizer CPU 18.810 vs 7.905 seconds
is an efficiency limitation. Supported-operator estimates do not certify true
complete hardware FLOPs. Source error arrays and frozen target results replay
exactly; 978 tests pass. All 20 weights/optimizer checkpoints and detailed scores
are retained in a checksummed workspace archive (PR #340 ARTIFACT.md), not
misrepresented as an independently downloaded GitHub Actions training artifact.

**Next construction objective:** retain ME061 as a development candidate and
learn reliable residual utility under real task shift, so the integrated
predictor can preserve persistence when physical/learned corrections are
unhelpful. Train against authentic future losses and calibration/risk targets;
require a newly registered target task to beat the strongest practical prior
as well as matched learned rivals. This is the next capability implementation,
not a return to an open-ended sequence of memory probes. Cost-sensitive
actions remain blocked on actual supported action/outcome/cost evidence.

### 2026-10-23: ME-062 selective forecasting and governed Woyengi handoff — SOURCE-ONLY

[Draft PR #341](https://github.com/FPC-effortless/TAC-OSM/pull/341) is a successor to ME-061, retaining bounded fast/slow/associative recurrence while adding a shared (TAC and GRU) causal horizon-conditioned trust gate over the learned physical residual, plus a positive heteroscedastic Gaussian uncertainty head and physical Gaussian training loss. Untrained model source and synthetic software contract tests only: **TRAINING=NOT_RUN; MODEL_MAE=UNKNOWN; REPO_REPLAY=BLOCKED; SUPERIORITY=UNKNOWN**. The new source code is not an observed improvement.

The branch also defines a versioned read-only Woyengi → TAC envelope for principal/workspace/entity-scoped, bitemporal (valid time vs recorded time), provenance-carrying authorized state histories and at most five candidate action sequences. TAC emits non-authoritative prediction proposals only, with checkpoint, state revision, trace and replay IDs, mandatory human review while uncertainty is unqualified; no model output authorizes an execution. Woyengi must authenticate, authorize and confirm snapshot freshness before and after inference. Veritas can stress-test RL/policies offline, but its generated worlds are **synthetic**, and model-simulated counterfactual scores cannot establish real action utility. Woyengi feature adapters, action governance integration, and Veritas trainer/evaluator adapters are NOT yet built by this TAC PR; absorbing SafeEdge/KoreFrame into Woyengi remains a separate authorized integration task.

ME-062 must be preregistered for a new untouched authentic physical task before any target labels are opened; compare equally trained/parametered/data/computed GRU, persistence, servo, and matched no-memory control; retain historical DEV as exploratory only and avoid old exposed TEST, clothes-hanging and tea-pouring as untouched confirmation. Measure 90% coverage and risk-width/coverage with new CAL-only Gaussian calibration, plus reset-state and separately trained k=1/4/16/64 memory ablations. Primary qualification requires beating the strongest practical baseline on the unseen real task with uncertainty-supported advantage. Simulated Veritas rollout success never substitutes. CPU runtime is disclosed independently, not required equal.

**New compulsory gate already effective:** no ME-062 fit can be declared complete unless original weights, optimizer and replay evidence for every compared arm/seed are in the TAC Git repository under `research/replay-bundles/ME062/`, retrievable and independently verified from a clean checkout. Before running any expensive fits, reserve that storage, register FIT provenance/schedules/compute/selection, and verify legal replay source access. The previously retained ME-061 local archive is historical evidence, not proof of newly required in-repository replay. See PR #341's [protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/me062-reliability-woyengi-handoff-2026-10-23/research/mathematical-engineering/me062/PROTOCOL.md) and [governed handoff](https://github.com/FPC-effortless/TAC-OSM/blob/research/me062-reliability-woyengi-handoff-2026-10-23/research/mathematical-engineering/me062/WOYENGI_HANDOFF.md).


### 2026-10-23: ME062 genuine original FIT training, repo checkpoint preservation, and prospective transfer

[Draft PR #341](https://github.com/FPC-effortless/TAC-OSM/pull/341) now has actual real original robot FIT-only training, **not merely source code**. [Original-FIT-only numeric preflight](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39881743894) audited 208 authentic episodes, parameter counts 37,427 TAC / 37,681 GRU (0.674% gap), and same-real-batch supported-operator update estimates 57,191,440 TAC / 56,827,806 GRU FLOPs (0.636% gap). All active parameters had finite preflight gradients; zero optimizer steps or heldout trajectories were opened in the preflight.

[Original ten-checkpoint FIT training run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39882865345) fitted 5 paired seeds 62001–62005, 600 AdamW updates per model, original source-sha `b55d0e1f894072c5738ef8e45aa283a3b464c1a1dc8f60c63594666d16b30a49`, all training sample orders frozen, followed by final numerical parity PASS. Physical validation data remained unopened during fitting. Original model+optimizer/RNG/checkpoint bytes for each seed/arm are now in **this TAC Git repository** under `research/replay-bundles/ME062/checkpoints/`; a separate GitHub Actions clean checkout verified the 10 SHA-256 identities and reloaded all models. `MODEL_REPLAY=PASS`. Fit CPU (efficiency only) ~30.46 s TAC vs ~13.83 s GRU; ~2.20x penalty. Final profiled FLOP receipts per model 34.314864 B TAC / 34.0966836 B GRU; unsupported operations and complete hardware FLOPs remain uncounted/uncertified.

The genuine different-task evaluation was prospectively registered using **metadata only**, selecting `line_trash_can_with_trash_bag` and 80 episodes at pinned publisher revision `9290f5a96dc91ea58256e81d9ce2dd9621890b51`. This task selection excludes historical source, clothes-hanging and tea-pouring research uses and did not read target trajectories, interventions or outcomes. It is the same robot interface and does not demonstrate operator/site independence.

**Scientific qualification remains pending:** trained checkpoints and model restore are separate from measured future-state MAE, uncertainty calibration, original heldout source DEV and registered transfer. Source DEV has historical development exposure and is not independent TEST. The [source+target scoring and independent replay workflow](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39883815447) subsequently completed successfully, reproducing original published real scores byte-for-byte from a distinct clean checkout with all ten Git original checkpoint bytes and **zero additional fits**: `MODEL_REPLAY=PASS`, `SCORE_REPLAY=PASS`, but `SCIENTIFIC_SUPERIORITY=FAIL`. Source DEV is not independent TEST, and same-robot different-task transfer is not independent site/operator generalization. Do not promote any ME062 architecture/transfer/decision claim until exact outputs, bootstrap intervals, strongest-prior comparisons, source calibration, score replay and replay receipts are independently verified and preserved in Git. The matched no-memory and k=1/4/16/64 context-length experiments are still unrun; unique associative/slow-memory importance is not proven. Veritas RL simulations and Woyengi governance interfaces remain separate from real intervention evidence. 


### ME062 exact scored negative result — frozen original FIT models, no post-test refit

The **five-seed ME062 registered accuracy gates FAILED**. Both original real cohort score files were generated from the ten previously Git-frozen and independently model-restored checkpoints: [source CAL/DEV and different-task scored summary](https://github.com/FPC-effortless/TAC-OSM/blob/research/me062-reliability-woyengi-handoff-2026-10-23/research/replay-bundles/ME062/REAL_WORLD_EVALUATION_SUMMARY.json). Source DEV (historically exposed, 76 episodes) physical MAE: native TAC **0.609373443944**, matched trained GRU **0.605164397358**, last-state persistence **0.702951822257**, servo **1.105522184081**; TAC wins **0/5 paired seeds**, TAC-minus-GRU episode-bootstrap interval **[+0.00195320, +0.00662016]**. The explicit reliability gate preserves a 13.31% source advantage over persistence but is inferior to GRU; do NOT credit native memory superiority or an independent test.

The frozen, metadata-preregistered *line_trash_can_with_trash_bag* target (80 original episodes) physical MAE: TAC **1.256390114265**, GRU **1.223529804823**, persistence **1.295032144382**, servo **1.031359254918**. TAC wins **0/5** paired seeds, TAC-minus-GRU interval **[+0.02227347,+0.04447820]**, and also loses to the *strongest practical servo baseline*. This is same robot interface with unverified site/operator independence; no target fitting, preprocessing, or calibration was permitted. Both superiority gates **FAIL**; preserve the negative result. The task-dependent switch from persistence-strong in ME061 tea pouring to servo-strong in ME062 trash bag is an observed baseline ranking change, **not** proof of causal source-shift mechanism.

Nominal 90% CAL-based whole-episode intervals for TAC achieve only **87.37% source DEV coverage** (mean physical width 12.6399) and **30.0% target transfer coverage** (mean width 12.2252); GRU whole-episode coverage is 85.53% source and 20.50% target. Observed transfer calibration catastrophically fails, and observational exchangeability remains unverified. Do NOT promote confidence-based auto-execution/abstention and do not tune on this now-exposed transfer. Gaussian trust-gating + shared physical recipe did NOT solve generalization; the stronger servo baseline makes the shift clearer. Matched no-memory and independently retrained context-length causal memory tests remain unrun. Research next must target robust physics/persistence selection under **a new untouched task**, with a genuinely independent heldout and action-support evidence before commercial/action claims. Preserve original in-repo weights, source failure receipts and subsequent independently replayed scores. 


**Post-evaluation replay completion:** [ME062 Git-tracked independent real-world score replay receipt](https://github.com/FPC-effortless/TAC-OSM/blob/research/me062-reliability-woyengi-handoff-2026-10-23/research/replay-bundles/ME062/EVALUATION_REPLAY_RECEIPT.json) records original ten checkpoints restored and CAL/DEV/80-target source results exactly recomputed; this satisfies the repository-owned preservation and *this dataset's* legal reproducibility lane. Do **not** confuse an independent computational replay with an untouched scientific independent holdout or qualified predictive superiority. Keep all negative scores and old frozen models available in the research branch.


### 2026-10-23: ME063 real FIT three-regime mixture result — both registered gates FAILED

[Draft PR #342](https://github.com/FPC-effortless/TAC-OSM/pull/342) (stacked on ME062, not merged) actually trained ten native TAC/GRU physical predictive models on publisher-original real FIT (208 episodes, frozen SHA b55d0e1f894072c5738ef8e45aa283a3b464c1a1dc8f60c63594666d16b30a49, five seeds 63001-63005 and 600 updates each), retaining original checkpoint/optimizer/RNG bytes directly in `research/replay-bundles/ME063/checkpoints/`. Supported-operator prefit resource parity: 37,177 TAC versus 37,431 GRU trainable params (**0.679% gap**); 56,895,406 TAC versus 56,531,772 GRU profiled forward/backward FLOPs/update (**0.639% gap**). Actual training final parity PASS; CPU update totals ~20.06 s TAC versus ~9.30 s GRU (efficiency limitation only; no equal CPU requirement). A separate clean GitHub Actions checkout verified all ten checkpoint hash identities and loadability **before any scoring**. The true full hardware/optimizer FLOP accounting remains uncertified.

**Architecture:** shared across TAC/GRU, a causal horizon-conditioned 3-way convex mixture of last-state persistence, original-FIT-identified physical servo state prediction and native recurrent neural prediction. Its uncertainty head moment-matches physical-prior spread, learned neural uncertainty, and component disagreement, then CAL-only source-episode conformal calibration. This is **not** a new general mixture-of-experts or adaptive calibration theory (see PMLR 2026 Dynamic TMoE and JMLR 2024 Gibbs/Candès); novelty question is empirical usefulness of native fast/slow/associative state with matched controls. Actions are logged factual covariates, not an intervention.

| Real original physical MAE (lower better) | TAC | Matched trained GRU | Last-state persistence | Original FIT servo |
| --- | ---: | ---: | ---: | ---: |
| Exposed source DEV, 76 episodes | **0.572882** | 0.573576 | 0.702952 | 1.105522 |
| Newly metadata-registered `fold_and_stack_scattered_clothes_neatly`, 80 episodes | 0.828290 | 0.829905 | 0.895391 | **0.793413** |

TAC narrowly wins 4/5 source seeds and 3/5 frozen transfer seeds, with **recording-block** TAC-minus-GRU 95% intervals [-0.003577, +0.001314] and [-0.008709, +0.005304], respectively. **Both intervals include zero, so neither TAC-GRU advantage establishes superiority.** On the untouched clothes-folding task, the servo forecast is better than both trained neural models (TAC 4.4% worse than servo); primary predeclared best-comparator requirement **FAILS**. No ME062-to-ME063 controlled paired improvement claim: architectures, training recipes and seeds changed, and the prior exposed trash-bag target cannot serve as new confirmation. This is a different recorded task on the same robot interface, not independently verified site/operator generalization.

Mean TAC router weights source DEV: persistence 7.33%, servo 40.89%, neural 51.78%; frozen transfer: persistence 10.36%, servo 55.09%, neural 34.55%. **Causal-input routing shifts toward servo in the new task without target fitting**, but the aggregate blend fails to match servo alone, and learned weights must not be called calibrated confidence. Source-trained nominal 90% episode conformal coverage: TAC **83.95% source DEV**, **72.25% new transfer**, with mean physical widths ~27.69 and 16.74. Both coverage and sharpness remain unqualified; exchangeability and independent-site claims unsupported. [Actual score source and checkpoint evidence](https://github.com/FPC-effortless/TAC-OSM/blob/research/me063-causal-regime-mixture-real-robot-2026-10-23/research/replay-bundles/ME063/REAL_WORLD_EVALUATION_SUMMARY.json).

**Research interpretation:** ME063 empirically shows learned *task-dependent weighting* and competitive but not demonstrably better TAC performance; physics prior selection remains unreliable. Scientific gate **FAIL**; memory ablation, long-history entity retention, interventional decision-cost utility, online verified adaptation, and calibrated policy abstention NOT demonstrated. Preserve all genuine original checkpoints and negative labels; do not tune on exposed source DEV or the now-open new transfer. The next experiment should isolate **causal forecast-correction necessity and out-of-support detection** on a separately registered, truly untouched real task, with proper stronger-prior/fair-GRU controls, rather than rediscovering mixtures or conformal methods.

**ME063 evaluation-replay audit complete:** [GitHub Actions run 39886390227](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39886390227) completed both original real evaluation and a separate clean-checkout replay. [Git-owned EVALUATION_REPLAY_RECEIPT.json](https://github.com/FPC-effortless/TAC-OSM/blob/research/me063-causal-regime-mixture-real-robot-2026-10-23/research/replay-bundles/ME063/EVALUATION_REPLAY_RECEIPT.json) reports source DEV and frozen clothes-folding transfer score files recreated **byte-for-byte by SHA-256** from the original ten Git-owned checkpoint bytes and legally accessed pinned publisher recordings; NO refits. Thus `MODEL_REPLAY=PASS`, `REAL_EVALUATION_REPLAY=PASS`, but `ARCHITECTURE_SUPERIORITY=FAIL`, `CALIBRATED_UNCERTAINTY=FAIL`, and `ACTUAL_POLICY_VALUE=NOT_TESTED`. Preserve all negative results and do not merge this branch as a model capability upgrade.


### 2026-10-23: ME064 true task-level leave-one-task-out training registration

[ME064 draft PR #343](https://github.com/FPC-effortless/TAC-OSM/pull/343) uses **five authentic previously development-exposed physical robot tasks**, sharing a 14D r1_lite recorded action/state interface, for *a single common jointly trained native TAC and matched GRU per paired seed*. Frozen source task cohort: clean-stains, fold-and-stack clothes, hang clothes, line-trash-bag, tea-pouring, with **32 whole FIT and 8 whole CAL episodes per task** selected by original publisher metadata hash, 4-episode recording block separation, 160 FIT original total, 40 CAL real total. Original clean-stains source TEST is forbidden from the new training cohort by reusing the original source FIT restriction. Original metadata-only selection of never previously evaluated task `install_spray_nozzle_onto_bottle` froze an 80-episode target before reading a single target trajectory; plug-charger and tissue-box remain unvisited.

Training is single five-task **multi-source, leave-one-previously-unseen-task-out** (NOT 5-fold task cross-validation nor a new hardware/site/operator split). Each of new paired seeds 64001..64005 receives the same 600 optimizer updates, batch eight, 64 state/action history and 16 future observed-action labels, **round-robin task** assignment (120 updates per source task per seed), independently sampled eligible origins within each source task, identical target/authority censoring, FIT-only normalization and servo prior identification, and a shared causal 3-way physical mixture head. Five seeds x TAC/GRU = 10 planned models. No learned target preprocessing, target labels in training, post-hoc checkpoint selection, source-calibration leakage or heldout retuning.

[Original five-task real FIT preflight](https://github.com/FPC-effortless/TAC-OSM/blob/research/me064-multitask-leave-one-task-out-2026-10-23/research/mathematical-engineering/me064/PREFLIGHT_MEASURED.json) measured: original 160 FIT robot episode source digest `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`, **37,177 TAC / 37,431 GRU parameters** (gap 0.679%), **56,895,406 TAC / 56,531,772 GRU PyTorch-supported operator FLOPs/update** (gap 0.639%). Parameter, supported-operator FLOP and all active-gradient source checks PASS. Whole optimizer/hardware FLOPs remain uncounted; CPU wall-clock equality NOT required. [Registered FAIR_COMPARISON](https://github.com/FPC-effortless/TAC-OSM/blob/research/me064-multitask-leave-one-task-out-2026-10-23/research/mathematical-engineering/me064/FAIR_COMPARISON.json) is frozen before any optimizer update. The scientific registered target requires TAC lower MAE >=2% versus each trained GRU, persistence and FIT servo with 4-episode source block-clustered paired CI upper bound below zero, and source CAL derived empirical nominal 90% whole-episode coverage/risk width reported. Same robot is NOT proven independent domain/operator/site.

Original checkpoint **bytes** (10 trainable model/optimizer/RNG and original source/schedule hashes) MUST be stored in TAC Git under `research/replay-bundles/ME064/`, clean-checkout restored before CAL/target opened, then independently reproduced from original legally available publisher trajectories. Preserve negative result or exact failure. No simulator task result qualifies real-world success. Production Woyengi integration forbidden absent independent model calibration and action utility.

### Woyengi guardrails verified alongside ME064

[Woyengi draft PR #53](https://github.com/FPC-effortless/Woyengi/pull/53) is a separate **source-only** TypeScript pure function and ADR based on the actual Woyengi `AgentGateway`, `CapabilityEngine`, `LocalComputeProvider` and governed effects semantics. It verifies Woyengi-origin read permission and snapshot freshness, task/request/checkpoint fingerprint and finite physical forecast receipt. Even a valid checkpoint forecast is `ADVISORY_REVIEW_REQUIRED` and `executionPermitted=false`; an unqualified ME062/063/064 forecast is `UNQUALIFIED_REVIEW_ONLY`; invalid references, stale/revoked permissions, malicious self-granted execute, and unknown model are blocked. This is NOT YET connected to a live API or production `AgentGateway.act()`; separate runtime reauthorization before any real effect remains mandatory, and no forecast can count as authority or canonical factual state. Woyengi CI Quality/Conformance and Container Smoke checked and PASSED on the draft PR. Veritas synthetic tests remain offline and never convert inferred counterfactual forecasts into real intervention evidence.


### 2026-10-23: ME064 real untouched spray-nozzle result — strongest practical baseline gate FAILED

[Draft ME064 PR #343](https://github.com/FPC-effortless/TAC-OSM/pull/343) trained 10 real original 5-task FIT-only model/optimizer/RNG checkpoints for paired seeds 64001–64005 under resource matching (37,177 TAC / 37,431 GRU parameters, gap 0.679%; 56,895,406 TAC / 56,531,772 GRU supported-operator FLOPs/update, gap 0.639%) and passed final actual-train preflight/compute parity. Five previously development-exposed tasks supplied 32 original FIT episodes each (160 total), with a separate 40 original CAL episodes (8 per task), identical round-robin task/window schedules and 600 optimizer updates per arm/seed. Original source FIT digest `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`. Ten original checkpoint bytes with optimizer/RNG source identifiers were committed in `research/replay-bundles/ME064/checkpoints/` BEFORE target outcomes, and an independent clean checkout verified 10 SHA-256 and restored 10 models in `CLEAN_CHECKOUT_MODEL_REPLAY.json`. Training is genuine original data, not synthetic or imagined.

The metadata-only preregistered entire untouched target `install_spray_nozzle_onto_bottle` (80 genuine publisher episodes, 20 four-episode blocks) was chosen from never-evaluated available tasks before opening ANY target trajectory. It was excluded from training, task normalization, source CAL and model selection. It is a different task on the SAME robot platform, not independent site or operator.

**Exact initial authentic physical holdout MAE** (lower better):
- **Last-state persistence 0.0123712880507810 (BEST)**
- **Native five-task TAC 0.135153214889579**
- Matched five-task GRU 0.148456336071249
- Source FIT-identified servo 0.918423870205879

The native TAC average score is about 9% lower than GRU, with paired original recording-block bootstrap 95% interval for TAC minus GRU **[-0.01809460,-0.00862360]** (uncertainty over the fixed five-model averaged episode predictions), yet TAC wins only **2/5** independent training-seed pairs, so superiority across independent model fits is NOT established. TAC versus persistence block bootstrap **[+0.11124416,+0.13384909]**; the last-state forecaster is approximately **10.9x lower-error than TAC**, definitively failing the registered all-baselines superiority gate. TAC's router assigned mean 51.06% persistence, 22.40% servo, 26.55% neural. This is not enough to avoid severe overcorrection during nearly static source task conditions. Mean source-CAL-based nominal 90% **whole-episode** coverage on untouched target reaches **100%** for both TAC and GRU, but is **extremely conservative**, with mean physical interval widths ~17.07 (TAC) and ~15.90 (GRU) versus TAC physical MAE ~0.135. Never imply this qualifies accurate, sharp risk estimates or safe autonomous decisions. Native memory necessity, independent-site transfer and real counterfactual utility remain NOT demonstrated.

**Honest reproducibility caveat:** initial [GitHub real outcome scoring run #39891151580](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39891151580) preserved original score receipts and returned correct negative gate; separate clean-checkout reproduction generated virtually identical figures but score JSON SHA-256 differs by small CPU floating-point differences (example TAC MAE 0.135153213376 vs 0.135153214890). Consequently the earlier **bitwise numerical score replay gate correctly FAILED**, despite 10/10 exact model checkpoint SHA and strict load PASS. A new separate objective numerical-tolerance replay audit is registered in `scripts/me064_validate_numeric_replay.py`, with exact nonfloating/source/schema identity, frozen original vs recomputed SHA receipts, absolute 1e-4/relative 1e-5 tolerances on individual score floats, tighter 1e-5 primary MAE/CI summary tolerance; this must **PASS** before `REAL_SCORE_REPLAY=PASS_WITH_FLOAT_TOLERANCE`. Do NOT falsely label byte-identical score replay, do NOT use unseen task for threshold tuning or refitting. The PR stays draft and unmerged. Preserve failed results and original checkpoint/source bytes.

**Research lesson:** ME062/ME063/ME064 confirm the strongest physical baseline changes by task and current mixture is not trustworthy even when routing weights shift. Stop treating a three-way blend as intrinsically calibrated. Before Woyengi production action use, native TAC must demonstrate useful selective **non-intervention**, reliable dynamics change detection, sharply bounded risk coverage, and decision value on separate authentic independent episodes/tasks with externally verified actions/outcomes. Woyengi's CI-tested advisory `tac-forecast-guard.ts` in draft [PR #53](https://github.com/FPC-effortless/Woyengi/pull/53) remains source-only, fail-closed for execution and not yet wired to production AgentGateway; Veritas simulations remain diagnostic only.


**ME064 frozen score-only posthoc diagnostic (NOT a new independent test):** [Git-owned original task error decomposition](https://github.com/FPC-effortless/TAC-OSM/blob/research/me064-multitask-leave-one-task-out-2026-10-23/research/replay-bundles/ME064/POSTHOC_FAILURE_DIAGNOSTIC.json) was generated strictly from preexisting original per-episode score receipts, without model inference, new target publisher downloads, optimizer steps, weight selection or recalibration. Using fixed *exploratory, after-exposure* physical persistence-MAE boundaries (<0.025 near stationary; 0.025–0.1 limited motion; >=0.1 moving), **78/80** original nozzle-install episodes are near-stationary (persistence MAE mean 0.011458 vs TAC mean 0.134010), **2/80** limited motion (persistence mean 0.048003 vs TAC mean 0.179749), **0/80** substantial motion by this band. TAC beats persistence **0/80** episodes, versus GRU **55/80** episodes. The per-seed original TAC/GRU mean errors show **2/5 TAC seed wins** with substantial outlier sensitivity: seed 64005 TAC 0.337241 vs GRU 0.491191, despite three GRU-favoring seeds. The episode-cluster CI for five-model averaged differences therefore cannot establish superiority across independent training seeds. This diagnostic's thresholds were selected AFTER the independent holdout's aggregate error was known and MUST NOT be used as model inference thresholds, policy gates, architecture tuning evidence or fresh generalization proof. It strongly motivates a **prospectively registered** real motion/nonmotion distinction on another untouched source, rather than additional post-hoc tuning on this target.


### 2026-10-23: ME065 comparative micro-world-model controls — source preflight only

[Draft ME065 PR #344](https://github.com/FPC-effortless/TAC-OSM/pull/344) establishes new **untrained** compact Gaussian stochastic RSSM and selective-diagonal-SSM recurrent controls, both using ME063's same 14D observed physical state/actions, persistence/servo/neural experts, causal history observer, horizon-conditioned forecast decoder and risk interface. Fully trained native fast/slow/associative TAC and matched GRU remain the original benchmark arms. This is NOT a full DreamerV3 agent, not Mamba, and does not turn native TAC into an SSM. Real-source parity has **not** yet been verified: the original five-task FIT-only PyTorch parameter/FLOP/gradient preflight is underway, must enforce <1% differences for every trained arm, and **must stop all full fits** if failed. ME056 concept-swarm/CED remains a separate fifth candidate and cannot join a fair comparison until its previously failed ~61.32% real-FIT FLOP preflight is corrected without dummy operations. No ME065 training, heldout test, skill, new superiority or agent-control claim follows from this scaffold. See `research/mathematical-engineering/me065/PROTOCOL.md` on the PR branch.

**Compulsory follow-on only after successful original source preflight:** preregister a new untouched genuine original publisher robot task using **metadata only**, never the already exposed ME064 spray-nozzle target; match original FIT data, trainable counts, optimizer-update/supported-compute budget, paired seeds and calibration protocol; preserve all original full model/optimizer/RNG checkpoint **bytes in Git** before opening new outcome labels; independently restore and score; compare native TAC and each eligible competitor against strongest physical persistence/servo as well as the neural alternatives and report **both episode-block uncertainty and independent-seed consistency**. No automatic Woyengi execution or reward/counterfactual intelligence assertion.

**Frozen failure diagnosis (exploratory, not qualifying):** [ME064 score-only posthoc diagnostic](https://github.com/FPC-effortless/TAC-OSM/blob/research/me064-multitask-leave-one-task-out-2026-10-23/research/replay-bundles/ME064/POSTHOC_FAILURE_DIAGNOSTIC.json) counted **78/80 near-stationary** original heldout episodes under its explicitly posthoc observed-future motion bands; mean TAC physical MAE ~0.1340 versus persistence ~0.01146 in those episodes. TAC beat GRU on **55/80** individual episode means but persistence on **0/80**. The routing weight to persistence was only ~51%; it failed to prevent large nonzero forecasts when little happened. These diagnostic classes are derived from FUTURE outcomes and must never feed a production gate or be reused to tune on those 80 original episodes. Future work must evaluate real observable forecast-time change evidence and selective non-intervention on another independently reserved authentic task. Previously reported CAL-based interval coverage was unhelpfully wide.



### 2026-10-23: ME066 four-arm authenticated real FIT training

**Observed training result, not a capability claim.** [Draft PR #345](https://github.com/FPC-effortless/TAC-OSM/pull/345) created and source-registered a four-architecture same-original-data challenge: native TAC ME063 (37,177 params), fully trained GRU (37,431), compact Gaussian deterministic+stochastic RSSM (37,319), and input-selective stable diagonal-SSM control (37,415, **not official Mamba**). ME065 [draft PR #344](https://github.com/FPC-effortless/TAC-OSM/pull/344) independently measured all on original real five-task FIT (160 authentic source episodes); reported supported-operator FLOPs per update were **56,895,406 / 56,531,772 / 56,423,631 / 56,841,027**, maximum 0.679% parameter and 0.829% FLOP disparities, all active gradients present, numeric prefit PASS. These are partial profiler FLOPs and **do not certify all-kernel hardware compute**, and CPU wall-clock equality is NOT compulsory. All active recurrent bodies share the genuine observed state/action and physical persistence/servo/neural forecast router/head, with disclosed .005 RSSM KL auxiliary.

ME066 separately registered 5 prospective paired original seeds **65001–65005**, **600 source-real AdamW optimizer updates × 4 architectures = 20 actual full FIT models** using exactly the same original FIT task-window order within each paired seed; source FIT SHA `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`. Git owns **all 20 original trained weight/optimizer/RNG checkpoint bytes**, their per-seed original hashes, and postfit numeric parity receipts under `research/replay-bundles/ME066/`. **Actual original postfit final parity PASS** for active parameters, source data/order and PyTorch-supported forward/backward estimate. The metadata-only, prefit-frozen new 80-episode original target is **`plug_charger_into_power_strip`**, explicitly different from the exposed ME064 spray-nozzle; **`replace_tissue_box_and_pull_out_first_tissue`** remains reserved. No source CAL or new target trajectory was opened during prefit or any of the 20 original optimizer fits.

**Exact model restoration still has a required audit:** first independent clean Git model load job failed *before loading any checkpoint* due absent `pyarrow` Python module in its CI verifier environment, **not** changed weights or failed metric. The separate repair installs the dependency and re-runs frozen checkpoint restoration **without retraining any model**. Until clean-Git SHA+load audit produces PASS, **ME066 new heldout scoring is strictly blocked**, and no performance/superiority/transfer/decision claim exists. Model and immutable physical publisher source bytes, original fitting and negative score evidence must stay Git-replayable. Once independently verified, score only the original metadata-reserved real charger task; compare TAC versus RSSM/SSM/GRU/persistence/servo, motion forecasting, source-CAL-only uncertainty/sharpness, 4-episode original recording-block bootstrap and all five seed pairs. Checkpoint hash replay and score replay are distinct: label any non-bitwise independent numeric score agreement with explicitly frozen valid float tolerance, never pretend bitwise equality. Real factual recorded actions do not constitute intervention policy utility; Woyengi remains advisory only. The previously unqualified **ME056 Concept-CED** (61.32% profiled FLOP mismatch) is not presented as a fair trained fifth arm.



### 2026-10-23: ME066 genuine sixth-task results against compact world-model controls

**First frozen authentic new-task result published, all-baselines superiority FAIL.** [Draft PR #345](https://github.com/FPC-effortless/TAC-OSM/pull/345), original scoring from [GitHub workflow #39894504075](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39894504075) on metadata-registered original publisher `plug_charger_into_power_strip` (80 genuine episodes, 20 four-episode recording blocks; same robot platform, distinct never-before-opened task). The unchanged twenty original real-FIT checkpoint bytes were source-trained on five original tasks and independently strict-hash and model-loaded **20/20 PASS** from clean Git checkout before any CAL or new target trajectory was opened.

**Frozen physical MAE** (lower is better):
- last-observed physical persistence **0.2453838432** (**best overall**);
- trained matched GRU **0.3815281845** (**best learned**);
- compact input-selective diagonal SSM control **0.3989837375**;
- compact Gaussian RSSM control **0.4034837283**;
- **native TAC fast/slow/associative 0.4105902042** (**worst learned arm**);
- source-FIT identified physical servo **1.2005298074**.

Native TAC was ~7.62% higher-error than GRU and ~67.33% higher-error than persistence; it beat GRU in only **1/5** independent trained seed pairs, while episode-block 95% CI of native TAC minus GRU **[+0.01766173,+0.04044166]** excludes parity against TAC. TAC relative to persistence original block 95% CI **[+0.11451411,+0.20985601]**, establishing factual holdout inferiority. TAC vs selective SSM CI **[+0.00276894,+0.01948439]**, unfavorable to TAC. TAC vs Gaussian RSSM CI **[-0.00715383,+0.02097459]** spans zero, insufficient to claim a reliable RSSM win over TAC. **Registered TAC superiority gate FALSE**. Seed-pair training variability is substantial and must not be hidden by episode-block resampling of averaged predictions.

Original source-CAL-only nominal 90% whole-episode uncertainty actual target coverage: TAC **77.75%**, GRU **79.0%**, RSSM **81.25%**, selective SSM **74.25%**. Mean physical interval width TAC ~**23.18** versus error ~**0.411** (width/error ~**56.45**), GRU ~**19.72**, RSSM ~**20.64**, selective SSM ~**20.25**. Hence all demonstrated poor interval sharpness and unqualified transfer coverage; neither any policy effect nor safe abstention follows. New task motion bands are **posthoc outcomes** (27 near-stationary, 30 limited, 23 moving) and may diagnose failure, not tune/test a new forecast gate on these same observations. The router assigned native TAC persistence/servo/neural **28.18% / 33.30% / 38.52%**, not a demonstrated correct regime classifier.

**Research decision:** no promotion or merging of ME066 as positive TAC architecture. Preserve the full negative outcome and all Git-owned original weights/optimizer/RNG with source episode/source hashing, original CAL/test scores, frozen budgets and source code. This matched four-arm original robot experiment does **not** show native tri-state memory superiority over GRU/RSSM/selective recurrence or physical persistence. Do not characterize the diagonal selective control as official Mamba, nor the Gaussian RSSM as full DreamerV3. ME056 Concept-CED remains independently **UNTRAINED / computational parity unqualified**, not an unseen challenger result. Independent separate original score recomputation is required; initial first-frozen score receipt is valid factual evidence but **do not say numeric replay PASS** until receipt is committed. Do not use these 80 original target episodes for model selection/tuning after exposure.

**Next research priority:** investigate why matched learning produces false motion and unsound uncertainty across real physical regimes, without rediscovering known mixtures/SSM math or adding cosmetic heads. Require a change/no-change detector supervised only on original FIT or freshly reserved authentic episodes, compare a stronger *honest* persistence/physics/robust-learning reference, test genuinely long-horizon and action-sensitive factual information in independent tasks, then verify memory mechanism against matched no-memory and bounded-history controls. Any new architecture/optimizer objective demands another **unused** originally authentic task and all usual 1% active parameter/profiler compute parity and Git-owned replay gates. Robotic logged-action forecasts remain observations, not evidence of identified do(a) counterfactual treatment effects or autonomous control value. Woyengi remains advisory-only.



**Independent ME066 original real score replay FINAL RESULT (2026-10-23): PASS_WITH_FLOAT_TOLERANCE.** A separate clean GitHub Actions checkout independently reloaded all 20 source-trained checkpoint bytes, refetched original legal publisher source CAL and untouched charger task trajectories, recomputed and audited all original score JSON schemas and source/target metadata (strict equality for nonfloat data). Original score JSONs were **NOT bitwise identical** due CPU variation: 2,168 of 579,784 float fields differed, maximum abs float difference **0.0000346899032593**, within prospectively declared abs 1e-4/rel 1e-5 per float and 1e-5 primary physical-MAE/CI summary limits. Verified receipt: [ME066 EVALUATION_REPLAY_RECEIPT.json](https://github.com/FPC-effortless/TAC-OSM/blob/research/me066-four-arm-original-robot-training-2026-10-23/research/replay-bundles/ME066/EVALUATION_REPLAY_RECEIPT.json); independent original [Actions run 39894504075](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39894504075) completed successfully. This is **numerical replay under an explicit bounded float tolerance, never exact score-byte replay**; original 20 trained checkpoint hashes/load PASS remained exact. Registered TAC scientific superiority FAIL is reproduced and the original negative physical result is final for this untouched task. Do not retune, reselect or retrain on these exposed results; PR remains draft and unmerged.


### 2026-10-23: ME067 frozen memory lesion result — fast/associative help, reliability shift hurts

**Original GitHub run [ME067 #39898699817](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39898699817) and draft [PR #346](https://github.com/FPC-effortless/TAC-OSM/pull/346).** The new registered `noncomparative_diagnostic` passed compulsory research governance, independently reused 20 original ME066 trained checkpoint bytes and the PASS_WITH_FLOAT_TOLERANCE frozen actual publisher physical scores, reread **only 40 already-exposed source CAL + 80 already-exposed charger task episodes** with verified original per-parquet SHA, retrained **zero** models and changed **zero** weights. All unlesioned per-model episode forecasts reproduced the original outcome origins, live-step masks and score within predeclared absolute 1e-4. This was a strict *posthoc latent-state readout intervention*, not another heldout task or trained variant.

**Original previously exposed charger task frozen native TAC MAE** 0.4105902042 vs persist 0.2453838432, GRU full 0.3815281844. Readout lesions (mean five independently originally trained checkpoints, 80 original episodes, lower is better):
- zero velocity AND innovation reliability traces: **0.2952071325**; delta to full **-0.1153830717**, favorable on **5/5** frozen seeds, block-bootstrap fixed-model 95% interval **[-0.14527494, -0.08512809]**.
- zero slow state only: **0.3811334452**; delta **-0.0294567591**, favorable **3/5** frozen seeds, fixed-model original-block CI **[-0.04103664,-0.01572630]**.
- zero 8×8 associative state only: **0.4428551179**; delta **+0.0322649137**, worse on **5/5**, fixed original-block CI **[+0.02438969,+0.04093000]**.
- zero fast state only: **0.6001573078**; delta **+0.1895671035**, worse on **5/5**, fixed original-block CI **[+0.16665579,+0.21248935]**.
- zero both fast and slow: **0.4844864206**; delta **+0.0738962163**, worse on **5/5**.
- zero ALL hidden/reliability: **0.4043588681**, delta **-0.0062313361**, interval spans zero.

**Exposed original source CAL (40 genuine episodes) acts as a counter-indicator against simplistic pruning.** Full native TAC MAE **0.7572043383**; zero fast **0.7930257016**, zero slow **0.7920301288**, zero association **0.7628351577**, zero reliability/kinematics **0.7590431777**. Fast/slow appear useful in source CAL while slow and reliability induce adverse transfer to the exposed charger task. These findings are consistent with a *domain-sensitive reliability/innovation-conditioning failure*, rather than a globally useless native memory. The frozen unmodified trained GRU also improved from 0.3815281844 to 0.3227768677 when the same reliability traces were zeroed on the charger target, with mixed seed directions; the vulnerability is **not exclusively TAC**.

**Interpretation and restrictions:** Each lesion is an out-of-training-distribution edit to the trained hidden belief, so it does NOT prove causality in physical dynamics, identify the correct new model weights, or qualify deployment performance. Posthoc ranking of multiple lesions on the *already revealed* charger labels cannot constitute a confirmatory gate. The known original trained winner remains matched GRU and the original physical baseline winner remains persistence. Do not use charger/spray-nozzle for fitting or threshold/model selection; preserve original negative ME066 outcome. New physical models require a prospective original FIT-only objective and equal-data/parameter/profiler-FLOP preflight, source CAL selection, complete repository-owned original checkpoints with clean-Git model+score replay, and a **genuinely never-opened third task** `replace_tissue_box_and_pull_out_first_tissue` (or a genuinely independent external physical source).

**High-value next integrated hypothesis, not another arbitrary 1-variable tweak:** native TAC's fast/slow/associative state can be useful but should learn *whether physical motion follows the current action*; separate observation-reliability/authority from the state-prediction readout, enforce an explicit train-only persistence-referenced **conditional innovation / selective no-change control**, and train jointly on observed real FIT action transitions. Compare matched native TAC + identical GRU control, stable persistence and servo on a newly sealed real task. Make the action-alignment timing explicit, stress long horizons and change/no-change, track sharp physically meaningful intervals and seed variance. Existing switching-state filters / innovation diagnostics are prior art; do not claim novelty simply for a mixture router. Independent physical reward/authorized intervention evidence is required for commercial control; Woyengi remains read-only advisory.


### 2026-10-23: ME068 source-physical action timing — 257,971 genuine autonomous transitions

**GitHub source-only action audit complete and governance PASS**, [draft PR #347](https://github.com/FPC-effortless/TAC-OSM/pull/347), [source-owned original result](https://github.com/FPC-effortless/TAC-OSM/blob/research/me068-original-robot-action-timing-audit-2026-10-23/research/mathematical-engineering/me068/SOURCE_FIT_ACTION_ALIGNMENT.json), [original Actions audit run #39899562213](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39899562213). ME068 read **ONLY the 160 original source FIT robot episodes** across all five previously registered physical tasks, identical original source digest `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`. No new model fit (zero optimizer steps), no source CAL, no previously exposed spray or charger task, no remaining reserved tissue-box task. Human-free shared measurement windows t−8..t+2 gave **257,971 authentic measured autonomous one-step robot transitions**. In those original FIT windows **91.2269%** had mean 14D physical joint change below an explicit hindsight diagnostic threshold 0.025. These are highly autocorrelated frames, NOT independent samples or a success rate.

**Real original FIT physical next-joint one-step MAE** (lower better, same common real windows): last observation **0.07019513195**, FIT-identified original servo with action at t **0.12046094284**, past action at t−1 **0.11914191368**, t−2 **0.12019357838**, t−4 **0.12590195195**, t−8 **0.13624281188**. The best sampled *causally already available* action offset **t−1** was also the minimum among all tested previous/current and future logged offsets in each of five original tasks, but improvement relative to t of only about 0.001319; it does **NOT** certify physical actuator delay or that action timestamps should be changed. Pure in-FIT action-lag optimization is not a heldout capability result, and FIT-identification of servo gain is itself optimistic in-FIT. Temporally FUTURE logged actions (negative k in registered convention) were used only as retrospective timestamp diagnostics and **are forbidden as real-time features**. Every tested servo lag remained worse than simple last-state persistence on the same authentic source windows.

**Cumulative inference from ME064→ME068:** cross-task physics cannot be repaired merely by more recurrent capacity. Native TAC fast/associative state genuinely influences the trained predictor on source/transfer, while the learned velocity/innovation reliability readouts show brittle domain shift (ME067); source original command-based servo is predictively poor even on original FIT (ME068); and rare meaningful motion is underrepresented in uniformly sampled frames. No independent decision utility, exact action-effect identification or robust calibrated risk is shown. The long-horizon ME066 evaluation conditions on original *realized* logged future action prefixes: valid as a factual action-conditioned rollout comparison, but do NOT represent the information available to an online agent unless that exact future proposed command sequence was already authorized and known. Future real decision benchmarks must explicitly pin decision-time knowledge, command/ack time, human authority and outcome availability, avoiding any claimed online prediction from unavailable realized future commands.

### Research next: integrated movement-preserving conditional physical dynamics (proposal, NOT trained)

Avoid rediscovering known switching filters, mixture-of-experts or state-space recurrence just to relabel them. Keep native fast/slow/associative state and verified postfeedback correction as the TAC research identity. Fix the specific real-world failure with **jointly trained, causal motion/no-motion prediction** and an exact persistence path, not another posthoc latent reset:

```text
 z_t  = NativeTAC.observe_only_past_verified_state_and_actions()
 q_{t,h,j} = sigmoid(motion_head(z_t, known_or_proposed_action_prefix, h))
 d_{t,h,j} = residual_head(z_t, known_or_proposed_action_prefix, h)
 xhat_{t+h,j} = x_{t,j} + q_{t,h,j} * d_{t,h,j}
 # if motion probability is low, mean prediction approaches exact x_t
 # never inject unverified postdecision outcomes or future unproposed commands
```

Model a conditional zero/near-zero versus actual-motion distribution using a proper calibrated likelihood or joint movement prediction, with physical sensor precision/noise and motion-target semantics fixed **from source FIT only**, not the exposed ME064/066 evaluations. The action-timing variable must be estimated/verified from actual command timestamps or action receipt metadata where available. Separate verified reliability/innovation feedback **writes** from uncertainty in **predictive readouts**; never give this candidate a missing action/authority signal its matched GRU control does not receive. No fixed domain oracle or hand-scripted target-aware expert switch. Retain true original physical persistence, causal AR/constant velocity and train-identified servo references; a strong matched GRU must share identical motion head/objective/source windows, plus eligible RSSM/SSM controls. Task time signatures, horizon errors, false change predictions, motion-conditioned scoring and sharp risk intervals are **primary planned metrics**, not opportunistic later analysis.

**Hard progression gate:** first independent physical publisher data/timing qualification, then authentic original FIT-only model numeric active-parameter/partial-FLOP/source preflight with <1% gaps, same original real data and optimizer budgets paired seeds, full original weights+optimizer+RNG at pinned retrievable Git paths before any final new target opening, clean independent checkout restoration, source-CAL-only selection, then a reserved never-opened real target (remaining `replace_tissue_box_and_pull_out_first_tissue` or stronger independent robot/site). Do not spend this last clean holdout to compare another untrained speculative router; select a sufficiently substantial causal physical model candidate and a competitive learned control in advance. CPU wall-clock inequality is report-only, not a fairness veto. A positive result must win against **persistence** and the best learned control across real tasks, 5 paired seed stability and original recording-block uncertainty, and useful calibrated interval widths. Factual logged actions do not establish do(a) intervention value or policy success. Woyengi remains fail-closed advisory, no runtime authority promotion.

Known foundations: [Yong et al. 2021 switched mode/input/state estimation](https://doi.org/10.1002/rnc.5306); [TD-MPC2 genuine model-based control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html). These mathematical ideas are prior art; the experimental question is whether a native TAC memory/controller can beat strong simple physical references under appropriately qualified actual data.


## 2026-10-23 ME069/ME070 evidence correction: no native-memory advantage yet

**ME069 (draft PR #348) is NOT a TAC-specific win.** On authentic BPI2012 (2,618 TEST cases, five paired seeds), native TAC and matched GRU share FIT empirical priors and learned mixture/wait heads. Native next-event NLL 0.122626 versus GRU 0.119182 (TAC **2.89% worse**; paired bootstrap interval positive), accuracy both 94.77%, native time MAE 6.3561h versus GRU 6.3618h, mean training CPU 35.251s versus 13.777s (~2.56x worse efficiency). Both beat the 0.348946 first-order prior, but that **does not isolate TAC memory**. Positive value over an underfit simple reference is not architectural superiority. Real stateful workflow inference up to 32 events is implemented, **not** persistent 64/256-event capability. PR #348 retains all ten original Git model/optimizer/RNG checkpoints; local/GitHub independent replay evidence includes two documented fixed-tolerance failures and later passes—no universal deterministic promise.

**Retrospective original TEST audit** now measured in [draft ME070 PR #349](https://github.com/FPC-effortless/TAC-OSM/pull/349), with all seven diagnostic software tests passing in [Actions run 39920681790](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39920681790). Paired correct/wrong disagreements are only 11–36 cases per 2,618 per seed (exact McNemar p 0.227–1.0). Mean 230h waiting-time interval width conceals a pathological mixture: median **4.8–10.6 seconds**, p90 **109–197 hours**, p99 **2,134–4,605 hours (89–192 days)**, maximum **21,087–33,638 hours (~2.4–3.8 years)**. This is evidence of **severely uneven predictive dispersion**, not evidence that all cases have 230h uncertainty, nor that tighter intervals are automatically better. Direct interval/Winkler/pinball scores and next-activity lifecycle label frequencies cannot be computed from currently saved per-case statistics alone; require legal original-source replay and full endpoints. This diagnostic uses **already exposed** ME069 TEST; no model selection, refitting or new holdout scientific promotion from this information.

**Next source-first capability hypothesis:** BPIC2018 ([original DOI](https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972)) has published ~2.51M events/43,809 cases and reported max trace length 2,973, but independent eligible 256-event **case cohort counts and source licence/hash must be verified**. [ME070 protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/PROTOCOL.md) requires >=50 FIT and >=25 TEST distinct real cases with a factual next event following 256 observed events, 60/20/20 chronological original-case splits and no crossing-case leakage. Otherwise **STOP** and qualify another authentic source. No new model training until this gate and a **new fully preregistered architecture-comparison FAIR_COMPARISON** pass. The current ME070 registration explicitly identifies **noncomparative source/retrospective diagnostics only**.

**One falsifiable 64-event kill gate, not another endless ablation loop:** compare native TAC, matched GRU, meaningfully parameter/compute-matched native-associative-memory ablation; include shared-head/transition-only, smoothed n-gram, and a genuine process-monitoring LSTM or SuTraN comparator. At real online-observable prefixes (not a retrospectively selected midpoint), pre-register rejection of a broad native memory advantage unless TAC earns >=1% case-mean NLL improvement vs GRU at >=64 original events, clustered paired-case 95% upper CI <0, >=4/5 paired seed wins, and a positive incremental benefit of native associative memory. A 256-history claim also independently needs adequate true >=256-event evidence. Measure calibrated and sharp waiting intervals by quantile/Winkler proper scores, positive-gap risk, coverage and width distributions against empirical conditional quantiles, plus rare/cross-lifecycle transitions. Preserve full in-repository checkpoints and independent clean-checkout replays; CPU is **efficiency only**, not a fairness parity condition. ME055's older 5% CPU acceptance bound was historical and superseded for prospective science by the 2026-10-22 owner rule; its original recorded failed gate is not erased. No unsupported operator cost, causal benefit, Woyengi authority or commercial readiness claim.


**ME070 verified source-lifecycle supplement (2026-10-23):** original publisher-byte-hash-verified BPI2012 was reloaded using the ME069 source adapter and analyzed in [Actions run 39921891424](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39921891424). Its original true case-length maxima are FIT **161**, DEV **94**, TEST **109** events (FIT mean 17.80, TEST mean 18.27); therefore **zero TEST cases can provide 256 previous events**. On the retrospective ME069 TEST midpoint targets, **38.43%** remain in the same underlying activity and **13.06%** are exactly START→COMPLETE of that activity, while the majority target label is only **28.23%**. These lifecycle effects require difficult-transition and lifecycle-stratified baselines; they are not proof that all 94.77% accuracy is trivial. Original checksummed aggregate results were committed under [ME070 EXPOSED_ME069_LIFECYCLE_AUDIT.json](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/EXPOSED_ME069_LIFECYCLE_AUDIT.json). The original BPIC2018 long-source downloader attempted three connections in [run 39920980632](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39920980632) and timed out; dataset hash/256-event case eligibility remain **UNVERIFIED**, so ME070 training is **STOPPED** until lawfully accessible real data qualify. This is data-access failure, not a measured insufficient-cohort outcome.


### 2026-10-23 executed ME070 non-neural higher-order controls — original real BPI2012

[Draft ME070 PR #349](https://github.com/FPC-effortless/TAC-OSM/pull/349) now contains **actual FIT-only trained and Git-preserved 1/2/3/5-event-count predictors**, not merely a proposed control. From 7,096 original FIT cases (126,285 authentic events), fixed smoothed backoff (alpha=5) was fitted exclusively on FIT; original Git-owned count state was frozen at commit `6b711ff` **before** any historical DEV/TEST scoring, scored at `6f34537`, and the identical scores independently replayed from a clean repository checkout in [Actions run 39924308857](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39924308857). The original state/checkpoint and score bytes remain in `research/replay-bundles/ME070/practical-ngrams/` on the draft ME070 branch. Nine fixture regression tests passed. BPI2012 TEST is **already exposed** by ME069; this is **retrospective real-data diagnostic** rather than new independent qualification.

On the SAME 2,618 original ME069 TEST cases at the original hindsight-midpoint selection, empirical first-order Markov NLL/accuracy **0.348946 / 84.76%**, second-order **0.198247 / 92.25%**, third-order **0.172033 / 93.58%**, fifth-order **0.159743 / 93.96%**. Historical native TAC **0.122626 / 94.77%**, historical matched GRU **0.119182 / 94.77%**. Fifth-order FIT count memory explains **~54.2% of the first-order log-loss gap without a neural model**, so never credit improvement vs first-order baseline to native TAC memory. Both neural models still beat the fifth-order practical baseline but **GRU remains stronger than TAC**, and relative architectural contributions are unresolved. N-grams are explicitly unmatched practical references, not capacity-matched architecture comparisons. The original source also has only FIT=174 / DEV=15 / exposed TEST=61 cases with actual next labels after 64 observed events; after 128: FIT=4, DEV=0, TEST=0; after 256: zero across all. [Source eligibility receipt](https://github.com/FPC-effortless/TAC-OSM/blob/research/me070-long-horizon-enterprise-qualification-2026-10-23/research/mathematical-engineering/me070/EXPOSED_BPI2012_HISTORY_ELIGIBILITY.json).

Original BPI2018 source **metadata**, not its bytes, has now been independently resolved from the official Figshare article **12688355** (file **24025820**, size **158270992** bytes, publisher MD5 `66e54ddaa6b50d81a0166766d6f7a5d4`), but the original 4TU URL and ALL tested official Figshare downloader/API file routes returned timeouts/HTTP **504** in GitHub Actions. Source SHA256 and honest 256-event cohort counts remain **UNKNOWN**, thus the fully matched new ME070 neural model train/test remains **BLOCKED**. Never replace missing real source with simulated data or report a published maximum history as cohort evidence. Stop automatic failed downloader workflow dispatches until original access changes.

**Capability status unchanged:** observational real enterprise forecasting exists; native long-memory advantage, 256-event context, real counterfactual action utility, and enterprise deployment **NOT DEMONSTRATED**. Continue the pre-registered 64-event kill test only with independently qualified real data and repository-original fitted checkpoints plus clean-clone replay.

## 2026-10-23 ME074: actual same-cohort END-aware forecasting baselines and frozen memory lesions

**Original real BPI2012 executed diagnostic, NOT matched architecture superiority or independent TEST**: [ME074 draft PR #356](https://github.com/FPC-effortless/TAC-OSM/pull/356), original [Git-owned all-case baseline result](https://github.com/FPC-effortless/TAC-OSM/blob/research/me074-real-baseline-memory-diagnostics-2026-10-23/research/replay-bundles/ME074/REAL_DEV_END_AWARE_BASELINE.json), [original memory lesion receipt](https://github.com/FPC-effortless/TAC-OSM/blob/research/me074-real-baseline-memory-diagnostics-2026-10-23/research/replay-bundles/ME074/EXPOSED_MEMORY_LESIONS.json). 5 original 99,781-param ME073 model checkpoints and Git-owned source/preprocessing/output arrays unchanged. No neural FIT updates, no original TEST, no unknown future labels used at inference. FIT-only originally fitted ME070 count states were reused, with ngram 1/3/5 and an empirical per-current survival prior making absorbing **source END-of-record** predictions; END is NOT verified completed business or known at decision time. ME073 original scores were independently recomputed from frozen outputs and exact same actual query-age case identities before comparison.

**Six-observation prefix, 555 original already-exposed DEV cases; joint activity OR END NLL on ALL 555 at each horizon** (lower better, individual history ages have different case populations): 

| horizon | native ME073 five-seed mean | exact iterated bigram | 3gram 256 paths | 5gram 256 paths |
| -- | --: | --: | --: | --: |
| 1 | **0.246678** | 0.935693 | 0.566093 | 0.566272 |
| 2 | **0.405964** | 1.501591 | 0.964559 | 0.977954 |
| 4 | **0.716333** | 1.540770 | 1.180823 | 1.173752 |
| 8 | **0.885717** | 1.458240 | 1.071582 | 1.067201 |

**Crucial negative result: two-observation prefix** (925 original exposed cases) 3gram **beats ME073** on h1 NLL **1.146131 vs 1.416383**, h4 **1.966705 vs 2.145845**, and h8 **1.382027 vs 1.676677**. At age16, 165 cases, native beats 5gram h8 NLL **1.882833 vs 2.018601**, but age16 has only 106 true future event8 outcomes; joint scoring includes all age-eligible cases at each horizon.

**Frozen original five-seed native ME073 first-horizon fast/slow/association readout lesions on same exposed DEV:** Original joint h1 activity/END NLL for query age2 **1.416383**, age6 **0.246678**, age16 **0.231778**. Zero fast state before readout: age2 **1.191974 (improves 5/5 seeds)**, age6 **1.163264 (worse 5/5)**, age16 **0.581138 (worse 5/5)**. Zero associative 8x8: age2 **1.428893**, age6 **0.247267**, age16 **0.238165** (small and mixed seed directions). Zero slow: age2 **1.349949**, age6 **0.250235**, age16 **0.303922**. Zero ALL hidden: age2 **1.130832**, age6 **0.676848**, age16 **0.435481**. Every unlesioned per-case native h1 probability vector exactly reproduced the frozen original ME073 file within 1e-5 before a lesion was scored. Out-of-distribution hidden-state readout interventions diagnose sensitivity, not beneficial normally trained ablations or proof of a uniquely useful 8x8 associative memory.

**Inference:** Native ME073 predicts real process sequences substantially better than practical FIT activity-only Markov count models once enough observations are available, but it **fails** early-history practical reference on several horizons. No matched learned GRU/direct multi-horizon/causal-Transformer 100k future-path trial has been executed, so no evidence autoregression or TAC memory is a winner. The five trained TAC seeds are repeated cases, not five independent real-world source institutions. The baseline uses activity only versus native resource and timestamps, plus Monte-Carlo approximate later ngram marginals: practical comparison, NOT matched-input/params/compute controlled architecture. No case-paired uncertainty for these practical comparisons yet. Full interval calibration and trajectory path legality still unqualified.

**Synthetic proposal resolution:** ME074 contains deterministic 16/64/256 distractor key/value *fixture-integrity tests* verifying leakage-free query and correct END math, **no models trained on synthetic data**, hence no synthetic test of learned associative recall or controlled 100k GRU/Transformer superiority. The real-world-first policy explicitly forbids treating synthetic benchmarks as capability evidence or spending stand-alone architecture research optimizing synthetic accuracy. A fully trained synthetic diagnostic would require an explicit separate research exception, no research-success promotion, and 100k ±5% live model guard with ≤1% between-arm parity. Original BPI2012 has **zero** original 256-history cases with genuine future labels; never claim 256-event native retention from it. Publisher-authenticated BPI2018 256-event source remains access blocked (HTTP 504); no unauthorized mirror or synthetic substitute.

**Required prospective next study:** matched ~100k native-autoregressive vs matched GRU-autoregressive vs causal Transformer vs direct multi-horizon heads, five paired seeds, real FIT-only original events, equal genuine data/steps/profiled compute, identical query-time inputs, every original model/optimizer/RNG committed in Git before *new authentic source/TEST* scoring. Include simple FIT ngrams and survival baselines without artificial parameter padding. Score ALL-cases activity/END proper NLL and Brier, conditional activity accuracy with true denominators, source EOR versus actual verified completion separately, real trajectory-likelihood and free-run path consistency, per-horizon interval coverage **and** sharpness/pinball, short-history robustness, training-seed and independent entity uncertainty. Memory needs actual real long histories and clean learned matched controls, not OOD lesion alone. Prior ME069 matched GRU beat native on TEST log loss (0.119182 vs 0.122626), not a proof against a different later ME073 autoregressive architecture. Preserve Woyengi advisory-only permissions.



## 2026-10-23 DIAGKEY001: associative-memory hypothesis challenge (owner-requested, synthetic mechanism only)

**Status: prospective narrow mechanism diagnostic, NOT real-world qualification or a replacement for primary real observations.** [Draft PR #357](https://github.com/FPC-effortless/TAC-OSM/pull/357) implements native fast/slow/8×8 association, reallocated-capacity no-association native control, GRUCell and two-layer causal Transformer: **101554 / 101560 / 101500 / 101616 actual trainable and deployed parameters** respectively, within the compulsory [95000,105000] individual band and <1% relative difference. Matched fixed five paired seeds, 640 equal optimizer updates, batch16, same stateless generated train sequences with four variable key/value writes, 16/64/256 unrelated-key distractor writes and withheld query answer (32 possible values, chance3.125%); 512 untouched synthetic diagnostic TEST examples per delay/seed, explicit exact-match correct bits, NLL and original Git-owned optimizer/model/RNG checkpoint bytes. Source [diagnostic protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/diagkey001-100k-associative-mechanism-2026-10-23/research/mathematical-engineering/diagkey001/PROTOCOL.md). This is a narrow explicit owner-requested exception as a controlled software-memory diagnostic, **NOT an amendment to the standing ban on synthetic evidence for claims of real world TAC capability**. Synthetic model-training comparisons cannot constitute primary architecture-superiority proof.

**Rejection of overclaim from ME074:** ME073 native fast/slow dimensions are **64/64**, not 32D. Original preexisting BPI2012 age6 h1 all-case NLL: full0.246678, fast-zero1.163264, slow-zero0.250235, association8×8-zero0.247267. Age16: full0.231778, fast-zero0.581138, slow-zero0.303922, assoc-zero0.238165. Zero-association adds only ~0.24%/2.75% to NLL at ages6/16; memory is a weak measured contributor, but OOD frozen lesion is not proof it is universally dead weight. The huge fast-zero effect at mature prefixes is expected for a primary recurrent state and not TAC-specific evidence relative to a matched GRU. At age2 fast-zero improves NLL, but this **does not prove uninitialized noise**; model calibration, deficient early conditional training and prior interference remain competing hypotheses. Do not hard-code 4-event fast-memory warmup or replace native associative memory without a prospectively trained matched control and real independent validation. N-gram practical gain is confounded by source activity-only inputs versus TAC activity/resource/time; it does not isolate feature attribution.

**Synthetic-mechanism decision:** measure original paired 5-seed TAC vs GRU and **retrained no-association TAC** at delays64/256, compare against causal Transformer, report exact-match, chance and NLL and explicit supported-operator/FLOP budget gaps. A >=5 percentage-point native advantage over BOTH GRU and retrained noassoc with positive paired-case and all-five-seed evidence plus verified full computation parity would support a *narrow ability to learn associative recall on this designed diagnostic*, never commercial or real-life skill. Absent such win, say **associative-mechanism advantage NOT SHOWN in this experiment**, not logically rejected for all datasets or scales. If all models underlearn near chance, diagnose undertraining and do not claim uselessness.

**NEXT authentic primary challenge, regardless of synthetic score:** 100k matched GRU native-autoregressive and direct multi-horizon + causal Transformer on authentic independently sourced verified long sequences with identical activity/resource/time available inputs, genuine jointly scored event/absorbing-source-EOR probabilities, proper nll/Brier and calibrated delay intervals, case-paired bootstraps, 5 paired training seeds, truthful original model/optimizer/RNG + score replay, and no existing exposed BPI2012 TEST promoted. In shorter-term exposed analysis, replace particle-derived 3gram horizons with **exact order-three dynamic marginalization** over 37 original activities, add paired case bootstrap CIs, and restrict early/mature query ages to the **same 165 cases reaching 16 events** to distinguish cohort shifts from prefix information. This no-fit analysis stays exploratory. BPI2018 official bytes remain inaccessible; no fabricated real 256-history population. Woyengi predictive outputs remain advisory-only.


### DIAGKEY001 original unqualified synthetic pilot outcome — preserved negative result

**Important: NOT qualified training or FAIR architecture comparison.** On [draft PR #357](https://github.com/FPC-effortless/TAC-OSM/pull/357) first five paired seeded original synthetically generated four-write associative recall models were actually fitted (native TAC fast/slow+8×8 matrix; retrained capacity-reallocated no-association TAC; GRU; 2-layer causal Transformer), with **101554/101560/101500/101616 learned parameters** respectively, exactly same 640 AdamW updates/batch16 for each original seed75001–75005 and identical FIT generated examples per seed/step, held-out **synthetic** per-arm tests 512 random new key/value mappings each at 16,64,256 intervening distractor writes. Value chance **3.125%**. Original five-seed mean **exact answer recall %**:

| Original synthetic distractor delay | Native TAC | Retrained no-assoc TAC | GRU | Causal Transformer |
| -- | --: | --: | --: | --: |
| 16 | 3.75 | 3.40 | 3.79 | **98.98** |
| 64 | 3.32 | 3.71 | 3.36 | **98.83** |
| 256 | 3.55 | 3.75 | 2.93 | **98.79** |

**DO NOT PROMOTE this scoreboard.** First actual five-seed FIT used raw PyTorch `torch.optim.AdamW`, unguarded `optimizer.step`, and direct `torch.save` rather than compulsory whole-model live pre-optimizer budget and checkpoint-bound receipts: the source-controlled scientific CI **FAILED**, and the subsequent first Git replay attempted a moved guarded result path and **FAILED**; it did not independently confirm the original pilot. Original per-seed score JSON and model/optimizer/RNG bytes are preserved under `research/replay-bundles/DIAGKEY001/seed-75001...75005/`, **quarantined as exploratory nonqualified original results, not selectively erased**. The parameter-only parity passes, but measured **PyTorch *partial supported-operation* forward+backward FLOPs** at original batch2/delay256 differ by approximately **20.9%**: native202222328, noassoc180257072, GRU219454912, Transformer227950656. Unsupported fused ops make these partial FLOPs, so total physical compute parity is NOT verified, and the existing 1% compute fairness requirement FAILS. Runtime also differs materially. No model inference or original checkpoint is available to establish production agent intelligence, 256 real-event memory, or real generalization. The almost-chance native/no-association/GRU predictions with near-perfect transformer strongly suggest either a representation/optimization limitation under the chosen 640-step algorithm or a genuine task-specific associative deficit; **they do not distinguish these explanations**, and do not justify claiming that the matrix is universally useless. Transformer full-prefix attention has a different memory-access budget and inductive bias.

**Corrective scientific response:** The [owner-requested narrowly whitelisted original synthetic-only guard](https://github.com/FPC-effortless/TAC-OSM/blob/research/diagkey001-100k-associative-mechanism-2026-10-23/research/mathematical-engineering/diagkey001/FAIR_COMPARISON.json) and actual live model/optimizer/weight-checkpoint receipts were implemented on the same draft branch, quarantining new guarded outputs in `DIAGKEY001_GUARDED`. **No guarded new original scores have been verified yet; do not claim new guard run PASS.** Since partial FLOP mismatch is already known, even guarded repeats **cannot** establish qualified architecture superiority without a new frozen fair-compute design. Any next experiment must preflight both active live parameters and supported FLOP/real training budget before fitting, preserve score labels and five-seed all-case comparisons, and continue treating synthetic work as software diagnosis only. Do not hardcode a four-event warmup or remove all native associative memory based on OOD ME074 readout lesions or this pilot. The decisive research remains authenticated genuinely long-history real world data and a matched 100k native vs GRU/direct/Transformer head-to-head plus independent unopened test.


### DIAGKEY002 — write admission versus retention versus query read, no-fit original model ladder

**Draft [PR #359](https://github.com/FPC-effortless/TAC-OSM/pull/359), synthetic-only no-fit original pilot diagnosis**. Do NOT claim architecture superiority, real-world capability, or invalidate primary real-evidence requirement. Native TAC already implements a rank-one error-corrected delta update (`M ← M + β(v−Mk)kᵀ`) and global matrix-norm protection; **absence of a delta rule is NOT the observed defect**. Actual native code writes every event including distractor WRITE and final QUERY unless its learnable gate suppresses them, and learned keys/readout may be misaligned. First isolate actual write admission, retention under distractor collisions/global normalization, and query-time overwrite, rather than growing 8x8 or hardcoding a fixed early-history gate.

**Independently locally executed math-only idealized 8×8 onehot oracle**: four orthogonal target associations; per delay1000 randomized trials at 0/1/2/4/8/16/64/256 unrelated distractor writes. Correct nonlearned oracle protective write admission recovers exactly 100% at delays0/16/64/256. Unfiltered writes recover 100% / 20.4% / 13.1% / 12.3% respectively; intentionally unfiltered query-time zero write recovers ~13.6% / 14.0% / 12.8% / 11.4% (8-valued oracle chance12.5%), because it erases the queried key. Global whole-matrix norm retention with unfiltered writes similarly loses earlier signal. These arithmetic tests reveal that the same 8×8 *capacity* can retain four ideal orthogonal key-value pairs if appropriate writes are protected, but they do **not** demonstrate that current learned TAC identifies which writes to suppress. A learned 64-key/32-value problem is materially harder. Original local result file `diagkey_write_retain_read_oracle_results.json` created, and full Git-owned independent no-fit oracle/frozen pilot implementation in PR359 must pass source CI before promotion even as software diagnostic.

**No-fit original trained pilot ladder pre-registered**: reuse and SHA-verify all 20 quarantined DIAGKEY001 original synthetic fitted models at ~101.5k params. Score 512 new answer-free synthetic mappings per length/arm/seed on 0,1,2,4,8,16,64,256 write distractors with Wilson95 per original sampled test, NLL, all five original seed trajectories, and autograd norms to initial four WRITE tokens, final QUERY token, write-map and query-map. **Do not call new ladder0–8 learned-success/learned-failure confirmation:** the frozen original networks were trained only on16/64/256 and these shorter sequences lie outside their fitted training distribution. The original pilot failed the live optimizer science guard and had >20% between-arm partial-FLOP mismatch, so any replay remains informative only, not a qualified fair benchmark. DIAGKEY001 later guarded follow-up reported GitHub `action_required` and was not independently verified; `action_required` can indicate GitHub Actions approval/permissions rather than an architecture bug.

**Next actual learned mechanism gate (not yet passed)**: training all new ~100k models with honest equal input examples, useful param count and **matched profiled/total training compute**. Stage zero-distractor learning with pair-count1/2/4 to isolate write+read; fit the whole model under parameter guard and paired 5 seeds; only once zero is learned allow curriculum1/2/4/8/16 and LR sweep (pre-register control of total optimizer/FLOP budget). Include an *actual retrained* no-association TAC, GRU and a learned 8×8 gated-delta positive control; always compare independent held-out generated associations, never claim this synthetic milestone qualifies a real world-model. Distinguish structural capacity limits from optimization floors, and audit gradient to actual write/read weights and key collision. If no variant exceeds chance, the experiment is underpowered and not a test of whether association generally helps; if native solves easier condition but loses as distraction increases, directly test selective gate/orthogonal write protection at the same ~100k scale before expanding memory.

**Primary real-world priority continues**: exact order3 N-gram marginals (currently approximate with finite particles), activity/resource/time feature-matched real baselines, identical same-case prefix comparisons at ages2/6/16, paired original case bootstrap, and genuinely untouched long-event histories from independently sourced authentic records for 100k native versus matched GRU/Transformer/direct-head forecasts. Forecast timing coverage, interval width and actual decision utility are separate mandatory gates. Do not touch ME073 original weights or the untouched robot tissue-box real holdout for software-only tests.


### DIAGKEY002 executed no-fit oracle and frozen original 20-model ladder — five original seeds, NO training

[DIAGKEY002 PR #359](https://github.com/FPC-effortless/TAC-OSM/pull/359), [original detailed frozen no-fit diagnostic](https://github.com/FPC-effortless/TAC-OSM/blob/research/diagkey002-write-retain-read-ladder-2026-10-23/research/replay-bundles/DIAGKEY002/ORACLE_AND_FROZEN_LADDER.json), [SHA-linked compact evidence](https://github.com/FPC-effortless/TAC-OSM/blob/research/diagkey002-write-retain-read-ladder-2026-10-23/research/replay-bundles/DIAGKEY002/COMPACT_AUDITED_RESULTS.json), original detailed SHA256 `f89d44e85a29faba5f0827610d46bf597355d3cb2e3eec6511bad2b3d8ba9b25`. **GitHub actual no-fit score run PASS**. Scientific governance for original first DIAGKEY001 model FIT **FAILED** and the >20% supported-operator FLOP mismatch remains; no real-world capability or compute-fair architecture head-to-head is established. Original old checkpoints bytes and exact current suite are preserved; no optimizer steps or authentic TEST scoring. This is a retrospective controlled software probe on source models trained only with 16/64/256 distractors, plus arithmetic oracle.

**Idealized 8×8 onehot oracle, n1000 per original delay**: four orthogonal target key/value writes can be perfectly recalled if a nonlearned oracle masks any subsequent writes into those protected target keys: **100% exact-value retrieval** at distractors 0/16/64/256. If distractor writes are not masked, exact retrieval instead falls **100%/20.4%/13.1%/12.3%**; query-time overwriting of the queried cell by a zero-valued association returns **13.6%/14.0%/12.8%/11.4%**, compatible with the 8-valued random chance 12.5%. This is not proof of learned TAC gate intelligence: the oracle has explicit ground-truth protected keys; the real task has 64 learned key IDs, 32 distinct value categories, dense nonlinear learned projection, global memory norm bounding and no hardwired target-memory protection. Directly inspect actual learned gate outputs and query-to-write-key alignment before changing the architecture.

**Frozen original fitted 101k architecture models, 2,560 generated diagnostic queries/model/delay across five original seeds** (lower compute scientific qualification blocked; random label chance3.125%):

| original trained diagnostic arm | 0 distractors (outside FIT) | 4 (outside FIT) | 16 (inside FIT) | 64 (inside FIT) | 256 (inside FIT) |
| --- | ---: | ---: | ---: | ---: | ---: |
| native TAC | 2.73% | 2.97% | 3.32% | 3.32% | 3.24% |
| retrained noassoc TAC | 3.20% | 2.54% | 3.95% | 3.55% | 3.16% |
| GRU | 3.20% | 2.30% | 3.36% | 3.36% | 2.89% |
| causal Transformer | **78.01%** | **66.88%** | **98.83%** | **98.44%** | **98.63%** |

The 0/1/2/4/8 challenges were NOT present during fit of the original models. Even the successful full-context Transformer degrades early, so do NOT call native failure on zero distractors a proven inherent deficiency without a **fresh trained zero-distractor fit**. Five fixed-seed gradients for the original native pilot show the path remains differentiable; for seed75001 gradient batch at length256 the first four WRITE encoder L2 gradient **0.00386803**, final QUERY **0.02760562**, associative write-map parameter **0.13410404**, query-read weight **0.02991140**. None of these establish correctly learned key binding, read reliability or good update-gate admission. A live learned gate/write/read trace is implemented on PR359 and requires its own CI qualification before interpreting actual gate magnitudes.

**Scientific conclusion:** A simple rank-one delta rule **already exists** in native TAC; 8×8 *mathematical* capacity is sufficient for the idealized four orthogonal pair retrieval, but the original 100k TAC training did not learn the synthetic 64-key/32-value 4-pair query. The direct learned cause remains open: encoding/query misalignment, indiscriminate distractor or QUERY overwrites, global L2 normalization, gradient optimization or their combinations. Not a sound proof of universally passive associative state. Record negative results even if expensive future architecture proposals seem attractive. Stage next honest 0-distractor trained curriculum under live model budget and separately verified computational parity; do not prematurely add a new DeltaNet clone. The authentic real-world fitted + heldout comparison remains the only primary research promotion gate.


### DIAGKEY003 ACTUAL GUARDED FIVE-SEED TRAINING — native zero-distractor associative learning, replay PASS, promotion gate FAIL

**Implemented and trained**, [DIAGKEY003 PR #361](https://github.com/FPC-effortless/TAC-OSM/pull/361), [independent original five-model Git-owned replay evidence](https://github.com/FPC-effortless/TAC-OSM/blob/research/diagkey003-100k-zero-distractor-fit-2026-10-23/research/replay-bundles/DIAGKEY003/INDEPENDENT_SYNTHETIC_ORIGINAL_ZERO_REPLAY.json), [GitHub Actions original run #39976266996](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39976266996). User-requested **synthetic software mechanism only**, NO real-world evaluation, no multiarchitecture training and no comparative FLOP superiority. One unchanged `Native` fast/slow/8×8 learned delta recurrent model, 101554 actual active/deployed params, five seed75001–75005 fully independent 640 original whole-model **live guarded** AdamW updates each, batch16 on four distinct randomly assigned WRITE key/value observations and one immediate answer-free QUERY, no distractors. Each seed saw 10240 synthetic FIT sequences from its registered deterministic original SHA source schedule. Original whole-model checkpoint+optimizer+PyTorch/NumPy RNG state, exact original binary SHA and live first/every-update model budget receipts are preserved in repository. Clean Git checkout original checkpoint hashes and TEST score replay **PASS** for all five original models, without any refitting. Compulsory scientific preflight **PASS**; independent synthetic original replay **PASS**. A separate repository full-suite CI failure was the stale mutable test-count provenance, **1135 passed / one failed**, corrected count 1136 in the study branch; require a genuinely later green suite before stating full CI pass.

**True separate original fresh 512 synthetic TEST associations per original seed (chance 3.125%, no distractors):**

| original guarded seed | correct/512 | exact accuracy | original TEST cross-entropy NLL |
| --- | --- | ---: | ---: |
| 75001 | 83 | 16.21% | 2.70904 |
| 75002 | 154 | 30.08% | 2.22689 |
| 75003 | 178 | 34.77% | 2.15446 |
| 75004 | 167 | 32.62% | 2.14686 |
| 75005 | 180 | 35.16% | 2.08729 |
| five trained seed mean | — | **29.7656%** | — |

**Prospective hard decision `all five original seed TEST accuracy >=90%`: FAIL**, all five significantly below the fixed target. Therefore **DO NOT** start fitting a new 1/2/4/8/16+ distractor curriculum or claim native associative advantage over any GRU/Transformer. Nonetheless this fitted zero-delay ~30% result, above the 3.125% chance oracle, **refutes the too-strong interpretation** that the current trained architecture cannot learn any query binding even without distractors; the prior 0-distractor old-weight posthoc ~2.73% was OOD because original long-delay fitted data had no zero-delay cases. Real model gradient and gate traces from DIAGKEY002 show nonzero actual differentiable write/read pathway, near-zero learned query/original-write-key cosine and continued distractor writes. But even after directly training on the simplest four-pair retrieval task for 640 guarded steps, no original seed learned reliable accurate lookup. Original DEV zero-delay learning curves were still rising late in training (e.g. seed75004 DEV 23.4% at step512 vs 43.75% at640), so distinguish training-budget/optimizer/gradient distribution from structural/8×8 slot limits. If changing optimization, register a genuinely NEW synthetic-only diagnosis with new independent generated splits; do not tune against these exposed TEST labels and call it a clean confirmation.

**Next decision now:** No reflexive expansion to 256 distractors or wholesale conversion to a DeltaNet or SSM. Analyze learned `WRITE`/read `key alignment` and allocation of four targets within eight key directions using teacher-forced target-key diagnostic as a positive control. If undertaking a 0-delay LR/curriculum/structured association supervision test, pre-register independent source+TEST generation, identical active ~100k capacity, measured compute budget and original five-seed checkpoints before fitting. Persist the original negative result without rewriting historical weights. Only authentic externally gathered long histories and balanced real-world matched ~100k neural controls can qualify operational world-model intelligence for Woyengi or robotics.


### ME077A original exposed robot movement-discrimination audit — 5 frozen native seeds, no fits

**Actual new original genuine-real no-fit result:** [ME077A draft PR #367](https://github.com/FPC-effortless/TAC-OSM/pull/367), [Git-owned score receipt](https://github.com/FPC-effortless/TAC-OSM/blob/research/me077a-real-motion-risk-latest-base-2026-10-23/research/replay-bundles/ME077A/ORIGINAL_REAL_MOTION_RANKING.json), [successful original GitHub execution #39994637279](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39994637279). Five ORIGINAL 102,503-param native System1 ME077 checkpoint predictions, original exact query and source-file hashes, all already-exposed historical true physical next states, zero model fits or new trained architecture, no sealed TEST or tissue box. Real eligible population 960 actual next-state queries across 40 genuine original robot recording episodes, 84 actual movement-rich and 876 near-stationary (retrospective >=0.025 mean14D displacement diagnostic). Five seeds predict same original episodes and are NOT 4,800 independent real queries. Original SOURCE remains historically exposed DEV.

**New factual capability finding:** auxiliary observed-source motion-head probability has modest original-real discrimination, not reliable early warning. Five-seed mean ensemble **ROC-AUC 0.644121**, predeclared 2,000-resample original episode-block 95% CI **[0.593817, 0.696739]**; **average precision 0.170556** (95% episode-block [0.124249,0.242110]) against observed event rate **8.75%**. Predicted probability is only **10.84%** for truly moving cases vs **9.82%** for stationary. Highest-ranked original 96/960 predictions catch **20 of 84** actual movements (23.81% sensitivity, 20.83% precision). Registered threshold0.10: **53 TP,401 FP,31 FN,475 TN**, recall63.1%, FPR45.8%. At threshold0.25 or0.50, **0/84** moving cases identified. All five original fitted seeds individually had ROC AUC 0.597875–0.651106; this is a small effect, not cross-site generalization.

**Probability usefulness:** ensemble motion Brier **0.0786957** vs **0.07984375** for the hindsight mean REPORT motion-rate constant oracle, just **1.44% relative improvement**; 40-episode bootstrap for Brier difference **[-0.002015, -0.000307]** conditions on original exposed cases and fitted weights. The 8.75% true REPORT prevalence is a **retrospective oracle baseline**, NOT available to a real deployed observer as an empirical prior: this comparison is descriptive, not practical matched online baseline proof. Fixed reliability bins show forecasts clustered entirely 0.05–0.2; the 14D coordinate-wise mixture-component probabilities are NOT global motion-event probabilities (proxy ROC AUC only0.5765). Motion label identifies observed next displacement, not an independently annotated hazard, motor torque effect or causally controllable event. Factual no-fit diagnostic at real robot *does not* qualify near-term movement anticipation, action selection, SLA, device latency or customer benefit.

**System 1 lane state as of ME077A:** Original ME077 integrated selective physical predictor itself improved exposed all-state MAE from ME076 0.080158 to0.066476 and improved 3.19% over last-state persistence0.068668 on same actual 960 queries, with original 40-episode paired MAE advantage. BUT **original preregistered >=5% all-five and moving subgroup primary gain gates remain FAILED**: movement-rich MAE ME0770.741981 vs persistence0.743241 (~0.17% relative), FIT servo0.704517 substantially better in that outcome-conditioned stratum; gross 90% coordinate interval coverage93.96% with mean original 0.30094 physical width. New ME077A aux event rank audit confirms subtle movement-state limitation despite physical stationary forecast gains; report as `MIXED_TRADEOFF`, never general intelligence or safe reflex.

**Next System 1 capability test, not another reflexive no-fit cosmetic tweak:** Train new pre-registered original ~100k physical model to distinguish **when change is likely** and **how much conditional residual matters**, with a real decision-time prior FIT-derived event probability (no hindsight REPORT prevalence), genuinely observed actual dynamics, a calibrated global change risk, and physical state median/uncertainty from one consistent distribution. Evaluate actual precision/recall and false-alarm burden as prospective primary objectives alongside movement-rich and stationary MAE; compare original practical physical persistence and servo, old ME077 on exact same exposed DEV case IDs first, and reserve independent untouched source for confirmation. If wanting an ARCHITECTURE superiority claim, add strictly parameter+training-data+compute-matched GRU/SSM controls at 95–105k and original five seeds. Preserve training no-policy rule and noncausal logged command timing disclosure. The fully independent enterprise lane requires its own appropriate authentic event source and equal-input GRU/Transformer/direct horizon evaluation, and cannot be counted as qualified based on ME077A.

## 2026-10-24 — native computational-graph development, ME082–ME086

Owner priority remains improving the **trained complete model**, not attribution.
[Draft PR377](https://github.com/FPC-effortless/TAC-OSM/pull/377) implements the
native graph continuation. ME082,ME083,ME084 each completed five cold-start real
fits,2048updates,B24,T32,104539learned parameters. All15 originals, optimizer/RNG,
sampler receipts and8/32-frame scores are Git-owned and frozen before scoring.
They all **FAIL the full unchanged predictive gate / MIXED_TRADEOFF**. Primary
H1 MAE0.0530266/0.0527303/0.0544514 versus ME0800.0593931; actual-moving
H1 0.408234/0.397165/0.390429, but stationary H1 0.018966/0.019702/0.022234
remains far worse than ME0770.001701. Strict four-frame and paired cluster-CI
requirements also fail. No successful general intelligence or architecture
superiority is inferred from improved averages. See the [complete attempt
ledger](https://github.com/FPC-effortless/TAC-OSM/blob/research/me082-equilibrium-parallel-2026-10-24/research/equilibrium-search/RESULTS.md).

Full float64 reference inference using original trained float32 values resolves
strict external original replay for ME082/083 at original rtol1e-5/atol1e-6:
27600forecast records each. This is a new qualified numerical path, not a
retroactive change to any earlier failed tolerance or FP32 throughput claim.
ME084's independent receipt is preserved in the same study branch.

A separately prospective bounded ME085/086 search is now training the
99731-parameter **ClockValueWorld**: native elapsed-rate fast/slow/8x8 memory,
parallel time-decayed local filters, a shared horizon head and learned convex
retrieval of recent *raw observed values*. Same actual source and protected
cohorts; only past/current capture elapsed is added. Future labels, future
commands/capture times, absolute epoch, task/episode IDs are not features.
Publisher nominal retained-frame timestamps differ from recorded capture time;
FIT contains325capture gaps, but only138/180324eligible training windows contain
a gap, so clock aliasing is **not proven to explain the prior failures**.

Two4096-update five-seed cold-start recipes were fixed before fitting: NLL
weight1.0 then0.2, stop at first complete gate plus original independent replay.
Keep every ME081 gate and the stronger ME082 conditions; additional99%paired
CI addresses these2+3declared attempts, not all historical adaptive research.
Exposed DEV tuning remains exposed. At this update **no ME085 REPORT score or
ME086 fit exists**; do not claim a learned gain from graph engineering.

The real-FIT pure tensor export has2579nodes, all20parameter tensors receive
finite nonzero training gradients, and chunk/step/causal-prefix equivalence
passes. Encoder, time features, local filters, retrieval and horizon queries
expose parallel axes; native nonlinear recurrent state remains sequential.
The explicit caller-owned inference cache verifies entity separation and
idempotent retries without changing model weights. This is execution correctness,
not long-history prediction, autonomous policy intelligence or Woyengi deployment.
Protected tissue-box/CAL/TEST remains CLOSED; no merge/deployment/customer contact.

### ME085 frozen outcome and ME087 substantive correction — 2026-10-24 continuation

ME085's five99731-parameter originals completed4096guarded updates each and
were Git-frozen atf869884e before all REPORT scoring. **Full gateFAIL /
MIXED_TRADEOFF**, with H1/H4/H16 MAE0.05338952/0.15759004/0.68190478.
Actual-movingH1 is0.38013820 but near-stationaryH1 remains0.02205745. Four
conditions fail: quiet precision, stronger ME081 H1target, original95%episode
CI and additional99%CI. Physical marginal NLL and interval-score guardrails
pass at all horizons. Exact original scores/weights remain in draftPR377.
Independent GitHub replay40067423426 passed27600original forecast records;
committed B2,T32/B2,T1 compiled reference programs also restored and agreed on
actual FIT inputs. Same-model compiled timing1.83xprefill/1.61xstep is scoped to
those profiles and shared-workspace measurements, not a production SLA.

The separately frozen ME086 recipe is executing unchanged. ME087 was registered
before any of its updates as a **new integrated architecture/training objective**,
not a hidden replacement:99731active/deployed parameters; quiet raw-value
borrowing softsign-bounded by FIT noise*sqrt(h), every coordinate; same native
elapsed-rate state/filter/associative dynamics; same coherent mixture; actual
median directly trained through the implicit CDF derivative dq/dtheta=-F_theta/f.
The original ME085 error-location audit finds95.18%of quiet absolute error in
the two gripper coordinates; this is location, not proof of mechanistic cause.
No gripper-specific inference condition or future-outcome router was added.

The implicit derivative's initial zero-CDF derivative draft failed a software
check before fitting; its no-fit registration is preserved and the corrected
safe piecewise derivative passed independent finite differences before v2
source freeze. Real-FIT export2593nodes, all20parameter tensors finite nonzero
gradients. Five cold starts,4096updates,B24,T32, proper CRPS+0.2NLL+0.2motionBCE
+1.0proper median absolute error, final-only weights. All prior gates stay
fixed; additional99.1667%paired CI includes all6declared attempts, not arbitrary
historical adaptation. Both currently scheduled studies remain **UNSCORED**
at this update; no new gate pass is claimed. No additional fit under completed
ME085; no protected source use, winner, control, deployment or customer claim.

### ME086/087 completed; ME088 fixed-index native fine tuning — 2026-10-24

ME086 also completed all five originals, frozen atd9780695, and failed its full
predictive gate: H1/H4/H16MAE0.05168648/0.15890228/0.68835176; quietH1
0.02097012 and both original95%/additional99%paired CI conditions failed.
Independent strict original27600forecast replay passed40068971618.

ME087's five originals completed4096updates each, frozen at526a14ae before
REPORT, raw scores frozen atcac5aad9. **Full gateFAIL/MIXED_TRADEOFF**:
H1/H4/H16MAE0.05088844/0.14812261/0.68210592; actual-movingH1 0.36353597,
quietH1 0.02090854. Original95%episode CI(new−ME080) is
[-0.01673979,-0.00027164], but the additional99.1667%CI includeszero. Quiet
precision remains far above the unchanged1.10*ME077target (~0.00187161).
Every other numerical original/stronger forecast gate passed. Independent
whole original replay and derived compiled-profile restoration passed
[40070245236](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40070245236).
This is useful development progress, not a full qualification or unseen-site
intelligence. No further fits under ME086 or ME087.

A **single separately registered ME088 fine-tuning recipe** now starts from
ALL5fixed ME087 final originals, one-to-one seed-index pairing; no best-seed
selection or ensemble. Every99731weight remains trainable; new optimizer/RNG,
2048additional updates each,4096old+2048new=6144total lineage updates. Original
weights/optimizer/RNG remain unchanged. At most two independent one-thread seed
processes run; no shared gradients/RNG, no extra seeds, no performance-based
scheduling. Final-only weights and all5originals before REPORT remain compulsory.
Source code, warm-start SHA and objective were Git-frozen atbbf30a57 before fit.

The model separates predictive innovation units from quantized-observation noise
(max(FITquiet_noise,0.01FITdelta_scale)), uses a smaller known-horizon nonlinear
trust factor at one step, adds factual coordinate-change routing supervision,
and an explicit8xone-step quiet-median precision penalty alongside prior proper
scores/implicit median. These are training labels, never future inference inputs.
The quiet-preservation multiobjective does not guarantee unweighted posterior
calibration: unchanged NLL/coverage/proper interval gates still apply. All seven
declared attempts are disclosed; additional99.2857%CI is descriptive of those
attempts, not independent historical-adaptation correction. Protected tissue-box/
CAL/TESTstay CLOSED. ME088 is training and has no REPORT score at this update.
No architecture-winner, causal action, production, merge or customer-value claim.

### ME088–ME090 completed negative joint gates; visual source qualification — 2026-10-24

All 15 original fixed-parent fine-tunes completed, five per registration, each
from its corresponding ME087 original. Every fit used 2048 additional updates
(new optimizer/RNG; base lineage 4096+2048); no best-seed selection. Original
weights were committed before REPORT scoring. Exposed DEV only; sealed sources
remain closed. None passed the unchanged full joint predictive gate.

| Study | H1 MAE | Moving H1 | Quiet H1 | H4 MAE | H16 MAE |
|---|---:|---:|---:|---:|---:|
| ME088 | .06462415 | .72320615 | .00147245 | .14623004 | .60987751 |
| ME089 | .05846031 | .62616197 | .00402316 | .15295355 | .62642236 |
| ME090 | .05521041 | .53075936 | .00960983 | .15017947 | .61691413 |

ME088 meets quiet precision but loses moving accuracy. ME089 adds a causal
coordinate-local router. ME090 additionally predicts future recorded references
from past information; its reference MAE is worse than last-reference persistence
at every scored horizon. This is a preserved negative capability result, not
causal action intelligence. ME090 originals frozen at327593b3; original forecasts
and complete compiled graph at889f71ae. All 45 trained versions ME082–ME090 remain
preserved. Independent strict original/compiled replay passed for ME088
[40087528374](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40087528374)
and ME089 [40087528353](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40087528353).
ME090 independent score replay remains pending at this update.

The next substantive capability direction is authentic synchronized visual
observations, if source alignment/acquisition qualify. Pinned publisher metadata
lists overhead and both wrist cameras, each360x640 RGB video. Camera existence
is verified; usable alignment, exact source hashes, role-limited access and
resource feasibility must be established before any visual training. Contact or
stopping information absent from qpos is a hypothesis, not an established cause.
No new visual model has trained and no gate has been relaxed. Added modalities
would preclude an information-matched architecture-superiority claim.

### ME091 integrated causal visual/native candidate — training, 2026-10-24

ME090 original and compiled replay now independently PASS
[40089178025](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40089178025),
without changing its failed predictive gate. ME082–ME090 preserve45 fitted
versions and every negative outcome.

ME091 is a new complete multimodal capability recipe, frozen before fitting
at14cd30e5, with exact licensed input pixels/runtime effects added at0acab771.
**104225 active trainable/deployed weights**: retain94907 original values via
explicit paired SwiGLU width116→108 mapping; initialize7792 shared visual weights
and1526 common-location weights. All five fixed ME087 parents are used without
selection. Each gets4096 new AdamW updates (8192 cumulative retained-base updates;
new visual/location weights get4096), final-only, at most two independent workers.
No selected original scores have been computed at this update.

All160 overhead videos from the SAME120FIT/40exposedREPORT episode roles passed
publisher SHA256 and retained frame-count checks. Full source size4633584032bytes;
exact32x32 grayscale observed inputs114832136bytes are now Git-owned withCCBY4.0
BodenAI/RW-RL attribution. No CAL/TEST/tissue-box, new episode or future image was
opened as an inference input. Recorded frame alignment does not prove sub-frame
physical synchronization. Added images preclude an information-matched winner claim.

The complete graph shares batch/time image encoding and numeric projections,
retains elapsed-time bounded native fast/slow/delta memory, batches horizon
readouts, and adds a shared bounded location translation to BOTH mixture means.
The translation preserves one coherent proper distribution while allowing point
location to change without forcing probabilities/shape to perform that role.
No extra memory rollout, generated observation or action selection is used.
The unfitted104508-parameter draft is separately retained; no fits under it.

Real-FIT export2804nodes, all29learned tensors finite/nonzero gradients. Full
local regression1192PASS. Functional chunk/step/prefix causality and explicit
camera-aware cache retry/revision/entity isolation passed. These are software/
execution evidence, not predictive gate success or validated intelligence. The
fixed joint quiet/moving/multi-horizon gate remains compulsory, plus original95%
and additional99.5%paired intervals for ten declared recipes; adaptive exposed
DEV is not independent confirmation. All original models must be Git-frozen
before scoring, then independently replayed at unchanged tolerances.

### ME091 full negative result; ME092 shared-gradient refinement — 2026-10-24

ME091's ALL5 originals frozen at925c8c5f, all original forecasts0aaa8cd4.
**Full gateFAIL/REGRESSION**. H1/H4/H16MAE0.06244503/0.14998120/0.62834285;
H1moving0.66996371, quiet0.00418982;0/5paired H1wins againstME080. Original95%
episode CI[-0.00506261,+0.01176421], additional99.5%CI[-0.00878291,+0.01544684].
NLL also worsens againstME081 at every horizon. Images/common-location therefore
do not establish useful added visual intelligence. No extra fits under ME091.

Independent original27,600forecast replay, normalized compiled program replay
and raw physical/pixel-to-physical-output boundary restoration all PASS
[40092527317](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40092527317).
Fixed-profile physical graph3904prefill/1590stepnodes; first-original shared-CPU
speedups1.606x/1.593x, not all-shape or rival speed evidence. Local1193testsPASS.
Every previous outcome is preserved; ME082–ME091 now total50trained versions.

ME092 is ONE new integrated optimization recipe, Git-frozen at0ca5ed1d before
fits. Same104225parameter visual/native architecture and exact same observed
image/numeric inputs; no new episodes, future inputs or protected source access.
ALL5fixed pairs donate94907native values fromME087 and7792learned visual values
fromME091;1526common-location values start atzero. No best-seed choice or ensemble.
2048new updates each, all trainable; root4096+visual-pretraining4096+new2048=
10240pipeline updates. Retained native and visual values each receive6144updates
along their own lineage; fresh location2048. All discarded parent training is
still charged and originals retained.

The original importance-weighted CRPS+0.2physicalNLL+medianMAE+0.2globalmotionBCE
score is partitioned into quietH1/movingH1/long-horizon tasks, exactly once per
valid label. Coordinate-component BCE and extra quiet penalty are removed;
a mixture component is not an observed movement class. One shared forward and
one batched3task VJP provide PCGrad directions (Yu etalNeurIPS2020), with rotating
deterministic projection order,clip1,AdamWlr0.0001. Three-task projections/AdamW
do not guarantee joint descent, calibration or success. Actual CPU cost is retained.

Real-FIT preflight verifies the scalar/gradient partition against the independent
original proper-score function, within original1e-5/1e-6gradient tolerances.
1196local testsPASS. All5fits are running; no ME092REPORT score at this update.
Every unchanged full predictive gate plus original95%/additional99.54545%paired
intervals and strict independent original replay remain compulsory. No frozen
TEST, architecture winner, autonomous control, deployment or customer claim.

### ME092 completed; ME093 three-regime predictive architecture — 2026-10-24

ME092's five originals were frozen at d3665d4e before REPORT scoring at01b55060.
Full joint gate **FAIL/MIXED_TRADEOFF**: H1/H4/H16MAE
0.05126858/0.13945950/0.61481598; H1moving0.35612416, quiet0.02203585.
All NLL/coverage/interval/task/seed and secondary32forecast conditions pass,
but quiet error and both paired confidence-interval conditions fail. Original95%
H1difference CI[-0.01670057,+0.00032829], additional99.54545%
[-0.02104607,+0.00407439]. No extra ME092 fits;55versions ME082–ME092 now preserved.
Independent strict original27,600forecast replay, normalized compiled restoration
and raw physical/pixel boundary restore completed successfully in
[40095957627](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40095957627),
head01b55060. REST named-step receipts preserved; signed Azure stdout downloads
were403, so no unobserved per-field log measurements are claimed.

ME093 is a new integrated native architecture recipe, frozen **51120067** before
training. It retains native fast/slow/associative recurrence, real observed elapsed,
shared causal numeric encoder and CNN, and one shared SwiGLU horizon readout.
It replaces two competing modes with a **single proper hold/local-noise/transition
Laplace mixture**. Current observation anchors hold, a learned convex combination
of current and causal3value median centers local noise, and the retained native
physical predictor centers transitions. Positive ordered scales permit exact
finite-mixture CRPS and stable likelihood; all quantiles invert that same CDF.
No independent point-forecast averaging, component-class labels or hardcoded
future movement. Normative means, gradients and densities are independently tested.

Actual96,595active learned values:82,699native,7,792visual,1,526dynamic-location
retained from ALL5fixed indexedME092parents;4,578newlocal-noise values. Remove112old
lag-logit rows; no dummy parameters or frozen learned encoders.3,072new updates/seed,
retainedLR0.0001/newLR0.0003, whole-model AdamW and original proper3task PCGrad.
All5originals/optimizer/RNG must enter Git before REPORT. Pipeline13,312updates,
retainednative/visual9,216ownlineage,dynamic-location5,120,newfields3,072.
1201local tests and real-FIT active-gradient/loss-partition preflight PASS.
Five original fits are running at this update; **no ME093 result yet**.

All original predictive thresholds remain compulsory, with original95%and additional
99.58333%paired intervals for12declared recipes. Same120FIT/40exposedREPORT,
exact Git-owned licensed overhead pixels; protected CAL/TEST/tissue-box unopened.
No new independent generalization, architecture winner, policy intelligence,
customer value or deployment claimed. Full mathematics/input-to-output graph:
[ME093](https://github.com/FPC-effortless/TAC-OSM/blob/research/me082-equilibrium-parallel-2026-10-24/research/mathematical-engineering/me093/MATHEMATICS.md).

### ME093 complete negative; ME094 explicit median architecture — 2026-10-24

ME093ALL5originals5ec26e40/postfit e5a2f36b, original scores efc5e625.
Full gate **FAIL/MIXED_TRADEOFF**: H1/H4/H16MAE
0.05053248/0.13904939/0.59860214; H1moving0.36073998,quiet0.02078656.
Only quietH1 and additional99.58333%paired interval fail. Original95%CI
[-0.01717568,-0.00049938] passes, stricter interval[-0.02099906,+0.00287236]
does not. No extra ME093fits. All60ME082–ME093versions are preserved.
Independent27,600forecast original replay, normalized compiled and raw physical
restores PASS [40112348639](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40112348639),
head efc5e625; exact audit payloads retrieved with connected GitHub app. ME092's
previous signed-blob log403 was similarly resolved; exact original full audit is
now retained. Reproducibility does not change either predictive failure.

ME094 registered **c87dc63c** before fits: one new101,173parameter integrated
native model, ALL5indexedME093parents,96,595retained+4,578new decision/retrieval
values, alltrainable. Add time/content relevance over the causal eight-value bank
and an independent learned point median m. For proper multimodal base CDF B and
its median q, translate every component by m-q: P(y)=B(y-m+q), so P(m)=.5.
Preserve implicit derivatives through q; do not detach shape-centering gradients.
Probability weights/scales no longer directly determine the point; shared physical
means still participate in both. No missing uncertainty or separate unrelated bounds.

Complete objective CRPS+0.2physicalNLL+20medianMAE+0.2factualglobalmotionBCE;
pointweight increases for BOTH quiet and moving examples. Same3taskPCGrad,
retainedLR.0001/newLR.0003,AdamWclip1.4,096newupdates/seed;17,408chargedpipeline,
retainednative/visual13,312ownlineage,location9,216,oldlocalnoise7,168,new4,578values
4,096updates.1206local tests and real-FIT all33activeparameter-gradient preflight
PASS. Maximum absolute partition-gradient difference9.155e-5 passes unchanged
relative1e-5/absolute1e-6contract; it is not an absolute-error-below1e-6 claim.

Five fits are running; **no ME094REPORT result yet**. Same120FIT/40exposedREPORT,
exactlicensedpixels, protectedsourcesunopened. All old predictive conditions plus
original95%/additional99.61538%paired intervals for13declaredrecipes and strict
original replay remain. Scoring uses numerically checked final-only decode to
avoid discarded prefix readouts/centering roots; dense-prefix training unchanged.
No independent/site, architecture-winner, policy/control or customer claim.

### ME094 derived computation reuse, predictive result pending — 2026-10-24

Two originals completed4,096updates each; remaining portfolio still training at
this update. No ME094REPORT forecasts or predictive-success claim yet. Same101,173
parameter full model; all original thresholds and five-model Git freeze remain.

First-original real-FIT normalized and physical-input compiled graphs reproduce
the original quantile/state calculation under unchanged1e-5/1e-6. Physical fixed
B2T32prefill9.013506ms, B2T1incremental3.463949ms,1.61–1.66xversus their own eager
paths. This is MOREabsolute computation thanME093, not an across-model speed win.
Actual restored complete graph has4,927/2,489nodes; serialization canonicalization
changes counts versus pre-save graphs, not arithmetic work. One folded183x108
projection and final-only native representation projection are verified.

The explicit median permits a query-selective derived view: if a caller requests
only median/risk/newstate, compiler dead-code elimination removes the CDF solves.
The SAMEpoint/risk/all13state outputs match the complete original calculation.
Actual first-FIT fixedB2T1point-view0.739726ms vscomplete3.434416ms(4.64x);
B2T32prefill5.325116vs8.259853ms(1.55x),31alternating shared/unpinnedCPUrounds.
Video decode/handler/network costs excluded. Not single-entity/all-shape/hardware
promises or learned adaptive computation. Complete probabilistic capability stays
available; no new smaller trained arm, parameter-budget exception or equal-output
rival comparison. Independent clean restoration remains pending all5frozen models
and original scores. Numerical and replay qualification stay separate from learned
capability success. Full1,207local testsPASS; all training-source hashes unchanged.

### ME094 completed negative; ME095 selective context/state architecture — 2026-10-24

ME094all5originals182b3397/postfit907afd48 preceded original scores1ad5e625.
Full gate **FAIL/MIXED_TRADEOFF**. H1/H4/H16MAE
0.05005783/0.14339711/0.63722543; movingH1 0.36156072,quietH1 0.02018769.
QuietH1 and additional99.61538%CI fail. Original95%CI
[-0.01764886,-0.00102579]passes; stricter[-0.02172001,+0.00241638]doesnot.
No more ME094fits;65originalME082–ME094models preserved. Independent27,600forecast,
normalized/physical compiled and point-query restore PASS
[40118547708](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40118547708).
This verifies reproducibility, not full predictive success.

ME095 registered **00295abc** before fitting: one104,689parameter native model,
ALL5indexedME094parents101,173retained+3,516newlearnedvalues. Add known requested
operation context (five types), current capture-clock sin/cos at fixed1/4/16second
periods, and a bounded selective physical observation filter. No episodeID,
curator-relative age, total duration, ending, future clocks or same-frame command.
Task/context is an information advantage over old models, so no equal-information,
parameter- or compute-matched architecture superiority claim is eligible.

For input-only encoded event e and observed positive rho, retention
A=exp(-softplus(W e+b)*rho), filter k'=A*k+(1-A)*[observedlevel;rawvelocity].
28filter scalars carry level/velocity. Affine segments compose associatively as
(A2*A1,B2+A2*B1); a causal prefix scan exposes logarithmic branch depth without
unstable division. Its values/gradients match independent serial computation.
The native coupled fast/slow/associative recurrence remains nonlinear/serialT.
A filter-evidence projection enriches its input; filtered velocity enters the
physical expert. Complete state15fields/1,359scalars. Global model bounds are not
physical-safety or useful-long-memory proofs.

2,048newupdates perseed, alltrainable; retainedLR.0001/newLR.0003, original proper
ME094objective/3taskPCGrad. Pipeline19,456updates; native/visual15,360ownlineage,
location11,264,oldlocalnoise9,216,decision6,144,new3,516values2,048. Full1,214local
testsPASS and realFIT all40parameter tensors finite/nonzero gradients. Earlier
maxabsolute gradient-partition difference7.248e-5passes unchanged relative1e-5/
absolute1e-6contract; not an absolute-error-below1e-6 claim.

Fits underway; **no ME095REPORT score yet**. All original numerical requirements
plus original95%/additional99.64286%paired intervals for14recipes and strict
independent replay remain compulsory. Allfive original bytes beforeREPORT.
Same120FIT/40exposedREPORT and exact licensed pixels; protectedsourcesunchanged.
Typed cache now keys clock/operation context as well as observed numeric/image
inputs and model revisions; raw physical execution includes clock preprocessing
and task-conditioned state initializer. No new policy, deployment/customer,
independent generalization or general-intelligence claim.

## PNDS v1 mathematical-engineering authority for all TAC-OSM research lanes — October 24, 2026

**PNDS is the authoritative, versioned mathematical representation and proof-engineering repository for TAC-OSM**; it is not a competing predictive model. The [cross-repository foundation bridge](PNDS-UNIFIED-MATHEMATICAL-FOUNDATION-v1.md) traces Core/DIAGKEY, System 1, physical, enterprise, Temporal, Edge-Lite, Counterfactual, and ES-WAM to a [source-commit-pinned PNDS v1 registry](https://github.com/FPC-effortless/pnds-lean/blob/research/unified-tac-osm-mathematical-foundation-v1-2026-10-24/foundation/v1/registry.json), [mathematical computational graph specification](https://github.com/FPC-effortless/pnds-lean/blob/research/unified-tac-osm-mathematical-foundation-v1-2026-10-24/foundation/v1/ARCHITECTURE.md), and append-only historical positive/negative research evidence ledger. [Draft PNDS PR #5](https://github.com/FPC-effortless/pnds-lean/pull/5) adds explicitly restricted exact-real Lean proofs and live CI audit; the original PNDS v0.2 source and prior independent formal PR#1/#2/#4 are preserved. Historical DEV/ME/DIAGKEY/ESWAM ranges not yet individually indexed remain **explicit open archival gaps**, never silently lost or claimed covered.

All model weights, optimizer/RNG, source/TEST hashes, actual real results and negative research gates remain authoritative in original TAC-OSM experimental branches. A Lean operator proof does not establish whole-model float implementation equivalence, real-world intelligence, causality, architectural superiority, safe autonomy or customer ROI. The prospective 95k–105k learned-parameter and <=1% comparator parity, same real training data and COMPUTE (not CPU wall time), original independent checkpoint replay and authentic independent-source evaluation remain compulsory. No model is merged or trained by writing this bridge.


## Binding PNDS ↔ TAC-OSM dual-language mathematical engineering (2026-10-24)

For all architecture variants and input-to-output computation graph changes, the compulsory rule is [PNDS MATHEMATICAL_ENGINEERING_RULE.md](https://github.com/FPC-effortless/pnds-lean/blob/research/unified-tac-osm-mathematical-foundation-v1-2026-10-24/foundation/v1/MATHEMATICAL_ENGINEERING_RULE.md), mirrored here as [research/PNDS-MATHEMATICAL-ENGINEERING-RULE.md](PNDS-MATHEMATICAL-ENGINEERING-RULE.md). PNDS is the source of versioned mathematical representations, **both Lean 4 and Python reference formulations**, proof premises and open proof obligations; TAC-OSM remains the actual PyTorch/training/authentic-score/checkpoint authority. Every version must appear in the reciprocal [source-pinned variant links](pnds-variant-math-links.json); historical negative findings must never disappear.

Engineering-gate levels are distinct: source-linked/partial math, kernel-verified exact theorem, tested Python math, actual original-source numerical/gradient equivalence, then independent authentic experimental capability. The first three do NOT imply the last two. No completed whole-model verification is claimed for the 17 historic indexed versions. The prospective capability-first and fair parameter/data/training-compute rules remain unchanged; **equal CPU seconds are not required**.

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
