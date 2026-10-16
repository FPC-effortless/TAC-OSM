# DEV-033: PNDS trace conformance and optimization continuation

The PNDS source is pinned to
[`3962fd11a75420481c12502ab16c95c2d064e0aa`](https://github.com/FPC-effortless/pnds-lean/tree/3962fd11a75420481c12502ab16c95c2d064e0aa).
The [protocol](../research/dev033/protocol.json) records hashes of all six Lean
modules. Sources were inspected; Lean was not built or re-proved in this study.

## Completed backend diagnosis

The [frozen comparison](DEV-032-BACKEND-COMPARISON.json) finished in private
[Kaggle version3](https://www.kaggle.com/code/eweewee2/tac-osm-dev032-frozen-gru-diagnosis/versions/3).
All three backends used identical finite pre-update parameters and the same
fingerprinted training batch at attempted update286, with no optimizer steps:

| Backend | Forward loss | Non-finite gradient elements | Maximum forward-logit difference from CPU |
|---|---|---|---|
|CPU FP32|0.57120311|1,802,668|0|
|CUDA/cuDNN FP32|0.56964409|1,802,668|0.36030912|
|CUDA unfused FP32|0.57403910|1,802,668|0.11981130|

Saved gradient-file hashes and every reported non-finite count were independently
checked on CPU. The failure is not confined to cuDNN or CUDA at this point.
Forward differences show backend sensitivity; they do not establish its cause,
the first unstable update, or an implementation bug. The exact same-runtime
replay requirement remains unchanged. DEV-031's original trial remains failed.
This is not an architecture-capability result or a complete numerical analysis.
The earlier RUNNING status snapshot is historical and superseded by this report.

## Mathematical claim boundaries

The two-level softmax expression is exact when all relevant scores share one
level and distractors share another. A minimum relevant margin gives a sufficient
bound more generally; the corresponding threshold need not be necessary when
relevant scores vary. Top-k recall is measured separately, including the case
k is smaller than the relevant set. Scoring every candidate still costs work:
constant selected width is not constant indexing, writing or total computation.

The decay of a product of calibrated conditional step probabilities does not,
by itself, force a length-dependent admission threshold. If p is a correctly
specified path-correctness probability conditional on all decision information,
then accepting p >= 1-epsilon bounds conditional error by epsilon. The same
threshold can work across lengths if that conditional calibration holds within
each length. A fixed threshold does not establish calibration, and aggregate
calibration can hide length-specific errors. Length-stratified held-out
calibration remains necessary to test the proposed empirical guarantee.
PNDS's current constant threshold and definitional product identity do not prove
that guarantee. We do not modify or claim a new Lean theorem here.

The cardinality identity in Scaling does not prove retrieval or asymptotic total
cost. Causal.do_op and causalContrast remain axioms, not an executable SCM.
Registration's conjunction is a contract, not evidence its Boolean inputs measure
utility, stability, integrity and cost. The three-seed sign-test bound concerns
that exact seed-level sign test, not an impossibility theorem for every statistical
test. Twenty seeds also do not guarantee adequate power or control selection bias.

## Executable conformance layer

`pnds_conformance.py` provides independent measurements of softmax distractor
mass, minimum relevance margin and selected-index recall; false admission and
contamination with their different denominators; a fail-closed four-gate shadow
decision; and Brier score, binned ECE and false-commit fraction by path length.
Empty denominators return unavailable. Incomplete semantic labels leave population
admission rates unidentified. Empirical rates and ECE are descriptive, without
confidence bounds or a population calibration guarantee.

The shadow gate requires explicit Boolean measurements or unknown values.
It does not calculate utility/stability from verifier probabilities, certify
evidence, or change PersistentStore admission. Online four-gate enforcement
remains unfinished. Proposed measured gates need prospectively fixed utility
comparisons, fresh-seed stability estimates, independently audited provenance,
and marginal cost measurements, including uncertainty and held-out selection.

The DEV-032 trace adapter deliberately leaves correctness labels unknown.
Its public `accepted=True` receipt is not validity. Forced verification fixtures
are excluded from the ordinary trace population. This adapter cannot establish
verifier calibration, contamination control or learned four-gate registration.
Eight contract tests cover analytic softmax values, incomplete top-k recall,
Bayes denominators, missing labels, gate type safety and calibration strata.
These are deterministic software checks, not learned routing experiments.

| Property | Executable status | Still needed for a learned scientific claim |
|---|---|---|
|Relevance and scaling|Metric implementation; actual relevance labels absent|Defined relevant targets, distractor sweep, recall, indexing/write/scoring/execute costs|
|Four-gate registration|Shadow conjunction; all actual gate evidence unknown|Measured frozen gate policy, real admission enforcement, independent audit|
|Path verification|Stratified descriptive metrics|Held-out semantic truth, confidence bounds, frozen calibration and thresholds|
|Causal planning|No calibrated action-outcome forecast adapter|Explicit simulator SCM abstraction, learned outcome decoder, paired action interventions|
|Complete-system contribution|DEV-032 engineering loop only|Trained matched alternatives, component ablations, fresh seeds and physical costs|

Python trace tests are not a proof that PyTorch refines the Lean specification.
A formal correspondence requires an explicit abstraction, transition semantics
and a proof connecting implementations to the theorem assumptions.

## Registered optimization progress

The next run is an existing DEV-029 slot: GRU, LR0.0005, seed311,
128_then512 curriculum,2160 updates, original source
`4a8728ef3f08e01d9c542374cf1e722bf8701240`. The first720 updates use H128,
then H512. Every saved216/648/2160 checkpoint receives original strict replay;
training is capped at420 seconds. All partial files and failures are retained.
Kaggle created it at
[`tac-osm-registered-gru-128-then512`](https://www.kaggle.com/code/eweewee2/tac-osm-registered-gru-128-then512).
No changed optimizer, clipping, pruning, recipe selection or scientific test access.
One additional recipe does not complete fair tuning or demonstrate convergence.

## Recorded conformance evidence

The [trace report](DEV-033-PNDS-TRACE-REPORT.json), frozen at source
`ac8f73094ca82171c66e72132e08a1a64b23f5c7`, inspected56 ordinary rollouts and96
actions (the three fixture actions are excluded). All96 public receipts were
accepted, but zero semantic correctness labels were available. False-admission
and contamination rates are therefore unavailable; all four gate measurements
are unknown and the shadow policy does not admit. This does not change actual
DEV-032 commits. The [backend tensor audit](DEV-033-BACKEND-TENSOR-AUDIT.json)
verified all three retained gradient files and per-parameter counts.

Validation:1151 local tests passed, including eight new contract tests. No
scientific test generation, learned relevance evaluation, online admission
enforcement or formal implementation proof occurred.

## Completed curriculum slot

The [curriculum summary](DEV-033-GRU-CURRICULUM-SUMMARY.json) and
[execution manifest](DEV-033-GRU-CURRICULUM-EXECUTION.json) record completion
through2160 and three original strict same-runtime replays. Initial parameter
hash and runtime match the earlier LR0.0005 fixed512 slot.

| Updates | All-role primary (96 cases) | Cancellation (48 cases) |
|---|---|---|
|216|52.08%|50.00%|
|648|53.13%|50.00%|
|2160|54.17%|50.00%|

The final decision policies again service all1152 entity opportunities, with
zero failures,562 unnecessary services, no inspections and cost1.0 per entity.
Cumulative training took121.40 seconds; execution including replay176.09 seconds
(startup excluded). The curriculum did not resolve cancellation tracking for
this initialization. This does not establish convergence, an architecture limit
or an architecture ranking. The other registered recipes/seeds remain necessary.

The [next-stage proposal](DEV-034-MASKED-MTSK-AND-DECISION-PROPOSAL.md) preserves
entity-local MTSK and decision supervision as separate candidates. No fused-model
training or ActionOPD replication is claimed.

CI [38964662462](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38964662462)
passed the full suite, count/lint checks and conformance report generation.
Downloading its artifact returned HTTP403 from the storage endpoint, so no local
inspection of that CI artifact is claimed. The locally archived frozen report
and independent backend tensor audit remain available above.
