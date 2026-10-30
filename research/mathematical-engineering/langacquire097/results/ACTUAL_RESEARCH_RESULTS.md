# LANGACQUIRE097: actual lexical-reuse and rule-transfer results

All 40 genuine approximately 100K-parameter models trained and completed the
frozen evaluation. No family passed the registered lexical, ARC, joint or
practical-reference gates. This is an internally measured negative result for
the tested objective intervention and budget, not a disproof of all persistent
architectures or a general-intelligence measurement.

## Prediction and useful outputs

The table uses carried state. Lexical NLL is for the **first target byte** on
32 positive identifier-reuse cases, averaged over five seeds; lower is better.
The exact columns show the result **in each seed**, for both objectives.
They do not treat repeated seeds as additional independent cases.

| Family | Full-loss lexical NLL | Query-loss lexical NLL | Exact four-byte reuse, full/query | Exact ARC grids, full/query |
| --- | --- | --- | --- | --- |
| TAC | 3.798880 | 3.883340 | 0/32, 0/32 | 0/8, 0/8 |
| GRU | 4.189302 | 3.533227 | 0/32, 0/32 | 0/8, 0/8 |
| Selective SSM | 4.260555 | 3.665843 | 0/32, 0/32 | 0/8, 0/8 |
| Transformer | 4.037934 | 3.476952 | 0/32, 0/32 | 0/8, 0/8 |

The untrained 1,928-byte raw-history reference completed **8/32** positive
identifier cases. Copying the last observed ARC demonstration output solved
**1/8** tasks, with valid format on all eight. These are practical references,
not parameter- or FLOP-matched neural architectures. Uniform and negative code
strata, all interventions and all seeds remain in [RESULTS.json](RESULTS.json);
uniform-stratum successes do not replace the registered positive cohort.

TAC's registered full-minus-query NLL improvement was **-0.084460 nats**,
95% cluster interval **[-0.312126, 0.143205]**. Query reset-minus-carry was
**-0.011081**, interval **[-0.164028, 0.075117]**. Donor-minus-carry was
**0.001397**, interval **[-0.040948, 0.038212]**. None establishes the required
content-specific memory benefit. TAC's lower FIT losses did not become useful
held-out lexical reuse.

There are partial predictive positives worth preserving: GRU, SSM and
Transformer query-versus-full NLL gains were 0.656075, 0.594711 and 0.560982
nats, respectively, with positive registered cluster intervals. Transformer's
reset-minus-carry gain was 0.121531, interval [0.052988, 0.182172], improving
in all five seeds and all four code owners. Its donor-control interval crossed
zero and its exact completions remained zero, so it also failed the complete
registered state-use gate. These observations do not qualify an architecture
winner; absolute hardware FLOPs and cross-family tuning parity remain open.

ARC is a bounded diagnostic of a human-authored abstract benchmark, not a test
of general intelligence as a whole. TAC produced no valid carried ARC grids
under either objective. Across all families, only one of the 320 carried
seed/task/objective outputs had valid grid format, and none was correct.
Failure to emit the required format prevents cleanly separating rule-learning
failure from output-execution failure. Five predeclared unavailable ARC donor
controls remain inconclusive; they were not replaced or dropped.

## Mathematical limitation and next improvement

The current objective averages CE over all selected target bytes. Each batch
contains two lexical examples with four future bytes each and two ARC examples
with variable answer lengths. The algebraically audited mean ARC coefficient
mass changes from **60.6729%** under full loss to **83.1349%** under query-only
loss. Lexical future-target mass nevertheless rises from **1.58897%** to
**16.8651%**. This does not establish lexical starvation: individual future
bytes receive more weight. The intervention combines removal of prefix loss,
future-target upweighting and domain reweighting. It cannot isolate acquisition
learning alone. These are loss coefficients, not measured gradient shares.

A cleaner next experiment would hold future-target coefficients fixed and
vary the prefix term separately:

\[
L_{\lambda,w}=\sum_{d\in\{\mathrm{code},\mathrm{ARC}\}} w_d
\left[\operatorname{mean}_{i\in d}\operatorname{mean}_{t\in Y_i}\ell_{it}
+\lambda\operatorname{mean}_{i\in d}\operatorname{mean}_{t\in P_i}\ell_{it}\right].
\]

Fixed domain weights and per-case means remove the current answer-length
weighting confound; changing lambda then changes the prefix term while keeping
the explicit future-target coefficient fixed. Optimizer/clipping effects still
require measurement. This is a proposed intervention, **not a newly trained
or verified capability improvement**. A subsequent study needs a fresh frozen
FIT/DEV registration, formatting-success diagnostics and the same complete
memory and raw-history controls. The present negative scores remain unchanged.
No larger model or new memory component is justified by these results alone.

The exact formulas, five-seed allocations and source/profile hashes are in
[OBJECTIVE_ALLOCATION_SUMMARY.json](OBJECTIVE_ALLOCATION_SUMMARY.json).
[The calculator](reproduce/audit_objective_allocation.py) uses FIT metadata,
stdlib and NumPy only, with no neural forwards or DEV reads.

## Executed work, resources and verification

- Four families, five seeds and two objectives: **40 actual fits**, 256 AdamW
  updates each, **10,240 optimizer updates** total. All 20 original zero-update
  initial archives, 40 original trained archives, optimizer/RNG states and ten
  leg seals were preserved. TAC has 100,521 active/deployed parameters; controls
  have 100,404–100,661, with no dummy parameters.
- Instrumented fit children summed to **285.78 CPU-job minutes**. Each leg took
  19.17–34.57 minutes within its 40-minute bound. CPU models differed across
  runners; all-step operator tracing is included. These are not deployment
  timings or matched hardware-FLOP comparisons. Setup is recorded separately.
- Original evaluation took **228.87 seconds**, with zero optimizer updates,
  GPU minutes or protected TEST access. Recorded phase costs and native state
  payloads are in [EVALUATION_COST_DIAGNOSTICS.json](EVALUATION_COST_DIAGNOSTICS.json).
  TAC used less persistent tensor storage than this Transformer control but
  more than GRU/SSM, and was slower in the measured acquisition, query and
  generation phases. State metadata and auxiliary control states are separate.
- Independent archive audit rechecked all 40 initial/trained pairs, every
  optimizer slot at step 256, RNG and source/data/sampler/parameter bindings.
  All five seed pairs had identical initial weights and matching recorded
  forward/backward and optimizer operator work. No local neural inference or
  optimizer updates were used for these audits.
- All **40 fresh-process query replay pairs** matched bitwise. Independent
  outcome calculation reconstructed **11,520 main rows**, **320 binding rows**
  and all intervals/gates, retaining **200 unavailable donor rows**. Maximum
  numeric report discrepancy was **2.22e-16**; raw-reference log probabilities
  matched exactly. Five seeds are averaged within cases before 5,000 cluster
  resamples over four code owners or eight ARC tasks.
- The dedicated postfit compatibility suite passed **56 tests** in Actions;
  the full study's **138** focused source/objective/probe/resource checks also passed.
  Local collection found 1,849 tests. A broad run in the incomplete local
  historical checkout recorded 1,723 passed, 90 failed, 32 errors and four
  skipped, including absent contracts/artifacts/data; it is not represented as
  a full-suite pass. No new097/probe/parity test appeared among those failures.

The independent checks are code/tensor/outcome reviews within this internal
research process, **not external qualification**, whole-model Lean/IEEE/AD
certification, safety certification, checkpoint promotion or an AGI claim.
Public DEV has prior research exposure, and its few source/task clusters limit
generalization. Full results and the failed/inconclusive cases are preserved.

## Original verifier failure and explicit format review

The original run trained all forty models successfully, then stopped at
`final-parity-tac` **before opening DEV**. The original parser rejected two
empty-tensor AdamW profiler markers as non-ATen names. Its original failure,
raw ledgers and source bytes remain preserved.

The separate source-pinned review admits only those exact balanced profiler
markers in the optimizer stage, with empty tensor inputs/outputs, and retains
their counts. All ATen arithmetic goes through the original parser. The review
also executes the unchanged original validator in-process so the explicit
compatibility binding is used; it does not represent the original parser as
having passed unchanged. Unknown operations and arithmetic/count mismatches
still fail. Models, data, actor, scorer and thresholds were unchanged, and the
original source files were rehashed after evaluation. All four final family
resource gates independently passed this scoped review; architecture
superiority eligibility stays false.

## Source and original artifacts

- Original training source: `d21c7e7b385ca164eac9046fafca43fc9a3762b1`.
- [Original 40-fit run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41137837235).
  Its final verifier failure is preserved, despite all ten fit jobs succeeding.
- Evaluation-only review source: `dbe3f9edba9a89037b3e8c6bfe2e775f81777ae4`.
- [Successful original-checkpoint evaluation](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41141983424).
- [135,590,835-byte original result archive](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41141983424/artifacts/13019296161),
  SHA256 `0e8c425a8cc99483cc70f5b9cde238614d348c340762f848f5ca4c59e07a5b63`.
  It includes original initialized/trained weights, optimizer/RNG states,
  predictions/replays, frozen data, tracked source, leg-root logs and receipts.
- [Scoped PNDS mathematics](https://github.com/FPC-effortless/pnds-lean/pull/96):
  nine theorem axiom checks and eight numerical references passed; composed
  model proof and canonical main/master registry integration remain open.

The [outcome auditor](reproduce/audit_outcomes.py) recomputes saved predictions
without importing neural models or the registered scorer. The
[checkpoint auditor](reproduce/audit_checkpoints.py) verifies original tensor
archives and replays without model forwards. Their source hashes match the
saved audit receipts. Pass explicit local source/data/profile paths to the
CLIs when reproducing outside the original workspace. The result-archive
receipt and all machine-readable audit outputs are stored alongside this file.
Report-only publication closes the completed evaluation request and does not
repeat either training or inference. Both research PRs remain draft/unmerged.
