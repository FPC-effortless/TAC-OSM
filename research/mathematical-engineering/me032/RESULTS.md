# ME032 completed — centered state training improves teacher, fails native gates

**9new common-head fits +27memory fits completed. Overall candidate FAIL.**
Source/protocol frozen and pushed at7c4fea4 before fitting. No failed fit, restart,
hyperparameter change or endpoint selection. Nine historical ME031 R-encoder
checkpoints were authenticated and reused only for shared R-context embeddings/
preprocessing; no prior S-trained head/state weights entered this study.

## Mathematical change and actual training

The new readout subtracts its zero-state prediction:

`delta(h,C)=w'h+v'[tanh(Bh+AC+a)-tanh(AC+a)]`.

This guarantees delta(0,C)=0 and excludes an independent pure-context correction
at zero state. Native32fast/32slow equations and RMS remain unchanged. All student
memories start from scratch and receive real S-user supervision end to end. The
conditional arms freeze the R-trained common head; the joint native control can
adapt it. The pooled control learns a permutation-invariant mean-event memory.

All27memory fits have **299/300 steps with measured nonzero memory gradient**;
zero first-step credit follows the registered zero-output initialization. The
18conditional common heads remain bit-identical;9joint heads change. These are
verified actual gradient/allocation behaviors, not just equations on paper.

## Source and evaluation boundary

10776 original KuaiRec FIT windows,376users, targets beforeAug01 2020,128genuine
past events. Same3user-disjoint R/S/T rotations as ME031,3seeds14001/2/3. R trains
common representations/head, S trains new memory, T evaluates. All models were
frozen and hash-indexed before any T score. Each user's predicting model excludes
that user from its parameter/normalizer fitting. Their observed past is allowed.

This is entity-separated crossvalidation in previously used FIT: roles cover the
same historical period, not an out-of-time operational forecast qualification.
Causal prefix features exclude next response/later events, but model-fitting dates
are not a prospective temporal split. Historical DEV is only source-fingerprint
checked, never scored; future/other reserved cohorts stay closed. Prior design
exposure to FIT remains. No independent generalization or deployment claim.

## Fixed300 S endpoint, three-seed pooled means

| Predictor | Brier ↓ | Log loss ↓ | AUC ↑ | ECE10 ↓ |
|---|---:|---:|---:|---:|
| Prior32 | .216722234 | .625206467 | .653172745 | .042291698 |
| R-only common teacher | .170203796 | .513921993 | .806423730 | .026479599 |
| Conditional native | .165824448 | .501253355 | .816014833 | .020218687 |
| Joint native | .165650858 | .501249195 | .817362630 | .023492645 |
| Conditional pooled | **.163399886** | **.494946892** | **.821181332** | **.017887879** |

Conditional native minus each reference, Brier:

| Reference | Mean difference | Per-seed14001/2/3 | Conditional user-cluster95%interval |
|---|---:|---|---|
| R teacher | **-.004379348** | -.004757215 /-.004334020 /-.004046809 | [-.006271716,-.002613042] |
| Joint native | +.000173589 | +.000061636 /+.000367298 /+.000091834 | [-.000383425,+.000737254] |
| Conditional pooled | +.002424562 | +.002109263 /+.001909285 /+.003255138 | [+.001706774,+.003151889] |

Native conditional correction improves on the R-only teacher in all3pooled seeds,
passing the registered materiality requirement for that reference. It does NOT
beat joint native in any pooled seed, nor pooled memory in any pooled seed. The
joint contrast interval includes zero, so do not claim a population-level harm
from freezing. The pooled contrast favors the pooled model under this recipe,
but actual-compute parity is unqualified, not a fully matched architecture win.

Fold wins conditional native vs joint/pool are [1,0],[1,1],[2,0] for seeds14001/2/3.
Gain-vs-joint, gain-vs-pool, fold consistency and joint bootstrap gate FAIL.
Teacher-gain and calibration guards PASS. **Overall engineering candidate FAIL**;
no native variant promoted. All individual metrics/contrasts in results.json.

## What this resolves and what remains open

ME031 froze representation while fitting probes. ME032 confirms that a centered
correction can train memory from residual prediction error and improve a fixed
R-only common predictor. Neither forcing memory credit nor freezing the common
head establishes a native recurrence advantage. The centered algebra is a valid
engineering invariant, not proof of unique information or useful chronology:
a constant nonzero state can still support a context-dependent correction.

Pooled memory learns its own S-specific event embeddings; its gain could reflect
embedding/context adaptation rather than chronology or longer retention. The R
teacher has no S fitting, while students do. A pure common-head S-adaptation
reference with identical current protocol is missing, so teacher gain alone cannot
be attributed to learned memory. Do not add a post-hoc fit to this closed matrix.
ME031's historical S-trained common base had Brier.162070604 under a different
normalization/training procedure: it is a warning against declaring a program-wide
accuracy breakthrough, not a matched new-study result. ME031 and ME032 scores
must not be presented as controlled before/after architecture gains.

The next unexecuted design should isolate S-specific common/embedding adaptation
from ordered-memory effects before modifying recurrence: preregister an S-adapted
common reference and freeze/unfreeze embedding controls with matched information
and exposure. Any new state allocation or learned pooled-summary integration needs
its own fixed-capacity/cost test. Compression, hierarchy and adaptive reasoning
remain gated on demonstrated native-state value. No causal/conditional-MI claim.

## Costs, parameters and parity

Active S trainables:136236conditional native,137441joint native,136772conditional
pool; max global gap.8845%. Nonembedding counts8236/9441/8772 disclose allocation
change. Native stored total265441 incl frozen R embedding/common head; pool265977.
Centering adds no inactive output bias; all1260correction coefficients participate.
Historical R pretraining135377parameters served all3arms and is fully charged.

| Arm | Mean new S-training CPU sec | Mean charged pipeline CPU sec* | B128 full pipeline ms** |
|---|---:|---:|---:|
| Conditional native |19.4244|41.3754|12.1718|
| Joint native |19.5368|41.4878|12.2301|
| Conditional pool |1.5736|23.5246|1.4440|

*Includes original R pretraining plus new source/context/teacher costs charged
fully per arm; shared stages physically computed once. It is not new physical
study spend. New study physical CPU377.0623sec, wall377.0971sec. All stage/pipeline
caps satisfied. Receipt construction/serialization excluded as registered; all
per-stage clocks and whole-process RSS/hardware are preserved.

**Seed14001rotation0, CPU thread1,3warmups/9samples; raw128events -> frozen context
and trained memory -> four prefix predictions, final endpoint reported. These
are implementation-specific measurements, not scalable throughput guarantees.
Profiler S-step forward/backward operator subtotals for8histories/four targets:
42,843,391conditional native;42,920,447joint;1,889,047pool. Unsupported/fused/
nonlinear/elementwise work, Adam and frozen-context preparation are incomplete;
these are lower-coverage estimates, not full FLOPs.

Parameters PASS, data/exposure PASS, selection PASS, identical historical-user
evaluation PASS. **Actual-compute parity FAIL/unqualified** as preregistered;
there is no fully matched architecture-superiority claim or compute-matched lane.
Same caps do not mean equal spend. No new real decision-cost/action capability.

## Original-artifact verification

- Rebuilt10,474,272context values from authenticated R-only source and preprocessing.
- 36optimizer/Adam/sampler-RNG records and12,600step receipts verified; all teacher
  freezes/joint updates, zero-state anchor and actual memory gradients verified.
- 27student checkpoints reproduce96,984forecasts bit exactly; supplementary audit
  verifies9teacher checkpoints/32,328forecasts bit exactly. All36new checkpoints
  therefore have direct forecast replay.
- 45export/reference paths reproduce161,640forecasts bit exactly.27fresh Torch-only
  raw-input pipelines reproduce108forecasts exactly without repository imports.
- Corrupted checkpoint rejected before source loading.936tests+12subtests pass.

See replay.json (fresh-process/profiler audit), verification.json (expanded teacher
checkpoint coverage), [figure](conditional-training-results.png), protocol/math,
logs and original archive restoration records. All new fits/checkpoints retained.
