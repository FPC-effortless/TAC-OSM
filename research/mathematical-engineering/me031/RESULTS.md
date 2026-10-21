# ME031 completed — cross-fitted state residual fails predictive-value gates

**36 original fits complete; finite-family state candidate rejected.**
Corrected source frozen/pushed at50bdadb before any model fit. Protocol a80a101,
resource accounting completed5937696, scalar-schema preflight correction50bdadb.
The first source-only launch failed before creating a model; its receipt is
preserved in PREFIT-FAILURE.md. No training retry, tuning or threshold change.

## What was actually isolated

Original authenticated KuaiRec FIT only:10776 windows,376 source users, targets
before Aug01 2020,128 genuine historical events. Three user groups have
3655/3591/3530 windows and126/122/128 users. In each rotation, disjoint groups
train the representation, fit the prediction probes, and evaluate. Every user's
prediction uses an encoder, feature normalizer, common base and probe trained
without that user's records. Their own observed history remains legitimate input.

Each seed14001/2/3 has fresh encoders/base/probes per rotation:9encoders,
9common bases,18residual probes. Four causal targets during fitting; ONLY final
128-event forecast during evaluation. All36models were frozen and hash-indexed
before any evaluation metric was computed. No historical DEV scores or reserved
cohort access. This partitions previously used FIT, not an independent test;
previous architecture-design exposure is not erased by within-study isolation.

## Fixed endpoints, pooled predictions in original row order

Three-seed means, lower Brier/logloss/ECE and higher AUC are better:

| Predictor | Brier | Log loss | AUC | ECE10 |
|---|---:|---:|---:|---:|
| Prior32 reference | .216722234 | .625206467 | .653172745 | .042291698 |
| Frozen common-feature base | **.162070604** | .492364352 | **.823693907** | .013947100 |
| Base + common-only residual | .162310484 | **.492265800** | .823365563 | .015713609 |
| Base + native-state/context residual | .164980411 | .499343869 | .817534112 | **.013006566** |

State minus common residual Brier per pooled seed:
**+.002154751,+.002355957,+.003499073**; mean **+.002669927**.
State loses in **all9 seed/rotation cells** (0/3 fold wins in each seed).
User-clustered95% bootstrap interval **[+.002095950,+.003187102]**,
conditional on fitted models and the reused FIT source. Rotated training sets
overlap; bootstrap does not refit representations/probes or capture that uncertainty.

Against the frozen base, state residual differences are
**+.002467207,+.002338086,+.003924128**. Mean common-only residual also fails to
improve the base's Brier. No post-hoc decision substitutes an earlier checkpoint.
The base is a practical reference, not a separately tuned winning architecture.

Four predictive gates FAIL: material gain over common residual; gain over base;
consistent fold wins; bootstrap upper bound<0. Calibration guard PASSES:
mean state-minus-common ECE10 **-.002707043**. Overall candidate gate **FAIL**.

## Interpretation and architectural consequence

A nonlinear state-aware residual fails to extract extra forecast value under this
frozen representation/training recipe with out-of-user representation generation.
This is stronger isolation than mean-state substitution, which changes the input
distribution. It is still a finite-model probe, not proof of zero conditional
information, useless chronology, or universally inferior native memory.

Trainable residual counts1197(state) vs1205(common) differ<1%; both have nonlinear
interfaces, but state hidden width13 vs common43. Their approximation families
are not identical. Representations are frozen ME028 encoders trained on a smaller
user subset than earlier studies. Source/role shift, head expressiveness, finite
sample learning and optimization remain possible explanations. The common base's
.162070604 cannot be compared directly with ME030 DEV scores: source targets and
training protocol differ.

No architecture is promoted. The next unexecuted hypothesis is **conditional state
training**: train the native representation itself to improve residual prediction
over a separately trained/frozen common predictor, with gradients reaching memory
and controls receiving equal information/representation exposure. Freeze all
roles, controls, capacity and compute before fitting. ME031 only trained residual
heads on frozen state; it did not execute that representation-training change.
Compression, hierarchical states and adaptive latent reasoning remain gated on
useful state, as in the architecture research plan.

## Resources and parity

Exposure matched: per arm/rotation, representation300x128 histories/4targets;
base500x128/4; residual500x128/4. Both residual arms use identical samples,500steps,
AdamW.003 and clipping; no search/selection. Representation135377 parameters,
base1205, residual1197/1205. Stored total137779/137787. Encoder's original401head
parameters serve pretraining and are unused by probe prediction; disclosed rather
than counted as active residual coefficients. See protocol/math for deployment
and shared encoder branch differences.

Shared representation/base stages computed once, charged fully to BOTH arms.
Mean charged pipeline CPU **22.5235sec common**, **22.5043sec state**. Physical
whole-study CPU **205.7634sec**, wall **205.8120sec**,single-thread CPU. These are
different accounting quantities, not a claimed400sec actual study spend. Every
stage and full pipeline remained within its preregistered caps. Full RSS/hardware,
all clocks and per-cell consumption are in results.json/original payloads.

Post-fit profiler residual-only forward/backward subtotals,128historiesx4targets:
2,422,272(common),2,436,608(state). Unsupported nonlinear/elementwise operations,
Adam and the encoder/base pipeline are not fully represented; not full FLOPs.
B128 full shared-encoder pipeline median inference **12.4303ms common**,
**12.4637ms state**,3warmups/9samples on seed14001rotation0. Both execute native
feature extraction; no optimized common-only runtime or broad speed claim.

Parameter budget PASS, common sample/exposure PASS (state access is the intentional
ablation), selection PASS, identical evaluation PASS for this historical FIT-user
diagnostic. Actual-compute parity remains **unqualified/FAIL under registration**:
charged CPU is close, but this was registered as an exposure diagnostic with no
full-compute contract or full FLOP audit. No architecture-superiority claim.

## Verification

- Rebuilt23,534,784 feature values from original source/frozen encoders and S-only
  normalization; independently verified every role assignment.
- All36 Adam counters, exact sampler RNGs and16,200 per-step clock/loss receipts
  pass. No evaluation user enters that model's fitting stages.
- 18 checkpoint residual paths reproduce64,656 predictions bit exactly;
  36export/reference paths reproduce129,312 predictions bit exactly.
- 9 fresh Torch-only processes load the full encoder/base/residual pipeline and
  reproduce72real forecasts exactly without repository imports.
- Corrupted checkpoint fails before source loading.930tests+12subtests pass.

See replay.json, source/role/result manifests, [figure](crossfit-results.png),
protocol/math and original checkpoint archive records. Reserved cohorts remain
closed. No deployment, merge, action utility or conditional-MI claim.
