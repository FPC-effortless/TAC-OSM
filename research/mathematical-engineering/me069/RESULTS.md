# ME069: authentic enterprise event prediction

Ten original models were trained on BPI Challenge 2012 loan-process records:
five native TAC seeds and five matched GRU seeds. This adds a trained
categorical/time-aware enterprise adapter to TAC's native fast/slow and
associative memory. It is a forecasting capability, not an enterprise control
or commercial qualification.

## Frozen temporal TEST results

One deterministic midpoint prefix per case; 2,618 later-starting cases;
mean of five separately scored fitted seeds. Lower loss/error is better.

| Model/reference | Next-event log loss | Next-event accuracy | Waiting-time MAE, hours | Long-wait proxy Brier |
| --- | ---: | ---: | ---: | ---: |
| Native event TAC | 0.122626 | 94.77% | 6.3561 | 0.070188 |
| Matched GRU | **0.119182** | 94.77% | 6.3618 | 0.070767 |
| FIT empirical transition/time prior | 0.348946 | 84.76% | 6.4953 | 0.077039 |
| Zero waiting-time reference | — | — | 6.8074 | — |
| Global-median waiting-time reference | — | — | 6.8088 | — |

The repeat-current-activity reference had 0% accuracy at these midpoint
targets. The global next-activity prevalence reference had 3.078338 log loss.
They are unmatched practical references, not parameter-matched architectures.

Native TAC reduces log loss by **64.86% versus the empirical transition
prior**, winning all five seed comparisons. Native-minus-prior paired case
bootstrap 95% interval: **[-0.243067, -0.210038]**. It therefore passes that
registered predictive-usefulness subgate on this protocol.

Against GRU, native TAC has **2.89% higher log loss**, winning only **1/5**
paired seeds. Native-minus-GRU difference is +0.003445, with paired case
bootstrap 95% interval **[+0.001552, +0.005243]**. The registered joint
superiority gate **FAILS**. No TAC memory advantage is established.

Waiting-time MAE is 2.14% below the empirical time prior and 0.09% below GRU;
the latter is a small secondary difference, not a registered superiority
result. The delay-proxy Brier score is 8.89% below the empirical prior. The
threshold is the FIT-only upper quartile of positive inter-event gaps. It
is not an observed SLA breach, failure, monetary loss or policy reward.

Native nominal 90% intervals cover 92.28% of TEST cases, with **230.12 hours
mean interval width**. GRU covers 92.04% with 220.32-hour mean width. Coverage
alongside such wide intervals does not establish useful operational certainty.
No conformal calibration, paid inspection or abstention utility is claimed.

DEV is also negative against GRU: native 0.136391 versus GRU 0.130535 log
loss, compared with empirical prior 0.303391. No DEV outcome selected model
weights, architecture widths, hyperparameters or checkpoints.

## Training, fairness and preservation

| Resource | Native TAC | GRU |
| --- | ---: | ---: |
| Trainable/deployed parameters | 80,296 | 80,966 |
| Common supported-operator FLOPs/update | 483,601,874 | 482,228,192 |
| AdamW updates per seed | 800 | 800 |
| FIT prefix endpoints sampled per seed | 25,600 | 25,600 |
| Mean optimizer process CPU seconds per seed | 35.251 | 13.777 |

Parameter gap is 0.8275%; profiled FLOP gap is approximately 0.2840%.
Source bytes, cases, sample schedules, dense nonpadding target counts and
selection privileges match. The final machine-checkable parity gate passes
under its registered estimate convention. The profiler excludes unsupported
operators and optimizer work; full hardware FLOPs are **not certified**.
CPU time is an efficiency measurement, and native training costs about 2.56x
GRU here. Neither CPU parity nor an inference-speed advantage is claimed.

Each model received 411,528–417,662 valid dense next-event/time targets,
depending on its shared seed schedule. All 10 original checkpoints, optimizer
and Torch RNG states, learning traces and measured resource manifests were
committed at `491f085b75aa1b207dd5a69620900d07dda5905d` before any DEV/TEST
scoring. A separate clean checkout strictly restored those original models
and optimizers before first TEST inference. Original case scores were then
preserved at `7b458208` before the independent scoring replay. See
RESTORATION.json, REPLAY_AUDIT.json and ARTIFACT.md for executed audit evidence.

The local clean-checkout score replay is **bit-identical PASS**. GitHub's
different CPU runner recomputed both cohorts but **FAILED** the fixed per-record
waiting-time score tolerance (first reported at DEV / GRU seed 69005 / row
520). [Original failing run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39905268211).
This is a numerical portability exception, not a relaxed tolerance or an
all-environment replay pass. Original weights, scores and acceptance thresholds
remain unchanged. The branch remains draft and unpromoted.

Full repository regression: **1,005 tests passed**. Numerical fixtures are
engineering evidence only. GitHub CI additionally loads the repository-owned
originals and recomputes real DEV and TEST scores when the bundle exists.

## What this changes, and what remains open

TAC now has a trained enterprise event adapter, learned discrete regime
arbitration, elapsed-time dynamics and a probabilistic waiting-time interface.
Its 94.77% next-event accuracy is useful relative to the registered empirical
transition reference. The equal-resource GRU's lower log loss remains a
negative primary result for the native architecture.

This one-institution historical log and bounded midpoint protocol do not
establish general workflow intelligence, action consequences, lifetime state
retention, a contemporary process-monitoring leaderboard position or useful
customer deployment. Resource embeddings are observed anonymized covariates;
there is no case-ID feature or outcome-linked future attribute. Excluding
cases crossing temporal cuts changes the case-duration population. End of
record is not independently verified completion. Source timestamps are
assumed observable as recorded, without separate acknowledgement timestamps.

The next integrated model should improve event-history use and sharp,
well-scored waiting-time forecasts on **FIT/DEV only**, then qualify on a
separately registered untouched cohort or source. This TEST is now exposed
and may not be reused as prospective proof. Preserve the authentic enterprise
lane alongside the archived physical negatives; keep the robot tissue-box
holdout untouched. Do not turn this failed superiority result into a reason
to stop architecture development or repeat diagnostic-only memory studies.
