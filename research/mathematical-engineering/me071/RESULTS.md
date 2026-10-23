# ME071: integrated native event-model development

**Implemented, genuinely trained and Git-preserved:** five native TAC models,
93,770 trainable parameters each, six complete FIT passes, 715,134 supervised
next-event/time targets per model. Mean optimizer process CPU: 88.87 seconds
per model. Training uses authentic publisher-verified BPI2012 records. No
comparator models, attribution fits or new TEST evaluation were run.

## Capabilities added to the complete model

| Capability | Executed implementation |
| --- | --- |
| Persistent case history | Fast, slow and associative state carried across detached 32-event chunks; reset at a new case, not every chunk |
| Context-conditioned event prediction | Learned log-linear blend of native event probabilities and FIT-only five-event history prior; .05 label smoothing |
| Probability adjustment | Scalar temperature fitted on distinct TEMP cases and frozen before REPORT |
| Waiting-time quantiles | Ordered .05/.50/.95 log1p-second quantiles trained jointly by pinball loss; positive-delay occurrence head retained |
| Interval calibration | Separate CQR cases, finite-sample quantile rank, float64 conversion to seconds |
| Concurrent entity streams | Independent timestamped state bank, explicit resets, rejection of time reversal |
| Missing observations | Masked assimilation with accumulated elapsed time; hidden observations excluded from history priors in real probes |

The native recurrence remains fast64 + slow64 + 8x8 associative state.
The SwiGLU decoder is width128. Stateful execution and truncated-gradient
training are distinct: backward credit remains bounded to 32 events.
There is no claim of a new Transformer/SSM, natural-language generation,
online weight learning or autonomous counterfactual policy optimization.

## Registered development readout

Predict the next record after **exactly six observed events**. Of 925
chronologically later DEV REPORT cases, **367** have a following target at
this query. TEMP has 240 eligible cases, separate CQR has 235. Short/ended
traces have no eligible target; they are not fabricated successes/failures.
The whole DEV population has prior research exposure. This is development
evidence, not new independently held-out qualification.

Mean of five frozen native seeds; raw and postprocessed outputs of the same
model, without architecture ranking:

| Metric | Raw model | Frozen temperature + CQR |
| --- | ---: | ---: |
| Next-event log loss | 0.420459 | **0.401741** |
| Next-event accuracy | 81.58% | 81.58% |
| Multiclass Brier | **0.225696** | 0.234267 |
| ECE, ten equal-width bins | **0.071837** | 0.073577 |
| Waiting-time median MAE, hours | 2.0573 | 2.0573 |
| Nominal 90% interval coverage | 91.39% | 93.35% |
| Mean interval width, hours | **48.15** | 51.69 |
| Mean interval score, hours | **48.75** | 52.28 |

Temperature adjustment reduces mean log loss by **4.45%**, while Brier/ECE
worsen. Temperatures are approximately .75–.82, sharpening these smoothed
models; this is not proof that historical ME069 was overconfident. Accuracy
is invariant to positive scalar temperature. Do not claim an across-metric
calibration win.

Three fitted CQR corrections are zero, while two widen intervals. Coverage
increases but mean width and interval score worsen. Calibrated median width
is approximately 0.78h; p90 approximately 229.42h and p99 approximately
284.47h. Broad tails remain a practical limitation. Log-space modeling and
CQR do not guarantee tight intervals or exact coverage under time shift.
Full per-seed proper quantile/interval scores are in ORIGINAL_SCORES.json.

The fixed-six-event target/population differs from ME069's hindsight midpoint.
Do **not** compare this 81.58% with historical 94.77%, or these interval widths
with historical 230h, as a controlled improvement/regression. Such a claim
would conflate different prediction tasks and case selection.

## Streaming and implementation checks

On 16 original, previously exposed real DEV cases per trained seed, interleaved
versus isolated processing gives **zero maximum state difference**, verifying
entity routing. Masking four intermediate records worsens mean next-event
NLL in all five seeds on this small probe. Features, elapsed gaps and history
priors were recomputed from visible events. This is a posthoc missing-input
stress check, not evidence that TAC uniquely reconstructs hidden facts or
that a GRU would fail. See STREAMING_PROBES.json.

Nine new mathematical/causality/gradient/calibration tests pass. Full repository
regression: **1,023 tests passed**. The inherited frozen documentation count
was synchronized after the initial count-only failure.

## Preservation and scope

All five original model/optimizer/RNG checkpoints were frozen at `7c503e95`
before calibration, with final single-architecture exposure gate PASS and
no architecture-superiority eligibility. Calibration froze at `080976ee`
before REPORT scoring. Original predictions and scores are Git-owned at
`1973df6d`, with a checksummed REPLAY_MANIFEST and legal source acquisition.
No original TEST predictions or weights were used for selection.

A first readout incorrectly reported raw pinball values in the calibrated
column. That file is preserved as FIRST_READOUT_PINBALL_REPORTING_ERROR.json.
The corrected calculation scores the adjusted quantiles; all model weights,
calibrators and prediction arrays remained unchanged. It does not affect
the NLL, accuracy, coverage or width figures above.

An inherited ME052 CI job unexpectedly began historical robot training when
the shared validator changed. It was cancelled; at least 20 updates had been
reported, with no completed fit claimed. Partial weight recovery is
unverified/not preserved because its temporary artifact download returned
HTTP403. This limitation is explicit in LEGACY_CI_INCIDENT.json. Historical
training is now opt-in; it cannot be silently relaunched by PR validation.
Those interrupted legacy calculations are separate from the five preserved
ME071 originals.

Independent original-model restoration and real REPORT scoring replay are
recorded in REPLAY_AUDIT.json after a clean checkout and the scoped GitHub
workflow. Replay does not retrain weights or refit calibration.
**REPO_REPLAY=PASS:** [GitHub run 39929979913](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39929979913)
restored all five originals and recomputed 1,835 case-forecast vectors and
all raw/calibrated REPORT metrics under the preregistered numerical contract.

This cycle delivers a more complete native predictive architecture and a
measured NLL improvement from its fitted probability interface. Useful tight
uncertainty, reliable long-memory advantage and decision intelligence remain
unqualified. Continue improving the integrated model using development data;
reserve a separately sourced untouched population for future qualification.
