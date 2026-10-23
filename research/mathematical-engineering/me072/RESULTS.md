# ME072: trained native multi-horizon marked futures

ME072 delivers one integrated trained native model that forecasts 1, 2, 4 and
8 future recorded events, a source-record boundary probability and a single
coherent zero/positive waiting-time distribution. It retains native fast64,
slow64 and 8x8 associative state and adds horizon-conditioned latent readouts.
There is no new comparator or attribution-fit matrix.

Five original models, **123,377 trainable parameters each**, completed six
authentic FIT epochs. Each saw **757,710 prefix exposures** from **126,285
distinct observed prefixes** and 3,030,840 correlated horizon-target exposures,
including source-boundary proxies. This is repeated training, not three million
independent transitions. EXPOSURE_COUNTS.json separates actual future pairs
from boundary targets. Mean optimizer process CPU was **143.37s per model**.

## Actual multi-step predictive capability

Registered query: after six observed events. Of the exposed DEV REPORT's 925
cases, 555 reach this query. All 555 receive activity-or-boundary scores;
conditional activity/timing scores use only actual future records. Mean of
five original seeds, with no posthoc temperature/CQR or weight selection:

| Events ahead | Actual future records | Boundary proxies | Activity accuracy given a future record | Conditional activity NLL | Waiting median MAE, hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 367 | 188 | 82.02% | 0.383476 | 2.129 |
| 2 | 326 | 229 | 68.47% | 0.712665 | 3.380 |
| 4 | 275 | 280 | 59.64% | 1.067113 | 6.582 |
| 8 | 189 | 366 | 54.18% | 1.378439 | 27.845 |

Activity-or-boundary NLL across all555 cases is .254050/.424659/.731650/.849001
at those four horizons. Boundary Brier is .000006/.000507/.065736/.134330.
Do not credit correctly predicted source endings as correctly predicted
actual future activities; the table separates them.

The learned model now supports genuine multi-step event/time prediction, but
predictive quality deteriorates into more distant futures. These development
scores do not establish unique native-memory value or architecture superiority.

## Coherent uncertainty, still uneven usefulness

One normalized marked law supplies source-record existence, activity mass,
zero-time probability, positive lognormal-mixture density and quantiles. Its
construction enforces decreasing survival and zero mass, increasing component
locations and nondecreasing timing quantiles across horizons. This solves an
internal consistency problem, not calibration or operational usefulness.

Six-event query, nominal90% conditional clock intervals:

| Events ahead | Coverage | Mean width, hours | Mean interval score, hours |
| --- | ---: | ---: | ---: |
| 1 | 84.74% | 16.81 | 24.16 |
| 2 | 94.23% | 37.27 | 44.25 |
| 4 | 92.95% | 78.17 | 89.65 |
| 8 | 89.63% | 200.41 | 252.98 |

One-step intervals under-cover; longer-horizon intervals can be very broad.
At query age16 / horizon8, mean width is **3,536h** with 86.79% coverage.
Coherent mixture modeling did not establish tight, reliably calibrated timing.
Full density, zero-delay Brier, multiclass Brier, width quantiles and proper
interval scores are preserved in ORIGINAL_SCORES.json and RESULTS.json.

## Variable observed histories

| Observed query age | Reached cases | Actual next records | Next-activity accuracy given a record |
| --- | ---: | ---: | ---: |
| 2 | 925 | 925 | 40.00% |
| 6 | 555 | 367 | 82.02% |
| 16 | 165 | 157 | 90.70% |
| 32 | 52 | 49 | 82.86% |
| 64 | 2 | 1 | Too sparse for qualification |

These are different surviving case populations, not a controlled history-length
comparison. Early-branch accuracy is weak, distant activity prediction and
time uncertainty remain limited. Training carries complete real-case state;
only two REPORT cases reach64 observations, so no64/128/256-event memory
advantage is established. The owner's capability-first direction permits
continued model construction without such a superiority claim.

## Boundary and evidence limits

EOR means only that the h-th future record is absent from the published case
suffix. Collection censoring/true business closure cannot be separated from
this alone. Its availability cannot be backdated to the last event timestamp.
Consequently boundary-head results are **retrospective record proxies**, not
prospective completion, interruption, failure or causal-stop performance.
No future time is fabricated when the h-th future record does not exist.

Conditional mark/time independence given latent state is a modeling restriction;
the readout does not guarantee globally valid business-process paths. Latent
future states do not overwrite actual observed memory. No real intervention,
action value, online weight learning, autonomous reasoning, independent-site
transfer, Woyengi deployment or commercial qualification follows.

REPORT was already exposed in prior work. No original TEST scoring, BPI2018
retry, posthoc calibration or model/seed selection occurred. Query ages and
horizons were fixed before fitting. Repeated ages and five seed predictions
are not independent workflow observations. Comparisons with earlier different
tasks or training budgets are not architectural-superiority evidence.

## Preservation and validation

Eight new mathematical/causality/gradient tests pass. Full repository regression:
**1,032 tests passed**. Normalization, coherent inverse-CDF quantiles, monotone
horizons, state carry and no invented time labels after record ends are
engineering checks, not scientific performance by themselves.

All five original model/optimizer/RNG files were Git-frozen at `447d06dc`
before REPORT scoring. Original distribution arrays and scores were preserved
at `0d926ba7`. REPLAY_MANIFEST.json indexes exact bytes, loaders, dependencies,
frozen preprocessing and legal publisher-source acquisition. Actual parameter,
exposure and storage receipts pass the single-architecture final gate; this
gate explicitly provides no architecture-winner eligibility.

The scoped GitHub clean-checkout workflow restores the five original models
and optimizers and recomputes all case/query/horizon arrays and metrics. Its
executed receipt is retained in REPLAY_AUDIT.json. No retraining substitute,
calibration refitting, tolerance relaxation or original-output replacement
is allowed. See ARTIFACT.md for exact commands and fields.
**REPO_REPLAY=PASS:** [independent GitHub run39941010981](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39941010981)
restored all five originals and recomputed8,495 case/query vectors containing
33,980 horizon distributions and their real REPORT metrics. Those are repeated
queries/seeds over925 original cases, not8,495 independent workflows.

Next integrated capability work should improve difficult early branches and
longer-horizon mark/time coupling and tail quality, then qualify a frozen
candidate on a genuinely new authentic source. Do not replace that work with
another default component-attribution milestone.

Operational relevance and evidence-stage boundaries are recorded in
[the capability-to-value card](CAPABILITY_TO_VALUE.md); no customer value was measured.
