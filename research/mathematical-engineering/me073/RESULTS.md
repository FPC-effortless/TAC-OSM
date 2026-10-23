# ME073: trained native autoregressive event/time trajectories

ME073 trains one integrated native candidate with a shared eight-step future
rollout. A proposed activity and conditional delay update a separate native
fast64/slow64/8x8 associative state before the next step is predicted. This
replaces independent horizon readouts with dependent imagined sequences;
it does not guarantee legal business paths or causal intervention effects.
Factual observed memory is never overwritten. The positive clock conditions
on activity and uses bounded per-step FIT-anchored locations/scales rather
than cumulative unconstrained log-location shifts. Untimed requested-amount
metadata is excluded; initialization uses the first actual observed event.

## Original training and evidence

Five original **99,781-parameter** models completed six authentic BPI2012 FIT
epochs, each 757,710 observed-prefix exposures and 4,998,618 active suffix-factor
exposures from 126,285 distinct FIT prefixes. Five seeds do not multiply the
number of independent records. Mean optimizer CPU **196.61s per model**.
No new comparator, ablation, DEV tuning/calibration, seed selection, retraining
or original TEST scoring. All originals frozen in Git at **100671c7** before
REPORT scoring; original predictions/scores frozen at **190c2bc8**.

The 99,781 live trainable and deployed learned parameters satisfy the newly
fetched master 95k–105k band. Its policy arrived during already registered FIT;
original prefit commit4afb9131 already measured this count, with all weights
trainable. The transparent supplemental schema/actual-checkpoint scale audit
is preserved. Neither model nor registered training/evaluation changed.

## Actual free-running forecast scores

Query after six observed events:555 of 925 historically exposed DEV REPORT
cases reach this point. Separate source-boundary scores use all 555; conditional
activity/time scores use only the actual future records below. Mean of all
five original models,256-particle deterministic QMC marginals; h1 categorical
probabilities are analytic before registered tiny mass smoothing.

| Events ahead | Actual future records | Conditional activity accuracy | Conditional activity NLL | Median forecast MAE, h |
| --- | ---: | ---: | ---: | ---: |
| 1 | 367 | 81.85% | .372472 | 2.047 |
| 2 | 326 | 68.83% | .690375 | 3.342 |
| 4 | 275 | 62.11% | 1.067092 | 6.577 |
| 8 | 189 | 53.44% | 1.523066 | 29.842 |

| Events ahead | Nominal 90% time coverage | Mean interval width, h | Proper interval score, h |
| --- | ---: | ---: | ---: |
| 1 | 85.99% | 11.38 | 18.97 |
| 2 | 90.37% | 18.13 | 27.65 |
| 4 | 88.95% | 43.85 | 55.68 |
| 8 | 90.26% | 232.36 | 239.62 |

Timing is coherent with the generated paths but is not consistently calibrated
or sharp. One-step intervals under-cover; mean 8-step width is almost ten days.
Exact factual-suffix density likelihoods, boundary/joint Brier, time pinball,
24h-delay Brier and width quantiles are preserved for every seed/age/horizon.
Teacher-forced suffix NLL is a different proper task, never free-run accuracy.
Later categorical marginal scores are finite-particle approximations, not exact
analytic path marginalization. No guarantee of population interval coverage.

## Goals that remain unmet

**Early next-event prediction did not improve:** after two observations all
925 cases have a real next event; accuracy remains **40% in every seed**, mean
NLL **1.416298**. Nominal 90% time coverage there is 82.29%, despite very narrow
intervals. This is a failure of the intended early-stage improvement, not a
reason to present later surviving cases as success across all workflows.

**Distant prediction is mixed:** at query age16/horizon8 (106 future records),
mean median-MAE is62.74h, width636.94h and coverage88.11%. Mean widths vary
471.55–827.96h across seeds; p99 widths reach 1,584.12h in a seed. Bounded
parameters do not make long-horizon uncertainty operationally tight.
At age32/horizon8 only 29 actual futures remain, accuracy 24.14%, coverage 77.24%.
Only two cases reach64 observations and one has a next record: no memory
qualification. Query ages compare different surviving populations.

For historical developmental context only, ME072 had six-prefix one-step
NLL .383476, interval score 24.16h, age16/h8 width 3,536h and MAE 100.05h.
ME073's respective values are .372472,18.97h,636.94h and62.74h. However,
ME072 age2 next-NLL was1.134370 and six-prefix h8 NLL 1.378439, both better
than ME073's 1.416298 and1.523066. The early and distant probability goals
therefore remain unresolved. This is **not a matched architecture comparison**:
parameters, target factorization/exposure, objectives and inference differ.
Do not infer a TAC module effect, overall superiority or improved intelligence
from selected favorable scores. Preserve ME072 as the reference candidate.

## Path and operational interpretation

The same generative law yields conditional marks, positive/zero clock draws
and absorbing source-record termination. Sampler paths are conditional record
continuations with cumulative survival weights; they do not assert that all
eight proposed events occur. Every generated elapsed clock is nondecreasing.
Conditional-on-survival quantiles may reverse across horizons as the survivor
population changes. Eight paths per query are preserved by fixed sample index,
not selected for accuracy; full marginal generation uses 256 paths.

EOR is retrospective absence from the published suffix, **not verified actual
completion, failure, cancellation or censoring-free survival**. Its label
availability cannot be backdated to the last timestamp. No missing future
time is invented. No policy choice, action effect, online weight adaptation,
independent source transfer, customer pilot/value or production claim.
The Woyengi spec is a read-only research draft;24h-delay probability is not
an actual SLA-breach label, and>.80 is not an approved routing threshold.

## Preservation and validation

Full local regression **1,043 passed**; ten dedicated trajectory checks and
an absolute-scale guard are included. The engineering-check ledger preserves
initial stale-count and obsolete10k synthetic-fixture failures and repairs.
These tests verify contracts, not learned predictive performance.

Original model/optimizer/RNG, exact source/preprocessing, training schedule,
actual budgets, generated distributions/paths, proper scores and 37-file SHA
index are in the Git replay bundle. **REPO_REPLAY=PASS:** [independent GitHub run39948180370](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39948180370)
restored all five original models and recomputed 8,495 case/query vectors,
33,980 horizon distributions, 67,960 representative weighted paths and all
registered exact suffix/real REPORT scores without refitting. These are
repeated predictions over 925 original cases, not thousands of independent
workflows. The actual executed receipt is in REPLAY_AUDIT.json.
No new fitting, calibration, tolerance relaxation or original-output replacement
is permitted. See ARTIFACT.md and CAPABILITY_TO_VALUE.md.

Next integrated work should improve early probability quality and the
training of free-running distant event/time forecasts, then qualify a frozen
candidate on a new lawful authentic source. Another mandatory component-
attribution campaign is not the next deliverable.
