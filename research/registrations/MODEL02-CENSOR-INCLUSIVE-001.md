# MODEL02-CENSOR-INCLUSIVE-001 — First *eligible* drive decisions, explicit unknown labels

**Evidence tier:** source-own authentic original Backblaze October/November 2013 **previously exposed DEV**, frozen two classical references; **no fitting, protected TEST access or independent qualification**. Added after observing older first-*matured* model ranking results, so this is post-exposure sensitivity research, not a preregistered confirmation.

## Why the earlier decision comparison was insufficient

The earlier operator-budget view chose the first **matured and labeled** seven-day decision per drive. That excludes an initial prediction with missing follow-up and instead may pick a later, outcome-complete day. A deployed operator cannot know which drive will have an available outcome seven days later at decision time. This is evaluation selection by future observability, even when the algorithm never sees future inputs.

New rule: every source ID/model pair first gets seven consecutive observed nonfailure daily records (unchanged past-only eligibility); take its **earliest eligible November 2013 DEV day** exactly once, before checking any outcome. Load each original FIT checkpoint unchanged. The outcome is positive only if a publisher-recorded failure is observed within seven continuous observed following days, negative only if every follow-up day exists with no failure, otherwise CENSORED/UNKNOWN. Do not classify missing drives as safe, failed, or zero-valued sensor readings.

## Authentic source results

[Original CPU-only source run #40483393103](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40483393103) passed **9 constructed contract tests** and retrieved original Backblaze ZIP SHA256 `7f5a53e79b16e695b4b034955806bb3bb194534b169f6eca460dfd3dc48096fe`. There are **26,311** distinct first-eligible DEV drives: **31 known positives**, **25,618 known negatives**, **662 unknown/censored**.

| Reviewed top-k drives | Model | Known positive | Known negative | Unknown | Precision worst–best case |
| --- | --- | ---: | ---: | ---: | ---: |
| 263 (1%) | 8-feature original logistic | 16 | 246 | 1 | 6.08%–6.46% |
| 263 (1%) | 2-feature original lookup | 10 | 253 | 0 | 3.80%–3.80% |
| 1,315 (5%) | 8-feature original logistic | 22 | 1,292 | 1 | 1.67%–1.75% |
| 1,315 (5%) | 2-feature original lookup | 15 | 1,299 | 1 | 1.14%–1.22% |

Bounds are exact for the already-selected top-k candidate subset **conditional on all unavailable seven-day labels becoming either positive or negative**. Worst precision=known positives/k, best=(known positives+unknowns)/k. They are *not* probabilistic confidence or calibrated survival intervals, nor observed avoided losses. The total number of original physical drive failures among all censored drives is **not identified**, so prospective recall and sensitivity cannot be claimed. Risk from informative censoring, manufacturer/age group confounding and lack of historical as-of publication evidence persists.

Compared with older retrospective first-*matured* analysis, the 1% capacity denominator changed from 256 to **263** precisely because 662 additional eligible physical drives are now retained instead of missing-label exclusion. That change matters even though the logistic's selected known positives are the same 16.

## Reproduce

`python specialist_models/model02_censor_inclusive_alert_bounds.py --out /tmp/original_model02_censor_bounds.json`; requires permitted public Backblaze publisher ZIP access. Frozen original Git-owned model hashes are required, neither refitted. Complete source result committed at `research/source-evidence/MODEL02-CENSOR-INCLUSIVE-001-ORIGINAL.json`, without source serials or original raw CSV.

## Next capability gate

**Do not** promote original logistic, lookup, or native TAC. Establish source-published/available-at timestamps or compatible genuine operator-telemetry rights, distinguish retirement vs failure, prospectively freeze a **new** source cohort and censor-aware evaluation metric, stratify by manufacturer/drive age and assess fixed alert economics. Prove utility on new independent physical drives before autonomous warnings or customer ROI. TAC neural comparisons still require 95k–105k, ≤1% matched parameters, equal data/history/source identity/budget and Git-owned original checkpoints.

## Historical action cost envelope (2026-10-27; pure hypothetical arithmetic)

One model forecasts that a drive has a recorded failure within seven days; this **does not imply** reviewing or replacing that drive would have prevented physical failure, data loss or cost. No operator reviewed a drive, no alert was sent, no prevention effect, labor/hardware cost, downtime cost, intervention harm or actual economic benefit is observed in the publisher source.

To prioritize evidence requirements without inventing an ROI study, define C as *all-in cost of reviewing one flagged drive*, B as *actual avoided loss if a preventable failure is averted*, e as intervention effectiveness (0–1). In a review budget of k alerts where p known future positive drive labels and u unresolved/censored outcomes, even under the **maximally optimistic assumption** that all recorded future positives are actually preventable, expected net benefit would require `B/C > k/(e * prevented-eligible count)`; the optimistic count lies between p and p+u. At e=0 no finite ratio could break even. The source provides **no validated e, B or C**; all ratios are conditional engineering planning thresholds, not customer value or a release gate.

[Actual Git-owned original-source hypothetical calculation #40485708597](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40485708597), **8 algebra/receipt tests PASS**. Top 1% first-eligible review k=263: logistic 16 known positives, one unknown => under e=100% conditional B/C threshold spans **15.47–16.44**; two-feature lookup 10 positives, no unknown => **26.30**. Top 5% review k=1,315: logistic 22 known positives and one unknown => threshold **57.17–59.77**; lookup 15 known positives and one unknown => **82.19–87.67**. At a hypothetical 50% prevention probability, all thresholds double. Lower means easier break-even under those unverified assumptions; no claim of actual predictive value or saved costs.

Full repeatable receipt at `research/source-evidence/MODEL02-FIRST-ELIGIBLE-HYPOTHETICAL-REVIEW-COST-001.json`, computed only from original source-backed data and **without fitting, protected TEST, signed business invoices or external operator actions**. These estimates are post-exposure development evidence. Never infer that original positive publisher labels are truly avoidable or that historic source timestamp and missingness qualification gates have passed.

**Next decisive capability experiment:** new genuine source-authorized operator records with a physically meaningful delayed failure outcome and per-alert review effort, independent source collection dates, manufacturer stratification, observed interventions/actions, cost basis and proper censoring. Freeze an actionable budget before scoring, then benchmark classical strong controls and, only when source-valid and parameter/data/time controlled, a 95k–105k full TAC-OSM neural memory candidate.
