# Enterprise forecasting source preflight and prospective protocol

Status: **DESIGN_ONLY / BLOCKED**. This is a reviewable source-qualification and
experiment proposal, not an activated training registration. No new model family,
fit, outcome score, TEST access, customer contact or deployment is authorized by it.
Outcome: identify recoverable original longitudinal evidence, specify a fair
forecasting task, and make every missing prerequisite explicit before any fit.
Falsifier: if the purported historical features cannot be reconstructed from
versions actually published before a forecast cutoff, the historical task fails
qualification regardless of model scores.

## Original source evidence acquired

Original public metadata bytes, URLs, retrieval intervals, byte counts and SHA-256
hashes are retained alongside this file in `metadata-acquisition-receipts.json`.
They contain no service-request or loan event rows. Acquisition time is our
observation clock, not independent authentication of a publisher's historical clock.

| Source | Newly observed evidence | Remaining limitation |
| --- | --- | --- |
| NYC311 `erm2-nwe9` | Official metadata; archival replication active at data version **194240**; public changelog exposes revision **37**, archive data version **187190**, successful replacement ending **2025-12-27T01:36:53** as publisher text | One visible archived revision does not establish daily row history, timezone of the naive completion timestamp, or historical field availability before that release |
| NYC311 archive 187190 | Read-only archive status reports `done`, gzip and `refSize=3476668601` | Status is not downloaded bytes, a verified full export, or permission to redistribute; size accounting is publisher-defined |
| Original BPI Challenge 2017 | Version **1**, DOI `10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b`, published 2017-02-06; original file ID **24044117**, **29,658,747** bytes; publisher-reported MD5 `10b37a2f78e870d78406198403ff13d2` | Metadata only; file not acquired or SHA-verified; event time does not prove recorded/available time; rights listed as **4TU General Terms of Use**, not assumed CC0 |

The changelog response requested five revisions and returned one with `next=null`.
This establishes that response only; it does not prove there are no hidden,
legacy, differently indexed or additional historical versions. Archive status
was queried with GET; no archive generation, enrollment, restoration or full
export was triggered. A preliminary unprefixed changelog request returned 404;
the working frontend uses `/api/publishing/v1/revision/.../changes`.

**Correction to the earlier research assessment:** NYC's general terms warn that
older data need not be retained, but the official [Open Data 101 guide](https://www.nyc.gov/opendata/get-started/user-guides/open-data-101)
documents optional per-dataset archives. [Tyler's archive documentation](https://support.socrata.com/hc/en-us/articles/9486838238743-Introducing-Dataset-Archiving)
explains version export and warns attachments are not restored. Live metadata now
establishes actual 311 enrollment and one visible archive. Therefore historical
recovery is **PARTIALLY IDENTIFIED**, not categorically impossible and not yet
qualified. The 2010–2019 partition is a partition by event year, not proof of
historical knowledge versions.

The original [BPI publisher record](https://research.tue.nl/en/datasets/bpi-challenge-2017/)
describes applications filed in 2016 and subsequent events through February 2017.
It is a separate fallback source, not permission to open existing protected BPI
cohorts, and not an independent second observation of NYC outcomes. Any derived
offer-only subset requires an independently checked selection/coverage audit
before being used as a denominator; this audit retained only original-log metadata.

## Source decision and recovery procedure

Prefer NYC publisher snapshots for a public-service workflow task if useful
version coverage can be established. First obtain the archive's column schema,
rights/privacy review and lawful storage plan. The reported multi-gigabyte full
export may contain fields outside Forge's minimal six-column capture, including
sensitive text/location. Do not publish or download it blindly. Establish a
publisher-supported minimal projection or approved isolated processing and
immediate restricted-field disposal before acquisition. Retain lawful original
bytes with SHA-256 and release identity; an expiring Actions artifact alone is
not a permanent evidence store.

Next identify at least two recoverable, ordered publisher releases and verify
release times/timezones, schema, deletions, corrections, case identity and archive
coverage. Only an observation of version v available before cutoff t may become
an input at t. Never substitute an archive's event `created_date` for availability.
The archive 187190 and the October 2026 25-row latest-record query do **not** form a
matched longitudinal cohort: query membership differs and follow-up is unknown.

If historical coverage is inadequate, use a separately frozen prospective
collection design: enroll open cases at first observed eligible snapshot, freeze
their IDs independent of later outcome, and requery those exact IDs rather than
repeatedly taking the latest 25. Record missed acquisitions and missing IDs as
unknown; never silently replace disappeared cases. No scheduler is enabled here.
Define capture frequency, minimum outcome coverage and source-supported status
semantics before enrollment. A prospective dataset needs elapsed follow-up time;
it cannot be manufactured in this session.

BPI2017 remains an alternative source audit. Before opening original event rows,
independently resolve prior exposure/protected-cohort ownership and rights, then
verify whether trace-level attributes were finalized later and whether actual
recording/availability clocks exist. Exclude late/undated trace attributes. If
only event timestamps exist, label any permitted study **retrospective log-prefix
prediction under an explicit availability assumption**, not historical operational
capability or independently verified business completion. Do not switch sources
after reading scored results under one registration.

## Proposed primary task and denominator

Lane: **Enterprise**, within the existing six families. Target model skill:
source-available state/history-conditioned multi-horizon workflow risk.
Primary NYC task: predict whether an enrolled open request is **publisher-reported
closed at the fixed seven-day horizon**. One landmark per case. For prospective
collection, use acquisition completion after eligible open status is observed.
For recovered historical replay, use independently evidenced publisher release/
availability time after open status is observed in that version; today's archive
retrieval is never substituted for that historical landmark. This measures a
publisher status, not verified physical problem resolution, SLA compliance or
causal intervention benefit. Any eventual actual-resolution study needs distinct
independent evidence and a new target contract.

Allow only first-observed, timestamp-qualified agency/category, case age when
creation-clock semantics are verified, time-of-week, source age and missingness.
Raw identity is a join/group key, not a numeric predictive feature. No future
status, future closed date, eventual trace length, full-trace normalization,
outcome-derived cohort membership or later Woyengi claim selection in inputs.
Fit vocabularies/scalers on FIT only; unknown categories have a frozen fallback.
Reopenings/corrections remain versioned; horizon status does not imply first-ever
closure. Secondary 1/3-day horizons require their own observable horizon records.

Freeze an allowable horizon-observation tolerance and completeness rule from
source cadence before any labels are read. A missing horizon observation is
**unknown**, not negative. This primary status task requires observable labels;
it is not a survival estimator and must not discard missing cases without a
predeclared missingness/selection report. Report enrolled count, scored count,
missing denominator and worst-case score sensitivity; block promotion if missing
labels exceed the frozen completeness limit. Any separate time-to-closure study
must register censoring, interval observation, competing events and proper
survival scores before use; C-index alone is not proper probabilistic evidence.

## Splits, controls and statistical decision

Create immutable case-grouped chronological FIT, tuning DEV and untouched REPORT
cohorts from first eligible observation time. Freeze calendar boundaries and exact
ID hashes before fitting; actual dates remain BLOCKED on archive coverage. Keep
all observations/duplicate identities of a case together. Embargo at least seven
days plus the frozen observation tolerance between cohorts so FIT outcomes cannot
arrive after the next cohort's first forecast. Fit all transforms and classical
references without REPORT access. Publish exposure history; previously exposed
cohorts cannot become untouched again. Existing protected TEST stays closed.

Compare the integrated native Enterprise model with (1) global FIT event rate,
(2) smoothed agency/category/age FIT rates with a fixed global fallback and (3) a
regularized logistic practical reference. Those classical references are
unmatched utility references, not neural architecture controls. Optional matched
GRU is a **separate prospective comparison track**: 95,000–105,000 active trainable
and deployed learned parameters each, <=1% pairwise parameter gap, identical
inputs/IDs/order/exposure, measured training/tuning FLOPs and evaluation privileges.
If equal training compute cannot be evidenced, do not claim architecture superiority.
CPU wall time remains an efficiency outcome, not a matching requirement.

Primary proper score: per-case seven-day Brier score, with paired difference on
identical cases against the strongest reference chosen on DEV before REPORT.
Report log loss with a frozen numerical convention, calibration, positive/negative
support, source-age and agency/category subgroup scores. Seed means are not extra
independent cases. Use five preregistered seeds; paired bootstrap resamples frozen calendar blocks as clusters, retaining paired
cases and all seeds inside each sampled block, and reports all seeds and a 95%
gain interval. At least
100 independent REPORT cases is a floor, not power proof; determine the larger
precision requirement from FIT/DEV only. Do not choose a best seed.

A nonzero practically meaningful gain, completeness threshold, subgroup/calibration
noninferiority margins, bootstrap block length and all decision rules must be
signed and frozen in an activation addendum **before** REPORT outcome access.
They remain null in the companion JSON; no universal arbitrary percentage is
substituted for unknown operator costs. Until then this cannot pass an independent
gain gate. Development observations can be reported at their actual lower tier.

## Execution gates and preservation

Before any optimizer step, pin actual model/loader/source commits, PNDS reciprocal
Lean/Python record, full input-to-output graph and OPEN refinement obligations,
all hyperparameters, five seed values, update/tuning/FLOP budgets and live whole-
model parameter receipts. This proposal introduces no model variant and invents
no model/source hash. Reuse the existing parity validator and optimizer guards;
a descriptive JSON is not an enforcing training guard.

Independently audit source bytes and point-in-time assembly, split/normalizer
isolation and outcome access. Veritas must independently recompute source-status
labels from permitted original evidence; independent scoring is distinct from
independent confirmation that the city's physical work was completed. TrainLab
needs an actual standalone runner and receipts, not an inferred integration.

Commit every lawful original model/checkpoint, optimizer/RNG/sampler state,
prediction, case-level score, environment and replay material to TAC Git/LFS.
Independent clean-clone restore and score replay is required. `REPO_REPLAY=BLOCKED`
currently, because no fit exists. No old failures, protected splits or original
TAC evidence are rewritten. No inference of complete PNDS proof from shared lemmas.

## Capability-to-Value Card

- Skill: forecast source-reported workflow status at a useful future horizon.
- Operator hypothesis: help a human decide which open requests merit review;
  seven days is a candidate horizon, not validated intervention lead time.
- Data/time cut: NYC archive coverage partially identified; eligible IDs,
  available-time inputs, independent outcome ledger and rights remain BLOCKED.
- Evaluation: case-level proper scores and denominator above; no score observed.
- Original checkpoint/replay: NOT RUN / BLOCKED. Evidence tier DESIGN_ONLY.
- Customer/pilot/economics: UNKNOWN / NOT_MEASURED; no contact or pilot claimed.
- Stage unlocked: reviewable source-recovery and experiment design only.

## Activation checklist and accountable evidence

| Gate | Required artifact | Current state |
| --- | --- | --- |
| Lawful source | Independent rights/privacy disposition and safe acquisition scope | PENDING |
| Historical visibility | Recoverable ordered releases, release-clock proof, eligible feature audit | PARTIALLY_IDENTIFIED |
| Outcomes | Frozen target semantics and independent permitted per-case verification | PENDING |
| Cohorts | Exact case hashes, chronological boundaries, exposure and embargo receipts | PENDING |
| Experiment | Model/math/runner identities, actual counts, budgets, thresholds and seed freeze | PENDING |
| Original replay | Git-owned originals and independent clean-clone scoring | NOT_RUN |
| Release | Exact-head review and green required checks for relevant Forge PRs | Separate GitHub evidence; no merge/release granted |

Fail any prerequisite and keep training/scoring blocked. Recovery of one archive,
metadata hashes or successful integration CI never overrides the other gates.

Independent review verified all five metadata hashes and source claims; this
revision clarifies subset uncertainty, historical versus prospective landmark
clocks and calendar-block cluster resampling. Activation remains BLOCKED.
