# ME069 enterprise workflow lane — prospective source contract

Registered before downloading the original event log or training any model.
The objective is authentic enterprise next-event and waiting-time prediction,
not physical joints, causal intervention value or a simulated SLA claim.

Source: BPI Challenge 2012, published by Boudewijn van Dongen / 4TU on
2012-04-23, DOI10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f.
Original catalogue describes an event log of a loan application process.
Publisher metadata API:https://api.figshare.com/v2/articles/12689204.
Original file24027287, `BPI_Challenge_2012.xes.gz`,3,342,406 bytes,
publisher MD5`74c7ba9aba85bfcb181a22c9d565e5b5`.
Licence metadata says **4TU General Terms of Use**, not CC BY. Record that
exact attribution; do not redistribute original traces in this repository.
Only original source hashes, code, aggregates and derived model artifacts
are proposed for repository retention. No Woyengi customer log is assumed.

Retrieval receipt: the original download endpoint timed out. A pinned public
research mirror at Gength/generalization_analysis commit
`82d076ea0bb77c1164e6990b33b1f102540dacdc` supplied the exact original gzip,
matching publisher MD5 and size. Its SHA256 is
`5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3`.
This is byte-identity verification, not reliance on third-party relabeling.

## Frozen task semantics and time split

Read original case boundaries, timestamps, activities, lifecycle values and
anonymized resource labels. Case IDs identify trajectories/splits only and
never enter the neural input. An activity/lifecycle token and past event's
resource are observed covariates. Exclude no case based on forecast quality.
Cases with fewer than two events have no eligible next-event target.

Order cases by their first event time, tie-breaking by case identifier.
Cut at60% and80% of case starts. FIT contains first60% cases whose final
recorded event is strictly before the first DEV start. DEV contains middle20%
cases whose final recorded event is strictly before the first TEST start.
TEST is the last20% of starting cases. Record excluded crossing cases and all
cohort digests. This prevents training on outcomes not yet available at the
next cohort's start. No case appears in two roles.

The source parser necessarily reads the single original XES file to establish
case/time boundaries. It must disclose this ingestion access; TEST performance
and aggregate target distributions remain unexamined until model checkpoints
freeze. FIT-only vocabularies, timing statistics, transition priors and proxy
risk thresholds. Unknown heldout tokens/resources use UNK. No global outcome
statistics may fit the models. Do not describe file ingestion as sealed-byte
isolation of heldout records.

Predict (1) the next recorded activity/lifecycle and(2) time to that recorded
event, using at most32 observed prefix events and no future actions. Last
events have no next-event target and are excluded, not treated as success or
failure. Case end is only last recorded activity, not independently verified
workflow completion. No contractual SLA, actual timeout, intervention cost or
business failure label is assumed.

Model zero next-event delay exactly as a timestamp tie; positive delay uses a
proper positive waiting-time distribution. A FIT-only long-wait threshold may
be a prediction proxy, explicitly not a contractual SLA. Include latest-event
repetition, global prevalence and first-order transition/time statistics as
practical references. Compare native fast/slow/associative TAC against a
trained GRU with identical source access, exposure, supervision and resource
estimates. Freeze a full FAIR_COMPARISON before fitting. All original weights,
optimizer/RNG states must be repository-owned before TEST scoring, then
replayed from a separate clean checkout. Preserve the untouched robot
tissue-box task; do not open or repurpose it.

Engineering-only synthetic fixtures can check parser/causality and Woyengi's
advisory boundary. They do not establish predictive, commercial or financial
viability. Nothing here authorizes edits to Woyengi or Veritas.
