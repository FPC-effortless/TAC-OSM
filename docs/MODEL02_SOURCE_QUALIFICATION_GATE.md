# Model 02 — Source-backed qualification readiness, not a trained risk model

**Source:** Original [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) 2013 archive, SHA256 7f5a53e79b16e695b4b034955806bb3bb194534b169f6eca460dfd3dc48096fe. Backblaze permits derivative works with attribution but not resale of the dataset. Its recorded failure flag indicates a drive's last operational day before a recorded failure, while removal/upgrade may remove drives without a failure event.

**What this code accomplishes:** It independently reads Git-owned original publisher-source and rolling/fixed-cohort receipts, verifies their common archive digest, chronological split boundaries, exact per-phase denominator reconciliation, daily member SHA evidence and failure-event counts, and publishes the named remaining blockers. It cannot turn unsigned or post-exposure source fields into prospective authorization.

Current authentic exposed 2013 development source contained 35 distinct failing drives among 167 overlapping FIT positive windows, and 65 among 397 exposed DEV positive windows. FIT 175,657 of 287,772 eligible windows were censored (~61.0%), while DEV 180,532 of 384,905 were censored (~46.9%). The fixed first-day 900-drive cohort has no positive seven-day DEV windows. None of these counts are independent proof of learned memory, and positive windows share individual drives.

## Run

```sh
python specialist_models/model02_source_readiness.py --out /tmp/model02_readiness.json
python -m unittest discover -s tests -p test_model02_source_readiness.py -v
```

The readiness program reports status BLOCKED_FOR_QUALIFIED_TRAINING with explicit blockers; a zero exit means the audit ran successfully, **not** that neural fitting, protected TEST or model release is authorized. Its provisional distinct-event count and censor-fraction planning filters (50 FIT distinct positive drives and <=50% censoring) are engineering triage thresholds, not a power calculation, universal dataset rejection, or proof of predictive value. A future power analysis and source-specific survival estimator must independently replace these planning filters.

Real-source qualification requires authentic source data permission and publisher event meaning, correct record available-at-decision time, independent held-out population/period under a freeze adopted before seeing the results, transparent left truncation/rolling enrollment, proper censoring and rare-event effective sample reporting. A simple classical model may be developed in a separately labeled *exploratory* study on exposed source; it must not claim independent risk skill, prediction-time certainty, commercial value or TAC architecture superiority.

Even if someone edits the receipt's booleans to true, this specific already-exposed post-hoc cohort remains blocked for qualified claims. New source authorization and independent preregistration would require a **new study identity and receipts**, not edits that silently promote this historic study.

Related: stacked source and eligibility work [Model02 PR #482](https://github.com/FPC-effortless/TAC-OSM/pull/482), original typed-risk baseline [PR #455](https://github.com/FPC-effortless/TAC-OSM/pull/455), native memory research issue [#444](https://github.com/FPC-effortless/TAC-OSM/issues/444). No original raw serials, snapshots, model fits or protected evaluation rows are committed by this gate.
