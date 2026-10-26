# Authentic operational data candidate: NYC 311 service requests (2026-10-26)
**Status:** PUBLIC SOURCE IDENTIFIED, NOT DOWNLOADED OR SCIENTIFICALLY QUALIFIED. No new train/DEV/TEST cohorts registered; no training/scoring/checkpoint/publisher rights or independent outcome attestation reported.

## Publisher and authoritative public references

- Publisher: New York City Open Data, official [311 Service Requests from 2020 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9) dataset (Socrata ID `erm2-nwe9`); daily-updated operational source. The October 26 source review found created/closed dates, unique request key, agency and complaint-type fields documented.
- [Official dataset changes](https://www.nyc.gov/opendata/news/all-news/311-Service-Requests-Updates) note a December 2025 split between 2010–2019 and 2020–present and updated labels. API field identifiers do not necessarily change when display labels do.
- [NYC Open Data terms](https://data.cityofnewyork.us/stories/s/Terms-of-Use/k9k7-3cje) warn providers retain data version control, and public rows may be **corrected, overwritten or refreshed at any time; older versions are not retained**. The dataset source must therefore NOT be portrayed as an immutable history of what was visible at a historical cutoff.
- [Official Open Data FAQ](https://www.nyc.gov/opendata/get-started/FAQs) describes open data as usable without general restrictions, but check any provider-specific terms and actual privacy/redistribution conditions before a Forge release. Official open/public listing does not alone establish provenance of each snapshot or timestamped field change.

## Candidate task and scope

Enterprise subproblem: source-grounded forecast of **time to actual case closure** or open-request nonclosure risk conditional on information verifiably available before each decision cutoff.

- Candidate independent operational outcome: `closed_date` when original source supports it; requests remaining open at the observation cutoff must be represented as censored/unknown, **not closed=0 ground truth**. Changes/reopens/revisions require special handling.
- Eligible input time: `created_date` for case creation **only after source availability has been observed/verified**. Current `agency`, `complaint_type`, `status`, descriptors and geo fields might have been corrected later and must **not** be assumed present at creation without authentic historical snapshots or publisher documentation confirming immutable field semantics.
- Case identity: `unique_key`; other repeat-observation groupings by geography or agency may need separate evidence and leakage risk assessment. NYC 311 is a public service-request task, **not proof of an action-conditioned world model**, authorized agent control or actual intervention benefit.
- Models/baselines: time-to-event survival methods and empirical agency/category base rates, last-known status, matched GRU/SSM and proposed TAC entity/temporal model, with genuine equal parameters/data/training compute where neural superiority is claimed and separately reported unmatched classical baselines.
- Evaluation: temporal holdout with protected future cohorts and group/area/agency checks, missingness, changes/reopenings, observed censoring mechanism, C-index and proper survival likelihood/Brier, uncertainty/calibration, false alarms and operational intervention cost where independently known. Avoid choosing hyperparameters after viewing untouched validation/test results.

## Hard prerequisites before authorizing a Forge or TrainLab study

1. Resolve **real as-of input visibility**. Prefer pinned periodic original publisher extracts and independently recorded acquisition timestamps; ideally verifiable agency event logs or snapshots. A single current-row dump whose `status` was updated in 2026 cannot prove historical features known in 2022. A new scrape made today provides a legitimate **today-known** snapshot, not retroactively available data in the past.
2. Pin original legally obtained source bytes with digest, source URL/API query, retrieval UTC, data version, explicit rate/terms/privacy scope and generated checksums. Independently review column-sensitive details and avoid unnecessary free text/geocoordinates if privacy/exposure rules require.
3. Distinguish true `closed_date` observed outcomes from open/missing/censored cases; independently validate source correctness including reopenings and data revisions. Never manufacture actions, rewards or completeness.
4. Require sufficiently diverse independent case/site groups and time cohorts so Forge's actual grouped splits do not put essentially all cases into one group or leak near duplicates across subject identities; write and freeze the split plan before training.
5. Obtain original held-out outcomes through a separate allowed evaluation path, with Veritas independent recomputation and no hidden target through Woyengi/Forge/TrainLab/TAC feature, cache, normalizer or UI.
6. Preflight registered seed/compute/parameter gates and immutable checkpoints and meaningful baseline/uncertainty thresholds. **Existing S1 and ME110 full scientific failures remain unchanged**.

## Current decision

**NO TRAIN / NOT QUALIFIED**. Continue software integration CI using constructed cases only. NYC311 is a viable authentic operational *candidate*, not an already released, independently temporally reconstructable TAC training/evaluation dataset. A trustworthy 311-as-of archive/source event log is the gating artifact, alongside Veritas/Forge security review.

## Desired independently verifiable source receipt

```text
source_id: NYC311/erm2-nwe9
publisher: NYC Open Data
source_bytes_sha256: <computed by actual acquisition>
retrieved_at_utc: <actual source retrieval instant>
publisher_asof_or_snapshot_id: <real publisher or archived snapshot identifier>
row_version: <proven source metadata, otherwise UNKNOWN>
rights_and_privacy_review: PENDING
eligible_feature_proof: PENDING
outcome_verifier: PENDING
source_groups_and_frozen_splits: PENDING
trained_model: NOT RUN
independent_test_result: NOT RUN
```

These fields are intended to be filled from actual published/retrieved evidence. Do not fabricate source hashes or timestamps to satisfy the contract.
