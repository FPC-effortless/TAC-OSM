# Model 02 — Stateful delayed-risk source intake (Backblaze)

**Status: code + constructed fixture tests only.** No actual Backblaze daily files have been downloaded, no learned TAC-OSM checkpoint or trained probability is produced, and no delayed-failure superiority or genuine-risk calibration is established by this adapter.

## Source and permitted uses

Original publisher: [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data). Backblaze collects daily per-drive S.M.A.R.T. observations and a recorded `failure` indicator, with quarterly published collections and quarter-dependent schema. Backblaze requests citation and allows derivative works but **not selling the underlying data itself**. Validate actual source/download, terms, permissions, rights, license applicability and ingestion at the time of the study. Do not commit whole raw drive archives. A drive recorded as removed/disappeared is **not known to have failed**.

This implementation deliberately requires the operator to supply specific, hash-pinned daily `YYYY-MM-DD.csv` files and a publisher-data permission assertion. It performs no automatic downloads and no unrestricted search of records. It only allows FIT or previously exposed DEV extraction; it will reject files at/after the manifest's `dev_end` boundary.

## Cohort contract

Example `SOURCE.json` (constructed values, **not original publisher data or authorized cohort**):

```json
{
  "schema": "backblaze-entity-cohort-v1",
  "source_url": "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data",
  "permission": "review-required-no-redistribution",
  "enrollment_date": "2026-01-01",
  "fit_end": "2026-02-01",
  "dev_end": "2026-03-01",
  "horizon_days": 7,
  "smart_fields": ["smart_5_raw", "smart_9_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw"],
  "enrolled_drives": [{"serial_number": "REPLACE_WITH_PRECOMMITTED_ORIGINAL_SOURCE_ID", "model": "MODEL"}],
  "csv_files": [
    {"date": "2026-01-01", "path": "2026-01-01.csv", "sha256": "REPLACE_WITH_REAL_FILE_SHA256"}
  ]
}
```

All calendar days through the requested FIT/DEV boundary must be included, each file byte-hash verified and all enrolled drives genuinely present/functioning on the initial day. This checks **some** retrospective selection errors, not independent proof the cohort was fixed before failures became known. Before a claim of scientific validity, an independent verifier must attest the enrollment freeze, publisher source release timestamps, permission, schema changes, data coverage and exactly which SMART values were available to a real operator at the prediction timestamp. This adapter conservatively denotes the `observed_at` and `available_at` as end-of-daily-snapshot **proxies**, not independently authenticated operational or public-release availability times. Until genuine source-time visibility is verified, do **not** promote results as deployed historical predictions.

## Usage

```sh
python specialist_models/backblaze_censored_risk.py --manifest SOURCE.json --csv-root /path/to/verified/daily_files --phase fit --out-prefix ./private/fit
python specialist_models/backblaze_censored_risk.py --manifest SOURCE.json --csv-root /path/to/verified/daily_files --phase dev --out-prefix ./private/dev
python -m unittest discover -s tests -p 'test_backblaze_censored_risk.py' -v
```

The tool refuses to overwrite originals and writes:
- `.labeled.jsonl`: exactly `observed_at`, numeric `features` with `available_at`, and `outcome` 0/1 with actual `outcome_at`. Compatible with the classical typed risk baseline in [draft PR #455](https://github.com/FPC-effortless/TAC-OSM/pull/455).
- `.index.jsonl`: row-aligned hashed entity identity, actual source manufacturer/model subgroup, case dates and observed event status. **Never supply this auxiliary outcome/index sidecar as neural inputs.**
- `.censored.jsonl`: cases with missing daily future records or incomplete horizons; no fabricated negative label.
- `.receipt.json`: manifest file SHA256, output hashes and selected source-case counts, plus nonqualification status.

For a prediction on a daily non-failure record, outcome `1` requires an actual later publisher failure flag at or within H days **with continuous observed intermediate days**. Outcome `0` requires every day through horizon H to be observed and no failure. If a drive disappears or the quarter ends without a complete H-day history, the case is right-censored and **not** treated as a negative. Failures present on the prediction day are excluded as non-early-warning examples. The adapter never infers unrecorded drive failures or causal maintenance utility.

Model features are current `log1p` SMART raw values with explicit observed/missing indicators and one-day differences where both observations are present. Missing features are placeholders paired with masks, **not** evidence that the sensor was zero. The identity/subgroup metadata is held in the evidence sidecar, not injected into the classifier to give an unfair shortcut.

## Research and product qualification

1. Preregister source-licensed original daily publisher bytes, legitimate fixed enrollment, fit/DEV/untouched TEST periods, source-visible daily timestamps, censor/retirement policies, manufacturer/model subgroups, context horizons, secure original hashes and exposure ledger.
2. Establish strong present-snapshot SMART logistic/tree baselines, causal historical-feature baselines, survival/hazard references and GRU/SSM/gated-delta controls with **equal effective identity/history information**. Do not claim memory superiority from giving only TAC a persistent serial number. Keep TAC new neural 95k–105k active/deployed weights and <=1% matched architecture parameter/data/compute budgets. CPU time is an efficiency measurement.
3. Use event-rate appropriate scoring: Brier and log loss for fully matured labels, precision-recall/AUPRC, calibrated lead time and censor-aware survival metrics with assumptions. Report event/retirement ambiguity, per-model groups and independent-case clustered uncertainty. Overlapping drive-day windows are *not* independent drives.
4. First public qualified model may be a simpler classical risk baseline if it wins. Actual fast cloud or physical edge inference, latency, energy, memory and developer value must be separately measured.
5. No protected TEST, Woyengi action authority, production safety, source-run CI or third-party adoption is implied by passing constructed tests.

Current gap: no legal raw daily source bytes have been materialized into this repo; no prospective model fitted; no independent evaluation. Model 01 work lives separately in `docs/FORECAST_RELIABILITY.md`; six research families and historical ME111/negative results are unchanged.


## Authentic 46-day source audit and rolling-enrollment finding — October 27, 2026

[Original 46-day fixed-day-cohort audit #40470483261](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40470483261) fetched the SHA-locked Backblaze 2013 publisher original archive; exact publisher ZIP SHA256 `7f5a53e79b16e695b4b034955806bb3bb194534b169f6eca460dfd3dc48096fe`. The exposed October 1–November 15 source contains 134 original publisher failure flags, but only **four distinct failures of the 900 drives present and functioning on October 1**. The fixed-first-day cohort's seven-day FIT had only 16 positive overlapping observation windows and **zero positive DEV observation windows**. Those 134 source-wide flags cannot legitimately be inserted into the fixed cohort's labels.

To diagnose this sampling limitation, an alternative **past-only rolling admission** rule was implemented: an entity is eligible once its current and six previous *observed* daily records are continuously present and contain no failure, regardless of whether it appeared on day one. The rule was proposed **after seeing** the original fixed-cohort census and is explicitly POST-EXPOSURE development, not independent confirmation or real-time prospective admission evidence. No protected source members dated November 16 or later were parsed; the publisher compressed archive was downloaded in full.

[Actual original source rolling-admission diagnostic #40471363459](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40471363459) passed seven constructed safety regression tests and audited 832,041 publisher drive-day records (26,815 distinct source physical ID/model pairs).

| Exposed phase | Admitted drives in phase | Seven-day positive windows | Distinct drives associated with those windows | Observed negative windows | Censored windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| FIT Oct 1–31 | 25,304 | 167 | 35 | 111,948 | 175,657 |
| DEV Nov 1–15 | 26,311 | 397 | 65 | 203,976 | 180,532 |

These figures are *provisional publisher-recorded event eligibility* only, not independent confirmed disk failure outcomes, model predictions, trained capability, risk calibration or commercial viability. A seven-day window can overlap another window for the same failing drive; the 397 DEV positive **windows are not 397 independent failures**. Missing future source days are censored and cannot be treated as safe negatives. Substantial source population expansion from the first-day 900 drives and loss of follow-up can distort apparent survival probabilities, source-time priors and group comparisons.

Original result receipts are now committed in `research/source-evidence/MODEL02-BACKBLAZE-2013-46DAY-FEASIBILITY.json` and `research/source-evidence/MODEL02-BACKBLAZE-2013-ROLLING-EXPOSED-FEASIBILITY.json` with Actions links, source SHA, aggregated eligible counts and strict nonqualification labels. The former source 14-day per-file hash receipt remains preserved separately.

### Next scientific gate (not an unconditional training approval)

1. Resolve Backblaze original per-day record availability relative to an actual operational prediction point, archival licensing/attribution, data rights and original source-reported failure semantics; preserve source-visible timestamps and group ownership.
2. Freeze independent source enrollment and model-family/manufacturer-specific censoring policy before further source/evaluation exposure. Distinguish initial/day-late admission, seven-day warmup, left truncation, retirements and entity discontinuities. Do not use source-wide failure flag count as eligible positive cases.
3. Fit simple snapshot/history/rule/survival controls on **lawfully authorized, source-time-qualified** original FIT. Record exact original weights/checkpoints and replay in Git; report source cohort and unique physical-drive effective denominators, calibration, PR-AUC, Brier and lead time as appropriate. An actual classical model must precede any claimed trained memory skill.
4. If real separate DEV confirms value and frozen scientific constraints permit, test the source-owned native ~100k TAC memory model and fair matched recurrent/gated-delta controls with exactly equal effective history and identity information. No protected TEST opening, release promotion or customer claims based solely on these audits.
