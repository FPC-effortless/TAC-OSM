# PRODUCT-MODEL02-BACKBLAZE-2013-CLASSICAL — First original source-trained diagnostic

**Status: classical fitted risk model, genuine publisher source, exposed retrospective DEV ONLY. NOT TAC-OSM; NOT independently qualified early-failure prediction, production alert, causality, calibrated forecast or new source generalization.**

## Original source and rollout

- Original publisher Backblaze 2013 data: https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data; pinned ZIP SHA256 `7f5a53e79b16e695b4b034955806bb3bb194534b169f6eca460dfd3dc48096fe`. Publisher use/redistribution rules apply; no raw publisher rows, serials or full original archive committed.
- Only already exposed source-member records October 1–November 15, 2013 were parsed; future members November 16+ **not parsed** (compressed original archive downloaded). Sources may be released later than original measurement dates: historical true publication/decision-time visibility is unverified.
- Source-owned `(serial_number, model)` is the entity identity but never an input feature or committed raw identifier. Seven consecutive past-or-current nonfailure daily records are required for admission. Training FIT = October 1–31; scoring DEV = November 1–15. Outcomes are formed only with complete seven-day observed future publisher records **inside the same phase**. Gaps/removals/future phase crossing are right censored, not negative. Publisher failure flags are observational, not independently adjudicated hardware failures.
- Features: `log1p` current raw SMART 5, 9 and 197 with measured masks plus one-day differences for SMART 5 and 197, all supplied at original observation day. Original input metadata does not independently prove source availability by model prediction time. SMART 198 was absent on the earlier exposed publisher sample, hence excluded from this registered product diagnostic.

## Original real-data classical training and scores

[Original source FIT/DEV Actions #40476036439](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40476036439) **PASS**, including 7 fixture contract tests. Full-batch deterministic L2 logistic training 55 iterations, step .4, fit-only feature mean/std, 8 coefficients plus bias. This small classical product baseline is **not a new TAC neural research variant** and does not claim architecture superiority.

| Metric | Original source FIT | Original source exposed DEV |
| --- | ---: | ---: |
| Scored (noncensored) overlapping seven-day windows | 112,115 | 204,373 |
| Positive windows (not distinct failing drives) | 167 | 397 |
| Distinct drives with positive windows | 35 | 65 |
| Unscored right-censored windows in DEV | — | 180,532 |

Original exposed DEV classical logistic Brier `0.001928342376544673`, log loss `0.01259078687978418`, average precision over overlapping windows `0.19053700838199006`. A simple unchanged FIT-prevalence reference on identical DEV windows scored Brier `0.0019389584399990274`, log loss `0.014132219543851672`, average precision `0.001855737084201071`. These are exploratory **overlapping-window** metrics and are strongly dependent on 65 identified failing drives, not 397 independent cases. The constant-ranking AP implementation tie-order is arbitrary and should be treated as a prevalence-scale reference, not a robust independently evaluated AP baseline.

**Original checkpoint and whole original result record are Git-owned:** `ORIGINAL_CHECKPOINT.json` contains all original FIT learned weights, normalized feature values, source SHA, FIT/DEV date policy and model limitations. `ORIGINAL_EXPOSED_DEV_RESULTS.json` contains original scores and all scientific nonqualification warnings. The CI replay file `.github/workflows/model02-original-classical-replay.yml` reconstructs the original source, **loads learned Git weights without fitting**, reruns identical outcomes, and requires full JSON equality of the original result record; check the workflow result before marking this check complete.

Replay from clean checkout with Python 3.12, actual pinned source network access:

```sh
python specialist_models/backblaze_2013_classical_dev_fit.py --replay \
  --out-model research/replay-bundles/PRODUCT-MODEL02-BACKBLAZE-2013-CLASSICAL/ORIGINAL_CHECKPOINT.json \
  --out-report /tmp/model02_original_replayed_dev.json
```

## Promotion blockers

1. The protocol was designed after observing source cohort outcomes. Source publisher release times, true model as-of-input availability and original failure/retirement event interpretation need independent evidence; no unchanged external TEST has been evaluated.
2. Severe observed/nonobserved record censoring, observational selection and survival bias may distort risk. Conduct manufacturer/model group qualifications and physically drive-clustered uncertainty. The short DEV period with 65 distinct event drives cannot guarantee lead-time alert performance on new hardware populations.
3. AP computed across overlapping windows is an **exploratory ranking metric**. Meaningful interventions require independent cost-of-alerts, precision and lead time at a prespecified practical threshold, not post-hoc threshold optimization.
4. Do not reuse the exposed source periods as fresh independent confirmation. Freeze a genuinely new source-authorized cohort and honest controls on equal information. TAC-OSM memory hypotheses require a separate ~100k-parameter neural model training/replay and fair matched architecture controls, not this classical product result.
5. No cloud or physical-device serving of this risk model has occurred. Woyengi remains authoritative for actual decisions, and Veritas must independently qualify outcomes and access rules before a controlled commercial pilot.

**Evidence status:** ORIGINAL_SOURCE_FIT=PASS; ORIGINAL_CHECKPOINT_IN_GIT=PASS; NO_FIT_REPLAY=CHECK_WORKFLOW; INDEPENDENT_TEST=NOT_RUN; REAL_TIME_AVAILABLE_AT=UNVERIFIED; PRODUCTION=NOT_DEPLOYED.
