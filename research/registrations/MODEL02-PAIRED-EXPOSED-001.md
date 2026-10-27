# MODEL02-PAIRED-EXPOSED-001 — no-fit aligned references

This is an **exploratory paired-score diagnosis** on the previously exposed original Backblaze October/November 2013 source, not an independent benchmark or matched TAC-OSM neural architecture comparison. It must not be promoted to independent confirmation merely because it produces a new CI receipt.

Original publisher ZIP SHA256 `7f5a53e79b16e695b4b034955806bb3bb194534b169f6eca460dfd3dc48096fe`. Frozen original FIT models are Git-owned in `research/replay-bundles/PRODUCT-MODEL02-BACKBLAZE-2013-CLASSICAL/ORIGINAL_CHECKPOINT.json` (8-feature logistic) and `research/replay-bundles/PRODUCT-MODEL02-BACKBLAZE-CLASSICAL/ORIGINAL_CHECKPOINT.json` (2-feature categorical risk lookup). Both use October FIT only. No model is refitted and no protected source after November 15 is read.

## Fixed scoring and falsifiers

- Compare models on exactly the same 7-day-matured original **November 1–15 DEV** (drive ID/model, calendar observation date, actual source label) cases after 7 consecutive observed nonfailure days. Future missing or incomplete follow-up remains censored and is never a negative.
- Verify original case count **204,373** mature windows, **397** positive overlapping windows, **65** distinct positive drives and **180,532** censored windows before scoring. Any change is a hard source-evidence failure.
- Calculate Brier and log loss for both models and a prespecified **FIT-only constant prevalence** reference. Report the complete case set and a separate one-case-per-drive sensitivity metric using the **earliest chronologically mature noncensored forecast per physical drive**; never select the most positive or last-before-failure case using hindsight.
- The first-per-drive metric answers a different question: the outcome in seven days at the first eligible mature decision. It is not a survival estimator, unbiased incident capture or new independent event count. Score and case counts may be low for this conservative sensitivity cohort.
- Daily subgroup Brier for source-date consistency, without exposing original drive serial numbers or raw publisher cases. No architecture-superiority inference is permitted: the references have different feature sets and development choices on exposed data.

All inputs/source outcomes, design, checkpoint hashes and cohort were already exposed in preceding research. This is **a reconciliation of existing development evidence** and cannot qualify memory usefulness. No source publication-time availability verification, censor-adjusted risk or matched 100k TAC neural experiment is performed.

## Run

`python specialist_models/model02_paired_exposed_dev.py --out /tmp/model02_paired.json`

CI uses original publisher bytes and saves the summary JSON in an Actions artifact. Neither raw source data, secret keys, individual drive identities, patient-level events nor future protected dates are committed.
