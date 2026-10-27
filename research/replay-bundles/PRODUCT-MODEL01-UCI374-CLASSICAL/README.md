# PRODUCT-MODEL01-UCI374-CLASSICAL — Original real-source classical checkpoint and DEV replay

**Type:** standard-library fitted six-feature linear ridge model, NOT a TAC-OSM neural model. Original checkpoint bytes and original CAL/DEV results are retained in this Git repo. Commercial/independent model scientific qualification: NOT ESTABLISHED.

## Source and authority

- Original UCI dataset: [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), DOI 10.24432/C5VC8G; CC BY 4.0. Author: Luis Candanedo. Cite original collection and the 2017 paper.
- Original archive URL: https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip
- Original ZIP SHA256: 2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a; original ZIP 11,979,507 bytes.
- Original source split: FIT observation indexes [0,9867); CAL [9867,13814); exposed DEV [13814,15788); future protected TEST [15788,19735). Historical raw file was downloaded in full compressed form, but the protected TEST target rows were not parsed/evaluated or used in selection.
- Actual original source time labels are naive local 10-minute aggregate timestamps; source publication latency and live inference data freshness are NOT independently qualified.

## Original trained evidence

- [Original fit, CAL and DEV GitHub Actions #40467535435](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40467535435): 8 constructed invariant tests PASS, actual original source FIT classical ridge fit 9,717 examples, CAL 3,941, DEV 1,968. Five numerical lag/time features plus an additional lag feature; deterministic closed-form L2 ridge coefficients.
- `ORIGINAL_CHECKPOINT.json` contains all 7 original fit coefficients, FIT-fitted normalizer, source identity and algorithm registration. It is **not** a recipe in lieu of original weights.
- `ORIGINAL_DEV_RESULTS.json`: DEV MAE 44.79225268659299 Wh; RMSE 84.9138909041799 Wh; observed nominal 90% coverage 90.59959349593496%, width 185.065417371686 Wh and interval score 412.5286886865046.
- Earlier real original persistence DEV MAE 50.9400406504065 Wh, coverage 90.90447154471545%, interval score 620.7723577235772. This earlier DEV score was already visible when the ridge candidate was implemented; any comparison is exploratory/post-exposure, not independently confirmed architectural superiority.
- [Full no-refit Git original checkpoint replay #40467711128](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40467711128): exact original-source ZIP hash, original JSON checkpoint loaded without gradient fitting, all original metric fields compared for full equality. Status PASS.

## Replay without training

```sh
python specialist_models/uci_appliances_ridge_dev.py --replay-only \
  --out-checkpoint research/replay-bundles/PRODUCT-MODEL01-UCI374-CLASSICAL/ORIGINAL_CHECKPOINT.json \
  --out-report /tmp/model01_replayed_dev.json
python -m unittest discover -s tests -p 'test_uci_appliances_ridge_dev.py' -v
```

Requires Python 3.12 and permitted Internet access to the **same pinned publisher-original ZIP**. The script refuses source hash changes. Source bytes are not committed, respecting repository size; the public publisher is referenced and validated.

## What remains unproven

Does not qualify any TAC-OSM neural variant, conditional PM2.5 or NO2 gap intervals, physical mobile execution, production fast-cloud latency, independently new-site/generalization, customer ROI or action authority. Single-household correlated and overlapping forecast cases violate simple iid assumptions, and the underlying data are historical. This is a useful first developer product model candidate; future independent evaluation requires a genuinely new frozen cohort and expert review of actual field-availability clocks.
