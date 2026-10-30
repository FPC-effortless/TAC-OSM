# LANGMEM087 read-only reproduction

Download only original completed evaluation artifacts from Actions40984392435:

```
gh run download 40984392435 -R FPC-effortless/TAC-OSM --pattern 'LANGMEM087-EVAL-seed*' --dir raw087
python scripts/langmem087_aggregate.py --results raw087 --output reaggregated087.json
```

This path reads original JSON receipts/checkpoint bytes only: no optimizer,
model inference, original DEV rawtext, protected TEST or new evaluation. It
checks nine actual checkpoint fileSHA256s, exact160→640→1280parent chains,
parameter/cohort/source contracts and row-recomputed metrics. Seeds and
training durations share3584distinct sourcepositions. Do not count repetitions
as independent data. Preserve failures and original archives. Scripts/models
must be checked out at the explicit launch535fbfafe6202f5a8aaf53629150372e60bd7cb1
or documented reporter-only descendant. Ordinary report commits do not launch
new fits/DEV reads. No merges/promotions authorized by these receipts.
