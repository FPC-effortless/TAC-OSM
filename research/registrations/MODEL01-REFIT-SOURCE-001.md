# MODEL01-REFIT-SOURCE-001: source qualification for multi-household appliance channels

## Grounded publisher identity

Potential larger genuine dataset: [University of Strathclyde REFIT Electrical Load Measurements (Cleaned)](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/), DOI `10.15129/9ab14b0e-19ac-4279-938f-27f643078cec`, CC BY 4.0. The REFIT authors (Murray et al.) describe 20 households, original aggregate mains and nine individual appliance monitors per house, with readings roughly every 6–8 seconds. REFIT metadata and paper describe source cleaning and appliance channel changes.

**This study is source-engineering ONLY.** An original REFIT archive (publisher offers approximately 490 MB `.7z`, and the Zenodo version contains hundreds of megabytes per individual home) has **not been downloaded or scored**. These changes use publisher-shaped constructed fixtures solely to test data handling.

## Critical differences from UCI374 model target

- UCI374 training target: a building-specific `Appliances` value, ten-minute energy use in Wh, with a meter-specific definition.
- REFIT: each `Appliance1` through `Appliance9` channel is a separate monitored appliance's active power in Watts; monitored mappings differ by house, and some appliance assignments changed over time. Summing these is **not automatically** the same as UCI374's `Appliances` target. The whole-house REFIT `Aggregate` channel is a third, distinct target.
- The publisher-cleaned REFIT version has imputed missing readings: zero for gaps exceeding two minutes, forward-filled small gaps and zeroed spikes. A tidy 10-minute numerical block therefore **does not prove the source was physically observed throughout**. This weakness cannot be fixed from cleaned values alone. <https://doi.org/10.1038/sdata.2016.122>.

## Implemented candidate converter

`specialist_models/model01_refit_source_gate.py` requires explicit REFIT house ID and **one** selected IAM channel. It consumes source-shaped `Time,Unix,Aggregate,Appliance1,...,Appliance9` CSV, uses monotonic Unix seconds (not ambiguous local wall text) and integrates up to 16-second observed-sample-interval held power into ten-minute Wh. It never interpolates long timestamp gaps and emits only timestamp-complete publisher-cleaned ten-minute bins. Imputation provenance stays **UNVERIFIED** even for apparently complete bins.

The source intake is capped at 25 MB for controlled fixtures, preventing accidental full original archive download in CI. Building a genuine original publisher acquisition pipeline requires explicit chosen houses/channels, a separate streaming implementation, publisher SHA256 hashes, household/channel mapping tables with meter changes, source-time visibility receipts, gap masks from **raw** measurements where available, and a new frozen registration **before any original data rows are opened**.

## Research and business gate

**No model comparison, transfer score, new same-task generalization or customer success is established.** The existing REFIT metadata is a credible *candidate for further work*, not an actual qualified external test. The negative earlier UCI235 whole-house target-shift result remains unchanged; an apples-to-apples UCI374 appliance-only target is not yet identified in REFIT. A consumer-facing prediction must be independently validated by actual meter definition, data availability clock, source/copyright compliance, context, missingness, same-site vs unseen-site splits and a real operational decision.

Construction tests and CI validate only schema, integration, gap rejection, powers, source mismatch and no incorrect promotion. No original UCI374 TEST is read, no original REFIT rows trained/evaluated, no new TAC neural model is trained. This remains a draft research proposal and code adapter, not a released model.
