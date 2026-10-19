# ME-003: authentic dynamic-state source audit

**Registration:** 2026-10-19, before opening new MetroPT-3 source data. **No model fitting and no new TAC capability claim.**

## Motivation
Independent ME-001B audit established 6,379/6,379 positive changes on the chosen SCANIA first 24 sensors. Source-level direction degeneracy makes that particular metric unsuitable to qualify reasoning, change direction or causal state updates, although signed-log magnitude errors remain legitimate observations.

## Primary source for exploratory *real-data suitability*
- UCI [MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B), DOI 10.24432/C5VW3R, real train air-compressor telemetry: pressure, temperature, current, valve statuses.
- Download only from publisher UCI static archive link, checksum full archive. Do not silently accept alternative mirrors.
- Only **first 30 calendar days** of chronologically ordered CSV rows may be parsed; the remainder stays reserved and unread as a scientific future period in this audit. This train-era profile is not untouched final scientific evaluation.
- For consecutive authentic samples, discard gaps >30 seconds or invalid/duplicate timestamp intervals from transition analysis.
- Count positive, negative, zero and invalid per-feature changes using fixed absolute epsilon 1e-6, not thresholds fitted on future data. Report exact counts and observed timestamp intervals. A feature qualifies as observationally bidirectional for this screening only if at least 100 positive and at least 100 negative genuine train-era changes.

## Gate
Pass **source-suitability** only when there are at least 10,000 earliest-window real observations, at least 5,000 valid short-interval transition pairs, and at least three bidirectional features. If not, report the exact negative count, do not select a favorable source segment after results. Additional domains (e.g., Backblaze device failure outcomes) remain distinct study arms, not a substitute for this primary source's negative result.

**Important:** This is not a model-training, entity-discovery, compositional-generalization, or true counterfactual-action experiment. Per-feature label direction diversity does not prove difficult prediction, and pressure or valve readings from a single compressor do not establish general multi-entity world modeling. Previous MetroPT TAC experiments failed against persistence, so retain persistence as an obligatory next-stage control. No synthetic success metric qualifies the architecture.

Raw source CSV bytes, official reserved future readings, recorded actions/costs and verification of intervention effects are not published by this suitability audit. All output is derived feature-level counters, provenance and time-window metadata.
