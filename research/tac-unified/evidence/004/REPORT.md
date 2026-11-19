# TAC-UNIFIED-004AB hosted Veritas evidence

Frozen source faaca89526037fde3f8d4cda142897ceb3bad18f, Veritas eec439492407b0aa26922c23b5ae40d95289f4eb. [Actions run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/44578330060), [original artifact](https://github.com/FPC-effortless/TAC-OSM/actions/runs/44578330060/artifacts/14557279368), [draft PR](https://github.com/FPC-effortless/TAC-OSM/pull/735). This ZIP is the bit-identical original Actions artifact, SHA-256 verified against GitHub. Contains five actual TrainLab successful receipts, five clean verify results, five Research Lab imports and the lab SQLite database, memory, status, raw per-seed results, execution/compute plans, input pins and logs. Source and earlier evidence are in this branch. Unzip into a dedicated directory; compare against MANIFEST.json before inspecting. Do not treat imported evidence nodes as scientific qualification.

## Outcomes

Five seeds, 128 chains per seed/regime, 1024 actions, final-state success out of 640:

|Regime|Hard cert|Adaptive|Fixed5|Adaptive reads|
|---|---:|---:|---:|---:|
|exact8|640|640|640|2.4094|
|variable789|0|640|640|2.7308|
|heavy|0|431|432|4.4401|
|abrupt|0|529|529|3.3817|
|correlated|0|10|10|2.9225|

All 15 registered narrow gates passed. Heavy and common-mode errors remain. The rolling consensus quantile is not formally ACI or distribution-free coverage; adaptive timing includes decoding and simulated execution but not physical evidence acquisition. In exact8 the hard certificate was faster and acquired fewer reads. Adaptive shared corruption cannot deliver independent evidence.

B: 80/80 planted edges, zero false edges, 5/5 unsupported cyclic controls rejected; 320/320 joint goals solved vs 16/320 independent and 104/320 topological. 262208 simulator intervention observations per seed; graph hypothesis class supplied; no general discovery claim. Joint clusters are 16 bits; no measured hardware parallelism, arbitrary graph support or real-world outcome. Zero optimizer fits and GPU use. Native CSL, neural routing, full PNDS proof, scientific prefit and authentic capability qualification remain OPEN. Retain all negative results; do not tune on this cohort.
