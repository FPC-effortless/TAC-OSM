# LANGMEMORY056 — fixed-byte key priority improves K1, loses to raw history

**Original-source no-fit result — October 29, 2026.** [Draft PR #595](https://github.com/FPC-effortless/TAC-OSM/pull/595); [successful original GitHub CPU Actions #40851792177](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40851792177); [complete original artifact #12887386594](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40851792177/artifacts/12887386594) (~19.6 KB compressed). Original real-source published dataset and protocol, preregistration and safe compute decision were frozen in code **before** this run; preregistered no-fit source/gate preflight and **13 constructed tests passed**. Authentic original publisher source results and per-owner hashes are in RESULTS.json in that artifact. No new neural training, original checkpoint changes, GPU, DEV/TEST access, or model promotion.

## Source and causal experiment
Fourteen previously exposed original PSF CPython3.13.0 publisher **FIT** owners, original 3-byte history key at time t, prediction of actual next byte x[t] strictly BEFORE observing it. Comparable positions: 4928 at max 1536, 13245 at max 4096, 11784 at max 8192. Same source owners/positions, same byte budgets 1344B and 4096B, all original raw-ring and K1 replay anchors passed. Genuine publisher text, but **no untouched holdout**.

The new **two-way exact-key records are 8 bytes each**: 3 key + 1 Boyer–Moore candidate + 1 signed vote magnitude + 1 saturated key-frequency + 2 source-clock bytes. A separate 64-byte raw prefix ring (within the same nominal budget) is used identically across the keyed arms. Priority functions were fixed before results: frequency-only and `floor(128*frequency/(128+elapsed_source_positions))`. The prior K1 two-way table uses source-recency eviction; equal-total-byte raw ring scans original retained byte history and votes exact continuation majority. Bytearray storage is matched, **not lookup/update work or CPU time**.

## Exact original next-byte matches (all eligible positions)

| Source horizon | Positions | 1344B old K1-LRU | 1344B LFU | 1344B decayed | 1344B raw ring | 4096B old K1-LRU | 4096B LFU | 4096B decayed | 4096B raw ring |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1536 | 4928 | 781 | 836 | 821 | **1461** | 1169 | 1181 | 1184 | **1465** |
| 4096 | 13245 | 2262 | 2406 | 2431 | **4503** | 3524 | 3613 | 3612 | **4943** |
| 8192 | 11784 | 2148 | 2225 | 2307 | **4152** | 3343 | 3373 | 3400 | **4742** |

### Original long-history primary decision
- At 1344B, decayed LFU improved 2307 vs 2148 K1 (159 additional correct / 11784 original positions, relative K1 +7.4%) but lost to equal-byte raw ring 4152 (1845 fewer correct).
- At 4096B, decayed LFU improved 3400 vs 3343 K1 (+57 / 11784, +1.7% relative), but lost to raw ring 4742 (1342 fewer).
- At 1536 horizon LFU slightly outperformed decayed; no monotone frequency-decay gain is established.
- Both policies improved key-value retrieval only modestly under real fixed byte constraints, exactly consistent with prior LANGMEMORY048–051 findings that context-key *retention capacity* is crucial. Saturated 8-bit frequency and two-way hash-set competition still evict useful keys.
- **Falsifier triggered:** no proposed keyed policy beats the equal-byte raw ring at long horizon. Record a negative for this line of incremental fixed-table tuning; do **not** deploy/train a neural model on this basis.

## Limitations and next engineering choice
These fourteen publisher owners have been repeatedly inspected across LANGMEMORY045–051 and cannot count as a fresh independent benchmark. Original CPU Python lookup timings, probing, evictions and update costs are stored in the artifact; they are not mobile decode throughput, VRAM or energy measurements. There is no neural calibration/NLL improvement, no learned gate, no generalized memory gain, no Transformer/SSM/BDH superiority, no scientific source-heldout claim and no protected TEST result.

**Pivot recommendation:** freeze these memory negatives as research evidence, prioritize trained frozen-source TAC incremental inference graph parity, measured decode overhead and bounded-state optimization; only later consider a new trained memory architecture if it makes a major information-retention improvement against equal-byte raw ring on separate eligible publishers. Every prospective trained candidate must follow the repository's real-source/parameter-parity/checkpoint/PNDS formalization gates. No Kaggle access issue or repeated setup is a prerequisite for further CPU-only architectural engineering.
