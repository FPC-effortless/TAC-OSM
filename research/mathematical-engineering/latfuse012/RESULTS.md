# LATFUSE012 — real-source oracle recovery, matched wrong keys, writer residuals and initialization polarity

**2026-10-27 — original physically measured UCI501 FIT source. FROZEN original model checkpoint/zero optimizer diagnostic, CI PASS. No architecture changed or trained.** [Original source and mathematical CI #40455251782](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40455251782). First preflight run #40455084361 FAILED a Python metadata identifier SyntaxError before reading source, corrected with no changed method or authentic cohort and preserved in Actions. Full per-seed/H16,H64,H128,H512 original-source outputs and SHA evidence: \`RESULTS.json\`. Source archive SHA256 \`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8\`; original four LATFUSE008 trained reference checkpoints unchanged; study uses *only* the TWO associative-query trained originals and four independently initialized but UNTRAINED original-model seeds. Frozen live architecture still 99,390 parameters /1,696 FP32 values+16 source flags=6,800 nominal recurrent bytes. Protected 2016–17 and exposed 2015 original numeric source unopened.

## Cohort and honesty

Exactly the same original precommitted May–June2014 FIT-era physical high-PM2.5>=150 ug/m3 and ordinary<=35 ug/m3, **24 same-monitoring-station/same-hour-of-day physical event/control pairs**, genuine 13-channel real hourly original source masks/age/time. March2013–Feb2014 original strictly past-time normalization, unchanged original two learned checkpoint SHAs and source train/evaluation case order. This cohort has been examined previously and is NOT independent qualified heldout evidence. Repeated source episodes share physical weather conditions across monitoring stations; per-station block bootstrap 4,000 fixed replicates is descriptive, not a population validity guarantee. Multiple H and seeds require multiplicity consideration. The original model's historic event query receives past **clock** and current authorized state ONLY; privilege source-target write keys or values NEVER inserted into deployable model query.

## Primary source-directional oracle test

At each genuine observed source anchor, independently compute original physically source-written normalized learned two-head8D write keys, source-trained value16 and beta residual. After actual original physical H16/64/128/512 suffix, read memory with original old write key as **privileged ORACLE diagnostic** and compare resulting 16D vector with source anchor's original learned value. Same-site/same-clock MATHEMATICALLY WRONG other event/control anchor key and its written value supply source-grounded specificity negatives. The original historic model's actual learned clock/belief query remains separate.

| Real delay since true observed event | Privileged oracle read ↔ source original value cosine | Matched wrong write key read ↔ source value | Native historic learned read ↔ source value | Oracle correct-minus-wrong key cosine |
|---:|---:|---:|---:|---:|
| 16h | 0.9345 | 0.8083 | 0.3200 | 0.1262 |
| 64h | 0.9222 | 0.7589 | 0.2820 | 0.1633 |
| 128h | 0.9113 | 0.7230 | 0.2105 | 0.1883 |
| 512h | 0.9023 | 0.7270 | 0.2768 | 0.1754 |

Source anchor event and its matched ordinary anchor learned value16 **already average cosine0.6171**. Raw high oracle cosine can therefore be substantially inflated by common value direction and overwrite replacement. The correct-minus-wrong key specificity gap is positive but uncalibrated and not a source-event factual reconstruction score. Even a near-perfect cosine cannot prove recovery of event-specific source PM2.5 level or binary high event without objective decoders using only legal read-time information, source-heldout labels and adequate negative controls. Native learned read's cosine much smaller suggests a querying bottleneck, but neither missing training nor value representational overlap is excluded.

## Source-genuine event write residual and future writes

Actual fitted source twohead soft delta writer, per eligible head h:

\`M_new = alpha M + beta (v-alpha M k) k^T\`.

Capture the **true residual write component** \`beta(v-alpha M k)k^T\` before adding decay of older memory. Compare its Frobenius norm for the original physical high/ordinary anchor source step against actual recorded intervening real source update residual magnitudes along the following horizon:

| Source physical delay | Event residual write norm | Mean per-case median true intervening residual write norm | Mean event/median residual ratio |
|---:|---:|---:|---:|
| 16h | 0.09191 | 0.08267 | 1.15 |
| 64h | 0.09191 | 0.07238 | 1.30 |
| 128h | 0.09191 | 0.06444 | 1.49 |
| 512h | 0.09191 | 0.05534 | 1.72 |

Thus original source event write is **not tiny compared with typical genuine source writes**. But its value contents are learned and may be nondiscriminative: an adequately sized event write is not a correctly encoded historic target. Future beta directional overwrites and cumulative alpha loss still act strongly, as LATFUSE005 demonstrated. The average event step/write ratio changes with which intervening samples enter the median.

## Random-query and learned source null

Genuinely different source sample historic read queries were shuffled by source pair while keeping original writer and state; mean event-projection/other-content read norm:

| Delay | True learned query event/other ratio | Other-source shuffled query ratio |
|---:|---:|---:|
| 16h | 0.02094 | 0.01678 |
| 64h | 0.00829 | 0.00761 |
| 128h | 0.00548 | 0.00395 |
| 512h | 0.000687 | 0.000514 |

Both are low and similar: the actual historical source event is only weakly distinguishable from other matrix contents using the native historic read, without demonstrating that a random query achieves event-specific factual recall.

## Original trained vs UNTRAINED initial source event ranking

Same original physical24 matched FIT-event pairs, model source encoder, historical clock and exact original 6800B state (no model training). Untrained original seed initialization ranks true event high over ordinary as follows:

| Real delay | Init80701 | Init80702 | Init81901 | Init81902 |
|---:|---:|---:|---:|---:|
| 16h | 29.17% | 16.67% | 37.50% | 50.00% |
| 64h | 25.00% | 16.67% | 54.17% | 54.17% |
| 128h | 41.67% | 25.00% | 58.33% | 41.67% |
| 512h | 37.50% | 33.33% | 45.83% | 62.50% |

Trained original H64 seeds were29.17%/25.00%; original init **seed80701** at H64 was already25.00%, original init80702 **16.67%**. This refutes the simplistic inference that model learning must have created the reversal. Other initial seeds happen to rank >50%: inverse ranking is *not deterministic for all seeds*, so source-pair construction, parameter initialization and trained objectives can all contribute. This is an underpowered, observational diagnostic; no independent source test or significance-adjusted population inversion established.

Original 24 pair/source station bootstrap rank95% per seed and H is in RESULTS.json. At H64 the *trained* seed80701 bootstrap interval ~[16.7%,41.7%], trained seed80702 [12.5%,37.5%] (source-station resampling; geographic dependence remains). The source hour is matched, but event/control dates differ; source offset/date-gap, preanchor 24h true PM2.5 and current query-time PM2.5 descriptive 24-pair ridge coefficients are in per-seed H64 output. These descriptive regression coefficients are NOT causal confound adjustment or fitted predictor validation.

## Interpretation and next falsifier

**Do not promote "source memory stores complete fact, only query is broken".** Privileged old-write-key cosine is high, but matched wrong keys also have high similarity, and original source-written event value vectors themselves overlap. This strongly motivates **value-content specificity and legal query-conditioned retrieval training tests**, rather than blind time encodings alone. Source-trained original event residual writes are not very weak; the historic query has poor specificity and its source-real event signal/other-content norm is tiny. Both write-content specificity and learned read addressing are live suspects, as is undertraining (32 original optimizer steps).

**Option B, recommended NEXT as a preregistered isolated real-data training hypothesis:**

1. **Primary authentic target: explicit historical CONTENT reconstruction**, not raw key–query cosine. At legally requested lag/time and source stream, reconstruct source-recorded *past* physical PM2.5 event label or observed magnitude, where target is supplied to the LOSS only AFTER prediction, never inserted into historical read-query input. A fixed budget/past-time-normalized train protocol and genuine source masks must be used; no source data after read time or protected heldout. Use source-within-window meaningful hard negative past timestamps at same site, not easy random-site negatives.
2. Add **auxiliary within-window contrastive key/query alignment only as a separate factorial arm** *after* authentic content reconstruction gains are established. Otherwise making original keys/queries aligned geometrically can be a degenerate objective solved without learning physically useful value records. Contrastive negatives must correspond to other genuinely observed writes within same window/owner, no event label or future target as negative-feature oracle.
3. Before new neural fits require (a) paired actual SOURCE real-long-history yes/no and full magnitude targets, (b) source-owner/current fast/slow/clock shortcut controls, (c) explicitly registered top1 among actual window writes (at16/64/128/512) with target/negative source ID and true timestamp, (d) source future+6h MAE/NLL and channelwise coverage, (e) exactly 6800B recurrence, active trained/deployed ~100k with ≤1% parameter & verified FULL forward/backward/optimizer FLOP parity, identical source masks/sample ids/optimizer steps/time budget, >=2 fixed seeds, (f) fully replayable frozen original checkpoints and protected heldout approval.
4. A **time-addressed write-and-query pair** (Option A) remains separately testable only after proving content reconstruction/negatives are learnable. A possible capacity limit under two 8D key-heads is *eight linearly independent keys per head* for arbitrary exact associative mappings under ordinary linear assumptions, **NOT** a universal exact memory limit of eight events across the two-head 256-cell model or proof of no long-range learning. Source-conditioned gated-delta, recency and shared value directions change effective capacity.
5. No unvalidated source objective/architecture changes or fits in this read-only branch. Prior original checkpoints, trained source exclusions and negative research preserved in unmerged draft PR.

**Status**: all real original read-only checks passed. Better native historical source retrieval is still a falsifiable hypothesis, not demonstrated capability.
