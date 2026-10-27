# PAR-MEM-007 — correct historical WRITE key does not rescue fixed-size TAC memory

**Scientific conclusion:** Frozen TAC-OSM is a genuine persistent streaming architecture, but its existing 8×8 associative memory still **fails to provide useful historical PM2.5 reconstruction even with the exact privileged key originally used at WRITE time**. A shared deployable write/read key by itself is therefore NOT established as the solution. The original checkpoints and all prior negative results remain intact.

## Source and experimental integrity

Original SHA-pinned **UCI PRSA 2017 Beijing 12-station** public real-world measurements: SHA256 `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`. Physical inputs, true event chronology and observed historical PM2.5 targets are genuine. Five original full trained PAR-TIME-005 768-observation per-station recurrent TAC models and their original PAR-MEM-006 timestamp-query heads are SHA checked and frozen in read-only mode.

The 16/64/128-event historical target positions and source train/sample timelines were pre-registered by PAR-MEM-006 and reused without case or metric selection changes. The real May–June 2014 assessment records were **previously viewed**, so this is development evidence only and NOT a new independent test. The source adapter stops before opening any 2015–2017 numeric records.

Four separate **31-dimensional train-only ridge diagnostic probes** were evaluated on exactly the same authentic PM2.5 targets and original query dates. Each uses 14 current source observations, 8 known source event-time features, 8 vector elements from a selected matrix read (or zeros), and one intercept. The ridge lambda is fixed at **1.0**, fitted on original March–April 2014 real training query answers; no neural models were retrained. A ridge probe is a diagnostic, NOT a fully-compute-matched neural architecture comparison.

The four addresses:
- **Correct original historical writer key**, captured at the real past write event from the frozen core's content-conditioned address. This is PRIVILEGED and needs a full per-event key sidecar outside TAC's fixed-size memory. It cannot be used as a normal source-timestamp-only query.
- **Wrong adjacent historical writer key**, captured from the real next past event of that station, which also occurred BEFORE the query.
- **Original trained timestamp query key** from immutable PAR-MEM-006 trained weights and source-known query metadata, no true past measurements supplied.
- **No historical matrix read** (same feature dimensions, eight zeros).

Historical PM2.5 answers are fetched for the ridge target **only after** all frozen stream features and retrieval addresses are constructed. True full-history record lookup would achieve zero error only with unbounded external original history, so it is NOT a fair fixed-memory neural control.

## Observed real-source five-seed development results

Mean normalized historical PM2.5 MAE. **Smaller is better.** 3,096 original authenticated source DEV queries per seed: 1,032 each for lag 16, lag 64, lag 128.

| Read address | 16 old events | 64 old events | 128 old events |
| --- | ---: | ---: | ---: |
| **Privileged correct original writer key** | 1.264081 | 1.228065 | 1.247385 |
| Wrong adjacent source-event writer key | 1.253974 | 1.214386 | 1.230857 |
| Prior trained timestamp query key | 1.271975 | 1.199170 | 1.209572 |
| No associative matrix read | 1.264690 | 1.213126 | 1.236683 |
| Copy source current PM2.5 (historical-query baseline) | **0.690977** | **0.800042** | **0.814267** |

Across five frozen source-trained seeds:
- Mean cosine(correct old content-write key, adjacent wrong old event key) = **0.932538**. These real historical key addresses are highly similar. This is strong evidence of **low adjacent-event address discrimination** in the existing embedding, but does NOT alone prove unrelated distant source keys collide.
- Mean cosine(correct old content-write key, original learned timestamp-only query key) = **0.051146**. The address systems are poorly aligned in this study. Mean signs differ across the five seeds.
- Mean L2 difference between the final associative reads from correct and adjacent wrong keys = **0.053918** on these true real source queries.
- Giving the model the **exact original key still does not beat** wrong-key, no-read or copy-now baselines on the aggregate original DEV reconstruction metrics. This makes *write/read key mismatch alone* an inadequate explanation for failure.
- Ridge readout/feature/interference is NOT a proof that no physical information remains in the matrix: the 8x8 associative state may retain partial information not identified by this particular 31-feature ridge decoder. It is also not a bound on the theoretical memory capacity.

[Original source-regulated protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/parmem007-privileged-key-audit-2026-10-27/research/parallel-six/PAR-MEM-007-PROTOCOL.json) |
[All original per-seed real source results](https://github.com/FPC-effortless/TAC-OSM/blob/research/parmem007-privileged-key-audit-2026-10-27/research/parallel-six/PAR-MEM-007-RESULTS.json) |
[Source and exact write-key diagnostic runner](https://github.com/FPC-effortless/TAC-OSM/blob/research/parmem007-privileged-key-audit-2026-10-27/scripts/parmem007_oracle_writer_key.py).

## Changed scientific priorities — no superficial architectural rename

Before promoting a new input-only shared temporal address module:
1. Preregister real-source **writer-address diversity / effective rank** after 16/64/128/256 source-observed writes and different-distance wrong-key controls. Interference must be isolated from key mismatch, and outcomes must be compared on identical source anchors.
2. Compare short-lag write–read utility and real-key read–value decoding before attempting 128-event retrieval. A fixed 64-scalar superposition does NOT guarantee lossless retrieval of 128 independently varying source values.
3. Introduce a single **source-known-at-write and at-query event-identity/ordinal address generator** with the exact same address function and accessible query metadata, and a selective memory-admission/retention schedule. Quantify the memory-budget tradeoff and reserve a fixed number of slots for high-value historical facts. Do NOT smuggle unbounded per-event keys or old measurements into the fixed-state model. Introduce no new TAC family.
4. Verify batched affine scan vs streaming time-address operator, nonnegative elapsed-time constraints and total fixed-size state, with PNDS Lean and Python where appropriate. Preserve all existing trained checkpoints.
5. Preregister genuine independent longitudinal evaluation with true timestamp identities and physical observable outcomes. Compare against equal-parameter, equal-source/data/splits and **complete training forward/back/optimizer FLOP-matched** learned associative-read, source-time shared-address, matrix-blind, fully stateless, GRU/SSM, temporal-prior and source-copy baselines. Previous 2014 source DEV cannot be reused as new independent TEST.
6. Save weights, optimizer state, RNG, exact source order, streaming state and cold reproducible original results for all trained arms. If full parity cannot be closed, stop before making head-to-head model-superiority claims.

The new draft PR is strictly a diagnosis of the existing five authentic-source trained models, **not** an established remedy or new deployable historical-memory capability. Genuine source persistence and source-level clock tracking work, but *useful exact old-event retrieval remains unproven*.
