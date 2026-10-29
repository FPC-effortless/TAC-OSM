# LANGMEMORY048 — original publisher source verified-offset NEGATIVE result

**Real-source internal CPU measured 2026-10-29**. Draft [PR #579](https://github.com/FPC-effortless/TAC-OSM/pull/579); [successful Actions run 40826044659](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40826044659); [original raw output artifact 12876806909](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40826044659/artifacts/12876806909). Eight constructed owner/source-clock/full-key/buffer checks PASS, original publisher source/phone scientific preflight PASS. No model weights/training, GPU, DEV/TEST or release.

## Fixed original source and actual byte budgets

Original fourteen licensed CPython 3.13.0 FIT publisher documents, already exposed in 045–047. Strictly causal earlier-source 3-byte context, score genuine next-byte x[t] before updating source state. Anchor <=1536 prefix stride4: 4928 query positions; extension <=4096 stride4: 13245 positions; new genuine <=8192 stride8: 11784 positions. These are overlapping owners, not independent confirmation.

Single actual 1344B bytearray: raw1024B ring + 40 8B index offsets. Single actual 4096B bytearray: raw3072B ring +128 8B index offsets. 1-way and 2-way sets. Slot stores u16 hash fingerprint, u16 original continuation position, validity, 3 reserved bytes. Every retrieved offset is range-checked and the FULL original three-byte key reconstructed from the same retained raw ring. Wrong fingerprint/key and overwritten original source are rejected. The indexed output is the LATEST matching continuation, not the majority. Optional scan64 only on index misses. Python wrapper/allocation excluded from the nominal bytearray cost. Equal-total-byte raw ring predicts exact majority of retained source continuations but scans all candidate offsets. Unbounded exact index is a non-storage-matched reference.

## Correct genuine source next-byte positions out of all eligible targets

| Source-only algorithm | 4928 anchor | 13245 extension | 11784 longer-prefix |
|---|---:|---:|---:|
| Old 045A A+B sketch (nominal memory, not actually byte-bounded) | 933 | 2858 | 2710 |
| One-way verified locator1344B | 284 | 775 | 747 |
| **Two-way verified locator1344B** | **300** | **839** | **824** |
| Two-way locator1344B + bounded last64 scan | 437 | 1261 | 1255 |
| Prior keyed 2-way approximate-majority table1344B | 781 | 2262 | 2148 |
| **Raw ring1344B** | **1461** | **4503** | **4152** |
| Two-way verified locator4096B | 655 | 1879 | 1772 |
| Two-way locator4096B + last64 scan | 691 | 1997 | 1910 |
| Prior 2-way approximate-majority table4096B | 1169 | 3524 | 3343 |
| Raw ring4096B | 1465 | 4943 | 4742 |
| Unbounded exact original-history counts | 1465 | 4943 | 4861 |

On the longest original source cohort there are 7961 cases where the three-byte key occurred earlier than the previous64 observations. The 4096B raw ring no longer covers every preceding byte in all original histories; its 4742 correct falls below unbounded exact history 4861, unlike the earlier <=4096 experiment where equality was tautological.

## Measured work and failure (long prefix; identical original source and host)

| Algorithm | Python lookup ms | Update ms | Query work |
|---|---:|---:|---:|
| 2-way locator1344 | 54.32 | 593.24 | 23568 |
| 2-way locator1344 + scan64 fallback | 569.05 | 593.24 | 696912 |
| Previous 2-way table1344 | 306.65 | 405.48 | 23568 |
| Raw ring1344 | 5582.47 | 27.60 | 14500008 |
| 2-way locator4096 | 52.10 | 577.56 | 23568 |
| Raw ring4096 | 13466.22 | 23.28 | 34653424 |
| Unbounded index (not storage matched) | 13.65 | 112.16 | 11784 |

Actual two-way locator1344 suffered **84735 evictions** across 14 original streams and 8192-source horizon, 11 key-hash false candidates rejected before delivery. Two-way4096 suffered **70782 evictions** and 102 rejected fingerprint false candidates. Hash-hit counters include lookup probes from both ordinary and fallback arms; they are not distinct evaluation cases. No observed source byte outside ring was validly returned. Python timings are noisy and NOT a mobile/energy result; equal bytes is NOT equal read/write work or predictor semantics.

**Conclusion: failure of current architecture proposal.** The compact offset locator has cheap candidate queries but heavy writes, severe set contention, and only latest-value semantics. It falls far behind raw-history majority retrieval and even the previous 8-byte approximate-majority table, so it is not a quality/compute Pareto winner. Do not claim O(1) majority extraction, compressed information retention, new neural TAC memory, or production advantage. Preserve this negative, do not retrain/promote old TAC checkpoints.

Next: mathematically engineer the irreducible cost of retaining multi-continuation frequency under fixed source storage; compare carefully verified bounded multiple-candidate index or counting sketch at byte and CPU-work parity, natural source-prefix delays > retention, source identity lesions, previously untouched licensed publisher owner splits, normalized NLL fusion with frozen original trained TAC and random-uncertainty gates. Entropy/margin selection separately registered [LANGVERIFY049 #580](https://github.com/FPC-effortless/TAC-OSM/issues/580). Actual new neural variant requires independent PNDS Lean+Python typed graph and finite-precision/AD declarations, genuine five-seed matched original training budget, independent checkpoint replay. All formal neural assertions remain OPEN.

Archived 046/047 raw evidence ZIPs persisted beyond Actions retention in ChatGPT Library /TAC-OSM/Research Archives/LANGMEMORY046-047/; complete original successful CPU job logs committed under research/evidence/langmemory046_047/. Preserve original Actions artifacts/run IDs and hash receipts separately.
