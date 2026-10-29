# LANGMEMORY047 — actually measured fixed-bytearray memory vs equal-byte raw-source ring

**Original-source GitHub CPU experiment SUCCESS, 2026-10-29.** Draft [PR #577](https://github.com/FPC-effortless/TAC-OSM/pull/577); [Actions run 40824343151](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40824343151), [raw source results artifact 12875374689](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40824343151/artifacts/12875374689). Seven constructed causal/source-owner/buffer tests passed, source and phone planner preflight passed. The first CI attempt failed on a harmless Python raw-ring map iteration bug (4 tests passed,3 failed), which was corrected before any source measurement. No neural training, GPU, DEV/TEST, new original neural checkpoint, math certification or promotion.

## Actual original-source protocol

Fourteen genuine publisher CPython3.13.0 FIT source-document owners. All 14 had at least 1536 bytes; the 4096-byte extension includes 12 source owners with 4096 bytes, one with 3197 and one with 2417. At t>=128 divisible by 4, query original 3 prior bytes and score the original next byte. Only previous observations enter any memory. Each evaluated policy sees the identical query positions.

- One-way and two-way exact-3-byte-key majority-candidate tables: 8 bytes each per slot, with 64 bytes of actual recent ring, precisely **1344 B** or **4096 B** bytearray data payload per owner. Not unbounded source caches. Table collisions/evictions and Boyer-Moore value cancellations are retained negative evidence.
- Equal-budget **raw** rings: exactly 1344 B or 4096 B of original preceding observed bytes, naive context-match scan limited to byte positions truly retained in the ring; no older raw bytes available to lookup.
- Original 045A Python sketch + original recent64: **nominal** packed capacity only, not actually bytearray memory-bound; for anchor comparability 933/4928 correct as before.
- Unbounded exact-context histogram reference: source-legal but grows with input history, **not** at same memory budget.
- Actual bytearray payload sizes verified; wrapper/class metadata, interpreter heap, original source corpus/reference structures and CPU allocator are excluded from per-stream claimed fixed payload. No on-device inference measurement.

## Actual source anchor: first 1536 original bytes per owner, 4928 scored query positions

| Arm | Covered | Correct | Correct / all 4928 | Python lookup + maintenance ms |
|---|---:|---:|---:|---:|
| Original legacy A+B (nominal 1344B) |3078|933|18.93%|30.16 (lookup not separately timed)|
| Recent 64-byte ring |637|386|7.83%|229.79|
| 1344B one-way keyed table + recent64 |1169|729|14.79%|386.73|
| 1344B two-way keyed table + recent64 |1236|781|15.85%|385.52|
| 4096B one-way keyed table + recent64 |1693|1060|21.51%|360.76|
| 4096B two-way keyed table + recent64 |1840|1169|23.72%|364.40|
| 1344B raw source ring |2471|1461|29.65%|2738.48|
| 4096B raw source ring |2477|1465|29.73%|2780.16|
| Unbounded exact source-key counts |2477|1465|29.73%|48.83|

The 4096B raw ring covers **every source history queried** in the 1536-byte anchor by construction, so its equality to unbounded exact-history is expected. The 1344B ring also retains 87.5% of full length at the end, explaining its near-exact anchor performance. The original LANGMEMORY046 **always-index** 1528 correct counted **cheap fallback on exact misses**; this 047 unbounded exact-only reference 1465 correct explicitly abstains on misses and is not numerically inconsistent.

## Longer source-prefix extension: up-to4096 original bytes per owner; 13,245 evaluated query positions

| Arm | Covered | Correct | Correct / all 13245 | Python lookup + maintenance ms |
|---|---:|---:|---:|---:|
| Legacy A+B |9038|2858|21.58%|80.89 (lookup not timed)|
| Recent64 |1830|1137|8.58%|624.79|
| 1344B one-way keyed table |3351|2119|16.00%|1035.95|
| 1344B two-way keyed table |3561|2262|17.08%|1029.05|
| 4096B one-way keyed table |5136|3267|24.67%|967.97|
| 4096B two-way keyed table |5509|3524|26.61%|982.05|
| 1344B raw source ring |7598|4503|34.00%|10594.81|
| 4096B raw source ring |8406|4943|37.32%|18819.18|
| Unbounded exact source-key reference |8406|4943|37.32%|136.07|

The 4096B raw ring is still at least as large as any source prefix in the extended experiment, so its equality to the full exact-history reference is **tautological**, not evidence of superior compression. Critically, the 1344B raw ring is storage-matched to the 1344B table while retaining **4503** correct vs **2262** for 2-way, but reads/writes cost ~**10.3×** as much measured Python time (10.595s vs1.029s). It also beats the **4096B** two-way table's 3524 correct despite using one-third as much payload, at ~10.8× measured total Python CPU read/update time. This is a quality vs CPU work frontier, not a single winning algorithm.

There are 8,001 authentic long-history cases in extension (3-byte original context genuinely occurred before last64). The 1344B raw ring answered 7193 of them and correctly answered 4224; the 1344B two-way table answered 3157 and correctly answered 1985. Repeat data, original owner/delay prevalence, and source-specific code character sequences may drive this task.

**Hash congestion:** over 14 extended source owners, 1344B two-way tables suffered **37,251** set evictions; 4096B two-way tables **22,346**. These are total write evictions, not independent wrong-prediction labels. The table's simple single-byte majority candidate is insufficient to reproduce the exact continuation distribution; the 8B slot budget, hash collision eviction, and majority cancellation jointly limit quality.

## Mathematical and scientific interpretation

The data falsify the immediate hypothesis that these simple byte-budgeted exact-key candidate stores match full retained source utility on original language prefixes. The two-way tables improve over one-way under the same total byte count, but both trail raw source retention heavily. The raw ring takes dramatically more computation to query; the original unbounded exact keyed index saves query work by retaining more state. This is a legitimate trade-off, not a clear novel efficient model advantage.

*Design options for a newly registered study:* derive compact recency-aware exact-context lookup over a bounded raw ring (not full duplicate source); design a bounded index with generation counters, collision-resistant full-key verification, fixed-entry or grouped heavy hitter tracking, and explicit query/update budget. Compare against raw scanning at the same storage and counted read/write work, not merely an equally shaped Python tensor. Test natural delays beyond retained ring window (8192+ genuine source bytes), per-owner source-historical substitution/future poison and previously untouched owner-disjoint publisher documents. Only after objective retrieval/latency gain should one wire source-grounded learned TAC logits, reciprocal PNDS Lean+Python and matched five-seed 500K neural training. These are proposals, not implemented results.

## Nonclaims and inherited evidence

This was **no-fit diagnostic** work: no neural TAC, SSM, BDH or Transformer NLL/accuracy measurement, no learned query mechanism, no new model checkpoint, no independent source-heldout generalization, no live mobile energy, no protected TEST. Prior 045A source-only evidence, 046 negative selective gate, and LANGMEMORY042 missing long history generalization are preserved. No production or architecture superiority claims; existing PNDS maths does not certify these two candidate source-only Python caches as new neural variants. A fresh trained candidate needs a separate source-pinned full neural mathematical-engineering record (including Lean and AD/IEEE limits).
