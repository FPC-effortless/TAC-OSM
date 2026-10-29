# LANGMEMORY050 — authentic publisher source, multi-continuation compression NEGATIVE

**Original source CPU-measured 29 Oct 2026**. Draft [PR #581](https://github.com/FPC-effortless/TAC-OSM/pull/581), [successful GitHub Actions #40827506769](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40827506769), [original per-owner raw evidence artifact #12876384110](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40827506769/artifacts/12876384110). Original legal publisher FIT source/phone planner scientific preflight PASSED; **15 constructed source-causal/budget/hash/counter tests passed**, and all three actual original source replay anchor checks matched. No new training, neural checkpoint use, GPU, DEV/TEST, Lean whole-graph proof or model promotion.

## Design and assumptions

Original 14 CPython 3.13.0 publisher FIT owners; exact source byte x[t] as outcome only, cache state contains only x[:t], source owner never crossed. Same historical cohorts already used 045-048, hence no new independent generalization claim. All candidate tables have an actual single fixed-size bytearray index plus a 64B raw recent bytearray ring, totaling 1344B or4096B **allocated payload**, excluding Python instance/runtime bytes. Exact context key3B; K=2 (16B records) orK=3 (20B records) Misra–Gries approximate continuation frequency pairs, stored 16-bit counts, one/two-way FNV set associations and deterministic age eviction. After key eviction or MG decrement counts are not exact. This trial has <=8192 source bytes, no uint16 count saturation. Ideal fixed-resident-key MG heavy hitter guarantee under no overflow/eviction is NOT a full bounded-hash guarantee; PNDS proof correspondence remains OPEN.

## Real original source next-byte retrieval — correct / all eligible source positions

| Source-only arm | 4,928 anchor | 13,245 extension | 11,784 genuinely long context |
|---|---:|---:|---:|
| Old A+B source sketch (nominal, not bounded Python memory) |933|2858|2710|
| New K2 1-way 1344B |558|1647|1586|
| **New K2 2-way 1344B** |**545**|**1598**|**1564**|
| New K3 1-way 1344B |504|1465|1438|
| **New K3 2-way 1344B** |**489**|**1437**|**1432**|
| Previous 047 K1 keyed majority-like 2-way table1344B |**781**|**2262**|**2148**|
| New K2 2-way 4096B |957|2851|2703|
| New K3 2-way 4096B |881|2568|2454|
| Previous 047 K1 2-way table4096B |**1169**|**3524**|**3343**|
| Raw ring1344B, O(B) historical context scan |1461|4503|4152|
| Raw ring4096B |1465|4943|4742|
| Unbounded exact historical context-count reference |1465|4943|4861|

On the long prefix the 1344B K2 2-way record loses 584 correct matches vs earlier 8B table (1564 vs2148), and 2588 vs equal-byte raw ring (1564 vs4152); K3 falls still further. At4096B, K2=2703 and K3=2454 vs old 8B table3343, raw ring4742. The extra candidate counters **did not fix the original storage/retrieval loss**.

## Source-grounded failure mechanism and CPU cost (long cohort)

| Arm | Sum Python query ms | Sum observed-byte update ms | Hash key evictions | Misra–Gries decrements |
|---|---:|---:|---:|---:|
| K2 2-way 1344B |610.37|946.77|**77,107**|813|
| K3 2-way 1344B |604.53|906.95|**80,278**|129|
| Old 8B table 2-way 1344B |606.95|800.12|not recorded in this run|n/a|
| K2 2-way 4096B |610.60|845.38|**57,219**|2,655|
| K3 2-way 4096B |609.66|848.81|**61,610**|740|
| Old 8B table 2-way 4096B |591.21|667.10|not recorded in this run|n/a|
| Raw ring1344B |10550.36|64.92|n/a|n/a|
| Unbounded exact index (unmatched source storage) |24.73|229.54|n/a|n/a|

All sums across 14 original source owners; Python timings are implementation/environment dependent, **not** whole-model/mobile/energy latency. Equal byte payload does not imply equal source information capacity, equal compute, equal probability model semantics or independent language generalization. Key eviction dominance relative to MG counters is descriptive evidence of insufficient key residency, not a formal causal proof without no-eviction controls.

## Decision and next tests

**STOP promotion** of the K2/K3 exact-key fixed-size candidate family as a TAC memory improvement on this source task. The original K1 table already provides better source-continuation recovery at every tested horizon in the same bytes; raw-byte ring remains far better albeit slower to scan. No new SSM/neural/TAC NLL, recall of arbitrary distant semantic events or calibrated use of memory was shown. Freeze negative results, leave draft unmerged, original models untouched.

Instead, derive the **key-capacity versus conditional-value-entropy** frontier before allocating any new slots: estimate source-owner distributions of distinct 3-byte contexts, encounter frequencies, longevity, number of alternative next-byte values per context, and how much source oracle gain comes from repeat-heavy keys versus rare keys. Pre-register a no-training exact-source diagnostic with a *fixed key slot budget*, measured eviction/replacement and no-oracle FIT-only policies, plus a retention-only no-collision oracle identified as an **unrealistic upper bound**, not a deployable algorithm. If the source's value uncertainty is mostly concentrated in few high-frequency contexts, adaptive per-key capacity may be useful; if distinct-key identity is dominant, prefer compact raw source or bounded indexed raw storage. This decomposition must precede neural checkpoint edits.

Mathematical representation of complete TAC trained memory, Lean exact-key residency theorem, signed source-to-PNDS linking and IEEE/AD equivalence remain OPEN. Future real source/generalization claims need untouched owner-disjoint publisher documents; protected TEST remains sealed.
