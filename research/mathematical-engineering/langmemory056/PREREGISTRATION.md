# LANGMEMORY056 — frozen no-fit key-priority source experiment

**Registered before result acquisition; no GPU, no fit, no DEV/TEST.**
Parent/source: existing LANGMEMORY051 branch and authentic original CPython3.13.0 publisher 14 FIT owners (source receipts and licenses pinned in FAIR_COMPARISON.json). Prior owners were already repeatedly exposed; no generalization claim.

## Falsifiable question
Do compact exact 3-byte context-key occurrence counts change what survives collision-driven eviction enough to increase correct next-byte source retrieval at **identical physical bytearray budgets** 1344 and 4096, versus original K1 LRU, without losing to exact same-byte raw-history ring?

## Frozen algorithm and comparison
- Fixed two-way hash sets with 8 bytes/slot: key3, majority candidate1, Boyer-Moore vote1, saturated key frequency1 (zero=empty), last observed u16 clock2; remaining 64 actual bytes reserved for recent history, same as existing K1. No unbounded key dictionary, direct oracle lookup or future labels.
- **LFU** priority = observed (saturated) exact-key frequency. **Decayed** priority = floor(128 × frequency / (128 + elapsed source positions since last seen)). Replace minimum priority, then oldest, then lowest address. No parameter fit; constants registered before outcomes.
- Baselines: original equally sized real RawRing full scan majority; original FixedKeyTable K1 two-way LRU (both use identical recent64 exact-source fallback). Test all source positions t>=128, strides4 at cap1536/4096, stride8 at cap8192; byte x[t] evaluated after state queried and only then inserted.
- Original anchors expected: long horizon 11784 positions, raw1344 correct4152, old K1 table1344 correct2148. All arms must receive identical original target positions; fail if source/geometry drifts.

## Success/failure evaluation
Primary diagnostic: count correct among all positions, separately per 1344B/4096B and horizon. Secondary: coverage, exact key hits/misses, evictions, probe work, Python CPU query/update timing. A proposed policy is a useful *retrieval diagnostic* only if its correct count rises beyond both original K1 and equal-byte ring at the long 8192 source horizon, with no source/casual boundary breach; otherwise **NEGATIVE**. Even a positive result cannot claim neural accuracy, efficient production execution, generalization, or SSM/Transformer comparison.

The actual immutable source-label truth is evaluator-only; neither threshold selection nor policy is revised after reading these results. No GPU quota, training steps, publisher-hidden TEST or independent scorer are accessed. All raw empirical results/negative cases preserved in GitHub artifact.
