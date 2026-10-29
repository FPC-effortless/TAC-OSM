# LANGMEMORY047 — preregistered exact-storage-bounded source retrieval

**Status before first genuine-source CI run: prospective, 2026-10-29.** Descendant of original negative LANGMEMORY046 draft #576 and issue #574. Do not train, score DEV/TEST, promote checkpoint or imply improved neural TAC intelligence. This is an engineering retrieval falsifier.

## Central question

Can a fixed 1344-byte two-component memory genuinely recover useful older source-specific byte continuations better than simple raw byte retention at the **same actual buffer byte capacity**, and how quickly does its quality degrade with 4096-byte memory alternatives?

## Real source and frozen evaluation windows

Use the same original PSF-licensed, source-audited CPython3.13.0 FIT document owners as 045A/046. Two cohorts:
1. **Historical anchor** first at most1536 original bytes/owner, 4928 scored positions expected, and unchanged legacy original A+B = 933 correct (negative control).
2. **Extended original owner prefixes** first up-to4096 original bytes/owner (actual source lengths reported). This is an *exploratory extension on the SAME previously exposed source owners*, not independent confirmation. Do not tune on results.

At t>=128 every fourth source byte, query only x[t-3:t]. All memory tables/indexes/rings contain only previously observed events; observe x[t] AFTER scoring. No owner crossings, future target, artificial labels or protected TEST.

## Frozen comparator arms

- Legacy A+B from LANGMEMORY045A (exact recent64 and 256-bucket approximate sketch, **only nominal 1344B**, Python sets and dicts not bounded; retained solely to preserve historical numeric anchor).
- Real recent64 byte ring only.
- Bounded exact-full-3-byte-key majority-candidate table, one-way hash, with 64B retained recent ring: total **1344 B** and **4096 B**.
- Same fixed total-byte budget but two-way set-associative eviction, age-based local victim selection; total memory same as one-way.
- **Raw byte ring** of exactly 1344B or4096B storing only prior actual bytes, scanning the retained context+continuation history using no unaccounted older source. A strong simple comparator; O(retained B) reads per query.
- Unbounded original exact-context counter index (growing state, not a matched-storage competitor). Earlier LANGMEMORY046 full source exact-index results are preserved, not overwritten.

Actual bounded table storage layout eight bytes/slot: exact original three-byte context, candidate, majority vote, 16-bit time/age (valid for max4096 source prefix), occupied flag. 64B bytearray ring is included in each two-memory budget. For 1344B, remaining1280B gives160 8B slots; for4096B remaining4032B gives504 slots. Two-way variants halve the number of sets but keep identical slots and payload bytes. No Python dict or unseen history held in the fixed-budget lookup itself; wrappers/clock/allocator are reported as exclusions. Source loader and diagnostic exact index/legacy data structures may grow but cannot be accessed by a bounded-memory prediction.

## Report requirements and falsifiers

For each owner and arm: total queries, abstentions, correct among all eligible, accuracy given coverage, historical-context cases with a legitimate prior match outside recent64, their coverage/correct, counted contextual compare steps, measured Python read/write time, fixed bytearray bytes, true Python bytearray heap, occupied entries and hash evictions. Verify original legacy anchor exactly 933 successes and whole source owner isolation. Negative result remains reported if ring outperforms the new memory or if bounded mechanisms lose compared with cheap legacy on these previously studied owners.

**Cost fairness:** Equal 1344B/4096B allocated buffer is storage parity only. Table update+lookup and raw ring scans have different work; report these rather than declaring equal compute. Python timing has interpreter overhead and is not device/energy/whole-system benchmark. Unbounded exact index grows and never qualifies for bounded-state claims.

This source-only comparison does not establish generalization, neural next-byte NLL, semantic recall, or useful TAC long-history associative memory. No PNDS new model math is claimed: any new neural candidate requires reciprocal versioned source-pinned Lean+Python formalization with pointwise AD/IEEE limits marked OPEN, then authentic new owner-disjoint source, matched actual parameters/training budget, five independent training seeds, checkpoint replay and intact protected TEST seal.

Prior LATFUSE/GRAHP/041/042, 045A and 046 failures must remain preserved. No training, GPU quota, or merge. Advancement only on storage- and work-aware improvement that survives proper untouched owner evaluation.
