# LANGMEMORY046 — matched-call source verification versus random and growing index

**Actually measured** on genuine original PSF CPython3.13.0 FIT bytes, 29 October 2026. [Draft PR #576](https://github.com/FPC-effortless/TAC-OSM/pull/576), [SUCCESS GitHub CPU run #40823493340](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40823493340), [original unaggregated artifact #12876000667](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40823493340/artifacts/12876000667). Workflow 7 constructed causal tests PASS, source/phone/scientific preflight PASS. No neural training, checkpoint access/change, DEV/TEST, GPU, release or model promotion.

## Original observed-source protocol

Identical authentic 14 CPython FIT source owners as prior 045A, first up to 1536 original bytes per owner, 4,928 scored original next-byte positions. Each position uses only strictly earlier 3-byte content context and previously observed continuations. No future target enters the gate, index or random ranking; original source x[t] is added to index/sketch only **after** scoring x[t]. Five **precommitted hash-selection seeds** (11, 29, 47, 83, 131); they are five random selections **on the same data**, not five independent model training runs or held-out cohorts.

All policies except no-verify and full/periodic reference use exactly **885 costly verification calls (17.96% of events)**, matched per owner. Random-all selects SHA256-ranked candidates across all positions. Random-uncertain selects only from A/B missing/disagreement positions. The K per-owner random budget is determined retrospectively from the original observable selective gate, without consulting any target correctness; hence not itself an online deployable policy. Same number of calls does **not** guarantee the same total scan work.

| Verification policy | Correct of 4,928 | Verified calls | Scanned source positions (seed11) |
|---|---:|---:|---:|
| Cheap recent A+B, no expensive verifier | 933 | 0 | 0 |
| Random among all positions, seed11 | 1,045 | 885 | 731,909 |
| Original selective A/B gate | **1,050** | 885 | 737,845 |
| Random **among uncertain** positions, seed11 | **1,059** | 885 | 742,833 |
| Fixed every-fifth policy (not same call budget) | 1,049 | 980 | 810,460 |
| Exact index at same original gate | **1,050** | 885 index reads | 0 scanned |
| Exact index queried at every position | 1,528 | 4,928 reads | 0 scanned |

## All five frozen hash-selection seeds

| SHA selection seed | Random all correct | Random uncertain correct | Original selective correct |
|---|---:|---:|---:|
| 11 | 1045 | 1059 | 1050 |
| 29 | 1045 | 1052 | 1050 |
| 47 | 1056 | 1064 | 1050 |
| 83 | 1047 | 1055 | 1050 |
| 131 | 1050 | 1062 | 1050 |
| Arithmetic mean across same dataset | **1048.6** | **1058.4** | **1050.0** |

**Preserved negative finding:** the proposed deterministic t-mod5-and-disagreement selector does not meaningfully outperform even random-all at the same call count. It loses to within-uncertainty random on **all five** preregistered deterministic masks. Difference is +8.4 additional correct predictions for within-uncertainty random on average, or +0.17 percentage points across the common 4,928 original targets; no statistical independent generalization or tuned predictor is demonstrated. A/B expensive verification increases correct target count versus no verification, but the specific *selection rule* has no demonstrated merit.

## Exact index alternative — its cost is NOT free

An incremental source-owner keyed exact n-gram continuation Counter was rebuilt strictly from previous original observations. For all evaluated queries its answers agree exactly with independently recomputed naive historical scan, producing the identical selective gate 1,050 correct predictions. This is an exact dictionary/counting algorithm and **not a learned TAC associative neural memory**.

On this one GH Python CPU run, summed across all fourteen owners and 4,928 verification-probe positions:
- Naive source scan: **859.21 ms**, all query timings.
- Exact dictionary lookup: **3.26 ms**, all query timings, excluding index maintenance.
- Building/updating index at every originally observed byte: **24.80 ms** additional measured Python time.
- Combined lookup+maintenance: **28.06 ms**, ~30.6x less measured Python time than scanning all original positions on this host, **not** calibrated full application/runtime or production energy.
- Exact-context index serialized **lower-bound** across sources: **103,279 bytes**.
- Python recursive heap estimate for the index across sources: **3,540,408 bytes**, excluding runtime/allocator/transients.
- Raw original source prefix bytes held across the fourteen owners: **21,504 bytes**. The source index is thus much larger than raw source storage, and can grow as new distinct contexts/continuations arrive.

Measured per-query microtiming includes Python call/timer overhead and is not a cross-platform stable latency contract. These all-query times are comparable as source-only algorithm diagnostics; they are **not** an equal-work comparison to the 885 selectively selected scan calls. No SSM/Transformer/TAC model token latency is measured here.

## Scientific decision / next gate

The simple selector fails its pre-registered mechanism advantage test. Do **not** present the two cheap memories + modulo-five verifier as a proven efficient design or integrate it directly into native TAC-OSM. Preserve it as a negative control.

Next engineering gate should be **bounded growing-history lookup**, not yet learned next-token prediction: evaluate fixed-byte-budget indexed summaries (e.g. prefix-hash / collision-controlled buckets, eviction/aging, bounded exact key->counter records) against the current unbounded exact index, cheap A/B, P3 and sham history within source-tagged legal prior contexts, under matched stored-memory bytes AND read/write work. Test source-identity destruction, owner isolation, correctness by delay. Only when retrieval advantage survives controlled storage/compute and untouched publisher documents should a neural candidate with full PNDS Lean+Python actual graph, 5+ seed trained checkpoints, matched trainable parameters and training exposure be funded.

**Scope:** 4928 source-only lookup labels from previously seen FIT owners are not fresh language perplexity, semantic reasoning, conditional memory skill, mobile power measurements or independent scientific evidence. No protected TEST opened. Whole-model AD/IEEE and PNDS neural formulation remain OPEN. Source/maths history preserved.
