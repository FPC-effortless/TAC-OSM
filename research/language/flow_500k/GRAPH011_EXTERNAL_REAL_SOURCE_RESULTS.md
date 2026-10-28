# GRAPH011 — source-frozen calibrated approximate retrieval on eight unseen source files

**Date:** 2026-10-28. **Status: POSITIVE COMPONENT-LEVEL EXTERNAL-DOMAIN RESULT, NOT TAC ARCHITECTURE SUPERIORITY.** This experiment uses frozen original GRAPH010 source-query weights, never retrains on Python, and does not reselect the 0.1 mixture after looking at the external results.

## Before-scoring preregistration

The new external source manifest and policy were committed **before any scoring**, at TAC-OSM source commit `f14a0a51ded44d1a6a75ec2eb9bacdc092c05352`, `GRAPH011_EXTERNAL_EVAL_PREREG.json`. It pins eight real Python 3.13 stdlib files, their byte sizes and SHA256 digests, plus exact scoring rules:

- `email/parser.py`, `urllib/request.py`, `http/client.py`, `logging/__init__.py`, `re/_parser.py`, `asyncio/tasks.py`, `csv.py`, `textwrap.py`.
- None are among GRAPH009's six already-scored Python3.13 files: `argparse.py`, `ast.py`, `inspect.py`, `random.py`, `typing.py`, `doctest.py`.
- **8** deterministic 129-character source windows per file, using `np.linspace(0, byte_length-131, 8)`, with 128 next-character targets per window. **8 files × 8 windows × 128 = 8,192 real target characters PER seed.** Byte values >=128 are mapped to 63 identically across arms.
- Every window resets source memory; this is local online-document-context retrieval, not a multi-session 512-event streaming evaluation. All arms receive identical source and count events, same 128-class FIT16 smoothed prior, graph009 exact cache with tau=.05 and lambda=.6, and the same learned/untrained 2816-parameter 3gram query with **additional lambda=.1 only on exact misses**.
- The lambda **0.1** was picked POST HOC by inspecting the earlier 14 Vim FIT validation chapters, where original 0.6 failed. The eight Python files here were not scored during that calibration, so this is a new real external evaluation for the revised frozen policy; it does not undo the failed GRAPH010 prereg or create a formal sealed TEST set.
- All five fitted query checkpoint SHA256 digests remained unchanged from the GRAPH010 FIT-only run. No new weights, embedding updates or optimizer steps. This benchmark is a standalone FIT-prior + source-cache test: the 499,290-parameter TAC network was **not loaded**, and no SSM, GRU or Transformer was run.

## Eight-file real-source results

Mean next-character NLL (nats; lower is better) over five seed checkpoints:

| Identical source memory / retrieval treatment | External pooled NLL |
|---|---:|
| Exact 3-character count cache only (GRAPH009-style) | **2.844105** |
| Exact + 2816-parameter UNTRAINED context query, fixed additional lambda .1 | **2.814645** |
| Exact + same 2816-parameter FIT-TRAINED context query, fixed additional lambda .1 | **2.787104** |

**All five** trained checkpoints beat both exact-only and equally-sized untrained query controls. The mean trained versus exact-only reduction is **0.057000 nats/character (2.00%)**. The true learned-query increment over the identical untrained query is **0.027541 nats (0.98%)**. Do not interpret the full 0.057 gain as uniquely learned; untrained approximate neighbors already contribute some of it.

Per-seed external results:

| Seed | Exact-only | Untrained query .1 | Trained query .1 |
|---:|---:|---:|---:|
| 1701 | 2.844105 | 2.814171 | **2.780104** |
| 1702 | 2.844105 | 2.812883 | **2.785467** |
| 1703 | 2.844105 | 2.817814 | **2.791146** |
| 1704 | 2.844105 | 2.817730 | **2.792465** |
| 1705 | 2.844105 | 2.810627 | **2.786340** |

Across the **eight genuine individual source files**, trained + conservative 0.1 improved against **both** controls in **all eight documents** and all five seeds:

| Python source file | Exact-only NLL | Trained NLL | Untrained NLL |
|---|---:|---:|---:|
| asyncio/tasks.py | 2.89035 | **2.83211** | 2.85263 |
| csv.py | 2.65594 | **2.59709** | 2.62132 |
| email/parser.py | 2.98040 | **2.92955** | 2.95777 |
| http/client.py | 2.99988 | **2.94091** | 2.97362 |
| logging/__init__.py | 2.66426 | **2.60908** | 2.63848 |
| re/_parser.py | 2.67983 | **2.63309** | 2.65742 |
| textwrap.py | 3.05903 | **2.97734** | 3.01681 |
| urllib/request.py | 2.82313 | **2.77764** | 2.79910 |

These are within the same programming-language source domain. The independent source FILE selection and fixed calibration are meaningful, but this is **not new natural-language semantic or lifelong agent-memory qualification**.

## Prior negative result, math and architecture conclusion

**Original GRAPH010 failed** on real Vim FIT-only validation using lambda0.6:
- exact-only NLL **3.474112**, untrained + 0.6 NLL **3.761400**, FIT-trained + 0.6 **3.657214** (no improvement against exact-only in five seeds).
- 83.4% approximate-hit rate despite only 30.65% of approximated source positions increasing the true target's probability. Learned labels are not reliably helpful simply because neighbors exist.
- Decreasing the mixture to 0.1 selected post hoc on the already-seen Vim FIT14 gave **3.429566** vs same exact-only 3.474112; this inspired the separately versioned GRAPH011 external prereg.

For target y, `p_fused(y) - p_base(y) = lambda [p_query(y)-p_base(y)]`. For strictly positive lambda, any retrieved probability below the base's true-target probability reduces that target probability and thus worsens NLL. The separate PNDS `LanguageFusionRisk011.lean` states and machine-checks the exact-real risk laws, conditional positivity and log loss ordering; never infer a per-example safe weight from a future unseen target.

The candidate `source_query_hybrid_v7.py` retains GRAPH009 on exact hits and uses a fixed lambda0.1 on approximate misses only. The separate `test_query_hybrid_v7.py` must pass source-owner, streaming, exact-priority, 502106-total learned-parameter and normalized probability checks. Baseline GRAPH010 0.6 code remains unchanged.

## Replay limits and next experiments

- The local source-hash and checkpoint byte manifest, five trainable-query model+AdamW+RNG checkpoints, real source FIT training runner, FIT full measurements and post-hoc controls, precommitted external manifest and complete per-file/per-seed external JSON are preserved as a conversation ZIP. **GitHub cold binary checkpoint replay remains open** (binary files and source corpus not stored inside the repository), and the full original 500K neural checkpoint wasn't mounted or run in this separate source-component experiment.
- The local runner reproduced the committed GRAPH010 train/infer mathematics in a source-equivalent independent Python implementation rather than byte-exactly executing the published GitHub training script. Distinguish this from true cold GitHub checkpoint replay.
- Source labels/owner strings are provenance, NOT authentication. No measured 512-event long-horizon continual stream, legitimate cross-owner access control, semantic paraphrase, changed factual identity, latency/energy/serialized event I/O, or equal-cache pretrained SSM/TAC head-to-head.
- Before any TAC architecture promotion: load original 499290-weight frozen TAC and matched-parameter SSM or GRU with exactly the SAME external GRAPH009 cache and GRAPH011 query, identical quality/compute/source. Evaluate measured end-to-end NLL, true-entity memory, FLOPs, query overhead, owner authorization, uncertainty/abstention and protected independent source TEST. A proven count-cache-plus-kNN interpolation is not by itself a TAC-owned neural architecture advantage.

**Decision:** GRAPH011 passes its **preregistered component-only external source gate** (all five seeds and eight files) while the broader language architecture and commercial-superiority gates remain unqualified.
