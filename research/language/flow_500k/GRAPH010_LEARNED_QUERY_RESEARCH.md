# GRAPH010 — learned approximate-context query on GRAPH009 misses

**Status (2026-10-28): source + exact-real scoped math + constructed unit tests implemented. No completed FIT training, no new external result, no semantic-identity superiority or architecture promotion.**

## Research question

GRAPH009's causally indexed three-character continuation count posterior improved next-byte language loss on authentic Vim FIT validation and a six-file previously unseen Python source-code domain. That cache is a valuable external classical memory mechanism, **not evidence that TAC's neural recurrent state beats the same mechanism attached to SSM/GRU/attention**. A learned query must produce useful evidence where an exact context match does not exist, must preserve GRAPH009 on exact hits, and must earn its extra memory/compute.

This is a deliberately modest first test of **learned approximate character context matching**, not semantic entity retrieval. Character trigrams do not encode natural-language paraphrases or durable business facts. No semantic results can be inferred merely by training a similarity function.

## Implemented parameter-controlled candidate

- `learned_context_query_v6.py`: normalized shared learned lookup for each position's 3-symbol suffix, with trainable 128×16 token embedding and 48×16 projected query. **2,816 active learned parameters**; attached to the original frozen 499,290-parameter backbone the total is **502,106**, ~0.564% more than base, inside the prospective 500K band and within 1% count. The base weights are not changed.
- At an exact GRAPH009 three-symbol context hit, **original count-based posterior wins unconditionally**; no learned override. The candidate reads only when the exact cache misses, selecting at most eight *nonidentical* prior contexts. It scores their 16D cosine similarity and produces a count-like weighted next-character posterior with the *same positive FIT prior/tau and fixed weight 0.6*. No predictions are made from unseen future continuations.
- Cache retention is bounded to **512 prior continuation events per owner**, each containing source ordinal, 3-byte observed context and observed next byte, plus at most four trailing bytes. Exact count cache **still exists** in parallel; overhead is explicit, not free 520-float neural memory.
- `source_query_hybrid_v6.py` wraps the existing `SourceCalibratedTAC` (GRAPH009) and the new source learner; preserves exactly the original probabilities on exact hits and on abstention. Separately account for the additional 2816 learned weights, 512-event sidecar and top8 comparisons / source-key embedding overhead.
- `predict_window` evaluates differentiable teacher-forced source windows, with the same strict source condition as the streaming path. `scan` implements global ordinal + owner-tagged, chunk-resumable prewrite lookup. Owner string is *not authentication*; this remains an application responsibility.

### Crucial true-label and gradient constraint

Source label `x_(j+1)` from historical candidate j becomes readable only if **j+1<t** at query ordinal t; input x_t has been observed, and the model predicts x_(t+1). The queried future label is never fed into the query. Only the **FIT training loss** may compare the candidate continuation to x_(t+1).

Initial operator tests surfaced a real learning obstacle: when all candidate continuation labels differ from the true next byte, a normalized weighted vote cannot increase the true-label probability by re-ranking them. The query gradient was about 8e-16 on an intentionally unique-label fixture. Merely requiring a finite gradient would repeat the earlier GRAPH006 CED testing error.

The repaired training plan adds **positive-evidence contrastive ranking**: on FIT examples having at least one earlier distinct context with the same observed continuation, optimize

```math
L_{query}(t)
 = logsumexp_{j\in E_t}(s(h_t,h_j)/T)
 - logsumexp_{j\in E_t, x_{j+1}=x_{t+1}}(s(h_t,h_j)/T).
```

No-positive examples are **skipped and counted**, not hallucinated as retrievable. A finite sentinel prevents undefined gradients from logsumexp over empty sets; numerical tests require positive, finite gradients on actual-positive fixtures and explicitly demonstrate zero-gradient coverage on unique-label fixtures. This objective trains content ranking only; it does NOT guarantee a truth-bearing neighbor on all queries.

## Authentic-source registration, no external tuning

`GRAPH010_FIT_QUERY_PREREG.json` was committed **before any fit results**. `fit_learned_query_v6.py` is a SHA256-gated executable training and qualification recipe using only the archived original 30 human-written Vim9.1 **FIT** chapters; first16 are query-training and remaining14 are FIT-only validation (whole-document disjoint). All 36 original FIT/DEV document SHA256s must match the frozen original protocol before work begins; the six original DEV chapters are verified but not used for this candidate.

Five seeds 1701–1705; 256 AdamW updates each; batch4, length129 windows, 512-event bound, 8 top candidates, tau0.05, mixture weight0.6 unchanged from GRAPH009. Each sampled source filename/offset and observed bytes updates a per-seed SHA. Each learned query checkpoint includes model+optimizer, source manifest/protocol, sample SHA and RNG for cold reload; all five source-model checkpoint scores must be independently replayed before promotion.

The FIT-validation comparison uses **the same source windows** for:
1. FIT unigram (no retrieval);
2. GRAPH009 exact-three-symbol count cache only (same FIT prior/tau/horizon);
3. Untrained, seed-matched 2816-parameter approximate query on exact misses;
4. Trained approximate query on exact misses.

Report total and exact-miss NLL, exact/approx hit rates, positive-evidence training coverage, per-seed outcomes, query compute, serialized memory and original GRAPH009 match invariance. Any learned improvement must beat BOTH the exact-only and identical untrained-query controls in at least 4/5 seeds; only then freeze a new *unseen* external evaluation. The six already-scored Python3.13 source files in GRAPH009 can be used for debugging, but **not reused as new independent confirmation**.

The script is committed, but **it has not been executed to completion in this PR**. No fit scores, new checkpoint bytes or external gains should be inferred from code passing CI. The original 56MB GRAPH009 archive is separate from GRAPH010 and contains no trained new query weights.

## Fair architecture comparison gate

Before any assertion that TAC benefits more than the same memory on a stronger model:
- Attach the **exact same** learned query, smoothed count cache, 512-event and owner policy to an approximately 500K matched SSM and GRU (and a fair attention model if feasible), and report any extra state/I/O and learned weights.
- Give all candidates identical authentic source splits, query-side training examples, optimizer-update budget, and the same parameter band (competitors within 1% of each other), with full training FLOP differences recorded.
- Measure native model alone, with GRAPH009 count-only, and with GRAPH010 exact-first learned query, plus copy-only conditional-frequency and explicit source-index baselines.
- Primary question: **does TAC's neural internal state contribute an improvement after controlling for exactly the same external retrieval?** Extra retrieval alone must not be presented as TAC architectural superiority.
- General semantic-memory gate requires real source facts, paraphrased queries, entity isolation, temporal supersession, evidence citation and reauthorization. A tiny 3-character embedder cannot be scored as solving LongMemEval, NoLiMa or enterprise semantic retrieval.

## PNDS verification and limitations

Companion `PNDS/LanguageQuery010.lean` adds nine scoped exact-real named theorems for strict-past source eligibility, no future target access, exact-cache priority, no-hit abstention, conditional approximate-only fusion, normalized output, positive and normalized smoothed weighted posterior. The 129-theorem strict audit must be green on the latest commit before it is called verified. These theorems do not prove PyTorch's topK, event queue identity, logsumexp/softmax FP32 gradients, learned model accuracy or user authentication.

**Decision**: Preserve GRAPH009, mark GRAPH010 an untrained research candidate, and do not promote to a commercially differentiated semantic memory engine until genuine FIT and independent source-task results, fair SSM-cache parity, benchmark cost receipts and durable cold checkpoints support that conclusion.
