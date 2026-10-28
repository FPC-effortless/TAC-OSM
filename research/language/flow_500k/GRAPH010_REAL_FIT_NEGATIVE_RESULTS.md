# GRAPH010 FIT-only five-seed evidence — learning worked, fixed fusion failed

**Execution date:** 2026-10-28. **Status: NEGATIVE for the original GRAPH010 admission gate.** This is real-source **FIT-only** evaluation, not held-out protected TEST or an external independent source. Keep GRAPH009's exact count method as the qualified reference. No model promotion, no commercial claims or new semantic retrieval claims.

## Source provenance and controlled training

The local training used the original archived `TAC-LANG-GATE-003` `PROTOCOL.json` and authentic `/usr/share/vim/vim91/doc` files; **all 36/36 source chapter SHA256 digests matched**. Of the original 30 FIT chapters, first 16 whole documents trained GRAPH010 and the remaining 14 whole documents were FIT-only validation, with three fixed length129 windows per document. Original six DEV and protected TEST were not used. Non-ASCII bytes map to ASCII 63, following original code. The 128-bin strictly positive prior derives only from 16 query-training documents (this is not the separately selected prior from GRAPH009's full 30 FIT chapters).

Each seed 1701–1705 trained the **same 2,816-parameter** learned query (128×16 embedding, 48×16 bias-free projected 16D query, no backbone training), 256 AdamW steps, batch4, context129, learning rate .001, WD .01, global gradient clip1, genuine FIT source samples. Every seed kept sample/order SHA256, source hashes, optimizer/RNG checkpoint and passed cold reload measurement reproduction. The original **499,290-weight backbone was NOT loaded** in this local experiment; these are a valid standalone source-prior + exact-count/approximate-query component comparison, NOT full 500K TAC language NLL scores.

Local Python independently reproduced the same `LearnedContextQuery` train/inference subgraph as committed, and used an independently implemented exact causal three-character count oracle with strict past observed continuations and no window reset carry across unrelated documents. The local runner is included in the conversation ZIP, but is not byte-identical with the GitHub `fit_learned_query_v6.py`; this means the user's **repo-native independent cold replay gate remains open**.

## Fixed preregistered lambda=0.6 results (real FIT validation)

**Next-character negative log-likelihood** (nats per observed next ASCII character; lower is better), mean across five query seeds:

| Method | Mean NLL |
|---|---:|
| FIT-training unigram (no retrieval) | 3.698411 |
| GRAPH009-style exact three-character count cache only | **3.474112** |
| Exact cache + UNTRAINED seed-matched approximate query, weight 0.6 on exact misses | 3.761400 |
| Exact cache + FIT-trained approximate query, SAME weight 0.6 on exact misses | 3.657214 |

**Learned vs untrained: 5/5 seeds better. Learned vs exact cache alone: 0/5 seeds better.** Original GRAPH010's primary gate therefore **FAILED**.

| Seed | Exact-only | Untrained query | Trained query | FIT positive-evidence training rate |
|---:|---:|---:|---:|---:|
| 1701 | 3.474112 | 3.749992 | 3.655195 | 58.7449% |
| 1702 | 3.474112 | 3.757073 | 3.658744 | 58.6550% |
| 1703 | 3.474112 | 3.801167 | 3.669400 | 58.3411% |
| 1704 | 3.474112 | 3.774954 | 3.630253 | 58.6937% |
| 1705 | 3.474112 | 3.723816 | 3.672480 | 58.7813% |

The exact three-character cache hit **13.4673%** of prediction positions in these isolated 129-byte windows. The approximate matcher added candidate votes at **83.4077%**; its retrieved distribution improved the true-character probability against exact-only at only **30.65% of its hit positions**. On exact-cache misses, mean NLL worsened from **3.739049** to **3.950648** after .6 fusion. **Approximate query hit rate is NOT evidence of useful information retrieval.**

## Post-hoc fusion-weight sensitivity, NOT an independent success

On these very same already-scored 14 FIT validation documents, we varied the extra approximate-query mixture weight while leaving both the exact GRAPH009 cache and 2,816 learned query weights unchanged:

| Approximate mixture weight | Five-seed mean NLL |
|---:|---:|
| 0 (exact cache only) | 3.474112 |
| 0.025 | 3.450986 |
| 0.050 | 3.439516 |
| **0.100** | **3.429566** |
| 0.200 | 3.435775 |
| 0.300 | 3.463444 |
| 0.600 (originally preregistered) | 3.657214 |

The same weight 0.1 with a seed-matched **UNTRAINED** query gives mean 3.457492, compared with the trained mean 3.429566. The apparent gain over exact-only (**−0.044546 nats**) is interesting but **optimistically selected post hoc on precisely these validation labels**. It is a candidate for a new frozen protocol on genuinely unexamined data, not a confirmed GRAPH010 success or fair competitor win. Separately calibrated posteriors and genuine new source evidence are required.

The mathematical reason is structural: for true label y,
`p_mix(y)-p_base(y)=lambda * (p_query(y)-p_base(y))`. Positive lambda makes NLL worse whenever the query assigns less probability to the true label. The query hit frequency does not provide a lower bound on useful true-label probability. Inference cannot inspect the true future label to choose lambda; any learned gate must use ONLY available pre-target evidence and be evaluated independently.

## Persistent artifacts

- Conversation artifact ZIP **`graph010_fit_training_evidence.zip`**, size 207,520 bytes, SHA256 **`48e4b286c5ec02ee5558acdd499b1031fea9b527767d58dd580a15cfa57b7944`**.
- Includes the source-equivalent local training/query scripts, original source-hash-verified protocol, full per-seed `RESULTS.json`, all **five** cold-reloaded model+AdamW+Torch/NumPy RNG checkpoints, per-seed SHA256 sample orders, FIT post-hoc diagnostic and untrained-query control, independent archive SHA manifest. Copyrighted Vim source documents are NOT included.
- These binaries are **not committed into GitHub**; the original user rule requires repository-addressable independent checkpoint replay. That remains an OPEN gate. No pretrained GRAPH009/500K backbone checkpoints were loaded or modified.

## Decision and next controlled experiment

**Do not promote original GRAPH010 v6 with fixed 0.6 approximate mixing**. Source-causal and Lean probability contracts passed earlier, but empirical probability quality failed.

For a separately versioned GRAPH011 use a **FIXED, FIT-calibrated** smaller source-query mixture and a preregistered threshold/uncertainty condition, chosen without touching future targets or external test labels. Freeze all trained query checkpoints and both cache states; verify no change on exact hits. Evaluate against exact-only and **untrained same-size query at the SAME revised mixture**, with real documents not used to choose the revised weight. Measure source-hit precision, NLL and event/embedding query memory+FLOPs; compare identical sidecars attached to SSM/GRU/TAC. No superiority claim until this actually passes independent source evidence.

Status: implementation valid, source-gradient positive evidence present, but **fixed-weight learned retrieval scientifically failed primary FIT-only gate**. Negative evidence is retained as compulsory research history.
