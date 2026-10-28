# GRAPH009 — bounded Bayesian source-count retrieval: frozen authentic-source results

**2026-10-28.** Scientific status: **algorithmic retrieval improvement; learned TAC architecture superiority NOT established**. Stacked on #532; original GRAPH006–008 negative results and original checkpoint bytes preserved. Source modules: `bayesian_cache_scan_v5.py`, `source_calibrated_tac_v5.py`; numerical regression tests in `test_calibrated_retrieval_v5.py`; companion PNDS PR #57.

## Formal architecture and prior research

Motivated by Mamba-2 SSD's attention to associative transition algebra, and Gated DeltaNet's distinction between state retention and targeted updates; **the source-conditioned dictionary/queue is a classical finite count-cache and is NOT itself a novel SSM, an attention replacement, or a differentiable key learner**. Existing TAC base stays 499,290 learned parameters. The cache adds bounded *non-neural* source history, dated events and count state with higher memory than the original 520 floats; *neural parameter parity does not imply state-memory parity*.

For strict past observations under a caller-labelled owner:
`c_t(h,y) = sum_{j+1<t, t-(j+1)<=512} 1[h_j=h] 1[x_(j+1)=y]`, with suffix key `h_j` of length k. Update with a new eligible source event and delete the expired event; a queue and dictionary produce exact causal integer count recurrence in amortized O(k-orders) time, with an O(128) categorical output. No future target is admitted. Define `p_copy(y|h)=[c(h,y)+tau*p_FIT(y)]/[n(h)+tau]` for FIT-only strictly positive prior, tau>0. On a hit `p_final=(1-lambda)*softmax(TAC logits)+lambda*p_copy`, and on a miss use the base prediction. This is a valid normalized probability law; PNDS provides scoped exact-real algebra, not Python/FP32/full AD equivalence. Caller owner strings are *provenance labels, not authentication*.

## Before-scoring freeze

`GRAPH009_EXTERNAL_EVAL_PREREG.json` records the six **distinct human-authored Python 3.13 standard library** files and their SHA-256s *before external scoring*: argparse.py, ast.py, inspect.py, random.py, typing.py and doctest.py. Sixteen genuine Vim9.1 FIT chapters used for a 6-weight learned gate, 14 disjoint FIT chapters for FIT-only validation; all 30 source SHA checks. **The learned source-only 6-weight gate FAILED** to beat its initialized constant mixture: best FIT-validation epoch0, NLL **2.092433**. Keep that failed experiment; no learned gating benefit claim.

Separately preregistered `GRAPH009_BAYESIAN_CACHE_PREREG.json` selected from **84 FIT-only** candidate settings (3 orders × 4 positive Dirichlet smoothings × 7 mixture weights), using original frozen GRAPH008 seed1701. Frozen choice `k=3, tau=0.05, lambda=0.6` is in `GRAPH009_FIT_FROZEN_SELECTION.json`, committed prior to external scoring. Original 14-FIT-validation NLL: raw model **2.1926317**; previous k4 latest-match fixed mixture **2.0924330**; new count posterior **2.0600712**. These are FIT-only validation measurements, *not* independent TEST.

## External source-only next-character evaluation

All six Python source files SHA matched, 8 frozen evenly spaced 257-byte windows each, 48 real windows ×256 targets = **12,288 character predictions per seed**. Five original GRAPH007 and five original GRAPH008 **frozen next-only** checkpoints, base learned parameters each 499,290, **no new language training, no external calibration, no new checkpoint fitting**. The same causal count-cache is supplied to BOTH models, with controls no-cache and original last-match k4 FIT-calibrated mixture.

| Five-seed mean | GRAPH007 | GRAPH008 |
|---|---:|---:|
| No-cache NLL ↓ | 2.852715 | 2.858166 |
| Previous last-match-cache NLL ↓ | 2.541340 | 2.546398 |
| **New 3-gram count posterior NLL ↓** | **2.390480** | 2.395659 |
| No-cache perplexity ↓ | 17.3511 | 17.4439 |
| Last-match perplexity ↓ | 12.7005 | 12.7642 |
| **Count posterior perplexity ↓** | **10.9200** | 10.9765 |

New count cache hits **37.606%** of source positions, versus old last-match hit **30.745%**. FIT-only unigram prior on external Python has NLL **3.297686**, much worse. New count posterior improves against both baselines **5/5** seeds in both families. **GRAPH007 remains stronger** under equal retrieval, so do not attribute gains to GRAPH008's extra learned-memory capacity or claim architecture superiority. Held-out documents are genuine independent Python source files, but repeated analysis is exploratory, not protected formal TEST.

## Exactness, performance and replay qualifications

- Independently derived slow source-count implementation vs new expiry-queue recurrence: equal 128-bin integer counts at all positions for multiple capacities, multiple sequence realizations and chunk splits. Source owner mismatch and future poisoning fail/leave past unchanged in tests; deterministic prewrite source eligibility only.
- CPU timing of *cache component only*, two eager CPU threads, four 256-character contexts, five repeats: old reference **0.488 s/batch**, new count-scan **0.0243 s/batch**, ~20×, **NOT** an end-to-end TAC decoding speedup or GPU tensor-core scan.
- The base 499,290 learned parameters are unchanged; extra memory stores up to 512 dated events *per suffix order* plus owner/count metadata and intermediate counts. Comparing against SSM/GRU/Transformer without equal cache access and accounting for memory and FLOPs **is invalid**.
- `source_calibrated_tac_v5.py` returns probabilities and owner-tagged carry; exposed CI tests check streaming numerical parity, unit categorical sums, positivity, causal ownership and exact count oracle. Compulsory source-integrity/param-budget tests were green on PR #533's preceding head; inspect latest CI per commit before final claim.
- Local reproducibility archive in conversation includes 10 original frozen 500K next-only checkpoint binaries and source/replay scripts; **binary Git/LFS cold repository replay remains BLOCKED**. Original formal sealed TEST remains untouched.
- Scope of PNDS PR #57: exact-real count posterior positivity/normalization given mass premises, normalized mixture, no-hit identity and scalar affine operator associativity; not Python dictionary/FP32/AD/entire six-concept CED verification.

## Next architectural qualification

Keep benchmark controls strong: equip matched selective-SSM, GRU and attention models with the SAME cache; use multiple genuine code/language sources, full training FLOP receipts, exact checkpoint replay and protected TEST. Develop a **learned** source query and source-authorized read policy that beats its corresponding *nonlearned* cache reference, then demonstrate attributable gain to TAC's non-cache recurrent state. Do not present the classical count cache as unique intellectual property.
