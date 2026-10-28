# GRAPH015-TEST-001 — independent source first-open evaluation preregistration

**Frozen on 2026-10-28, BEFORE reading any CMake target bytes or model outcomes.** Model checkpoints and cache weights frozen before source choice. **No training, tuning, loss-weight adaptation or gate fitting after source open.** This is a first-open cross-domain exploratory confirmation exercise; verified independent original-replay and training-compute gates remain OPEN.

## Exact new authentic source
Kitware CMake **3.31 installed manual** original author-published RST source at /usr/share/cmake-3.31/Help/manual. Fixed eight documents selected solely using directory filenames and size metadata (not reading labels or outcomes):
1. cmake-buildsystem.7.rst
2. cmake-compile-features.7.rst
3. cmake-configure-log.7.rst
4. cmake-developer.7.rst
5. cmake-file-api.7.rst
6. cmake-generator-expressions.7.rst
7. cmake-language.7.rst
8. cmake-packages.7.rst

Each file must exist at the registered version; source bytes authenticated by SHA256 and package/license provenance; do not redistribute publisher text. If source rights are insufficient stop before evaluation. These documents are not part of original Vim9.1 FIT/DEV and did not participate in GRAPH011 Python query training. Prior-exposure risk for an unrelated system corpus cannot be independently attested from software receipts alone, so classify as **FIRST_OPEN_NEW_CORPUS**, not guaranteed sealed independent confirmation.

## Exact scoring protocol frozen before first open
Models: original historical five 499290-parameter WaveCED TAC checkpoints (seeds 1701–1705), new five GRAPH015 499290-param delta TAC, and new five GRAPH014 498288-param diagonal SSM, with identical corresponding historical source FIT sampling hashes. All fully frozen model and identical GRAPH011 query weights loaded with original binary hash checks when possible; no new gradients or optimizer steps.

Byte-to-class map: source file raw bytes in original byte order; UTF8 or other bytes >=128 mapped to ASCII '?' (63), **exactly as original model training**, 128-way categorical scoring. For each of 8 documents with byte length N>=129, calculate 8 *fixed* window starts `np.linspace(0,N-129,8,dtype=np.int64)` before inspecting model outputs, each window length 129, use first 128 as model inputs, score 128 next-byte labels; **1,024 real predictions/document = 8,192/seed = 40,960 model targets/architecture over five seeds** (repeated labels across seeds do NOT imply independent examples). Individual window state resets, ordinal start=0, so no long-memory claim; whole-document cache ownership is separate per window. Never let a model see future labels or other documents in cache.

Primary pre-registered contrast: equal-source **native NLL (delta TAC minus original TAC)**, five-seed mean per-character, lower is better. Significant meaningful positive improvement requires **at least 0.01 nat/byte absolute decrease** in delta TAC native NLL, improvement for >=4/5 seeds, upper endpoint of source-document-clustered two-sided 95% CI below -0.01, and no material calibration/rare-target regression. If it fails, label NEGATIVE even if another secondary metric improves. Source documents are correlated chapters, so bootstrap across eight *documents* is descriptive, not external-site generalization.

Secondary registered comparisons: new SSM, source-only count cache order3/capacity512/tau=.05, exact lambda=.6, trained query (fixed original five query checkpoints, topK8/temperature.25, lambda=.1 on exact misses only). **Same cache probabilities per observed prefix** for all three neural backbones; per-document NLL, Brier, top1 accuracy, number of cache exact/miss decisions, source-by-model differences. Report all per-seed results and any failures. Run one evaluation pass only, freeze output JSON SHA; no result-dependent architecture changes on this corpus; after reading, mark this corpus EXPOSED.

Independent architecture-superiority is still DISALLOWED by mismatched end-to-end training compute, incomplete original Git/LFS cold replay and no independent access authorization receipt. A first-open metric doesn't override those gates.

## Prior model and policy evidence
GRAPH013 original five TAC SHA receipts; GRAPH014 SSM five seed source archives; GRAPH015 new delta five source files/ledger, original used-source digests. Source-only replay script must fail on manifest drift. No fabricated third-party external validation. New CMake source is a test of cross-domain next-byte prediction only, not long-horizon memory, language reasoning, production or actual business value.
