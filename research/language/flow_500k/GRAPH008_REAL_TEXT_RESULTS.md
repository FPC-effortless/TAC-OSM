# GRAPH008 — SSM-inspired bounded source-slot memory: authentic results and failures

**28 October 2026 — exploratory, not promoted.** Exact model sources: `source_slot_memory_v4.py`, `tac_wave_ced_500k_v4.py`, `historical_500k_v4.py`, added on top of GRAPH007 without modifying #528/#529/#531. Complementary mathematics: PNDS PR #56. Research papers: [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2 SSD](https://proceedings.mlr.press/v235/dao24a.html), [Gated DeltaNet](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html). Those papers motivate structured state transitions and selective writes; this design is a straightforward indexed bank **not an assertion of new associative-memory novelty**.

### Actual new mechanism

Observed source ordinal t writes bounded learned 16D content `v_t=(.75+.25*sigmoid(g_t))*tanh(W_v*e_t)` only to slot `t mod 512`. Strict prewrite reads require `0<=q<t`, `t-q<=512`, admitted source and matching stored absolute ordinal. Sources are tagged by caller-supplied owner; mismatches are rejected, but owner strings are not authentication. The same read is attached to the primary next-char decoder through existing `concept_out` weights. No new trained weights: **499,290 base; 503,514 historical-head**. The representation expands from 520 FP32 to **8,712 FP32 + 512 int64 ordinals + owner metadata**. This is **not** a solution within the old 520-float memory budget.

Scalar/slot noninterference, boundedness, 512-window no-address-collision, past-only access and owner equality have seven scoped exact-real Lean statements in companion PNDS; full PyTorch/IEEE/AD/decoder claims remain open. Five local numerical test groups plus a 101-step wraparound test passed; exact current source committed; CI receipt must be checked separately.

### Frozen protocol, exploratory training

30 genuine Vim9.1 FIT and six previously exposed DEV manuals; 36 original source byte SHA-256 match. Five seeds 1701–1705, each initialized from original GRAPH007 corresponding base checkpoint. Paired 120-step AdamW follow-up, batch4, context256, LR0.0006, exactly matched original real FIT sample SHA. History arm optimizes next CE + .3×mean lag64/192 source reconstruction; next-only arm optimizes next CE. History adds 4224 weights (0.846% gap). Objectives/forward FLOPs differ; **not a compute-equal architecture-superiority test**.

| DEV metric (five-seed mean) | GRAPH007 historical | GRAPH008 historical | FIT-only smoothed unigram | GRAPH008 wrong-key | GRAPH008 zero-read |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lag64 historical NLL ↓ | 4.22580 | **3.18118** | 3.53225 | 4.06701 | 4.77721 |
| Lag192 historical NLL ↓ | 4.70200 | **3.23433** | 3.63324 | 4.12548 | 4.78262 |

At lag192, old source slot erasure at token128 produces historical NLL **4.78262**, equal to zero-read; correct source latency/history contributes a genuinely useful representation in each seed. Historical top1 **55.53%**, vs true exact-index byte-reference **100%** by construction. Model is only asked to reconstruct old observed content when a **correct ordinal is supplied**; it has *not* discovered a semantic content query.

**Downstream primary prediction gate FAIL:** v4 history-supervised next-char PPL **10.39261** versus v4 next-only **10.26072** on the same exposed DEV (history hurts every seed).

### Separate source-frozen, no-refit six-document evaluation

Pre-registered before scoring in [GRAPH008_UNSEEN_PRE_EVAL_MANIFEST.json](GRAPH008_UNSEEN_PRE_EVAL_MANIFEST.json): `autocmd.txt`, `builtin.txt`, `cmdline.txt`, `eval.txt`, `options.txt`, `syntax.txt`; none in original FIT/DEV. 48 fixed windows; all six SHA-256 checked; no training, tuning or threshold edits. Not the protected original sealed TEST and still the same Vim publisher domain.

| Model | Lag192 NLL ↓ | Next PPL ↓ |
| --- | ---: | ---: |
| GRAPH007 historical | 4.69746 | **13.70224** |
| GRAPH008 historical | **3.18907** | 13.82492 |
| FIT unigram prior | 3.55790 | — |

GRAPH008 wins historical source NLL against both references **5/5 frozen seeds** and loses next-token PPL **5/5**. Evaluation files are distinct, but architecture choice followed previous exposed results; this is within-domain evidence, not general real-language qualification.

### Qualification gates and exact scope

- True 512+ lag learned generalization **not evaluated**; training only 64/192 with 256-token contexts. A capacity-512 source bank is not proof of effective 512-token reconstruction.
- Query policy selection/discovery not learned; correct past ordinal is supplied. No access to future target or old target as a *new* input; observed historical source is legitimately in the buffer.
- Uncompressed indexed byte lookup supplied the same ordinal is a stronger oracle reference (perfect exact-copy). Any fair SSM, GRU, Transformer and n-gram comparison must also receive comparable indexed-store access and account for state memory and training FLOPs.
- Neither source identity strings nor state counts authenticate a real user's source authorization.
- Five original historical + five next-only newly trained checkpoints, optimizer/RNG/sampler receipts, exact model-source hashes, metrics, scored unseen documents, and no-fit local replay are preserved in the downloadable conversation archive. **Repo-backed original checkpoint bytes / independent clean clone replay remain BLOCKED** until they are committed/LFS tracked.
- Mathematical CI is separate; theorem checks do NOT establish full Torch/IEEE/AD proof. PR remains draft. Do not promote or merge on these scores.

**Scientific decision:** Specific old-source factual reconstruction improved on authentic documents by replacing low-rank interference with explicit bounded source slots. The result is a **memory-capacity tradeoff with an externally supplied address**, not a compressed or autonomously retrieving language model. The next real experiment must train a source-derived query policy and show incremental *next-token* gain versus cache-equipped rival models on newly unseen data.


### Additional post-hoc lag512 frozen diagnostic (not a new independent TEST)

After seeing the source-pinned lag192 results, the original five graph008 historical checkpoints were **not refitted**. A different 769-byte window framing of the **same six already-scored supplemental Vim manuals** was used to measure the extrapolation of the shared learned decoder to lag512. At 768 observed inputs, the known source ordinal `t-512` is supplied to the same bounded 512-slot memory.

| Five-seed mean | Lag512 NLL ↓ |
| --- | ---: |
| GRAPH008 learned historical decoder | **3.19650** |
| FIT-only character-frequency prior | 3.55445 |
| Graph008 zero-read ablation | 4.77881 |

All five individual lag512 checkpoints beat the FIT prior; mean correct-source top1 is ~52.57%. This tests exactly the **512-token bounded horizon**, not >512, autonomous key discovery, or a newly independent held-out data source. No optimizer, data augmentation or selection followed the outcome.

### State-space algebraic limitation and CPU engineering cost

The original lag-read matrix has value width16 and key width8. For **linear** source retrieval over N independent source positions, the read matrix factors as `R=M Q`, where `Q` is 8×N; hence `rank(R)≤8` (≤16 for two such heads). Thus it cannot reproduce **arbitrary higher-rank value matrices** by exact linear reads over N=192 or 512 distinct keys. This statement does **not** rule out nonlinear finite-precision coding of structured sequences. The indexed 512×16 slot state explicitly spends additional memory to eliminate these live-source address collisions.

Uncompiled PyTorch 2.10 CPU, two threads, full forward including lag64/192 history heads, 256-character contexts, 32 timed runs after warmup, frozen seed1701:

| Model | Batch1 p50 ms | Batch4 p50 ms |
|---|---:|---:|
| GRAPH007 historical | **12.092** | **37.777** |
| GRAPH008 historical | 13.371 | 41.903 |

GRAPH008 incurs ~10–11% higher measured eager CPU full-forward latency in this limited comparison, as well as the much larger state buffer. **Not a fused-kernel or single-token production benchmark.**

### CI verdict

New GRAPH008 TAC causal/gradient/source-slot numerical workflow, streaming baseline, prospective parameter-count checks and compulsory research-integrity workflow **PASS** on the current branch. PNDS #56 dedicated language theorem/axiom audit, prospective PNDS numerical-contract workflow and full Lean root build **PASS** at head `b1c6f7fd3a03886a9d292aefe779ea73a9e7c18e`. The compiled seven source-slot theorems are scoped exact-real invariants, not whole PyTorch/IEEE/AD refinements.
