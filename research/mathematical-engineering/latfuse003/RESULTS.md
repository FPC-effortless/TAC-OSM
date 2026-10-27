# LATFUSE003 — trained four-head selective delta research

**2026-10-27: ORIGINAL GENUINE SOURCE TWO-SEED FIT AND CLEAN REPLAY PASS. SCIENTIFIC MEMORY/FORECAST QUALIFICATION NOT ESTABLISHED.** This is a separate exploratory candidate; original LATFUSE002 fitted checkpoint/source untouched.

## Mathematical model and adaptation

Four *independent* value16 by key4 heads use all 4*16*4=256 original FP32 associative matrix scalars, NOT four small4x4 matrices (which would retain only64 cells). Same fast80,slow80,matrix256,16 slots x80 and16 Boolean flags:1696 FP32 recurring values +16 flags = **6800 nominal bytes/stream unchanged**. Base98775 trained params; candidate99747 (+972, below1% gap and in95k..105k band).

Source-conditioned top-2 competitive writer leaves nonselected heads unchanged exactly. Normalized4D per-head keys/queries, read-weight softmax over heads, source-real within-head observation-distinct key-decorrelation and write-load balance penalties were implemented. Temperature applies to HEAD competition only; there is no history attention softmax or KV cache. Keys/queries ALREADY were normalized in LATFUSE002; normalization alone is no remedy. Recurrent head h update:
M_next^h=alpha_h M^h + beta_h(v-alpha_h M^h k_h) k_h^T
= M^h alpha_h(I-beta_h k_h k_h^T)+beta_h v k_h^T.
That input-scheduled operator is right-affine/associative for known schedules, but no complete model optimized time-parallel scan or IEEE/AD formalization has been demonstrated.

Relevant prior work: ICLR 2025 Gated DeltaNet https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html ; Gated DeltaNet-2 https://arxiv.org/abs/2605.22791 . These are inspiration only, not evidence TAC success. No tokenizer/language softmax head is required for the original continuous air-quality observations.

Original trained model source SHA256 f20981d42da33609e87fc7acf1b56f9626f8bbec40653aaddb1c1c04340a063b. Source FIT origin zip d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.

## Actual verification receipts

1. [Genuine-source nofit preflight run 40435551572](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40435551572): 24 focused regression tests PASS; all39 candidate live parameter tensors had finite source-gradients. Bitwise original LATFUSE002 inherited initial weights, unselected head write isolation, retained recurrent shape passed. First preflight40435392218 failed an overstrict query permutation requiring bitwise FP32 equality for reductions differing by8.94e-8; reclassified to tight floating tolerance. It was NOT a training attempt.
2. [Registered FIT, two original trained checkpoints, original postfit receipts and clean checkout no-fit replay run40436214348](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40436214348): PASS. Exactly seeds310301/310302,128 genuine original UCI501 optimizer steps each, batch8, history24, six-hour actual measurement target exposed ONLY after forecast. AdamW lr0.0005 wd.001 clip1; FIT losses .8masked Gaussian NLL+.2normalized MAE+.25source-distinct selected-head key hinge+.03head-balance, fixed before training. FIT March2013-Aug2014, chronological validation Sep-Dec2014, previously-exposed 2015 DEV opened only AFTER both original FP32 checkpoints frozen; numerical protected 2016-17 rows NOT parsed. Original optimizer/RNG, all loss values, source/sampling hashes, validation+DEV arrays retained in research/replay-bundles/LATFUSE003. No extras/restarts, no checkpoint selection.
3. [Read-only original trained source memory-head diagnosis 40436676996](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40436676996): PASS. Original 2014-Jun 256 FIT source hourly observations ×12 stations, no new optimizer, true future labels, 2015 or protected numerical rows. Result in TRAINED_HEAD_DIAGNOSTIC.json with source original model SHA checked before/after.

## Honest training outcomes

Previously-exposed 2015 development:312 physical source station-hour origins;4024 originally measured channel targets. Same practical unmatched last-observation reference PM2.5 MAE33.3725, all-channel normalized MAE0.559802:

| Exposed 2015 metric | Seed 310301 | Seed 310302 | Persistence |
|---|---:|---:|---:|
| Physical PM2.5 MAE ↓ | 26.4292 | 24.2369 | 33.3725 |
| All-channel normalized MAE ↓ | 0.576912 | 0.559940 | 0.559802 |
| Normalized Gaussian NLL ↓ | 5.5855 | 6.2080 | not probabilistic |
| Pooled nominal 90% interval coverage | 88.82% | 87.62% | n/a |

PM2.5 improved relative to a practical reference by20.8%/27.4%, but BOTH all-channel aggregate MAEs failed to beat it (seed2 nearly tied); Gaussian NLL/coverage not qualified. 2014 chronological validation108 origins,1392 actual targets: PM2.5 MAE31.0106/32.7280 vs persistence33.1038; all-channel normalized MAE0.550310/0.546504 vs persistence0.475008; coverage90.23%/89.66%. **No global predictive or calibration pass.**

**Compulsory fairness boundary:** Source-FIT zero-update Torch CPU supported operator forward/backward FLOPs original LATFUSE002=58,640,568 versus candidate with routing/key auxiliary=61,714,798: **5.24% higher versus base**. Optimizer/unsupported operations excluded. Parameters close (<1%) but training compute is not. This is prospective single-model training, explicitly NOT an equal-resource architecture comparison or win. Already-exposed 2015 outcomes are not independent.

## Trained head collapse: why concatenated key separation is NOT the real solution

Source-different pairs require >=3 originally jointly measured sensor variables and original normalized RMS sensor difference>=1. On same genuine FIT scope, 6558 eligible distinct source pairs per model. Trained concatenated16D key abs cosine>.9 incidence0.046%/0.701%; encoder80 incidence0.274%/0.808%. But real writer/reader normalizes and accesses EACH 4D head, so these whole-key numbers cannot prove content-correct memory retrieval.

| Source write-head selected in genuine 2014 FIT hourly observations | Head0 | Head1 | Head2 | Head3 |
|---|---:|---:|---:|---:|
| Seed310301 | 99.97% | 70.41% | 23.24% | **6.38%** |
| Seed310302 | **1.99%** | 36.46% | 86.85% | 74.71% |

Exactly two heads selected per observation, so each row totals about200%, not100%. Seed310301 learned reader mean head weights66.26%,22.08%,6.94%,4.72%; seed310302 learned reader0.14%,9.12%,73.40%,17.33%. Heads receive highly unequal and seed-sensitive workloads: **head starvation and route collapse despite weak auxiliary load balancing**.

Real source-distinct pairs co-routed into each 4D head still collide: seed310301 head1 >.9 absolute cosine incidence9.34% of2634 eligible co-routed pairs; seed310302 head2 21.87% of3539, head3 14.65% of3317. A zero-pair or2-pair co-routed sample says nothing about head collision rate. Zeroing head0 in seed310301 locally changes normalized forecast location by0.016, zeroing head3 by0.000264. In seed310302 zeroing head2 changes0.02175, zeroing head0 by just0.0000064. These are OOD FIT-local forecast sensitivities, NOT factual retrieval accuracy or an attributed causal effect.

Thus four separate matrices do not guarantee learned independent semantic viewpoints. Forcing uniform load may waste scarce capacity when source regimes differ; forcing every historical key orthogonal is impossible in a 4D subspace after many events and may separate semantically related physical observations needlessly. Source key decorrelation is a regularizer, not memory-skill proof.

## What to try next (NEW prereg required; no silent refit)

1. Instrument actual FIT source identity/time and delayed source-verifiable information retention per head; require retrieval of recorded facts or improved genuine future prediction, not just lower pooled key cosine.
2. Probe balanced 2x(16-value x8-key) =256 original matrix cells as an alternative to 4x(16x4); larger key subspace costs no recurrent bytes, but offers fewer perspectives. Use source-aware smooth competitive gating that trains infrequently selected heads and does not silently erase old heads. Add learned edit-strength/erase distinctions only after no-fit original-source stability checks.
3. Establish full measured TRAINING-FLOPs equality alongside <1% parameters, same original real sample order/split and actual optimizer schedule across frozen candidate and monolithic comparator; at least two seeds, independent original source integrity, original checkpoint/replay. Do not count this candidate's unequal-FLOP single-model results as a fair head-to-head.
4. Separate uncertainty/NLL and rare pollutant outlier diagnosis, with genuine missingness/tails and 2014 FIT-era cohorts. Do not tune on exposed 2015 DEV or open 2016-17 protected TEST without new prospective approval.
5. Keep whole-model Lean source/AD/IEEE/probability refinement open, retain six-family native TAC architecture as a distinct research program.

**Decision:** Multi-head source-gated delta was implemented and genuinely trained. It has preserved recurrent memory and some practical PM2.5 gains, but important original concerns—useful associative recall, head diversity and probabilistic reliability—remain unresolved. Keep LATFUSE003 research draft and original checkpoints immutable; do NOT merge/promote as a qualified breakthrough.
