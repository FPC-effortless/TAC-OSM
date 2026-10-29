# LANGSCALE070 — 250K / 500K / 1M native language TAC FFN-capacity scaling

**Original empirical evidence:** [TAC draft PR #609](https://github.com/FPC-effortless/TAC-OSM/pull/609); [GitHub Actions original nine fit and replay run #40871491842](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40871491842); [original verified full aggregate JSON artifact #12896077841](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40871491842/artifacts/12896077841). All nine independent CPU FIT jobs + strict 9-source checkpoint aggregate job succeeded. Companion [PNDS draft #71](https://github.com/FPC-effortless/pnds-lean/pull/71), [Lean CI #40871261163](https://github.com/FPC-effortless/pnds-lean/actions/runs/40871261163) PASS for exact FFN count arithmetic, NOT whole graph.

This is the first genuine publisher-original-source **same-architecture FFN width** scaling pilot, **not** parameter-matched architecture superiority or a universal model/data/compute scaling law. Previous trained checkpoints and 100K general parameter policy intact; only LANGSCALE070 has exact owner-authorized 3-tier exception.

## Fitted conditions and source control

- Original publisher-owned licensed CPython3.13.0 source version/hash, 14 FIT owners including verified Unicode supplement; four previously exposed DEV owners. No protected TEST.
- Same 129-byte original input/target windows selected by exact per-seed original 035 RNG sampler; 160 AdamW steps x batch8 x 128 next-byte targets = **163840 real original FIT labels per model**. Three seeds 1701,1702,1703 and all 9 combinations, no hyperparameter selection. All three widths share original CONTEXT026 first-stage, same graph-free 48D recurrent upper, same native8x8/2 aligned/wave/6 concept slots, same 256 byte output; only the SwiGLU FFN hidden width (150/455/1067) varies.
- Each original trained checkpoint includes model, optimizer, Python/NumPy/Torch RNG and original source sample-order SHA, is immediately cold reloaded, and verifies identical scored DEV NLL/top1/Brier/top5. Aggregator independently verified nine checkpoint SHA256, trainable/deployed counts, exact same original sampled labels for matched seeds, CPU receipts, full original 4096 exposed DEV labels and all 0-GPU/no-TEST bounds.

## Original exposed DEV (three-seed mean)

| Trainable / deployed | FFN width | Mean DEV NLL ↓ | DEV top-1 ↑ | DEV Brier ↓ | Mean CPU process training seconds |
|---|---:|---:|---:|---:|---:|
| **250,133** | 150 | 2.426841324 | 35.4899% | 0.773787252 | 15.60976 |
| **499,623** | 455 | 2.317014445 | 38.1348% | 0.753575506 | 21.50243 |
| **1,000,239** | 1067 | **2.239726254** | **39.7380%** | **0.736944116** | 33.70295 |

Mean NLL decreases by **0.109826879 nats/byte** at 250K->500K, another **0.077288191 nats/byte** at 500K->1M, total **0.187115070 nats/byte** (7.71% relative NLL reduction vs the 250K starting point). The trend exhibits diminishing *incremental absolute* improvements per added FFN capacity over these three registered sizes; this is NOT an estimated asymptotic scale-law exponent. Mean top1 increases from 35.49% to 39.74% (+4.25 percentage points). Mean CPU process training time increases ~2.16x from 250K to 1M; different hosted CPU jobs have environment/measurement variance and CPU time is not a fixed-FLOP comparison.

## Actual independent seed source-matched NLL pairs

| Seed | 250K | 500K | 1M |
|---|---:|---:|---:|
| 1701 | 2.413500136 | 2.313106534 | 2.256180705 |
| 1702 | 2.419363817 | 2.295354095 | 2.211759930 |
| 1703 | 2.447660019 | 2.342582706 | 2.251238126 |

**All three seeds improved** at each scale increase; none omitted. Per-seed NLL differences 250->500: -0.100394, -0.124010, -0.105077; 500->1M: -0.056926, -0.083594, -0.091345. Three seeds are not sufficient to claim statistical population certainty or generalization.

## Strict scientific limitations

1. **Only FFN width changed**. Does not show that larger native/associative/wave/concept state, longer context, or additional layers improve. First-stage and upper recurrent memory are **fixed**, and full trained-model streaming interface for 250K/1M is not qualified.
2. Equal authentic data order, 160 optimizer updates and target exposure, but ***not equal training FLOPs***; larger models have more active weights and compute. Interpretation is performance-size-cost under **fixed training examples and updates**, not matched-compute architecture superiority or compute-optimal scaling. Short 160-step training may underfit all arms.
3. DEV is the original **previously EXPOSED** publisher dataset of 4096 targets; no previously unseen independent TEST or new publisher corpus evaluation. No inference speed/energy measurement for new large models, no Android.
4. [PNDS #71](https://github.com/FPC-effortless/pnds-lean/pull/71) verifies only exact parameter arithmetic `127433+818 W` and fixed upper48 state geometry in Lean/Python, not the entire actual original PyTorch forward/gradient/IEEE/mask/source graph or full reciprocal variant manifest. Whole source math remains OPEN.
5. New trained checkpoints are experimental and unpromoted; original 500K previous checkpoints, negative decoder results, original source and protected partitions untouched. No Kaggle GPU minutes consumed.

**Next preregistered question:** perform original checkpoint *no-fit* context-length scaling (16/32/64/128/256 input bytes) with identical next-byte targets across all sizes and seeds to isolate longer context from added FFN capacity. Later consider genuinely new independent original-source DEV and matched SSM/GRU/Transformer training at each size/budget before any comparative architecture claim.
