# LANGCOMPARE031 — Authentic CPU three-arm language experiment results

**Run:** [GitHub Actions #40780215655](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40780215655) — **15/15 training jobs SUCCESS**.  
**Study PR:** [#559](https://github.com/FPC-effortless/TAC-OSM/pull/559), draft stacked after the immutable 16-update pilot [#558](https://github.com/FPC-effortless/TAC-OSM/pull/558).  
**Date:** 2026-10-28. **Status:** internally measured, original publisher source, **not independently qualified**.

## Frozen protocol

Three genuine 256-original-byte next-byte models, no Mamba-3:
- Unchanged `CONTEXT026` TAC-OSM, **500,122 parameters**.
- Official pinned Pathway BDH, **501,504 parameters**.
- Causal four-layer Transformer, **496,960 parameters**.

Each trained from scratch on CPU in a separate Actions job at the registered original seeds 1701–1705; exactly **320 AdamW updates per model/seed**, batch 8, 128 original next-byte predictions per source-owner window, for **327,680 authentic FIT target presentations per seed**. Same official 13 CPython tutorial FIT files plus frozen authentic UTF-8 FIT supplement, source commit `60403a5409ff2c3f3b07dd2ca91a7a3e096839c7`. No document merges or future labels; all three models have *identical sample order and original source byte SHA256* for a given seed. Every checkpoint includes complete original weights, optimizer, Python/NumPy/Torch RNG states; saved binary SHA256 was independently checked and its cold-load predictions reproduced the original scores. **15/15 verified.**

**Evaluation:** the exact same frozen 4×8 windows = **4,096 next-byte targets per trained model**, from four **already-exposed** original DEV publisher files; NO protected TEST, no GPU usage.

## Per-seed exposed-DEV NLL, nats/byte (lower is better)

| Seed | TAC-OSM | BDH | Transformer |
|---|---:|---:|---:|
| 1701 | 2.123229 | 2.575036 | 2.556374 |
| 1702 | 2.070750 | 2.682433 | 2.570692 |
| 1703 | 2.095374 | 2.567127 | 2.579999 |
| 1704 | 2.108064 | 2.590533 | 2.548533 |
| 1705 | 2.101159 | 2.458057 | 2.544183 |
| **Five-seed mean** | **2.099715** | **2.574637** | **2.559956** |
| NLL standard deviation | 0.019252 | 0.079875 | 0.015075 |

## Additional descriptive metrics

| Model | Top-1 | Top-5 | Brier | Bits/original byte | Mean CPU process seconds |
|---|---:|---:|---:|---:|---:|
| **TAC-OSM** | **43.81%** | **74.98%** | **0.697355** | **3.029249** | 44.17 |
| BDH | 35.62% | 65.64% | 0.804242 | 3.714416 | 272.37 |
| Transformer | 30.25% | 64.41% | 0.818940 | 3.693236 | 27.11 |

TAC has lower exposed-DEV NLL than each neural comparator in **5/5 seeds** in this exact single-study execution. However, the previously frozen same-source deterministic P3 n-gram baseline's NLL **1.877306112** remains stronger than TAC's **2.099715**. These results are **not** an architecture-superiority claim: actual per-architecture training FLOPs were **not matched or verified**, DEV documents had already been exposed, comparison configurations were not jointly tuned, and no independent unseen test set was opened. Training CPU process seconds are measurements, **not equal-FLOP budget evidence**.

## Original trained checkpoint SHA256 by model and seed

| Seed | TAC-OSM | BDH | Transformer |
|---|---|---|---|
| 1701 | `897435d2db494a5de10ee46a159f0050a651c916f67718165e08f894f6c84e5e` | `516e82b75792de6727940832acf24e5d3b411758d8c8efc5c68cdc24e498786d` | `ab36031debf5b68abc25dfc3c8ca362110854b49877e2b2f19e3df806e7786ae` |
| 1702 | `2d95b6d9005138b2bca474bda959318f929a24a08bc6f541dc5e48eaeaf4125f` | `c913db67e9a0dd9afce0e3b4df5dfc5895d672a62dfc9cd73ff60c756588ba4b` | `1829705335b2ee9b4bd87d6ebb6614aff64e1204e12f4982097172492ca1d8e1` |
| 1703 | `fe8e002a3a0d064b6ce21755593c6ddbad90fb2380fdbf1ae351d795d1b41a20` | `07f3e77058a3384b4d454ad4e311f680ee27ac26bbaa2fd93db554604bbbbdc9` | `dfccf0c40d6817f4f3ddab50f9f4fedd9aa105809dbb1ecd93314d43158a6d62` |
| 1704 | `b7321f781fd9e5e47bda8357e32df5a3c3522bed4bf2da20b061c248f8b4ea5c` | `a899207aca4548e48bd00745c52c803f3cdeb5bf716c08ba35734416efaffc77` | `c8498cee9bd8f5e49be50f40fe5a89cf4e58e3e2753364fbdb57cd5d0322098a` |
| 1705 | `fc1496a813e7758dc5ec1bdc06a8b3e4883ab707ac0e14a5f06626d3352f2bf6` | `306e9def8e857c5f708d7d39f1f265bf6ed9de51973ddea859d22df603338bb1` | `0145ae7203b870258b4d6d7abd6ac3865dccaec7a113a26988eeddfa35b5f042` |

Artifacts containing trained binary checkpoints and original full JSON receipts are attached to the individual 15 matrix jobs in [run #40780215655](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40780215655), with 90-day retention. This committed result record preserves their immutable cryptographic identities and measured metrics; a separate archive strategy is required for durable checkpoint binaries. No original research evidence was overwritten.

**Scientific next gate:** actual model training-FLOP accounting, fresh disjoint real publisher corpus, stronger classical and neural controls, and independent replay on the immutable 15 checkpoints before any promotion or commercial claims.
