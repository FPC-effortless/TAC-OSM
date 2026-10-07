# TACOSM-PLM-INTEGRATED-E2E-007 — Result

## Status

**Measured; primary criterion passed.**

E2E-007 tests the capacity-selected hidden_dim=64 integrated PLM after a residual-linear persistent-state write was selected on a separate development split because the prior XOR diagnosis localized the principal error to state-content recovery.

## Primary result

| Seed | q2 accuracy |
|---:|---:|
| 0 | 0.9350 |
| 1 | 0.9325 |
| 2 | 0.9375 |
| 3 | 0.9425 |
| 4 | 0.9275 |

Mean q2 accuracy: **0.9350**

Seed-level bootstrap 95% interval: **[0.9305, 0.9395]**

Minimum seed: **0.9275**

Registered primary threshold: **0.80**

Therefore **primary_pass = true**.

## Controls

| Control | Mean q2 | Drop from normal |
|---|---:|---:|
| normal | 0.9350 | — |
| no_memory | 0.6215 | 0.3135 |
| shuffle_image | 0.8350 | 0.0990 |
| text_only | 0.6010 | 0.3340 |
| image_only | 0.6060 | 0.3290 |
| audio_only | 0.6170 | 0.3180 |

The controls materially degrade performance. Diagnostic mean memory drop is **0.3135** and image-alignment drop is **0.0990**.

Per-operator q2 performance is highly asymmetric in a different way from E2E-006: XOR is now **1.00** at every seed, OR is **1.00** at every seed, XNOR is **0.99–1.00**, while AND is the remaining weak operation at **0.71–0.77**.

## Integrity

All registered gates passed:

- classifier decision consistency: zero mismatches across all seeds;
- gradient-surface coverage: passed;
- training/evaluation semantic overlap: zero;
- all previous sealed/development/confirmatory composition sets excluded from E2E-007 training;
- evaluation generated after training;
- identical evaluation episode objects reused across controls;
- no auxiliary payload/entity supervision;
- oracle q2 accuracy: **1.0000**;
- independent result validation: passed before artifact upload.

## State-content diagnosis and selection

The development-only XOR diagnosis found that XOR errors were caused by state-content recovery rather than Boolean execution: mean queried-bit joint accuracy was **0.8320**, while XOR execution error conditional on both bits being correct was **0.0**.

A preregistered paired development comparison then evaluated three state-write interfaces on the same development episodes:

| State write | Mean q2 | Mean XOR q2 |
|---|---:|---:|
| linear | 0.8470 | 0.8320 |
| residual-linear | **0.9080** | **0.8920** |
| residual-MLP | 0.9020 | 0.8760 |

Residual-linear was selected before the E2E-007 confirmatory test. No E2E-006 held-out outcome was used for this selection.

E2E-007 therefore tests a specific development-selected repair on a new held-out composition set; it is not a comparison against the E2E-006 test set.

## Fresh held-out set

E2E-007 uses eight new cross-modal compositions:

- xor(1,6), xor(2,10)
- and(4,11), and(5,9)
- or(2,8), or(6,10)
- xnor(1,10), xnor(3,9)

These are excluded from training, along with:

- the sealed E2E-005 held-out set;
- the E2E-006 held-out set;
- the registered development set.

## What is established

The evidence supports the following bounded statement:

> A hidden_dim=64 integrated PLM with the residual-linear state-write interface selected on development data successfully solves a second fresh synthetic multimodal persistent-computation benchmark at 0.9350 mean q2 accuracy under the registered leakage and integrity gates.

The development evidence additionally supports:

> In the registered paired development comparison, residual-linear state writing improved mean q2 accuracy from 0.8470 to 0.9080 and improved mean XOR q2 accuracy from 0.8320 to 0.8920 without changing the encoder, CASM, optimizer, training budget, or evaluation episodes.

These results do **not** establish real-world multimodal understanding, semantic memory, learned semantic addressing, learned operator discovery, scaling laws, or hardware efficiency.

## Current bottleneck

The previous XOR weakness is largely removed. The next operator-level diagnostic target is **AND**, now the dominant residual failure on this fresh benchmark.

Before changing AND execution, the correct next experiment is another development-only causal diagnosis separating:

1. queried-bit recovery for AND;
2. Boolean execution conditional on correct bits;
3. operator dispatch;
4. class/base-rate effects.

No E2E-007 held-out outcomes should be used for that diagnosis or subsequent selection.

## Provenance

- Experiment: TACOSM-PLM-INTEGRATED-E2E-007
- Workflow run: 37564404866
- Commit: e3eb9675c242c74c143bff3af6319172fc62f735
- Artifact SHA-256: sha256:32cd3901489d5c6ccd7768a18a7ece1ec3c5ced74dd3002bc2d39970c3eaefc5
- Contract SHA-256: 2d3e78ca07ecfd308613293d7ab2650ddb3cc03567dac76390468a2af1eaabc5
- Benchmark SHA-256: 5dadfe67d77944c3bcacfdf44b6e09cec76a11d20995b8a7eccb737ea2be3226
