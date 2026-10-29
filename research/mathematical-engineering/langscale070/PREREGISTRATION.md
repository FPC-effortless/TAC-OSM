# LANGSCALE070 — genuine language TAC-OSM FFN capacity scaling, CPU-only exploratory

## Before fitting: scientific hypothesis and failure conditions

The current native language architecture (original CONTEXT026 base + graph-free state48 upper recurrent block) may show improved original publisher-source next-byte DEV NLL as the **SwiGLU FFN hidden width** increases, at fixed original recurrent memory geometry, input context, optimizer, sample schedule and number of updates.

This is **one-axis head-capacity scaling**, not scaling of all state/concept/memory dimensions. The 48-coordinate upper recurrence, native 8x8 matrix, two aligned banks, eight complex wave modes, six persistent concept seeds, vocab256, 280D fused feature input and 128-token input context remain the same. No invented performance or claims of a universal Kaplan/Chinchilla law.

Registered untrained constructors:
| Tier | FFN width | Actual required trainables |
|---|---:|---:|
| p250k | 150 | 250133 |
| p500k | 455 | 499623 |
| p1000k | 1067 | 1000239 |

Counts MUST be checked on actual live PyTorch source **before the first optimizer step and after cold reload**. All three tiers have the same learned state operators and only a larger/smaller fully active projection/gated head. Any parameter-count mismatch means blocked fit, not rounded-in acceptance.

Historical global 100000 ±5% policy is not rewritten. **Narrow owner instruction dated 2026-10-29 for scaling** is represented by a unique special-case manifest validator for exact study LANGSCALE070, with registered contract, same study_type=single_model_training, exactly these three unpadded counts, full original source receipts. Any other study/incorrect contract/changed count continues to FAIL prefit. Previous 500K-scoped waiver LANGLAYER040 unchanged.

## Authentic data, frozen protocol and resource controls

- Dataset: 14 original PSF licensed CPython3.13.0 source FIT owners including original UTF8 supplement; original recorded SHA dd5aabf90646be409e3f18ecf763dbcefc84b6575e12ff54f4a7195f92ff150e. Four original exposed publisher DEV documents, 4096 original next-byte targets; TEST sealed.
- Same exact 129 byte windows for each seed in all three tiers, 8 source windows per AdamW step, 160 optimizer steps per arm/seed, batch8 x 128 targets, **163840 original next-byte targets per fit**. Three seeds (1701,1702,1703), 9 fits. `iter_batches(fit,seed,steps=160)` original saved sample index/source SHA checked in FIT. AdamW lr=.001, grad norm clip1, zero hyperparameter selection.
- GitHub Actions CPU ubuntu24.04, intraop2, no GPU. Each fit produces model+optimizer+PyTorch/NumPy/Python RNG/seed/source-order SHA checkpoint and **immediately cold reloads** it to independently verify exact original exposed DEV NLL, Brier, top1 and top5. Save per-job original source, exact parameters, 160-step loss curve, CPU wall and process time, RSS and source hashes.
- Metrics: original exposed DEV NLL in nats/byte (primary), top1, Brier, top5; final/initial FIT loss, CPU seconds and checkpoint weight bytes. Paired seed differences and visible variance reported, but no significance claims with 3 seeds. Same updates/source does **NOT** mean equal training FLOPs; 1M arm performs more computation. Any gain means only better next-byte performance at larger size / greater compute in this narrow pilot, not novel architecture superiority or compute-optimal scaling.

## Negative gates and prohibited interpretations

No future-token leakage, no document boundary crossover; source/weights must match exact original SHA. All active parameters must receive finite gradients; any NaN/Inf, missing checkpoint, failed cold reload, absent original source owner or failure on any arm is retained. CI success records runs but does not assert monotonic scaling. **If NLL improves not consistently across seed pairs, report mixed/non-monotone scaling, not a victory**. No comparison to historical 500K models trained on different exposure without matching conditions. Actual Android streaming ability for larger models remains unimplemented; recurrent persistent bytes are not certified for new widths. Do not infer memory/edge runtime from parameter count alone. External independent validation, protected TEST, full PNDS original graph IEEE/AD/formal verification and architecture superiority all OPEN.

The design is a deliberately small original source scaling pilot that can falsify whether more FFN parameters improve this architecture under fixed 160-step exposure. Next phase, only if scientifically justified, is matched larger data-budget/compute-aware scaling, genuine source+fresh DEV and matched GRU/SSM/Transformer, not a posthoc redefinition of these measurements.

**Compute authority:** only bounded GitHub Actions CPU, no Kaggle quota/GPU consumption, no training outside this registered manifest, no merge, checkpoint promotion or protected TEST.
