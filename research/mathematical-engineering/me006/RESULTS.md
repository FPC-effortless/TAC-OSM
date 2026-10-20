# ME-006 — Future-conditioned CDL-PHR ablation: original real-source findings

**Actual original training run:** [GitHub Actions 39380424029](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39380424029), model/source SHA `b51c2cd49d63e471862bf25b0b714cabf5f3bd76`.
**Original artifact:** `12244879044`, containing 4 source-bound PyTorch checkpoint files, `results.json` and training log. Four checkpoint SHA256 digests were independently checked against that ledger.
**Independent runtime tests:** [run 39380423938](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39380423938), PASS.
**Formal proof companion:** [pnds-lean PR #4](https://github.com/FPC-effortless/pnds-lean/pull/4), successful Mathlib/Lean CI.

Data: official MetroPT-3 real compressor observations, 190,443 genuine readouts, 12 observed physical channels, same ME004 chronological train/val split (12,412 train, 3,235 previously-explored validation), future observation horizons 12/60. No March 2–9 or later future TEST data opened by this training. All original pretrained TAC weights hash checked and frozen, 0 pretrained refits.

## Measured original validation MAE (lower better)

| Relevance mechanism | Seed 606001, base 240401 | Seed 606002, base 240402 | Mean |
|---|---:|---:|---:|
| Original pretrained TAC | 0.268731087 | 0.271425009 | 0.270078048 |
| Direct future-supervised adapter | **0.263520032** | **0.265686721** | **0.264603377** |
| Fast Hebbian associative relevance | 0.268801689 | 0.271394551 | 0.270098120 |

Difference vs original: direct adapter improves **~2.03%** on mean normalized MAE; associative differs by ~0.007% worse. Direct improves in 2/2 seeds, associative improves in only one seed by a negligible amount. Mean future Gaussian conditional gain: direct ~0.01734, associative ~0.00147 (units according to ME006 fixed Gaussian surrogate). Both are measured with actual future labels only during evaluation, not in inference.

**Compute:** direct 1.74/1.76 CPU seconds per 160-step fit; associative 4.28/4.29 seconds, ~2.44× slower. Both report 1,082 trainable parameters, but only the associative arm actually uses its fast-weight update/query pathway; nominal parameter counts do not make the compute comparison matched.

**Mechanistic note:** Learned fast update rate ~0.00268 and decay ~0.00016 per observation (weak association), with 16×16 matrix updated for the 32-step history. These values were inspected POST-hoc and are not authority for score-driven hyperparameter tuning on this validation. The ability of the associative matrix to recall entity identities or improve eventual action cost has not been established. The future-target metric fixes the target-timing problem in ME005 contemporaneous `cdl_gain`, not the underlying performance limitation.

## Scientific decision

- **PASS** authentic real training execution and software/integrity checks.
- **FAIL** registered practical utility question for the current Hebbian associative memory arm: it does not beat the simpler direct future-supervised control, and is much slower.
- **KEEP** the direct adapter checkpoint as an exploratory working forecast-improvement candidate; no untested replacement of the ME004 or DEV067 agent.
- **NO PROMOTION** from reused validation, no independent causal-action test, no claims of learned symbolic reasoning, external verifier authenticity, or general-world-model superiority.
- **Next proper test:** Frozen weights on a *new* prospectively registered genuine later period or independent compressor; verify forecasting magnitude, sign and compute against a parameter-matched GRU and original TAC. Do not touch the already-opened original ME004F March 2–9 week and do not retrain from this negative associative result without a new hypothesis/protocol.
