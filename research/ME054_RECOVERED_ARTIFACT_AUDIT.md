# ME-054 original robot checkpoints — recovered, byte-audited evidence

Audit date: 2026-10-22. **Archived experimental evidence, NOT a new experiment.**

Primary GitHub Actions training run: https://github.com/FPC-effortless/TAC-OSM/actions/runs/39781336855

Artifact ID: `12429926482`, name `me054-stable-original-robot-ten-trained-models-real-world-predictions`.
Archive listed as non-expired with reported GitHub digest `sha256:40c8c44b0f20c9d73386e0abf7bddb43d8155276ee260d63bba755defbd51e08`.
The complete GitHub Actions artifact ZIP was downloaded through the connected GitHub integration and its internal JSON and checkpoint bytes inspected without deserializing executable pickle/model objects.

## Verification actually performed
- Original run's source/model preregistration preflight and train/eval GitHub Actions jobs both concluded **success**.
- ZIP contained original `results/me054/results.json`, `FAIR_COMPARISON_FINAL.json`, ten native/GRU trained checkpoint `.pt` files, ten training logs, and original real-robot training log.
- Counted exactly ten distinct (architecture, seed) trained checkpoint records, five TAC and five GRU, seeds 52001–52005; each reports exactly 600 optimizer updates.
- Recomputed SHA256 of **all ten ZIP checkpoint byte payloads** and verified equality to the archived per-checkpoint SHA256 manifest.
- Frozen source revision `9290f5a96dc91ea58256e81d9ce2dd9621890b51`; 325 original real FIT episodes, 80 DEV, 100 TEST. Checkpoints reported frozen before any DEV/TEST scoring.
- Full end-to-end training FLOP equality **not certified**: original final governance result states `FLOP_exact_consumption_verified=false` and `eligible_for_architecture_superiority=false`. This is developmental source-backed evidence only, not a scientifically fully parity-certified victory.

## Genuine physical next-state MAE (original robot joint units)

| Evaluation | Mean native TAC across five seeds | Mean GRU across five seeds | Last-state persistence |
|---|---:|---:|---:|
| DEV | 0.08218983 | 0.09444294 | **0.06977294** |
| TEST (previously exposed) | 0.07880869 | 0.09231612 | **0.06694929** |

All five native models have lower MAE than their paired GRU on the original TEST, approximately 14.63% mean reduction relative to mean GRU. **All five native models still have higher MAE than zero-parameter persistence**, approximately 17.71% mean increase relative to persistence; no useful-forecast qualification.

The original TEST was already opened by prior ME-052 and influenced development. Its scores are **not** an untouched independent confirmation. No operator/session generalization is asserted.

## Rare onset labels, policy capability caveat
Original DEV scoring supported 1 real takeover onset among 1,083 onset-evaluable observations. Original TEST supported 2 onset positives among 1,282. Even if BCE looks favorable, these sparse positives cannot support strong intervention policy or calibrated takeover risk claims. Factual pre-takeover evaluations do not establish autonomous policy benefit.

## Distinct new work
Successor anchored-native ME-056 in PR #334 is a *different* task and architecture and has **not** been trained by this artifact. Its successor ME-057 PR #335 fixes dense-target hindsight leakage, supports prospective CPU-independent parity-v3, and separately audits a genuinely independent 228-FIT episode task. Never relabel ME-054's old checkpoints as ME-057 runs.

**Audit boundary:** SHA256 verification confirms archive integrity relative to the archived JSON. No re-execution of all model weights on original physical trajectories or independently verified complete training FLOP count was performed during this artifact read.
