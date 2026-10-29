# LANGDECODE065 — three clean GitHub CPU repeats: PASS within defined scope

**Scope:** No-fit replication of the source-locked [LANGDECODE064 cached cap position candidate](https://github.com/FPC-effortless/TAC-OSM/pull/603), compared directly against the unchanged validated original `BatchedValidationTAC057`. No architecture mixing with negative LANGDECODE062/063; no training, optimizer step, GPU/Kaggle use, protected TEST or release/promotion.

- Original trained TAC checkpoint: 499623 parameters, seed1701 FIT320, immutable original SHA/source/optimizer/RNG receipts, original licensed CPython3.13 publisher FIT owners at 128B/512B, same 16 original greedy bytes.
- Original first successful checkpoint run: [#40866246875](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40866246875), 48 focused tests passed, median speed reductions 8.8428% / 7.7146% and p95 ratios .95945 / .99414. Only the **first** internally measured one-host acceptance.
- Independently preregistered clean-host follow-on: [three-job CPU run #40866782223](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40866782223), [aggregate complete JSON artifact #12894461050](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40866782223/artifacts/12894461050). Each job ran the **unchanged** `scripts/langdecode064_benchmark.py` with 2 warmup/7 alternating measured paired repeats. 48 source/guard tests passed in each of the 3 jobs. Jobs used the same hosted CPU *class* but different clean workflow runner jobs; not independently established physical host or mobile hardware diversity.

## Frozen acceptance outcome: REPLICATION_PASS_ON_GITHUB_CPU_ONLY

| Clean job | 128B median reduction | 128B p95 candidate/original | 512B median reduction | 512B p95 candidate/original |
|---|---:|---:|---:|---:|
| Repeat1 | 6.3365% | 0.8809 | 6.4015% | 0.9384 |
| Repeat2 | 6.7756% | 0.9261 | 6.3604% | 0.9251 |
| Repeat3 | 7.0184% | 0.9208 | 6.4436% | 0.9257 |

**Every** job independently passed >=5% median decode reduction and <=1.10 p95 ratio at BOTH prompt lengths. All 3 preserve original-vs-candidate **bitwise incremental logits and all nine persistent state tensors** (FP32 3904 bytes/owner). Original full growing-prefix max absolute discrepancies: 128B 6.67572021484375e-6 and 512B 5.7220458984375e-6 in all three original-source jobs, within preregistered 3e-5 absolute bound. Snapshot continuation and source/checkpoint receipt guards passed; no input/owner/clock/mask modification. Aggregator found exactly 3 correct result artifacts and emitted `REPLICATION_PASS_ON_GITHUB_CPU_ONLY`.

## Scientific and mathematical boundaries — still OPEN

This is a reproducible, original-trained **CPU decoder engineering** result on previously exposed authentic FIT source, not proof of better language-model NLL, accuracy, unseen-owner generalization, architecture superiority, GPU latency, mobile energy, physical devices or amortized memory cost. A cached engine-local 7D FP32 position vector (~28B) is additional nonpersistent process memory; persistent *per-stream* 3904B state remains unchanged. No new weights/parameters, no checkpoint mutation, no training or TEST.

Reciprocal [PNDS draft #70](https://github.com/FPC-effortless/pnds-lean/pull/70) [Lean CI #40866514418](https://github.com/FPC-effortless/pnds-lean/actions/runs/40866514418) passed a scoped exact-integer cap theorem plus executable Python reference. **Exact source IEEE FP32/FP64, AD, positional source-to-capped-time theorem, full model graph and append-only cross-repo variant registry remain OPEN.** Actual Android/mobile hardware and other CPU architectures also unmeasured. Promotion and merge **NOT AUTHORIZED** by CPU repeatability alone. Draft candidate should remain experimental, with preserved original failed gates for 062/063.

Next: PNDS/PyTorch cross-source formal/bitwise clock-bounds and actual mobile inference latency/energy; also benchmark next character/token decode against matched GRU/SSM/Transformer under a separately registered equal-parameter original data/conditions protocol, not assume an architecture win from this runtime speedup.
