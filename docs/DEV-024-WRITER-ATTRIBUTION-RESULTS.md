# DEV-024 audited results: writer learning helps short training; long-regime recovery fails

**Adaptive development; not confirmatory. Draft, unmerged.** All 60 cells, exact separate-process replays and independent audit passed. Source `c2b3e239fdddaac1ac96ad95ebc1dcc3672c833f`.

## Finding

Writer training contributes under the short recipe: both registered writer contrasts pass, and joint passes prediction, swap and selected physical gates. Under the long recipe, neither contrast passes and joint fails reliability. Thus DEV-024 does **not** establish writer attribution across both regimes. The previously successful DEV-023 long-regime result does not reproduce under these fresh initialization/training/evaluation namespaces; do not treat its five-seed pass as robust across namespaces. Schema41 references pass both-regime development gates here.

| Arm | H512 short | H512 long | H1024 short | H1024 long | Combined selected gate short / long |
|---|---:|---:|---:|---:|---|
| joint | 99.17% | 74.58% | 90.00% | 68.12% | True / False |
| frozen_writer | 70.21% | 56.25% | 72.29% | 58.75% | False / False |
| frozen_encoder | 94.17% | 63.75% | 90.00% | 63.54% | False / False |
| both_frozen | 65.42% | 56.46% | 68.12% | 56.46% | False / False |
| schema41 | 94.79% | 98.75% | 88.33% | 95.42% | True / True |
| schema41_frozen | 90.21% | 94.58% | 81.88% | 90.62% | True / True |

## Registered writer contrasts

Gains and exploratory paired five-seed 95% t(df4) intervals in percentage points. Required gain >=5pp and lower bound >0. Secondary encoder effects and interactions remain in the JSON; none can replace these contrasts.

| Regime | Contrast | Gain | Interval | Gain gate |
|---|---|---:|---|---|
| short | joint_minus_frozen_writer | 28.96pp | [5.94, 51.98] | True |
| short | frozen_encoder_minus_both_frozen | 28.75pp | [11.85, 45.65] | True |
| long | joint_minus_frozen_writer | 18.33pp | [-9.28, 45.95] | False |
| long | frozen_encoder_minus_both_frozen | 7.29pp | [-2.56, 17.14] | False |

## Per-seed primary reliability

Seed order 43,59,71,89,107; new numerical namespaces differ from DEV-023.

| Regime | Arm | Primary accuracies |
|---|---|---|
| short | joint | 100.00%, 98.96%, 100.00%, 98.96%, 97.92% |
| short | frozen_writer | 88.54%, 50.00%, 89.58%, 52.08%, 70.83% |
| short | frozen_encoder | 100.00%, 72.92%, 100.00%, 97.92%, 100.00% |
| short | both_frozen | 85.42%, 50.00%, 62.50%, 50.00%, 79.17% |
| short | schema41 | 100.00%, 92.71%, 89.58%, 92.71%, 98.96% |
| short | schema41_frozen | 93.75%, 87.50%, 85.42%, 87.50%, 96.88% |
| long | joint | 95.83%, 62.50%, 51.04%, 94.79%, 68.75% |
| long | frozen_writer | 57.29%, 65.62%, 52.08%, 50.00%, 56.25% |
| long | frozen_encoder | 71.88%, 50.00%, 61.46%, 79.17%, 56.25% |
| long | both_frozen | 52.08%, 51.04%, 58.33%, 69.79%, 51.04% |
| long | schema41 | 93.75%, 100.00%, 100.00%, 100.00%, 100.00% |
| long | schema41_frozen | 81.25%, 100.00%, 100.00%, 100.00%, 91.67% |

## Physical cost and failures

Means over five seeds below. The gates apply to **every seed**, so a favorable mean cannot hide an unreliable run. Thresholds remain failure<=.05, unnecessary<=.10, cost<=.80 and mean selected cost<=mean model-only cost, with no tuned tolerance.

| Regime | Arm | Selected cost/entity | Model-only cost/entity | Selected failures | Unnecessary service | Selected gate |
|---|---|---:|---:|---:|---:|---|
| short | joint | 0.532778 | 0.533247 | 0.008160 | 0.004514 | True |
| short | frozen_writer | 0.651146 | 0.687587 | 0.017882 | 0.082378 | False |
| short | frozen_encoder | 0.569523 | 0.583854 | 0.006424 | 0.037413 | False |
| short | both_frozen | 0.925174 | 0.982552 | 0.034722 | 0.308767 | False |
| short | schema41 | 0.526476 | 0.526476 | 0.003819 | 0.011719 | True |
| short | schema41_frozen | 0.524913 | 0.524913 | 0.002517 | 0.014063 | True |
| long | joint | 0.603490 | 0.613194 | 0.000087 | 0.095226 | False |
| long | frozen_writer | 0.908811 | 0.943316 | 0.009549 | 0.370660 | False |
| long | frozen_encoder | 0.911797 | 0.939063 | 0.014323 | 0.360069 | False |
| long | both_frozen | 0.960773 | 0.972917 | 0.004340 | 0.441667 | False |
| long | schema41 | 0.531736 | 0.531944 | 0.007812 | 0.004948 | True |
| long | schema41_frozen | 0.531328 | 0.531510 | 0.006944 | 0.007118 | True |

Long joint seed71 achieved51.04% primary accuracy, swap drop2.08pp, unnecessary-service rate.3746 and selected cost.8892. Seeds59 and107 also fell below80% accuracy. These failures are retained; low mean physical failure alone does not rescue a policy that performs unnecessary maintenance.

## Evidence and limits

- [Successful62-job run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38890378185), 2026-10-16 10:55:12–11:07:54UTC (12m42s wall time).
- 1,100 regression tests passed; six-arm smoke, freeze-gradient checks, source provenance and integrated memory CI passed.
- Complete artifact `12042216839`, 39,748,679bytes; ZIP SHA256 `b38cd15c89eeb86845c59a4399202d405f26ea942c7184881df0d84fc8caab1a`.
- Local archive hash, all 60 checkpoint/replay/source bindings, both schema-reference initial-state matching and raw metrics verified. Aggregate JSON reproduced byte-for-byte.
- [Audited summary](DEV-024-AUDITED-SUMMARY.json); [frozen protocol](DEV-024-WRITER-ATTRIBUTION-PROTOCOL.md).

Four TAC arms share identical initial tensors and the masked-local computation; active parameter counts are joint8674, frozen-writer5378, frozen-encoder7734 and both-frozen4438. Frozen means random initialization. This is a learning intervention under a fixed optimizer recipe, not optimization-independent mechanistic proof. Public-slot addressing and local aging remain bundled.

Only five seeds, adaptive synthetic fixed-eight-entity development. Training regimes match observed events but differ fourfold in episodes/labels/receipts. Initialization, training cases and evaluation cases all changed together relative to DEV-023; this result alone cannot locate the source of the long-regime discrepancy. Frozen-checkpoint cross-evaluation is the next cheap diagnostic. Native TAC overall capability and original attribution gates remain failed.
