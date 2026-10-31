# LANGDEPTH106: trained Tiny Shakespeare result

All six fixed-step fits and the frozen final evaluation completed successfully. At approximately one million weights, the four-layer allocation had lower TEST NLL in every paired seed.

| Mean across three seeds | Four layers | Eight layers |
| --- | ---: | ---: |
| Trainable/deployed weights | 998,024 | 999,784 |
| Primary TEST NLL, nats | 1.644815 | 1.709299 |
| Primary TEST bits/byte | 2.372966 | 2.465998 |
| Primary TEST accuracy | 51.53% | 49.96% |
| Continuous-carry TEST NLL | 1.623409 | 1.688134 |
| CPU decode, bytes/s | 206.50 | 106.11 |
| Carried tensor state, KiB | 289 | 578 |
| Training wall minutes | 12.28 | 13.36 |

Each fit used 4,096 updates, batch eight and 128-byte sequences, totaling 4,194,304 predicted TRAIN byte presentations. Three paired seeds received identical sampled offsets. Every held-out score contains exactly 55,769 targets. Final checkpoints were fixed before evaluation; TEST did not select checkpoints or settings.

Four layers use width 120 / FFN 224; eight layers use width 72 / FFN 120. The comparison changes depth and width together, and eight layers carry twice the per-layer state. The result favors four wider layers for this recipe; it does not show that deeper networks are generally worse. Continuous carry helps both, but carries local KV and all other state together, so it does not isolate associative-memory contribution.

All original checkpoint archives passed their original GitHub size/SHA-256 checks. The actual trained source passed parallel/recurrent TRAIN-only forward, state and parameter-gradient checks. The independent arithmetic audit passed 541 checks across 24 metric rows without rerunning TEST predictions. All six generated samples, their raw bytes, original loss curves, AdamW/RNG/sampler snapshots and fixed protocol are preserved. CPU timing includes the reference implementation and is not a GPU benchmark; full FLOPs and independent scientific qualification remain unestablished.

[Original run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41232595986) · [Draft PR #686](https://github.com/FPC-effortless/TAC-OSM/pull/686)
