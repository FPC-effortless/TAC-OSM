# CSL-125-R3 first CPU screen — NO FIT, 2026-11-06

Evidence: [GitHub Actions run 42242399316](https://github.com/FPC-effortless/TAC-OSM/actions/runs/42242399316); source commit `60d2a7dd4642b4e9ad2e55a86eeedc36d370045e`, benchmark job `127975759347`, full raw repeated samples archived at `CPU_NOFIT_2026-11-06.json`.

Procedure: hosted GitHub CPU runner, Python 3.12, Torch FP32, one thread, batch 1, sequence length 16, full exit, 2 warmups, 7 rotated-order repetitions per arm, same random untrained weights per scale. No optimizer steps, no fitted corpus/tokenizer, no learning or external-device testing. 1M and 10M are separately constructed sizes, **not slices** of a 125M checkpoint.

| Scale | Parameters | Base median | Top-32 revision median | Full revised median | Top-32 overhead vs base | Full overhead vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent 1M | 1,040,869 | 10.1559 ms | 10.7124 ms | 10.6252 ms | +5.48% | +4.62% |
| Independent 10M | 9,748,126 | 29.2048 ms | 30.3696 ms | 31.3435 ms | +3.99% | +7.32% |

Conclusions limited to this device, batch and reference implementation:

- Top-32 revision is modestly faster than full-vocabulary revision at 10M; at 1M it is modestly slower. **Both are slower than the base model**.
- The analytical MAC ratio for the additional revision pathway does not translate into the same total-model wall-time ratio. Top-K selection, gathering embedding rows, CPU overhead, and untouched earlier model layers remain.
- No trained error correction, calibration, novel accuracy capability, full 125M hardware speed, energy, GPU gain, production kernel performance, or compute/quality Pareto improvement is demonstrated.
- Baseline output projection is still paid in every tested mode; `candidate_repair_topk` can never retrieve a target absent from the base top-K.
- The separate inherited audit's **11/84 numerical gradient failures** remain failed. Do not merge or promote any memory/kernel or training change based on this no-fit screen.

Next scientific gate: resolve source/normalized-read gradient conditioning, source-qualified data and paired trained controls; only then determine if corrected output quality offsets measured extra compute.
