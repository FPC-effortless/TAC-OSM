# ME018 — Original real-data, weight-identical native recurrence execution results

**Result: PASS exact recurrence/gradient equivalence; positive native CPU efficiency improvement on the measured single-thread host.** Scientific world-model/policy advancement: **NONE**. No weights promoted or merged.

## Immutable provenance

- Successful [GitHub Actions run 39429721625](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39429721625), original result artifact [12266584995](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39429721625/artifacts/12266584995) named `me018-native-identical-real-fit-kernel`; source-specific original trained models from ME016 run 39412071932, original full training results JSON SHA256 `6870e254b65b74e8a55a242a3528aa478bf7bd08c6812ce92bf32a87fd4ba416`.
- Three source-hash-verified native_chrono_anchor genuine KuaiRec trained checkpoints, unchanged 135050 active trainables each and unchanged 600 historical original train updates. Original publisher ZIP MD5 `261550d472c48eff4990fb13c0e5bcf7`, SHA256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`, real original **FIT** cohort only. Registered first 128 anchors of 768 preselected evenly spaced true FIT histories. No DEV, future user1/2/3 or other independent label test opened; no new optimizer steps. All 3 original exact seed/checkpoint identities verified BEFORE authentic source parsing.
- First GitHub run 39429584854 failed a **TorchScript static-global lowering** issue (denominator `HISTORY+2`), which was fixed without changing mathematics using constant 34.0. The successful later run is the only valid performance measurement; initial failure retained as engineering evidence.

## Original pre-registered real FIT CPU measurements

| Model seed | Original validating eager forward ms/batch | Exact core eager ms/batch | Exact scripted native ms/batch | Eager/script inference speed ratio | Core eager forward+backward ms | Script forward+backward ms | Speed ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14001 | 6.1951 | 6.1421 | 3.7117 | 1.6548x | 20.9451 | 17.9838 | 1.1647x |
| 14002 | 6.2377 | 6.1579 | 3.7844 | 1.6272x | 21.2533 | 17.6210 | 1.2061x |
| 14003 | 6.2082 | 6.1906 | 3.7807 | 1.6374x | 20.3847 | 17.6708 | 1.1536x |

Paired three-seed medians of individual speed ratios: inference **1.6374x** (~38.9% lower steady-state compute wall-time) and forward+backward **1.1647x** (~14.1% lower wall-time). Every real checkpoint/real batch had **0 measured maximum logit absolute difference** and **0 maximum parameter-gradient absolute difference** compared with eager identical arithmetic on this CPU. Original PyTorch eager source-shape checks live outside the optimized trusted-input compute kernel, so the fair speed ratio compares core eager and core scripted with identical validation placement.

Original full inference and all training semantics are preserved to tested floating-point precision. JIT compilation overhead deliberately excluded; model initialization, data preprocessing, memory usage, total epoch optimization and GPU execution were **not** benchmarked. This is a software performance result, not faster abstract asymptotic complexity or causal action quality.

## Research decision

- **KEEP** TorchScript native core as a verified optional execution optimization for these exact existing weights and mathematical rules. No retraining, model promotion or replacement of canonical model in master.
- **REJECT** gratuitous recurrence redesign justified solely by slow-state gradient speculation. ME017 already shows slow state is productive on real 32-step FIT and the ME016 direct identity has failed a genuinely fresh real-user test.
- **NEXT SCIENTIFIC BOTTLENECK**: 32-step history is still the model's input-window boundary. Design a rigorously matched genuine-longer-history training experiment (32 vs 128 or longer, with native and ordinary GRU controls) but do not falsely call an architectural win if parameter/data/compute/time budgets differ. Use real history/outcome records, prospective sealed users or a genuinely new source, and no synthetic empirical success claims.
- **NEXT ENGINEERING GATE**: benchmark a full identical optimizer-step training loop with scripted/eager native and GRU using the same real batches, including compile/cold-start separately. Current no-optimizer backward timing alone is insufficient to establish fully matched training cost.

All ME001–ME017 evidence and master math plans preserved. No real-world safety/commercial deployment claims or PR merges.
