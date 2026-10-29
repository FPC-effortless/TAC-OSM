# LANGPROFILE060 — original trained decoder source-line inclusive CPU trace

**Original-source CPU-only run PASS**: [GitHub Actions #40862001664](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40862001664) on [draft PR #599](https://github.com/FPC-effortless/TAC-OSM/pull/599). Original CPython3.13 publisher FIT prefixes128/512, original 499623-parameter seed1701/FIT320 checkpoint SHA/source/optimizer/RNG checked, original full-prefix logits, all 9 carried state fields, snapshots, direct raw transition-vs-step bitwise. 3904 bytes persistent per owner, 0 GPU minutes, 0 new training, no TEST.

The previous [LANGPROFILE059](https://github.com/FPC-effortless/TAC-OSM/pull/598) PyTorch profiler stacks had **0% callsite coverage** and remain recorded as inconclusive. Unlike that method, this experiment used CPython `sys.settrace` line events for the *unchanged* `IncrementalTAC043._advance` source. 2 warmups + 24 traced repetitions on the same state/clock/token per prompt. **Values below are inclusive instrumented source-line medians, not original production latency**: they include called C kernels/Python nested functions and tracing overhead. They cannot be added/subtracted from LANGPROFILE058 ATen self CPU shares or LANGDECODE057 growing-sequence wall latencies.

| Original line | Source expression | 128B median µs | 512B median µs |
|---|---|---:|---:|
| 135 | upper.read_from_prior and upper.observe | 103.12 | 101.70 |
| 107 | embedding + positional phi transform + pos linear | 92.79 | 93.66 |
| 117 | current ordinal_code + rotation einsum + normalize | 88.61 | 89.46 |
| 118 | previous ordinal_code + rotation einsum + normalize | 79.92 | 80.78 |
| 136 | FFN up/gate/output and concept_out | 71.69 | 71.33 |
| 131 | 6×6 concept Laplacian solve | 37.68 | 38.22 |
| 140 | wave rotation/input recurrence | 29.21 | 29.31 |

These are same-state instrumented rankings only; no numerical or causal modification. Contrast with LANGPROFILE058's operator profile, which measured only ~1.2–1.3% of **instrumented ATen self CPU** in matrix factorization/solve—not a contradiction, because line131 includes Python expression overhead, data preparation and nested calls.

**Engineering target, separately controlled:** original `ordinal_code(indices,dtype)` in `context_transition.py` loops over periods (7,13,29,59), creating separate tensor remainder, dtype conversion, cosine and sine ops for both current and previous positions. A vectorized period kernel could reduce dispatch overhead; this is only a HYPOTHESIS. First benchmark exact original ordinal outputs against vectorized implementation on all frozen current/previous clock inputs without changing `_advance`. Report strict bitwise and max absolute error and independent CPU timing. Only if validated and faster should an optimized entire decoder with original checkpoint/state prefix/snapshot parity and PNDS Lean/Python source links be proposed.

Do not merge any prototype solely on trace timing. LANGDECODE057's mixed p95 failure remains OPEN; prior LANGMEMORY056 negative preserved. No claim of neural speedup, quality gain, model superiority, mobile CPU, battery or VRAM efficiency.
