# LANGKERNEL061 — bitwise-exact vectorized ordinal positional code, kernel-only

**Original CPU run [#40862636242](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40862636242) passed**; 10 constructed source identity tests, original protocol/no-fit preflight, original 8195 real-valued clock positions for FP32 and FP64 each. [Draft PR #600](https://github.com/FPC-effortless/TAC-OSM/pull/600), stacked on original-source traceback PR #599. No new neural training, protected TEST, Kaggle, model weights, source decoder modification or performance/capability claim.

The first automatic workflow [#40862601868](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40862601868) failed collection because test imports required the research path initialized by the kernel module. The import order was corrected without touching kernel/mathematical operations; original failure is preserved. The completed second run passed 10 focused tests.

## Numerical identity
`ordinal_code(t,d)` over periods 7,13,29,59 computes `remainder(t,p).to(d)*(2π/p)`, then cosine/sine interleaved. New vector code broadcasts all 4 periods and interleaves the same values. Clock positions 0..8192 plus high-clock stress cases (total 8195 unique positions), FP32 and FP64 both **`torch.equal` bitwise identical**, max_abs_difference **0**. At clocks128/512/8192, the pair of current and previous ordinal encodings was also bitwise identical for both new implementations.

## Original kernel-only CPU timings (µs per pair, unprofiled)

| Clock | Original two calls median / p95 | Vectorized two calls median / p95 | Vectorized pair median / p95 |
|---|---:|---:|---:|
| 128 | 105.91 / 170.62 | 60.12 / 84.92 | **36.58 / 45.24** |
| 512 | 106.54 / 133.56 | 59.97 / 81.66 | **36.30 / 42.40** |
| 8192 | 107.13 / 149.64 | 122.77 / 155.40 | **35.80 / 76.10** |

Two warmups, seven times sixty-four per-arm same clock repeats on GitHub hosted CPU, PyTorch 2.10 FP32 eager batch1 two threads. **No end-to-end decoder measured here.** The vectorized-two-calls arm was variable and slower at clock8192; do not cherry-pick it. The single vectorized-pair arm was approximately 2.90× / 2.94× / 2.99× faster in median at clocks128/512/8192, but no independent-host repeatability or whole-model speedup follows.

## Next gate
Full original frozen checkpoint TAC decoder experiment must require equivalent FP32 weights/logits, all nine 3904-byte tensor fields, original growing-prefix logits/argmax, causal masks and independent streams, original-format snapshot replay, equal parameter count and no training, and matched growing-sequence median/p95 repeatability across several independent CPU runs. The numerical kernel identity is a necessary precondition, not a graph source equivalence proof. PNDS Lean/Python ordinal vectorization math record required before any new graph promotion. Original LANGDECODE057 third-run p95 failure remains OPEN. Do not merge a neural model change on kernel-only evidence.
