# GP-UNC-002: bounded-memory rank-one GP update — constructed performance experiment

**Evidence type:** deterministic constructed arithmetic and matched single-runner CPU engineering benchmarks. **Not trained on original real data, not verified production latency, and not qualified model improvement.**

## Computational change

The existing GP-UNC-001 projected-process reference recalculates a dense m×m Cholesky factorization after each observed residual (O(m³)). GP-UNC-002 instead maintains the posterior covariance `P` and weight mean `w` of the inducing-feature Gaussian linear model:

```text
v = P φ
d = ν + φᵀ v
w' = w + (v / d) (r − φᵀ w)
P' = P − vvᵀ / d
```

With fixed inducing basis of m coordinates, this gives O(m²) arithmetic per update and O(m²+md) saved float state, following an O(m³) one-time kernel factorization. This is a covariance-form filter with finite-precision cancellation risk, not a numerically guaranteed arbitrary-horizon stable or trainable sparse GP. Neither this nor the original reference trains a neural TAC-OSM model.

Files: `src/tac_osm/gp_rankone_nofit.py`, `tests/test_gp_rankone_nofit.py`, `scripts/benchmark_gp_rankone_nofit.py`. The original `gp_sparse_nofit.py` reference remains intact.

## Verification

- 80 constructed independent filter-stream configurations (4 inducing sizes × 20 deterministic seeds), each with 18 chronological observations and three evaluation queries, agreed against the original precision reference for both mean and variance at absolute tolerance `1e-8`. These are not 80 independently sourced scientific trials.
- Eight constructed unit-test methods passed, including 512-step constant retained scalar count, out-of-order and future-label rejection, off-stream independent-state isolation, duplicate inducing points and nonfinite inputs.
- New dedicated GitHub [GP no-fit workflow 40482129800](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40482129800) completed SUCCESS, covering prior 11 exact GP test methods, 72 NumPy-reference configurations, 10 projected-process test methods, and eight new rank-one methods. No model fitting or protected TEST access.
- Current PNDS [Lean workflow 40481976580](https://github.com/FPC-effortless/pnds-lean/actions/runs/40481976580) completed SUCCESS with scalar rank-one covariance algebra, conditional nonnegativity and nonincrease, Python reference and axiom audit. No whole multivariate / source-to-PyTorch theorem was proved.

## Hardware-specific constructed timing

Exactly 100 sequential constructed observations per m, three repeated trials, alternating comparison execution order on the same GitHub Actions CPU runner; posterior parity at `1e-8` was asserted before generating the result.

| m | Precision-reference median (s) | Rank-one median (s) | Observed reference/new ratio |
|---:|---:|---:|---:|
| 4 | 0.003159953 | 0.002958173 | 1.0682× |
| 8 | 0.007472653 | 0.006102176 | 1.2246× |
| 16 | 0.024082831 | 0.014917234 | 1.6144× |
| 32 | 0.097708252 | 0.041845566 | 2.3350× |

Raw generated evidence: workflow `40482129800`, artifact `gp-rankone-constructed-cpu` (artifact ID `12723725437`), source-controlled seed `20261027+m`. These are comparisons of two *Python reference algorithms* on an uncontrolled cloud runner; they are not CPU performance promises, GPU measurements, throughput for six end-to-end TAC models, or a robust multi-machine benchmark.

## Qualifications and next real-world experimental obligations

- Existing LATFUSE019 original development rainfall/SO2/wind undercoverage remains unchanged and must not be used as a fresh TEST success claim.
- The rank-one update assumes correct stationary kernel, positive noise, intact source-time legal previous outputs and Gaussian conditioning. No external publisher evidence, entity IDs, source hashes, independent signed evaluator ACLs or original-source model residuals were fed to these tests.
- Source-model instrumentation, source-time feature/target pairing, numerical gradient audits, maximum safe updates before covariance degradation, full multivariate PSD/FP proof, actual full-model deployed parameter budgets (95k–105k), matched GRU/variance/conformal rivals, legal split/CAL/DEV/TEST, blind heldout replay, and causal counterfactual identification remain OPEN.
- Transfer to the six families is a program hypothesis, not observed six-family performance. Reviewer must preserve all failed historical gates and draft status.

Source: [TAC PR #491](https://github.com/FPC-effortless/TAC-OSM/pull/491). Formalization: [PNDS PR #45](https://github.com/FPC-effortless/pnds-lean/pull/45). Scientific tracker: [Issue #492](https://github.com/FPC-effortless/TAC-OSM/issues/492).
