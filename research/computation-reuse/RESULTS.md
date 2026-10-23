# Execution results: ENG_REUSE_001

**No new model training or predictive-intelligence gain.** Same five frozen ME080 models, each with 99,816 parameters. The optimized execution path adds zero learned parameters and leaves all original checkpoints and scores unchanged.

## Measured CPU workflow improvements

| Workflow | Median paired speedup across five fits | Range across fits |
| --- | ---: | ---: |
| All-prefix forward | 1.457× | 1.437–1.464× |
| All-prefix forward + backward | 1.369× | 1.359–1.390× |
| Final-prefix inference | 1.845× | 1.836–1.860× |
| Three quantiles together | 1.958× | 1.933–1.975× |

Single-thread CPU, batch 24, eight observed frames, five warmup rounds and 31 randomized paired rounds per workflow per original model. Raw paired timings, IQRs and per-model bootstrap 95% intervals are preserved in [ORIGINAL_EXECUTION_RESULTS.json](ORIGINAL_EXECUTION_RESULTS.json). Intervals condition on this host, fixed batch and original model; they are not independent-hardware evidence. The timing batch repeats one eligible original FIT transition 24 times, so this is a fixed-shape execution measurement, not workload-wide latency or throughput qualification. No CPU affinity, GPU, memory-bandwidth or energy claim.

## Mathematical and numerical preservation

All five frozen models were checked on all 960 original exposed real query windows each, across the five source tasks. All-prefix distributions, final-only distributions and joint quantile inversion matched the reference **bit for bit on the original local host**. A separate end-to-end check compares final optimized quantiles as well as raw distribution parameters. The engineering acceptance contract remains rtol 1e-5 / atol 1e-6; bit-exactness is a measured local result, not a cross-hardware promise.

Float64 software reference tests compare the whole-model loss and every parameter gradient. Cache tests cover entity interleaving, exact retries without encoder execution, conflicting retries, gaps, bounded capacity, defensive output copies, and invalidation after standard parameter/buffer/mode changes. No optimizer exists in this study; forward/backward timing accumulates gradients but changes no model values.

The external workflow verifies optimized/reference agreement on its own host. This is distinct from replaying historical ME080 outputs across CPUs: ME080 still has its documented three lower-quantile and one interval-score exceptions. Neither the optimized path nor its new tests can retroactively convert that original strict FAIL to PASS.

## Architecture and runtime boundary

The [mathematical specification](MATHEMATICAL_ARCHITECTURE.md) defines the exact native recurrence, shared tensor DAG, probability model, effect handler and complexity. Decoder batching preserves the nonlinear recurrence. Last-only execution omits unrequested decoder work. Joint quantiles share CDF constants and Python loop iterations. The bounded, versioned cache reuses state and exact latest-event results without hidden model-global state.

No arbitrary historical correction, concurrent cache service, learned long-memory advantage, better model accuracy, new action intelligence, compiler/GPU kernel fusion or customer value is established. The next model-capability improvement still needs its own prospective training and authentic outcome evaluation. ME080 remains MIXED_TRADEOFF with its full gate failed; this report does not supersede those findings.

## Reproduce

```sh
PYTHONPATH=src:. python -m pytest tests/test_reusable_execution.py -q
PYTHONPATH=src:. python scripts/verify_reusable_execution.py
```

Original timing collection is deliberately one-shot and refuses to overwrite the preserved benchmark artifact. For a new host/timing study, create a new dated registration/output rather than silently replacing original timings.
