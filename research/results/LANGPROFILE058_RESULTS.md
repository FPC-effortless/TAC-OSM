# LANGPROFILE058 — frozen original trained TAC decoding ATen CPU operator profile

**Original CPU-only evidence, October 29, 2026.** [Draft PR #597](https://github.com/FPC-effortless/TAC-OSM/pull/597) stacked on [LANGDECODE057 draft #596](https://github.com/FPC-effortless/TAC-OSM/pull/596). Successful [original GitHub Actions run #40859554128](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40859554128); [raw profile JSON + PRELIGHT/PLAN artifact #12890054337](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40859554128/artifacts/12890054337). The simultaneous [full regression #40859554110](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40859554110) passed **1455 tests, 3 skipped**; targeted original-checkpoint profiler and parity run PASS, CPU only. Provenance/planner checks PASS_DIAGNOSTIC_ONLY. No GPU, trained update, DEV/TEST access, model promotion or external scientific qualification.

## Research question and exact boundary

What consumes original FP32 CPU time when the unchanged trained TAC stream's `_advance` executes? Freeze all underlying model parameters, source and state. Contrast:
1. `step_with_guards`: original transition plus inherited checked input/output state validation and token/mask checks.
2. `transition_only`: the original **unchanged** `_advance` on the same fully validated state and original-generated next byte, bypassing only wrapper checks for diagnostic isolation.
3. `validate_only`: LANGDECODE057's state validation with no transition.

Original authentic PSF CPython 3.13.0 FIT owner prefixes128/512, original 499623-parameter seed1701/FIT320 SHA256 `0cbc1b89f53ea2a0fe1371a179a9ff14cc35ae84d09f76bdab1dffeac4ac86cb` with schedule/optimizer/RNG checked. Original 16-token growing-prefix/bitwise state/logit parity/snapshot checks inherited and PASS at both contexts. Guarded step vs raw transition logits and all carried state fields **bitwise equal** on tested input. Snapshot source and original weight hash unchanged. Persistent 3904 FP32 tensor bytes per stream, excluding temporary computations and model weights.

### Independent unprofiled wall-clock stage samples

GitHub hosted Ubuntu CPU, PyTorch 2.10 FP32 eager batch1, intraop threads2, interop1; each stage has 2 warmups followed by 112 separate timing calls on the same frozen state/token (7×16). Values in ms per invocation, **not** the growing-sequence decode protocol of LANGDECODE057.

| Stage | 128 median | 128 p95 | 512 median | 512 p95 |
|---|---:|---:|---:|---:|
| Guarded step | 0.800043 | 0.882462 | 0.759542 | 0.778765 |
| Raw unchanged transition | 0.539685 | 0.568268 | 0.547462 | 0.581418 |
| Batched validation only | 0.074843 | 0.081243 | 0.075038 | 0.082595 |

**Do not subtract these medians as a source-valid estimate of model-stage overhead.** They are separately sampled call distributions with Python overhead, repeated-state caching and host noise. The profiler was **disabled** for these wall samples.

### Profiled PyTorch ATen *self CPU* operator share within raw transition

Profiler with CPU activity, shape/memory recording, 12 calls on identical state, disjoint operator categories. These numbers are **shares of profiled ATen self-CPU**, not total unprofiled step wall time and **not** exact architectural-component shares.

| ATen category | 128-byte context | 512-byte context |
|---|---:|---:|
| Small matrix factorization/solve | 1.25% | 1.23% |
| Matrix projections/products | 21.50% | 19.57% |
| Tensor allocation/materialization | 12.13% | 13.56% |
| Nonlinearities/normalization | 5.50% | 6.00% |
| Other ATen ops (including elementwise, views and dtype conversion) | 59.62% | 59.64% |

Top raw-transition self-CPU operators at 128B include `aten::addmm` (845.83µs aggregated across 12 calls), `aten::mul` (544.76µs), `aten::_to_copy` (512.68µs), `aten::unsqueeze` (488.33µs), `aten::as_strided` (467.13µs), `aten::copy_` (370.63µs), `aten::linear` (306.14µs), `aten::empty_strided` (277.09µs), `aten::expand` (273.82µs) and `aten::matmul` (268.28µs). At 512B, `aten::addmm` (792.85µs), `aten::mul` (531.45µs), `aten::as_strided` (512.95µs), `aten::_to_copy` (453.31µs), `aten::unsqueeze` (434.58µs) were among the top contributors. **All times in this paragraph are instrumented cumulative self CPU, not per-byte wall latency.**

## Mathematical-engineering decision

**Falsified first intuition:** the visually expensive 6×6 concept Laplacian `torch.linalg.solve` accounts for only about **1.2–1.3%** of observed ATen self-CPU in this measured source, so an immediate replacement of that solver is not a first-priority speed intervention. No theorem about all hardware follows.

**Promising bounded next hypothesis:** per-byte small tensor views, dtype conversions, elementwise operations, memory materialization and separate dense projections may impose more runtime overhead than a single matrix solve. A later, separately preregistered candidate could reduce redundant tensor reshapes and avoid per-token constant constructors or fuse compatible affine projections only after (a) source-line/shape attribution; (b) explicit order-of-operations/FP32 error bounds and PNDS Lean/Python math; (c) original checkpoint bitwise/tolerance/greedy equivalence and snapshots, model hash, state bytes, causal masks, owner independence; (d) independently repeated growing-sequence p95/median on matched hosts. In particular, changing floating-point expression order can invalidate exact bitwise equivalence even when algebraically equivalent; avoid assumed source equivalence.

**Do not claim a speedup from this study**: it changed neither the source transition nor the neural graph. Do not claim that 60% of total real decode wall time is overhead, since the self-CPU grouping excludes Python dispatch and profiler perturbations. Source owner and checkpoint were previously exposed; no new independent evaluation, model quality improvement, mobile energy, CUDA memory or SSM/BDH/Transformer comparison. The previous LANGDECODE057 third-run 128-byte p95 regression **remains unresolved**; the improved median 057 implementation remains draft.

## Provenance and next gate

Original encrypted-independent checkpoint/optimizer/source receipt checks inherited from LANGRECOMP052/LANGDECODE057; SHA-checked original source and separate cpu-only stage benchmark script in this branch. Exact results, top operator tables, raw wall clocks, profiled operator times, profiler-memory metadata, plan and preflight are in immutable GitHub Actions artifact #12890054337. Backend accepts only original loaded checkpoint and constructed tests; new publisher TEST never touched. Historical PNDS LANGSTREAM043 [PR #66](https://github.com/FPC-effortless/pnds-lean/pull/66) remains a **partial** Lean/Python link with original source-wide IEEE/AD/gradient obligations OPEN.

Next research registration should measure which exact **source lines and tensor shapes** generate `aten::_to_copy`, `aten::unsqueeze`, `aten::as_strided` and `aten::mul`; only then choose one minimal mathematical code candidate. If no repeatable gains without violating equivalence, preserve this profile as an engineering diagnostic and do not merge a new architecture on this evidence.
