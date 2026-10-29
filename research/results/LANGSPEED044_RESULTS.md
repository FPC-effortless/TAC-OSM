# LANGSPEED044 — original trained cached CPU comparison

**Measured 29 October 2026.** Draft [PR #573](https://github.com/FPC-effortless/TAC-OSM/pull/573), [successful GitHub CPU run #40820746504](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40820746504), [original evidence artifact #12873944566](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40820746504/artifacts/12873944566) containing original unpooled `RESULTS.json`. Source/weights, model/optimizer/RNG and source-order were checked before inference; no fitting, DEV/TEST access or GPU.

## Key measurement

One hosted x86 Ubuntu CPU, PyTorch 2.10 FP32 eager, 2 threads, batch 1, seed1701. Genuine original CPython FIT owner, prompts 128/512 bytes, same prompts and growing-prefix semantics, 16 greedy updates/model (outputs may differ), 2 warmups+7 measured alternated repeats. Original trained TAC 48D graph-free 499,623p and original causal 4-layer KV-Transformer 496,960p. Training FLOPs not matched; no independent architecture superiority.

| Context | Arm | Median decode ms/byte | p95 decode ms/byte | Median prefill ms | Median prefill+16 updates ms | Median engine setup ms | Retained tensor state after prefill bytes |
|---|---|---:|---:|---:|---:|---:|---:|
|128|TAC streaming cap_all|0.863420|0.936965|4.590116|18.499072|1.484427|3,904|
|128|Transformer KV cached|0.289489|0.377566|1.895233|6.628215|0.009504|393,216|
|512|TAC streaming cap_all|0.867221|0.947351|10.762575|24.809551|1.499479|3,904|
|512|Transformer KV cached|0.334687|0.449714|11.071118|16.615656|0.012418|1,572,864|

Including generated 16 bytes the Transformer KV state becomes 442,368 bytes (128 prompt) and 1,622,016 bytes (512 prompt); TAC remains 3,904 bytes. Cache byte counts exclude model weights, metadata, transient allocations and runtime memory. TAC's current 3,904 bytes include cached logits (256 FP32 values).

**Conclusions:** Transformer present implementation decode 2.98x faster at128 and 2.59x faster at512; total prefill+16-update median 2.79x faster at128 and 1.49x faster at512. TAC retains ~100.7x less tensor cache at128 and ~402.9x less at512 (not a whole-process memory or energy claim). TAC prefill at512 is slightly faster in this single run, not robust superiority. Eager PyTorch implementation/kernel differences and test-host variability matter.

## Original graph parity and evidence

Constructed causal KV adapter tests: 5 passed. Prior full-prefix verification against frozen real weights at every tested step max absolute logit errors:
- TAC128: 6.67572021484375e-6; Transformer128: 2.384185791015625e-6
- TAC512: 5.7220458984375e-6; Transformer512: 3.337860107421875e-6

All within previously registered FP32 atol3e-5/rtol1e-4. Original Transformer binary SHA256 `ab36031debf5b68abc25dfc3c8ca362110854b49877e2b2f19e3df806e7786ae`; TAC original binary SHA256 `0cbc1b89f53ea2a0fe1371a179a9ff14cc35ae84d09f76bdab1dffeac4ac86cb`. Shared exact original FIT sample order SHA256 `45985ef86e7df644a3acfeee73fe781bbd8d2cb290ac6465dfe670579e6f702f`.

## Scope and follow-up

BDH cached **NOT RUN**: no qualified source-specific trained-checkpoint incremental interface. Official Mamba-3 CPU cached **NOT RUN**: kernel/version is not equivalent and the official BF16 path is GPU-constrained. No novel retrieval capability, fresh held-out generalization, phonedevice inference or energy measurement. Lean mathematical source-level / IEEE equivalence for Transformer KV remains OPEN; TAC PNDS66 scoped proof unchanged. All original LATFUSE/GRAPH/041/042 memory negatives preserved.

Follow-up memory hypothesis is preregistered in [LANGMEMORY045 issue #574](https://github.com/FPC-effortless/TAC-OSM/issues/574) and has **no trained or evaluation result**. Do not promote this checkpoint or merge draft #573 automatically.
