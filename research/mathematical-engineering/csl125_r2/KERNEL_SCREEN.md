# CSL-125-R2: bounded cumprod scan screen (NOT ADOPTED)

**Stage:** local CPU/FP32 no-fit engineering, 2026-11-05. Existing `CSL125R2.forward` continues to use the source-pinned stable associative `selective_affine_scan`. The experimental alternative in `csl125_r2_fast_scan.py` is explicitly **not** installed in the model.

## Derivation and bounded domain

The recurrent law is `S_t = beta_t*S_(t-1)+eta_t*z_t`, where `beta_t=exp(-rate*g_t)`, `eta_t=1-beta_t`. For within-chunk prefix `P_t=∏_{i<=t}beta_i`, exact-real algebra gives

`S_t=P_t*(S_before + Σ_{i<=t}(eta_i*z_i/P_i))`.

The source restricts `0<rate<=0.5`, `0<=gate<=1`, and chunk length `<=64`; therefore `P_t>=exp(-32)>1e-14`, strictly above FP32 normal underflow. This is a **bound on the denominator**, not a guarantee of small floating-point cancellation error, gradient parity for every input, or stable CUDA mixed precision. Arbitrary rates beyond the bound are rejected. The independent associative-prefix algorithm (no inverse powers) remains the trusted control.

## Observed tests (NO FIT)

- 49/49 new candidate tests passed across 12 random seeds × chunk lengths 1,7,32,64, plus an explicit denominator/domain test. Compare against independent float64 NumPy serial and stable associative scan, including input gradients, signed alternating values, zero write gates and long timescales. These are engineering probes, not mathematical worst-case proofs.
- Actual full **123,822,574p** R2 model, identical random weights, B1/T64 and all exits in a 20-layer inference call, with vocabulary projection disabled. A controlled, temporary module-level operator substitution for profiling (restored afterward) gives max logits difference **4.7684e-7** and max persistent-state difference **7.4506e-8**.
- One-thread local CPU, four repeated prefill calls after warmup: reference median **422.6015 ms** vs candidate **353.5540 ms** (approximately **16.3% lower** in this narrow CPU run). Not a replicated hardware-general speedup; the earlier CPU benchmark compares a different workload/methodology and should not be merged into this result.
- Separate CPU operator profile reveals more multiplication/concatenation operations in the associative-prefix R2 operator than in R1. This motivated the candidate, not an assumption that all speed issues are solved.

## Promotion BLOCKERS

Before replacing the model's source-defined operator: test exact same-weights forward/backward across full parameter tensors on source-qualified inputs, stress numerical errors near cancellation/near-zero occupancy, compile and benchmark genuine two-T4 CUDA for masked/variable chunks and mixed precision, run memory/RSS as well as latency and FLOPs, independently review Torch graph changes, and preserve original reference/source/checkpoint lineage. **No optimizer training, no model winner, no physical device/energy claim, no quantitative CUDA speedup in this package.**
