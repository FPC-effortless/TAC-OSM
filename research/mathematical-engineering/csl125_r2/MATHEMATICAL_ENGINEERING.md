# CSL-125-R2 — selective memory admission audit (2026-11-05)

**Status:** NEW SOURCE-ONLY/NO-FIT EXPERIMENTAL VARIANT, NOT A TRAINED IMPROVEMENT.
**Owner:** Woyengi Intelligence Labs. **Ancestry:** CSL-001 → CSL-125 → CSL-125-R1 (PR #712) → CSL-125-R2 (separate draft). Original R1 source/checkpoints/results remain immutable. Current master has `declared-target-v2` policy, not a universal 100k ceiling.

## Why this revision is necessary

R1 learns a role address distribution `p_t` but writes to every role on every token (`p_t,j > 0` under finite softmax). Its normalized numerator and occupancy are therefore continually updated even for non-evidence tokens. Increasing `τ` slows forgetting but also slows legitimate overwrites. Further, if the address distribution `p_t,j` and decay are time-constant and the state starts empty, every slot's ratio converges to the same token-value weighted mean, regardless of its address magnitude. This is an exact-real failure of the *constant-address* case, not a theorem that learned varying addresses cannot work.

## Typed computation and mathematical source

For each active block ℓ, current input hidden features `h[B,T,768]` are causal; there is no gold label, future target, privileged key mask, tool result or reward input. Existing role logits yield softmax address `p[B,T,R]` and bounded write candidate `v=tanh(W_v h)[B,T,E]`. The **new** 8-role write gate is `g=σ(W_g h+b_g)[B,T,R]` where `W_g[8,768]` and `b_g[8]` are learned and all active in training. Fast/slow learned rates `r=1/τ[R,E]` remain as in R1.

For each role and channel, let `η_t = -expm1(-r*g_t)` and `β_t = exp(-r*g_t)`. Numerator/mass share their update law:

```
N_t = β_t * N_(t-1) + η_t * p_t * v_t
D_t = β_t * D_(t-1) + η_t * p_t
read_t = Σ_j p_t,j * N_t,j/(D_t,j+eps)
```

- **Gate closed** `g_t=0`: `η=0`, `β=1`; exact-real stored numerator and occupancy stay unchanged, not just their ratio.
- **Gate fully open** `g_t=1`: `η=1-exp(-r)`, `β=exp(-r)`; original R1 recurrence recovered for exactly equal parameters and prior state.
- **Boundedness** if initial `D>=0`, `|N|<=D`, `p>=0`, `|v|<=1`, `0<=g<=1`, `r>0`, the update is a convex combination with `0<=β,η<=1` and `β+η=1`, maintaining `|N|<=D`. When `eps>0`, the per-role normalized read satisfies `|N/(D+eps)|<=1`. Not a bound on the full residual stream, which can exceed this range.
- **Identifiability gap:** with only a single write from empty state, gate scaling may largely cancel in `N/D`, so language loss need not identify the gate from that event. Interference and distinguishable histories are essential in a learnability study. The sigmoid gate can approach but not attain `g=0` with ordinary finite parameters; exact closed-gate tests are mathematical-operator controls, not learned model behavior.
- **Retention gap:** gate selection slows or stops state aging but does **not** create unbounded information capacity, prove novel key association, or supply explicit identity/version/absence semantics.

The scan uses an associative monoid of affine maps `(A,B)` where composition is `(A2*A1, B2+A2*B1)`. Inclusive doubling over chunks of up to 64 tokens avoids dividing by very small prefix powers. It has **O(T log C)** arithmetic work for chunk size `C≤64` and `O(log C)` parallel depth per chunk; chunk carry across time remains sequential. This is not a fused or memory-optimal CUDA kernel and may be slower than R1. The implementation computes the two rate channels in FP32; no formal floating-point error bound has been proved.

## Parameter accounting and runtime state

- **R1 baseline:** 123,699,534 unique learned parameters.
- **R2:** 123,822,574 unique learned parameters, including all twenty `(768*8+8)=6,152` new write-gate parameters = **123,040 added** (approximately 0.0995% increase; within 1% prospective comparator budget, subject to measured compute matching).
- **Headroom under the hard 125M ceiling:** 1,177,426 parameters.
- 32,768 vocabulary embedding and tied LM projection counted once; actual BPE tokenizer has **not** been fitted and has no independent learned neural parameters. Physical tokenizer file bytes still require separate accounting.
- Persistent R2 memory bytes at B1 / T64 / 2-,8-,20-layer depth in the local CPU reference: 77,824 / 507,904 / 1,171,456, exactly equal to R1 at the same depth. This includes KV cache; transient kernel allocations and process RSS differ.

## Actually executed checks (local no-fit CPU, random untrained weights)

1. Existing CSL-001/CSL-125/R1 + fixed symbolic binding test suites and new R2 suite: **75/75 passed**, including exact closed gate, opened gate≈R1 recurrence, 137-token independent NumPy serial reference at chunks 1/7/32/64, bounded mass, randomized values, future-token poison, streaming equivalence, invalid configs, bfloat16-autocast gradients and exclusion of label/oracle parameters.
2. Actual full 123,822,574-parameter FP32 B1/T8 nested depth forward/backward: **368/368 gradients present, nonzero and finite**, including 40 newly registered gate parameter tensors, 0 optimizer updates. See `FULL_GRAPH_NO_FIT.json`.
3. Paired no-fit CPU timings (one thread, FP32, B1, T64 prefill / 8-token cached decode, three replicates). R1 prefill tokens/s at 2/8/20: **1367.3 / 392.7 / 175.6**; R2: **1574.9 / 407.2 / 153.1**. R1 decode tokens/s: **154.7 / 42.2 / 18.9**; R2: **216.4 / 45.9 / 18.6**. Performance is mixed; no efficiency winner. Reference overhead and low replicate counts preclude hardware-general conclusions.
4. Mathematical **oracle-only** irrelevance example with one 1.0 write, 512 irrelevant -1.0 writes, τ=64: always-write final normalized read -0.99999; oracle zero-gating on distractors returns +0.999999. This demonstrates controllability **only** when the gate is externally specified correctly. No model achieved this learned behavior.
5. Standalone NumPy math reference: 3/3 tests passed. Lean proof obligations are a separate candidate; no claim of a locally compiled Lean module or source-equivalent whole-model verification.

## Scientific vetoes and next registered experiment

**NO_FIT remains mandatory.** Actual training is not authorized by a mathematical construction or a v2 parameter target. The next study needs source-qualified FIT/DEV with real values and rights, two-T4 operator approval, exact registered parameter targets, matched GQA and capable delta/SSM comparators, independent signed source/outcome/reward/ACL gates, real FLOP/exposure matching, source-to-checkpoint/RNG replay and the open reciprocal PNDS source mapping. Retain protected TEST sealed.

First discriminating hypothesis: *trained, input-dependent admission `g_t,j` learns to protect previously bound values from irrelevant tokens while still learning key updates.* Falsifiers: trained-gate performance no better than R1; shuffled gates give the same performance; absent-key answers leak from position/identity; or primary controls fail their own acquisition positive gates. Primary unit: **complete multi-answer key history** (not only per-query accuracy), across identity-disjoint TRAIN/DEV and several replacement/distractor schedules. Synthetic source-safe operator fixtures are diagnostics, not independently qualified language understanding.

Training objective: nested 2/8/20 language losses from causal predictions plus independently sourced task labels only when allowed. Optimize all active heads or explicitly report disconnected heads. Monitor per-depth learning, gate activation/mass distributions, gradient health, live state interference, train/heldout losses, time-to-quality, energy when measurable, and external outcome/verification metrics. A learned gate that always remains closed is also a **failure**: it cannot admit updates. No new scientific model-fit run was launched in this update.
