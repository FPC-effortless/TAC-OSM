# CSL-125-R1 — mathematical-engineering correction and research continuation

**Stage:** `ENGINEERING_ONLY_NO_FIT` · **Source family:** Woyengi Compositional State Ladder · **Date:** 2026-11-05 · **Ancestry:** CSL-125 original draft PR #711, source SHA-256 `0bf0ba779cf570084179675dd83a8b2a423867275282a129359a5f5a5828e85b` · **New variant:** CSL-125-R1, original candidate retained and never retroactively relabeled.

## Frozen objective and evidence

One shared 20-layer model; registered exits 2/8/20, width 768, FFN 1408, 8 role addresses, state width 256, 4 local GQA layers (12 queries / 3 KV), 32,768-token embedding tied to vocabulary output and separate tool/field/acceptance logits. Exactly **123,699,534** unique learned parameters for **each variant**, including the embedding/head, under the owner-selected **125,000,000** ceiling. Tokenizer fitted: **NO**. Checkpoint trained: **NO**. GPU: **NOT RUN**. Original local CPU engineering evidence remains distinct.

### Defects and falsifiers discovered in CSL-125

1. **Role collapse at initialization:** `roles ~ N(0,1/√E)` and scores `(Q·role)/√E` together give approximately `O(1/√E)` score standard deviation. A seeded 8-role construction reports legacy mean routing entropy **2.078800** versus maximal uniform **ln 8 ≈2.079442**, with average largest role probability **0.132008**. This is nearly uniform and may impair key-specific learning. The corrected initialization uses `roles ~ N(0,1)`, yielding initialization entropy **1.955903**, maximum **0.245394** under the same sampled input and seed. This tests initialization only; learned selectivity remains unmeasured.
2. **Steady-state amplitude attenuation:** with uniform `a_j=1/J`, constant `v`, and the original `N_j ← αN_j + (1-α)a_j v`, readout `Σ_j a_j N_j` tends to `v/J`. With J=8, the memory contribution is attenuated eightfold before downstream projection. R1 tracks a matching positive write mass and reads the normalized conditional mean. This eliminates the algebraic attenuation but does not itself learn a key-binding policy.
3. **Short direct-memory survival:** Original slow decay starts at `α=0.98` (half-life ~34.3 steps) and the scan clips all α to at most `0.995` (half-life ~138.3 steps). Old-state coefficients at 1024 and 4096 steps are ~`1.036×10^-9` and `1.153×10^-36` with the original initialization; the clipped maximum at 4096 is ~`1.212×10^-9`. R1 learns bounded log-time constants with fast initialization τ spanning 8–128 and slow 256–8192. An old numerator/mass pair survives 4096 steps with coefficient `exp(-4096/8192)≈0.607` in the slowest channel. **Retention of information under subsequent interfering writes is not guaranteed.**
4. **Training contract:** CSL-125's original optional `weights` allows nonfinite loss weights and lacks explicit valid-target masking. R1 rejects NaN/infinite/negative/zero-sum rung mass and accepts a boolean `[B,T-1]` target mask. The mask does **not** reset packed-document recurrent states; separate document-boundary controls remain necessary.
5. **Large-position rotary collision:** original `positions.float()` merges adjacent integer token positions above 2^24, corrupting relative rotary differences for very long-lived streams. R1 recomputes inverse frequencies and position phases in FP64 before casting sin/cos to the hidden dtype; a direct adversarial test reproduces the original collision and verifies stable pairwise relative scores in R1. There are no new learned parameters. FPGA/GPU speed and precision for actual multi-million-token sessions remain unmeasured.
6. **Invalid RoPE geometry:** the original config checks width divisible by Q heads but not the parity of head width required by its `_rope` implementation. R1 rejects odd head dimensions during config validation instead of failing in a forward pass.

## New mathematical operator (per active block ℓ, role j, channel e)

Let `h_t = RMSNorm(x_t)`, `v_t = tanh(W_v h_t)` and `a_t = softmax_j((W_q h_t)·R_j / sqrt(E))`, with `0≤a_tj≤1`, `Σ_j a_tj=1` and `|v_te|≤1`. Fix learned log-timescale parameter `u_j,e`, with `τ_j,e=exp(ln2+(ln32768−ln2)·sigmoid(u_j,e))`, `r=1/τ`, `α=exp(−r)` and `w=−expm1(−r)=1−α`. For each fast/slow bank, maintain an FP32 numerator `N` and FP32 positive mass `D`:

```
N_t,j,e = α_j,e * N_(t−1),j,e + w_j,e * a_t,j * v_t,e
D_t,j,e = α_j,e * D_(t−1),j,e + w_j,e * a_t,j
m_t,j,e = N_t,j,e / (D_t,j,e + ε)         (ε=1e−8)
read_t,e = Σ_j a_t,j * m_t,j,e
```

The two banks use different learned time constants and share the normalized role readout. No target, gold label, reward, future token, external tool response, or post-action state enters this predecision update.

**Boundedness invariant (exact reals):** if initial `N_0=D_0=0`, and `0≤a≤1`, `0<α<1`, `|v|≤1`, then by induction `D_t≥0` and `|N_t|≤D_t`; therefore `|m_t|≤1`, `|read_t|≤1`. This concerns the *state read*, not unbounded residual hidden activations, prediction accuracy, or learned information survival. FP32 source-to-NumPy comparisons and finite-gradient tests are implemented; a kernel-checked full-model Lean proof is **OPEN**.

**Stable chunk scan:** inside at most 64 tokens, `p_i=exp(−r·i)` and `S_i=exp(−r·(i+1))·S_in + w·p_i·cumsum(z_k / p_k, k≤i)`, with `z_k=[a_k v_k, a_k]`. Chunk carries are sequential. The algebra is exact over reals; floating-point errors are measured, not assumed zero. Using `expm1` avoids catastrophic `1-exp(−r)` cancellation at long τ. The scan retains `O(T·R·E)` arithmetic per bank, bounded per-layer carried state and **no quadratic global attention**. Four causal GQA modules continue to use a 128-token local window; at initialization, their usefulness has not been learned.

### Remaining mathematical and semantic weaknesses

- **No learned null/no-write slot and no delta overwrite:** every token assigns positive mass to every role, so irrelevant input can eventually interfere. R1 does not establish editable two-key memory. A separate write/erase/delta candidate should only proceed after the R1 learned-identity binding gate, and must compare contemporary Gated DeltaNet-type controls.
- **Dense FFNs dominate theoretical multiplication work:** per 768→1408→768 gated block, FFN weights number 3,244,032, or 64,880,640 across 20 blocks (~52.45% of model). Factorized/structured FFNs are plausible efficiency candidates, but any capacity loss must be measured. Do not silently claim an efficient kernel or no intelligence trade-off.
- **Softmax address entropy is not a binding score;** controlled identity-disjoint replacement and off-target abstention must be scored with an independent oracle.
- **Long timescales alone do not prevent interference or retain multiple unrelated values.** Time constants are shared learned parameters, not token-conditioned erase signals.
- **Stored state has strict stream ownership.** Per-entity batch separation, state reset/replay on depth changes and document boundaries are mandatory. Model-local token masks do not reset `N,D`.
- **Calibration and structured decoding OPEN:** `accept_logit` is untrained/untrusted; a grammar and independent tool authorization are external until explicitly implemented, fitted and evaluated.
- **Formal proof incomplete:** full Python/PyTorch forward/backward source equivalence, finite-precision error envelopes, Lean kernel build/axiom audit and synchronized canonical PNDS registry remain OPEN.

## Actual engineering tests and negative performance result

`PYTHONPATH=src OMP_NUM_THREADS=1 pytest -q tests/test_csl001.py tests/test_csl001_math.py tests/test_csl125.py tests/test_csl125_math.py tests/test_csl125_r1.py`: **49/49 PASS** locally, on Torch 2.10.0 CPU. New R1 tests cover independent NumPy float64 serial recurrence, 143-token chunk crossings, normalized boundedness and initial amplitude, causal poisoning, exact depth-state ownership, meta 123.7M count, 4096-token *oracle* routing retention, finite-difference gradient approximations and unmodified source controls. New generic `ResearchProbeAdapter` requires repository full-stack CI: **NOT YET EXECUTED locally**. Actual full 123.7M, T24 three-depth streaming vs chunked and future-target poison checks also PASSED; maximum observed logit discrepancy 1.8311×10^-4, maximum stored state discrepancy 7.1526×10^-7 (report `FULL_STREAMING.json`). A small CPU BF16-autocast forward/backward fixture also produced finite gradients, but no CUDA AMP, FP16 GradScaler or optimizer step was tested.

Fresh actual full-width 123,699,534-parameter forward+backward (B1,T8, synthetic next-token and head labels, 0 optimizer steps): **328/328 tensors finite, nonzero gradients**. Full vocab, all 3 exits. Source, device, result JSON and local wall time in `FULL_ENGINEERING.json`.

**Reference CPU, one thread, batch1, 64-token full-vocabulary prefill, 8-token streaming decode, 3 samples per cell; random weights, no trained quality:**

| Depth | CSL-125 control prefill tok/s | CSL-125-R1 prefill tok/s | Control decode tok/s | R1 decode tok/s | Control carried state | R1 carried state |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 766.3 | 747.8 | 80.5 | 73.8 | 45,056 B | 77,824 B |
| 8 | 409.9 | 346.0 | 34.2 | 35.8 | 401,408 B | 532,480 B |
| 20 | 206.5 | 166.8 | 19.0 | 17.3 | 892,928 B | 1,220,608 B |

**Negative result:** R1 is slower on prefill at every depth and slower on decode at 2 and 20 layers; the slight apparent R1 decode improvement at depth 8 is not a robust device-wide result. R1 uses more persistent state at all three depths in these CPU reference runs. It is *not* a proven efficiency improvement. Model weights dominate full model residency; these figures count only carried model state after the benchmark stream, not process RSS or quantized file sizes. Performance is unoptimized Torch, not a fair GPU/MCU kernel benchmark. A fused recurrence kernel and reduced data movement should be evaluated only after matching numerical tolerance and capability quality.

## Prospective research and development ladder (no fit authorized by this document)

1. **G0, source/math readiness:** original hashes and negative findings, author+independent code review, full-source and PNDS/Lean correspondence, cross-entity/causal/gradient/precision tests; maintain failures and opening dates. This draft provides partial source checks, not a completed PNDS gate.
2. **G1, identity binding falsifier:** TRAIN-only fixed two-key, distinct-key values, replacement, untouched and absent queries, address-neutral/null cases. Register oracle address and exact symbolic store positive controls, shuffled address/label negatives, full-history attention and credible delta/SSM rivals. R11's recent 8-fit negative result (0/384 complete histories for every tested arm) must remain in the control history; changing the model doesn't erase it.
3. **G2, authentic learning:** only source-audited FIT data and a FIT-fitted/roundtrippable 32,768-piece byte-fallback tokenizer; family/time/source-disjoint DEV; no repeated TEST selection. First enable preregistered v2 **1M/10M capacity pilots**, with real Transformer/delta controls, exact parameter-count parity ≤1%, identical token exposures/optimizer/tuning budget and ≥3 seeds. Measure all-rung NLL, identity binding, calibration and time-to-quality; avoid general superiority claims from tiny pilots.
4. **G3, 123.7M training preflight:** profile actual FP32/AMP optimizer steps on two physically observed T4s using approved provider budget and source; verify gradient scaling, memory, restored optimizer/RNG state and no silent unused accelerator. Do not submit until `FAIR_COMPARISON.json`, permissions, independently signed evaluator/source/ACL and reward evidence pass. The full 123.7M+comparators 20h compute requirement is **not automatically qualified**.
5. **G4, hardware deployment:** measure real CPU/GPU device latency, long prompt/decode, state buffers, logits head cost, model and tokenizer bytes, INT8/4/2 quantization quality, peak memory/RSS and joules per useful correct output. Verify grammar-constrained output and calibrated abstention. Edge/microcontroller suitability is unproven.

**Decision rule:** keep R1 only if independent source-trained capability gains offset measured state/throughput penalties. Otherwise revert to the original CSL-125 control or test a cheaper normalized-storage/fused recurrence architecture. No original runs/checkpoints/negative findings were changed.

**Prior art:** Gated DeltaNet (Yang et al., ICLR 2025) and Gated DeltaNet-2 (Hatamizadeh et al., 2026) already investigate targeted associative updates and gate separation; originality is not established by combining role slots, decay and local GQA. https://research.nvidia.com/publication/2025-04_gated-delta-networks-improving-mamba2-delta-rule and https://research.nvidia.com/index.php/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention . The Needle 3 intelligence-ladder precedent does not establish CSL's learned early-exit quality.
