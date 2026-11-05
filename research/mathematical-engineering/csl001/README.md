# CSL-001: Compositional State Ladder — source-only/no-fit candidate

**Owner:** Woyengi Intelligence Labs · **Status:** ENGINEERING PROTOTYPE, NOT QUALIFIED · **Date:** 2026-11-05

## Motivation and evidence boundary

A layer-sliceable low-power structured-output model is useful, but Needle 3's
published 9–29 MB CQ2, 2–20-layer claims and its grouped attention, convolution,
Monarch Hadamard MLP, engram storage and grammar engine are **not** implemented
or replicated here. See https://www.cactuscompute.com/blog/intelligence-ladders
and https://www.cactuscompute.com/blog/porting-needle . This code is an original
role-addressed recurrent/conv research candidate, with no Needle weight or
runtime dependency.

The WIL-PILOT003 code-source benchmark found hybrid test NLL 1.3777 vs
GQA Transformer 1.4232 and delta-only 1.3697 (3 seeds, equal tokens,
1.813M parameters); hybrid training was ~1.91x Transformer time. FLOCK148,
TAC106 and FUSION series failed reliable long-distance retrieval in the
reported Tiny Shakespeare tests. R9/R10 two-key replacement found no complete
learned-binding success. Do not claim role memory solves this; make it a
falsifiable target. Evidence context: personal joint research index,
https://chatgpt.com/space/page_cddf261bed8c819192349736f0a355bb,
plus https://github.com/FPC-effortless/veritas/issues/482 .

## Graph (per block ℓ; all dimensions typed)

Input `I[B,T]` consists of causal token IDs, 0≤ID<V. Shared `E[V,D]`:
`X0=embedding(I)`, with one residual stream `Xℓ[B,T,D]`.

1. `Hℓ=RMSNorm(Xℓ)`; local `Cℓ=DWConv_causal(Hℓ,tailℓ)` with K=3.
2. `Qℓ=Hℓ Wqℓ` and `Vℓ=tanh(Hℓ Wvℓ)`, both `[B,T,E]`.
3. `aℓ[b,t,j] = softmax_j(Qℓ[b,t]·Rℓ[j]/sqrt(E))`, J=4 role slots.
4. For each slot j and channel e, with 0<α<1:
   `Mfast_ℓ,t,j,e=αfast_ℓ,j,e Mfast_ℓ,t-1,j,e +(1-αfast_ℓ,j,e)aℓ,t,j Vℓ,t,e`.
   `Mslow` uses separate αslow, both owned per layer and stream.
5. `Rfℓ[b,t,e]=Σj aℓ[b,t,j] Mfastℓ[b,t,j,e]`; analogous `Rs`.
6. `gℓ=σ(Hℓ Wgℓ)`; `Uℓ=gℓ⊙Wrf(Rfℓ)+(1-gℓ)⊙Wrs(Rsℓ)`.
7. `Zℓ=Xℓ+γℓ(Cℓ+Uℓ)`; `Xℓ+1=Zℓ+δℓ Wdown(SiLU(G)⊙U)` where
   `[U,G]=Wup(RMSNorm(Zℓ))`. Shared final RMSNorm and tied vocabulary head
   produce token logits. Separate task/field/acceptance heads read that state.

`M_t` includes the current observation, not future observations. The only
prediction target is `I[t+1]`, so teacher forcing must not expose future
IDs. No encoder is allowed access to target, rewards or post-action fields.
The role address is soft and **not proved to bind multiple keys**.

`_bounded_scan` computes the exact real-valued EMA algebra by a prefix sum
inside each chunk of at most 64 tokens, propagating the final memory between
chunks. Formula for chunk position t (0-indexed), `p_t=α^t`:
`M_t = α p_t M_initial + (1-α)p_t Σ_(i=0..t) z_i / p_i`.
Real arithmetic is equivalent to serial EMA. FP32 numerical tolerance is
finite and measured, not bitwise equality. α is clamped to `[0.84,0.995]`
to bound inverse powers over 64 tokens. This is **not** a fused scan kernel.

## Nested depth training and execution

The 8-layer configuration has `(D,E,F,J)=(32,16,80,4)` and trained exits
`{2,4,8}`, instantiated parameter count **98,194**. The independent
20-layer geometry has `(20,8,56,4)` and exits `{2,8,20}`, count **97,814**.
The shared *output head and embedding* remain active at every rung.
For rung set R, hypothetical pre-registered language training objective:

`L = Σ_(r∈R) λ_r CE(Head(X_r)[:,0:T-1], I[:,1:T])`, λ≥0 and Σλ=1.

Separate supervised tool/field/accept objectives are implemented but have
**no training or calibration evidence**; acceptance scores must not authorize
external effects. A grammar-constrained JSON runtime is **not implemented**.
No stochastic early-exit policy is enabled; caller chooses an explicitly
registered depth. A different depth cannot reuse stale recurrent memory
without replay/reset. The same unsliced checkpoint may skip computation;
physical binary pruning/quantization is NOT implemented or qualified.

## Complexity and deployed-state accounting

At fixed J,E,K and width D, one block uses O(T(D·F + D·E + D² + J·E + K·D))
float operations up to constant factors; scan is O(T·J·E) arithmetic but has
O(T) parallel depth only across fixed bounded chunks; external chunk carry
remains sequential. The local path has no quadratic T×T attention matrix.
Internal state per *active* block: `2 J E` FP32 floats plus `(K-1)D`
activation elements. At B=1, state is 768 bytes per 8L block and 416 bytes
per 20L block in FP32. No honest end-to-end model energy, quantized-kernel,
Peak RSS or MCU suitability measurement exists. A 2.125-bit theoretical
parameter-only lower bound is **not** an exported CQ2 binary, and ignores
scales, codebooks, embedding overhead and memory-mapped runtime.

## Controls and release gates

This package contains **no optimizer fit**, real dataset, sealed TEST access,
production tool execution or paid GPU request. Reproducing the CPU receipt:

```
PYTHONPATH=src python -m pytest -q tests/test_csl001.py tests/test_csl001_math.py
python scripts/bench_csl001.py --variant eight --repeats 15 --seq 128 --decode-steps 24
python scripts/bench_csl001.py --variant twenty --repeats 15 --seq 128 --decode-steps 24
```

Evidence: `LOCAL_EIGHT_NO_FIT.json`, `LOCAL_TWENTY_NO_FIT.json`, and the tests.
CPU timing is specific to the recorded host, one thread, batch 1, PyTorch
reference implementation. Includes Python overhead and is not comparable
to Needle's C/CQ2 or a GPU-optimized Transformer.

**Outstanding mandatory gates before fit/public superiority:** original
source/rights/authentic FIT/DEV, preregistered 100k±5% individual models,
≤1% matched parameter gap across comparator arms, exact token order and
optimizer/compute/selection parity, positive controls, ≥3 seeds with paired
uncertainty, frozen checkpoint + state/optimizer/RNG replay, real task outcome
verifier independence and anti-leakage, independent review, protected TEST,
research/phone_queue plan and two-T4 proof for an approved Kaggle run. Scientific
TEACHER/DEV reward authority is not delegated to this implementation.

**PNDS:** `math_reference.py` and `CSL001.lean` cover only named operators;
Lean compilation/axiom audit and full source-equivalent gradient/mask/state/
precision proof remain OPEN. Link exact source commit/reciprocal PNDS registry
before mathematical release. `ResearchProbeAdapter` for this new family remains
OPEN, so cross-version probe dimensions are INCONCLUSIVE until wired.

**Ablation protocol before adding complexity:** comparator A, same-size GQA;
B, same-size delta-only; C, role gating disabled; D, memory disabled;
E, 1 vs 2 timescales; F, 2/4/8 vs 2/8/20; G, confidence/abstention
calibration. Primary outcomes: held-out task success including all key/value
queries, off-topic abstention, schema validity (independent grammar validator),
time-to-loss, decode+prefill latency, peak RSS, persistent state bytes and
joules on the target chip. For key-binding, include never-seen keys, two-key
replacement, miss/abstain and causal poisoning. Register falsifiers before fit.

**Limit:** a low parameter count or fast synthetic forward pass does not prove
better intelligence, deployability or inference efficiency on microcontrollers.
