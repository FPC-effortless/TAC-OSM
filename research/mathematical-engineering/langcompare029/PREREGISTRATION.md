# LANGCOMPARE029 — four-arm 500K byte-language comparison (proposed, NOT TRAINED)
Date: 2026-10-28. **Draft preregistration.** Parent: TAC PR #556 / PNDS PR #62.
No original checkpoints, formal proofs, or previously registered losses may be altered.

## Question
Does the *currently untrained* CONTEXT026 (500,122 active parameters) improve the
next-original-byte log loss relative to official Mamba-3 SISO, official Pathway
BDH (Dragon Hatchling), and a strictly causal Transformer at matched scale,
source/optimizer exposure, and evaluation? This is a narrow language-prediction
test, NOT evidence of action/decision world modeling.

## Four source-faithful arms (256 byte vocabulary)
- **TAC**: `research/mathematical-engineering/context026/context_transition.py`,
  `ContextTransition026(vocab=256,ffn_width=501)`: source-verified **500,122**.
  Do not substitute the older LANGTRAIN025 WaveCED weights or modify this graph.
- **Mamba-3 SISO**: upstream `state-spaces/mamba` official
  `MambaLMHeadModel(MambaConfig(...))`, `ssm_cfg.layer="Mamba3"`,
  `d_model=80,n_layer=6,d_intermediate=128,d_state=32,headdim=32,expand=2,
  is_mimo=False,chunk_size=64,vocab_size=256,tie_embeddings=False,
  pad_vocab_size_multiple=8,rms_norm=False,fused_add_norm=False`.
  **Analytical count 501,884 only**; independently instantiate and check the
  ACTUAL count and fused-kernel compatibility on target CUDA before a fit.
  Mamba-3 MIMO requires separate registration; never replace SISO with Mamba-2
  or a homebrew recurrence and call it Mamba-3.
- **BDH**: upstream `pathwaycom/bdh` original `BDH(BDHConfig(
  n_layer=6,n_embd=48,n_head=4,mlp_internal_dim_multiplier=69,
  vocab_size=256,dropout=0.1))`. Source-derived count **501,504**.
  Repeated depth shares trainable weights. No extra learnable adapter.
- **Transformer**: four causal `torch.nn.TransformerEncoderLayer` blocks,
  `d_model=96,nhead=4,dim_feedforward=384,dropout=0.0,
  batch_first=True,norm_first=True`; learned byte embedding, deterministic
  sinusoidal position encoding, independent 256-logit readout and final LN.
  **496,960** trainable parameters, locally instantiated in PyTorch 2.10.0.
  Causal mask must apply to EVERY attention layer; reject future leakage.

Expected preliminary spread: 501,884−496,960 = 4,924 (<1% of 501,884).
Source pins: BDH official upstream at `2b0d7a45b058d4309c84a10e0768d541fe18bdc2`,
Git blob `4cfff79df97fa6faafb33bf20739a027edd8f360`;
Mamba official upstream candidate at `e9594ce1c732d97440f0332fdc43170a2294dbfa`.
Pin the exact package/dependencies and first-source CI inspection before importing compiled Mamba modules.
**These are not final prefit counts.** Abort unless deployed/trainable actual
counts of all four are in the explicitly authorized 500K band and
(max−min)/max <= 0.01, with zero dummy/frozen compensating parameters.
Old universal 100K validator does NOT implicitly permit this new study; a
narrow owner-scale amendment and complete parity manifest must land before fitting.

## Real data, sampling and measurements
- Use **only** frozen PSF CPython 3.13.0 tutorial source at
  `60403a5409ff2c3f3b07dd2ca91a7a3e096839c7`,
  byte-hashed by existing `original_source`: 13 FIT documents + separately
  registered genuine UTF8 FIT supplement, four already-exposed DEV documents.
  Do not concatenate document owners or move source boundaries. 256 symbols,
  no BPE, learned frozen external embedding, external cache, or oracle targets.
- For each model/seed, share the identical **precomputed, source-owned** train
  sample schedule (source path and offset SHA256), same random seeds
  `1701,1702,1703,1704,1705`, exactly 320 optimizer updates,
  batch=8, 128 predicted target bytes per example, identical source windows.
  Optimizer AdamW lr .001, default betas/weight decay, global clip 1;
  identical permitted tuning opportunities (zero exploratory DEV tuning).
  Window reset at every sample; no cross-document or cross-window carried state.
- Evaluate exactly the **same** original 4096 DEV targets per seed with fixed
  `langtrain025.train_byte_native.windows` on all 4 owners (8 windows×128
  next-byte labels each); DEV is **already exposed**, so no TEST claim.
  Primary: pooled tokenwise next-byte NLL (nats); secondary:
  bits/original byte, top1, top5, Brier and per-document NLL.
  Include FIT-only deterministic P0 and P3 baselines, P3 historical DEV NLL
  1.877306112 as a benchmark reference only.
- Quantify wall time, GPU utilization, peak RAM/VRAM, optimizer steps, training
  token presentations and actual training FLOPs per architecture. Equal steps
  and exposure alone DO NOT establish matched training compute; report both
  exposure-matched and compute-matched lanes if necessary. Mamba CUDA BF16
  versus TAC solve FP32 numeric differences must be documented explicitly.
- Per-run original full model+optimizer+sampler/Python/NumPy/Torch RNG checkpoint,
  code/algorithm/source hashes, deterministic checkpoint reload on original DEV
  cases, then **independent clean Git replay**; no promotion. Preserve every
  failed run and original LANGTRAIN025 FAIL 0/5 vs P3.

## Hard gates / present evidence
- No-fit actual parameter/causal checks: TAC, Transformer and official BDH pass
  CI; five-seed FIT-only source sampling (320×8×128 per seed) is frozen and
  independently re-derived in CI. CI/receipts are not new training or NLL.
- **BDH** actual original 501,504p and causal prefix smoke PASS in CPU run 40775709259;
  **Mamba-3** actual model on qualified hardware still PENDING.
- **Hard hardware incompatibility identified:** official Mamba-3 SISO casts
  Q/K/V etc to BF16, but the current operator-required T4×2 venue is SM75
  without native BF16 Tensor Cores. A pre-kernel SM80+ guard now rejects it.
  The four-way experiment is BLOCKED on this execution venue; cannot pass by
  just changing the model input dtype. A separately approved venue providing
  SM80+ and matched execution resources is required. Never silently use a
  substitute, consume quota or claim a four-arm winner.
- Exact official CUDA Mamba-3 version pins, measured forward/backward, new
  appropriate venue and quota approval: pending.
- Same-source model-specific prefix and future-poison checks, source and target
  boundary controls, layer/weight gradient activity and replay: pending.
- PNDS Lean full-graph exact-real work is ongoing separately, with CED solve/
  executable correspondence/IEEE/AD obligations openly pending.
- Study-specific 500K rule amendment: implemented in this draft only, matching
  user-authorized current language scale; full machine-checkable actual source/
  FLOP parity manifest remains OPEN. No new neural optimizer step is currently permitted.
- Publication threshold: 5/5 seed complete, no source violations, matched
  counts/data/tuning/evaluation and measured compute; if compute differs, DO NOT
  call this a three-way fully matched superiority result.
- No protected TEST, no novel capability, safety, ARC, world-model or business
  claim follows from this benchmark.

## Sources
- https://github.com/state-spaces/mamba
- https://arxiv.org/abs/2603.15569
- https://github.com/pathwaycom/bdh
- https://arxiv.org/abs/2509.26507
- TAC PR #556; PNDS PR #62; LANGUAGE-SOURCE023 / LANGTRAIN025.
