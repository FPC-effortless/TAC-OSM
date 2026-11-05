# CSL-125: 123.70M-parameter Woyengi Compositional State Ladder

**Research stage:** no-fit architecture/source prototype, not trained or quantized.
**Parent:** TAC-OSM CSL-001 draft PR #710. **Variant:** new CSL-125, not a mutation of historical CSL-001 evidence.
**Total trainable:** 123,699,534 including token embedding, tied vocabulary head, all 20 layers, 4 local GQA modules and specialist heads. The full model and all exits share one trained weight file in principle; no learning or successful early-exit quality has been measured.

## Instantiated geometry and parameter reconciliation

| Group | Trainable parameters |
|---|---:|
| 32,768 × 768 embedding AND tied next-token projection, counted once | 25,165,824 |
| Twenty recurrent/conv/gated-FFN blocks (768 width; 1,408 FF; 8×256 role state) | 92,605,480 |
| Four local GQA modules (12 Q / 3 KV heads, 128-token window) | 5,901,316 |
| Tool (17-class including no tool), field (16-class), acceptance (1 logit) | 26,146 |
| Final RMSNorm | 768 |
| **Total** | **123,699,534** |
| **Headroom to 125,000,000** | **1,300,466** |

The count is verified by construction with `torch.device('meta')`, not estimated only from a formula. Config raises if >125M. The RoPE frequency vector is a non-trainable, non-persistent buffer (not an extra learned position embedding). There is no duplicate full-vocabulary LM head.

## Tokenizer contract

Use **32,768-piece byte-fallback BPE, identity Unicode normalization**, fitted only on corpus-verified `FIT` documents. The actual SentencePiece `.model` and byte-fallback tokenization were **not trained nor supplied**; all byte pieces and reconstruction tests must pass before a language fit. SentencePiece vocabulary/merges are static *metadata* (no neural trainable parameters); actual serialized tokenizer bytes **must be counted in exported binary footprint** and budget reports. The 25,165,824 embedding parameters already cover all 32,768 IDs (including reserved control/special/byte pieces). Reserve explicit no-tool/action/field schemas before tokenizer training; do not allow an unknown tokenizer to silently change V. No weights of a second embedding/head are hidden outside the cap.

## Computation

Text ID stream → tied 768D embedding → 20 residual blocks, each using causal 3-wide depthwise convolution and state-indexed fast/slow role read/write; additional local causal 12Q/3KV attention at one-based layers {4,8,14,20} → shared RMSNorm → selected token/tool/field/acceptance heads. Explicit exit depths {2,8,20} are supported in code, with independently saved state per layer. Training each depth by joint loss is only **implemented**; no exits are trained.

For a role `j` and state coordinate `e`, `M[t,j,e] = a[j,e] M[t-1,j,e] + (1-a[j,e]) r[t,j] v[t,e]`; the fast/slow copies use different decays. The attention mask is `t - 127 <= key_position <= t`. The local GQA reference executes prefill in blocks of up to 64 queries, with at most `window-1+64` candidate keys rather than a dense T×T mask. **This is not a compiled/mobile-optimized kernel.** Time/state memory bounds exclude heads, input vectors, allocator buffers, optimizer, tensor padding, runtime frameworks and external caches.

The byte/token LM head can be suppressed (`emit_token_logits=False`) for tool classification. Acceptance logits are *uncalibrated proposals* and cannot authorize real-world side effects. Schema-valid JSON grammar and tool signature semantics remain external/unimplemented. Actor identity and state depth may not change while reusing state; reset/replay first. The prototype does not establish two-key binding.

## Theoretical parameter-only storage lower bounds

| Precision | Pure parameter bytes (MiB) |
|---|---:|
| FP32 | 471.88 |
| FP16/BF16 | 235.94 |
| INT8 | 117.97 |
| INT4 | 58.98 |
| Ideal 2.125-bit packing | 31.34 |

A useful tokenizer vocabulary and merge table will consume additional bytes. Embeddings/norms may need more than 2.125 bits and per-group quantization scales also add cost. **Neither CQ2 export nor real CPU/GPU/mobile/MCU latency and RAM have been measured here**. A complete 123.7M-parameter model is not credible for a 12 MB RAM microcontroller. Layer slicing reduces compute and active state but not weight storage unless the runtime can selectively load or memory-map only the active prefix.

**Approximate stored serving state at depth 20 (B1):** 440 KiB FP32 role/conv state + 762 KiB FP32 or 381 KiB FP16 local attention KV buffers (128-token window). This excludes activations, runtime/framework and allocator cost. Early exit depth 2 retains approximately 44 KiB core state and no GQA cache.

## Tests, training boundary and formalization

- The local development harness exercised **16 no-fit CPU methods** against a copied CSL-001 compatibility fixture: parameter count, cap enforcement, stream/chunk outputs, causal future poison, live gradients, local KV bounds, head suppression, depth ownership and shapes. These are *not* original-repository CI, source equivalence to the real CSL-001 implementation or science.
- Operator-level NumPy reference exists, but **full-model math correspondence and Lean build are OPEN**. A matching PNDS source-pinned cross-repository record and reciprocal version registry are required before promotion.
- **Mandatory policy blocker:** current TAC-OSM `AGENTS.md` limits *new trained models* to 100k ±5% parameters. The 123.7M CSL-125 design is explicitly **NO_FIT**. Training must NOT start until the owner deliberately approves a separate size-governed policy/protocol, original authorized data, matched controls/receipts, source/reward trust boundaries, protected held-out policy and verified hardware/compute budget. No Kaggle/MoLab time is allocated here.
- Prior evidence: WIL-PILOT003 at 1.813M favored hybrid over configured GQA in NLL but delta-only had lower NLL and training time matters. R9/R10 learned identity binding failed. Four GQA layers are an *engineering hypothesis* to test, not a demonstrated cure.

## Reproduction in the source repository

```sh
PYTHONPATH=src pytest -q tests/test_csl125.py
python scripts/profile_csl125.py --meta-only
```

**Not delivered:** trained weights, fitted tokenizer, released model, quantized binary, runtime benchmarks, verified long-range recall, calibrated confidence, deployable MCU package, independent full-source proof. The draft must remain unmerged without review.
