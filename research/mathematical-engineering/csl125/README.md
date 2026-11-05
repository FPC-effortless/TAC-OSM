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

- The local development harness initially exercised **16 no-fit CPU methods** against a copied CSL-001 compatibility fixture. **Subsequently, the complete local suite of 31 methods passed against the actual original CSL-001 implementation archive** (original CSL-001 SHA-256 `45ef72bf9303849d3b1eebdae0cee87937ce837d3c5f8d4b91b942b299f3902f`, CSL-125 source SHA-256 `0bf0ba779cf570084179675dd83a8b2a423867275282a129359a5f5a5828e85b`); this includes 15 parent + 16 new methods. This is still **local** source-tree validation, not GitHub Actions CI, independent audit, optimizer training, proven whole-model mathematical correspondence or learned capability.
- Operator-level NumPy reference exists, but **full-model math correspondence and Lean build are OPEN**. A matching PNDS source-pinned cross-repository record and reciprocal version registry are required before promotion.
- **Current model-scale policy:** owner-approved `declared-target-v2` was integrated into `master` by [LANGCAP102 PR #680](https://github.com/FPC-effortless/TAC-OSM/pull/680) on 2026-11-05, merge SHA `8a28843550236cea9695552670d33491b2b95854`. The new scientific scale is registered per study; **no universal 100k ceiling remains** for genuinely new v2 studies. This stacked CSL-125 draft has an older base and must be reconciled with the merged v2 source before training. The **123,699,534**-parameter model remains **NO_FIT** pending reviewed and source-qualified `FAIR_COMPARISON.json`, model/optimizer/checkpoint/reward/verifier/anti-cheat gates, data and device authority, plus explicit GPU compute budget. This policy correction does not mean a tokenizer is fitted or model capability is established.
- Prior evidence: WIL-PILOT003 at 1.813M favored hybrid over configured GQA in NLL but delta-only had lower NLL and training time matters. R9/R10 learned identity binding failed. Four GQA layers are an *engineering hypothesis* to test, not a demonstrated cure.

## Reproduction in the source repository

```sh
PYTHONPATH=src pytest -q tests/test_csl125.py
python scripts/profile_csl125.py --meta-only
```

**Not delivered:** trained weights, fitted tokenizer, released model, quantized binary, runtime benchmarks, verified long-range recall, calibrated confidence, deployable MCU package, independent full-source proof. The draft must remain unmerged without review.

## Later local original-source no-fit capacity/timing receipts (2026-11-05)

Exact-source 31/31 tests passed locally with Torch 2.10.0+cpu and original parent archive. Running a full 123,699,534-parameter, **randomly initialized untrained** FP32 model, one CPU thread, B=1/T=16, tool-only outputs (full 32768-vocab projection disabled) gave median prefill times of **10.894 ms (2 layers), 62.922 ms (8 layers), 143.590 ms (20 layers)** over five timing samples each, following two warmups. Actual state after that 16-token prefix contained 45,056/180,224/450,560 bytes of FP32 role/conv state plus 0/49,152/98,304 bytes of FP32 retained KV cache. Process peak RSS 768,300 KiB includes Python/Torch runtime, model storage, intermediates and allocator, **not** a deployable model-only RAM result. This is an unoptimized Python/PyTorch reference timing on one host, not mobile/MCU/CUDA, not full text prefill, not training throughput, and not a competitor comparison. Raw details are preserved in `FULL_GEOMETRY_CPU_NO_FIT.json`. `PROFILE_NO_FIT.json` records separate all-model parameter-only meta profiling. No optimizer step or real task evaluation was executed.
