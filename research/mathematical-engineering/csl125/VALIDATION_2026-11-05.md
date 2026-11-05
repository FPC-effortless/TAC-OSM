# CSL-125 full-geometry validation — 5 November 2026

Stage: **local CPU, synthetic-input engineering verification; zero optimizer steps**. Source: `FPC-effortless/TAC-OSM`, draft PR #711, `src/tac_osm/csl125.py` SHA-256 `0bf0ba779cf570084179675dd83a8b2a423867275282a129359a5f5a5828e85b` (Git blob `4b7a60850e89480fddceeec74b7661350b6bb7bf`). Original parent CSL-001 Git blob `08e50da0a3d507202a35ae1773cc39755b81d4fa`.

## Policy reconciliation

The owner-approved `declared-target-v2` size amendment was staged in LANGCAP102 PR #680, but remained draft/unmerged. On this date it was merged to `master` as `8a28843550236cea9695552670d33491b2b95854`, with the new size section verified in the actual default-branch AGENTS.md. Model-size target is now prospective and registered; historical v1 preserved. **The rule change does not independently authorize training**: preregistration, authentic source rights, evaluator/reward and anti-leakage boundaries, realistic matched comparisons, GPU/compute approval, checkpoint replay and independent qualification remain separate requirements.

## Construction and testing

* Exact all-path learned parameter count: **123,699,534** (hard cap 125,000,000), including 32,768×768 tied vocab embedding/head, 20 role-state+conv+FFN blocks, four local GQA readers, final norm and specialist heads.
* Existing repo subset `PYTHONPATH=src python -m pytest -q tests/test_csl001.py tests/test_csl001_math.py tests/test_csl125.py tests/test_csl125_math.py` **31/31 PASS** on CPU (`torch 2.10.0+cpu`).
* Full-scale CPU FP32 batch1, sequence 8, all exits 2/8/20: combined random-label next-token/tool/field/acceptance loss finite; backward 2.458s, forward 0.450s. **328/328 parameter tensors had finite nonzero gradients**, zero absent; process peak RSS about 1,446,748 KiB. This is NOT a training result.
* Full-scale causal/streaming: sequence 24 split [7,6,11], at exits 2,8,20; maximum absolute logits deviation [6.10e-5, 1.22e-4, 1.22e-4]; state deviation ≤1.12e-8; KV deviation ≤5.07e-7; future-token poison effect on preceding 16 logits exactly zero. Synthetic numeric comparison only.

## Full-geometry CPU inference benchmark

One CPU thread, FP32 Torch reference, batch1, 64-token prompt and 8-token cached decoding, **full 32,768-token vocabulary projection enabled**, 3 independent repeated samples (median); model randomly initialized. All exits selected from one model instance.

| Depth | Prefill tokens/s | Decode tokens/s | Prefill samples ms | 8-token decode samples ms |
|---:|---:|---:|---|---|
| 2 | 671.2 | 61.7 | 189.003,95.349,91.709 | 134.678,129.730,123.404 |
| 8 | 337.9 | 29.5 | 216.861,189.432,178.769 | 283.408,265.767,271.319 |
| 20 | 177.3 | 15.0 | 426.620,360.985,335.893 | 517.836,560.460,531.940 |

The first broader benchmark was terminated at its execution timeout after depths 2 and 8, with no full-depth receipt. The replacement bounded benchmark was a fresh run with shorter 8-token decode and three samples; **it is not a continuation or retroactive completion of the timed-out run**. Peak RSS for the bounded inference process: 802,148 KiB. Do not compare directly with CUDA/optimized inference kernels, other machine configurations, CPU benchmarks with a suppressed LM head, quantized/Cactus runtimes, or model-quality claims.

## Important information-preservation bound

The current fast-state EMA initializes `alpha=0.90`, slow-state EMA initializes `alpha=0.98`, and the scanner **clamps alpha <=0.995**. The direct contribution of a memory cell held for `n` update events contracts by alpha^n: at 1,024 tokens, 0.98^1024 ≈ 1.04e-9, and even at the maximum alpha .995, 0.995^1024 ≈ .00590; at 4,096, .995^4096 ≈ 1.21e-9. Thus the **direct old-state contribution** cannot survive extremely long gaps unchanged through the EMA alone. Learned repeated refreshing or other paths could alter the effective system-level behavior, but these have NOT been shown to recover exact multi-key long-horizon information. This is a research limitation and falsifier, not a proof of global model incapacity.

## Next scientific and GPU steps, not executed

1. Integrate branch atop v2 scale policy without modifying historical v1 results. Register target size `123,699,534`, explicitly count all deployed/frozen weight paths, and preserve 1% matched parameter parity for contemporary GQA Transformer and competent SSM/Delta-only controls.
2. Authenticate a real licensed FIT-only corpus and fit byte-fallback BPE. Freeze module-family-disjoint FIT/DEV/TEST and causal information boundaries. Identify GPU runtime with owner-authorized budget; Kaggle plan requires exactly two verified T4s, using both for seed shards or DDP; no provider run launched here.
3. Profile GPU end-to-end FP32/BF16/FP16 with proper GradScaler as applicable, prefill, cached single-token decode, actual total RSS/VRAM and train step/s, FLOPs & 125M model optimizer state; test 2/8/20 loss balancing and state reset/replay.
4. Only then train multiple seeds on identical real tokens/optimizer budgets, grade delayed two-key overwrite/absence, JSON validity, tool routing and confidence calibration with independent verifiers. Preserve negative seeds and original source/checkpoint hashes. Verify real task progress before introducing more memory modules.

No real task accuracy, learned tool semantics, long-horizon binding, mobile device power, quantized binary, CUDA performance or model superiority is established by this report.