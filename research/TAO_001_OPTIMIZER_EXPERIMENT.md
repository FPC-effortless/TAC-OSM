# TAO-001 — explicitly routed adaptive orthogonal optimizer (2026-10-30)

**Status: ENGINEERING PROTOTYPE / NO NEW MODEL FITS / NO CAPABILITY CLAIM.**
Derived from published Muon/Moonlight, AdaMuon and NorMuon concepts and independently implemented for the TAC research workspace. The algorithm is architecture-agnostic; parameter-role routing is intentionally architecture-specific. This branch is stacked on the separate unmerged historical checkpoint/replay branch used for the 2026-10-30 memory investigations.

## API and default safety

Source: `src/tac_osm/optimizer_tao.py`; test: `tests/test_optimizer_tao.py`; synthetic CPU probe: `scripts/benchmark_tao_updates.py`. PyTorch is an **optional research dependency**, not a new import-time dependency of the pure Python TAC interface package.

```python
from tac_osm.optimizer_tao import build_tao

# Explicit, verified FFN-only starter route in CompactDualClippedTAC086
# and the related CONTEXT026 model. All unlisted learned parameters use AdamW.
optimizer = build_tao(
    model,
    orthogonal_names=("ffn_up.weight",),
    mode="normuon", lr=1e-3, weight_decay=1e-2,
)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
# torch.save({"model": model.state_dict(),
#             "optimizer": optimizer.state_dict(), ...}, path)
```

**Never use `p.ndim == 2` as a sufficient routing policy.** In these authentic TAC sources the embedding, native memory key/value and gates, recurrent fast/slow, aligned memory, concept graph and LM decoder have different meanings. `ffn_up.weight` is a reasonable conservative exploratory candidate; `ffn_gate.weight` is a second possible candidate, while `ffn_out.weight` and `concept_out.weight` are output heads and remain AdamW by default. Unknown or non-2D/4D names fail closed. Other architectures may explicitly select ordinary hidden MLP/projection matrices; their specialized state and tied output matrices must stay with AdamW until separately validated.

## Formula and variants

For each orthogonal matrix with `N=m*n` entries:

1. First moment `M_t = μ M_(t-1) + G_t`; Nesterov input `Z_t = G_t + μ M_t`.
2. `U_t = NS5(Z_t)`, based on the documented quintic `a=3.4445, b=-4.775, c=2.0315`, after Frobenius normalization. This produces an *approximate* polar update, not an exact orthogonal matrix.
3. `muon`: use U as-is. `adamuon`: NS5(sign(Z)) followed by elementwise `v_t=β₂v_(t-1)+(1-β₂)U²`, divide by `sqrt(v)+eps`. `normuon`: after NS5, use per-row `v_t=β₂v_(t-1)+(1-β₂)mean_cols(U²)`, divide each row by `sqrt(v)+eps`. `pre_ns`: estimate elementwise second moment of Z with bias correction, divide Z by its RMS estimate *before* NS5; this is an original, unvalidated comparison arm.
4. Every orthogonal branch renormalizes to `D_t = (0.2*sqrt(N) / ||U'_t||_F) U'_t`, giving update RMS 0.2 for nonzero directions, including imperfect NS outputs. Apply `W_t=(1-lr*wd)W_(t-1)-lr*D_t`.
5. AdamW fallback uses FP32 first/second moments, bias correction and decoupled WD. Orthogonal FP32 states: Muon ~N entries; NorMuon ~N+m; AdaMuon/pre-NS ~2N. This is **optimizer state**, not the inference-state footprint.

This code prioritizes a portable, auditable, single-device eager PyTorch reference, **not** the ultimate fused/distributed implementation. No BF16 NS kernels, Triton, FSDP matrix assignment, gradient synchronization, compiler support claims, or GPU profiling are implemented. Initial optimizer `lr` is shared between groups, but best learning rates must be re-tuned per experiment arm.

## Actual checks run

Run in a local CPU environment, Python 3.13.5, PyTorch 2.10.0+cpu, x86_64, `torch.set_num_threads(1)` for timings. Seven focused constructed numerical tests passed: four modes and rectangular/conv shape cases, exact nonzero update RMS tolerance, zero gradient, finite values, explicit routing/denial, fallback AdamW closeness, bitwise full-state restart continuation, and nonfinite-gradient preflight. These are **no-fit unit tests** only; CI after remote commit is distinct evidence.

Synthetic gradient update latency, milliseconds per matrix per step, medians of 3 runs with 3 warmup + 17 recorded steps each, NO forward/backward, NO authentic source, NO GPU. Hardware scheduling uncertainty not controlled:

| shape | AdamW fallback | Muon | AdaMuon | NorMuon | pre-NS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32x32 | 0.0863 | 0.2423 | 0.2353 | 0.2537 | 0.2883 |
| 64x128 | 0.1278 | 0.4486 | 0.4838 | 0.4692 | 0.4638 |
| 128x128 | 0.1757 | 1.0622 | 1.1304 | 1.1742 | 1.1530 |

After correcting redundant per-step temporary moment allocations, tests were rerun (seven functions for each of the direct and package imports, 14 passes total); the table reports the revised local probe. This is an unfused Python reference, **not** a like-for-like comparison with production fused AdamW. The **only numerical speed conclusion** from this probe: eager orthogonal update processing costs more than eager AdamW at these shapes. It does not establish net training speed, task quality, GPU throughput, NS accuracy under BF16, stability at realistic learning rates, or a TAC loss improvement. None of the optimizer modes has passed a real TAC training comparison.

## Proper next study (not executed on this branch)

Register a **five-arm** comparison: AdamW-only control, split Muon, split AdaMuon, split NorMuon, split pre-NS. For each, use original source-authorized FIT real-byte inputs, identical model initialization and actual trainable/deployed parameter counts, fixed equal token exposures and training-update budgets, matched per-arm LR tuning budget, multi-seed learning curves and held-out exposed-DEV NLL/accuracy/Brier. Measure end-to-end forward+backward+optimizer wall time, per-update optimizer time, peak memory, checkpoint size, update spectral spread, instability, and failure count. Evaluate with the standing prospective FAIR_COMPARISON/parameter/replay checks. No sealed TEST, new fit, Kaggle usage, paid GPU, CI parameter override, or merges were authorized or performed for this exploratory source branch.

**Promotion:** a mode is eligible for future default selection only if it improves paired source-based loss or time-to-quality across seeds, retains numerical stability and does not regress relevant capability/efficiency guardrails. Prefer an equal-quality cheaper simpler AdamW baseline when no advantage is observed.

## Primary implementation references

- PyTorch Muon: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- AdaMuon (paper and reference code): https://arxiv.org/abs/2507.11005 , https://github.com/Chongjie-Si/AdaMuon
- NorMuon (ICML 2026 and reference code): https://proceedings.mlr.press/v306/li26dh.html , https://github.com/zichongli5/NorMuon

The code should not be described as a bitwise reproduction of the published implementations. It contains intentional choices (FP32 NS reference, default shared LR, full-vector norm matching, fused kernels not yet implemented).
