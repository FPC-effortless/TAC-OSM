# GRAPH017 — native direct-horizon TAC: complete five-seed authentic-DEV negative

**28 October 2026 · System-1/language family · preregistration committed BEFORE training · not promoted.** [Research branch](https://github.com/FPC-effortless/TAC-OSM/tree/research/graph017-integrated-native-horizon-2026-10-28).

## Mathematical model

Original WaveCED500K recurrent/wave/concept dynamics preserved exactly; its causal 280-dimensional decoder feature `z_t` additionally feeds two trained, deployed low-rank future-target predictors:

`p_h(y|x_{0:t}) = softmax(V_h GELU(U_h z_t + b_h) + c_h)`, h ∈ {4,16}.

Original 1-step output uses its original decoder `p_1`. Total active trained parameters **512,634**, vs original **499,290** (2.67% larger, within the owner's 475K–525K new-model band; not <=1% matched for architecture superiority). Supervision from genuine original 129-byte source window only: for h∈{1,4,16}, at position t predict observed byte x[t+h], and score only positions t+h<129. Fixed objective `L1 + 0.15*L4 + 0.15*L16`. Mathematically, source-causal auxiliary heads do not modify native recurrent transitions. The new model CAN output normalized direct future marginal distributions; that does not make those forecasts accurate, and is not joint path forecasting.

## Authentic training and comparison

All 36 original human-authored Vim9.1 publisher chapters SHA256-verified: 30 historical FIT, six already exposed DEV. No new TEST was opened. Five seeds (1701–1705), exactly the original 120×8 random FIT source/offset schedule per seed (five original sample digests independently matched). Batch8, context128, AdamW lr=.001, gradient clip1, objective above. All original model, optimizer, PyTorch/NumPy/Python/sampler RNG, training sample hash and original observed development scores saved for each checkpoint. Five original checkpoint byte hashes verified and fully cold-local-reloaded; max NLL drift <1e-6. Checkpoint binaries remain outside Git/LFS: REPO_COLD_REPLAY=BLOCKED.

### Five-seed mean REAL EXPOSED-DEV proper scoring

| Horizon | Original/strong reference NLL ↓ | New 512,634-param TAC NLL ↓ | Decision |
|---|---:|---:|---|
| 1-step (6,144 targets per seed) | original TAC 2.537438 | **2.550268** | Guardrail FAIL, +0.012830 |
| 4-step (6,000 targets per seed) | FIT-only smoothed 3-gram 2.675518 | **3.507245** | Forecast utility FAIL |
| 16-step (5,424 targets per seed) | FIT-only smoothed 3-gram 3.195463 | **3.573829** | Forecast utility FAIL |

At h4: new Brier 0.938700 and accuracy 15.440%; FIT-only trigram Brier 0.794304 and accuracy 34.200%. At h16: new Brier 0.950978 and accuracy 14.919%; FIT-only trigram Brier 0.873116 and accuracy 24.115%. Neural horizons were evaluated against a FIT-only count model over whole source chapters with add-10 smoothed global prior. That practical reference sees more FIT historical positions than the neural model's 120 updates and is **not** a matched architecture-compute comparator. It is, however, a relevant task-matched hurdle for useful predictions.

| Seed | Original TAC h1 | GRAPH017 h1 | GRAPH017 h4 | GRAPH017 h16 |
|---|---:|---:|---:|---:|
|1701|2.584318|2.591598|3.493646|3.572571|
|1702|2.525547|2.538460|3.510112|3.535611|
|1703|2.518691|2.531740|3.534327|3.671091|
|1704|2.509429|2.526082|3.495336|3.530184|
|1705|2.549205|2.563459|3.502805|3.559686|

Every seed has a h1 regression, and the registered <=+0.01 mean NLL guardrail fails. Horizon gains fail in all seeds. This is a **complete trained negative**, not a selective failure of a single seed.

## Diagnostic: conflict at memory gates

Five frozen new-model checkpoints had per-task gradient direction measured on each seed's **first original authentic FIT training mini-batch**, without updating parameters. Six chosen shared parameter tensors were inspected. The native-gate weight's h1 vs h4 gradient cosine averages **−0.36645** (and h1 vs h16 **−0.09678**); learned basis gradients were positive h1/h4 at **+0.32115**, h1/h16 **+0.29613**. The weighted auxiliary-to-main norm ratios at native gate were only approximately 0.040 (h4) and 0.036 (h16). This indicates local conflict at memory gating but does NOT prove it caused the DEV failure; the tiny training corpus/schedule and bad horizon targets may dominate. All six tensor summaries and per-seed observations are preserved in GRAPH017_FROZEN_GRADIENT_DIAGNOSTIC.json.

## Mathematical/engineering verification

Six constructed numerical regression tests PASS: active learned 512,634 count; all three normalized finite 128-byte distributions; causal future-token poisoning; exactly aligned h=1/4/16 source targets and legal denominators; nonzero gradient paths for all active weights; **bit-exact original 1-step logits when the old original learned weights are loaded into the new model**, so any h1 regression results from optimization rather than a silent changed original decoder. Tests are software diagnostics only, not genuine predictive claims.

## Scientific gates and next math

- REAL_TRAINED + LOCAL_FROZEN_REPLAY = PASS, limited to authentic previously exposed DEV.
- ONE_STEP_PREDICTION_GUARDRAIL = FAIL.
- H4/H16_PRACTICAL_REFERENCE = FAIL.
- ORIGINAL_GITHUB_BINARY_CHECKPOINT_COLD_REPLAY = BLOCKED.
- FRESH_SEALED_TEST = NOT_RUN; GRAPH015 CMake and GRAPH016 Python sources are **already exposed** and may not be re-used for fresh qualification.
- EQUAL_TRAINING_FLOP_ARCHITECTURAL_COMPARISON = NOT ESTABLISHED; original vs new use different objectives and param budgets.
- WHOLE_MODEL_LEAN_IEEE_AUTODIFF_PROOF = OPEN.
- INTELLIGENCE/CUSTOMER ROI/DEPLOYMENT IMPROVEMENT = NOT ESTABLISHED.

**Next mathematically falsifiable revision, not yet trained:** test a *primary-loss-protected* gradient projection for the horizon auxiliary update at shared recurrence gates and proper FIT-only horizon prior calibrations. For main-task gradient `g1` and horizon gradient `gh`, define `gh_perp = gh - min(0,<gh,g1>)/||g1||² * g1` where ||g1||>0. Then `<g1, g1+λ gh_perp> >= ||g1||²` for λ>=0, a **first-order alignment fact** only. No promised finite-step NLL bound, no guaranteed horizon improvement, and no guarantee the severely undertrained horizon model catches a proper FIT-trained baseline. Any new training must be separately preregistered with source/parameter/compute and guardrails, and ultimately tested on independent newly authorized real outcomes.