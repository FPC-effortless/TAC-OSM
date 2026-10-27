# MLAG002/003 — The source-aligned bank's *actual* role and source-gradient repair

**Date:** 2026-10-28. **Research status:** authentic Vim 9.1 FIT/previously exposed DEV, no new independent TEST. Original trained 100,680-parameter neural architecture, 408-float neural core and all original checkpoint weights preserved. This work is **within the existing System-1/Language specialization**, not a seventh TAC family. The original separate TAC-LANG-CURRENT-005 control remains the SSM/bigram baseline, and MLAG001 remains the dual-horizon reference.

## Reason to prioritize diagnosis before pushing to 256

CURRENT-005 reported that removing the native associative read damaged next-character NLL much more than removing the aligned source read. It would be a scientific error to infer from this that the aligned bank is useless: the historical `hist_project` classifier reads only aligned memory. A trained 256-character objective needs input contexts **strictly longer than 256**, with a matched long-context control; it is not a valid query inside the 128-character source experiment. Diagnose both heads first.

## MLAG002: actual frozen historical and generation pathway

Verified 36 SHA-pinned real Vim 9.1 original chapters, ten original 64-only and ten MLAG001 dual-horizon checkpoint hashes, original Python model SHA256 `835212d74cc0d2d4d22ff7e75beb12771586cce58f2e63943690e5d859417457`, source protocol, 48 fixed DEV windows per checkpoint, 128 characters each. No training or source mutations. Original/MLAG001 baseline next/historical losses reproduced within 1e-5. All figures are mean NLL in nats/character, **lower is better**.

| Frozen path | Original 10 models | Dual-horizon 10 models |
|---|---:|---:|
| Full next-character NLL | 2.824216 | 2.827443 |
| Remove native associative read: NLL | 2.957112 | 2.960458 |
| Remove aligned source read: NLL | 2.827620 | 2.830128 |
| Native read removal delta | **+0.132895** | **+0.133015** |
| Aligned read removal delta | **+0.003404** | **+0.002685** |
| Historical lag96 correct-address NLL | 5.219261 | **3.127606** |
| Historical lag96 with entire aligned memory zeroed | 4.852031 | 4.852031 |
| Lag96 zero-memory lesion delta | **−0.367230** | **+1.724425** |

At original lag96, zeroing aligned memory **improves** historical loss; original model was not useful at that untrained delay. In the dual-horizon model, zeroing aligned memory **dramatically worsens** historical loss. The aligned pathway is therefore **not redundant for the separately supervised retrieval task**, even though it contributes little to next-character decoding.

A no-memory **lesion** is not an equal-parameter retrained no-memory model; this cannot establish architectural uniqueness. The historical output is source-aligned by implementation, while the native memory is principally used through next-character features. Checkpoints and prior-only baselines are archived separately; wrong-key tests restrict targets to the common valid positions and are NOT directly same-sample aggregate comparators.

### FIT-only source gradient diagnostic (real text, 20 frozen models)

Using a fixed source-hash-validated window from each of four genuine **FIT** documents (not DEV) and exact source PyTorch autograd at lag96:

| Gradient component | Original 10 mean: next vs historical L2 | Dual-horizon 10 mean: next vs historical L2 |
|---|---:|---:|
| Native-specific params | 0.3084 / 0.0000 | 0.3098 / 0.0000 |
| Shared aligned source encoder | 0.03970 / 5.28277 | **0.03494 / 1.08308** |
| Shared aligned encoder mean cosine | −0.07975 | **−0.07166** |
| Common embedding | 0.39362 / 2.10758 | 0.38170 / 0.41166 |

The new historical loss's source-gradient norm is **31.0×** the next-token source-gradient norm on this FIT slice (5.28277/0.03970≈133× in original). Mean cosine is slightly negative, indicating competing directions for this specific batch, not universal conflicting gradients. Native-specific parameters receive no history-head gradient by source graph structure. The `read_router` is shared between historical and normal read, so a label like 'history-only readout' must not be attached to the whole router group.

**Interpretation:** a useful source-bank historical auxiliary loss is not automatically reflected in the normal next-token decoder. This motivates modifying *training gradient allocation and the read-to-generation interface*, not blind deletion of source-aligned storage.

## MLAG003: five-seed real FIT experimental repair

Precommitted `MLAG002_003_PREREG.md` before new fitting. **Same SiLU architecture** (100,680 parameters), same original genuine 30 FIT chapters, source hashes, 5 paired seeds (1701–1705), identical per-seed `sample_sha256` to each frozen MLAG001 checkpoint, 120 AdamW updates, batch8, context128, objective `L_char + .15 L_hist64 + .15 L_hist96`, global norm clip1, and the same exposed DEV documents. Only change: for six named source-shared learned tensor groups, if `<g_char,g_hist><0`, replace the historical gradient by `g_hist - <g_char,g_hist>/||g_char||² * g_char`. This is parameter neutral; extra autograd passes mean **training compute is NOT matched**.

| DEV metric, SiLU five-seed mean | Original dual horizon (MLAG001) | Source-gradient guard (MLAG003) | New−old |
|---|---:|---:|---:|
| Next character NLL | 2.827523 | **2.826805** | −0.000718 |
| Historical lag64 NLL | **3.375578** | 3.387505 | +0.011928 |
| Historical lag96 NLL | **3.112097** | 3.126281 | +0.014185 |
| Lag64 accuracy | **19.4596%** | 19.0625% | −0.3971 pp |
| Lag96 accuracy | **25.1563%** | 24.8958% | −0.2604 pp |

Project rule triggered in 60, 70, 58, 63 and 64 of 120 steps by seed (315/600 = **52.5%**). Next-character DEV NLL improved in 4/5 seeds; BOTH historical NLL metrics worsened in 5/5. A five-seed paired bootstrap with 100,000 deterministic resamples gave exploratory new−old next-character NLL 95% interval **[−0.001200, −0.000196]**; 96-NLL **[+0.00946, +0.01806]**. Bootstrap cannot correct repeated source DEV evaluation or architecture-selection adaptivity.

### Formal scope and candid decision

Raw-gradient first-order geometry:
```text
g = ∇_shared L_char; h = ∇_shared (.15 L_hist64 + .15 L_hist96)
h' = h - min(<g,h>,0) / ||g||² * g   if ||g||² > 1e-20
<g,h'> = max(<g,h>, 0)
<g,g+h'> >= ||g||²
```

PNDS `LanguageGradientGuard.lean` verifies scalar overlap algebra, non-conflicting overlap preservation, nonnegative adjusted overlap and first-order directional alignment, **assuming** positive exact-real squared gradient norm and the stated dot geometry. Python tests independently exercise the actual named-tensor backward and 100,680 learned parameter budget. This is *not* a theorem for AdamW's adaptive preconditioner, clipping or full neural inference correctness.

**Preregistered exploratory numerical gate:** mean next-NLL ≤ MLAG001+0.001, historical64 and 96 each ≤ MLAG001+0.02, at least 4/5 seed next-NLL noninferior. These numerical thresholds were met. However the gain is a tiny development-only improvement bought by consistent degradation of historical accuracy and additional backward compute. **Do not promote the gradient guard** as an engineering or commercial victory. Keep the original dual-horizon reference as the baseline while investigating a better transfer from aligned memory to next-token output.

## Longer-horizon next experiment: registered source-real 256 characters

The logically next test is **new context320 or greater**, NOT simply asking 128-context checkpoints to read lag256. In two matched fresh five-seed arms: (A) 64/96 objective with context320, (B) 64/128/256 objective with context320. Hold active learned parameters (~100680), same authentic real source data/splits, sample orders and 120 update budget; explicitly record extra compute and historical target counts. Include a separately retrained aligned-lesion control, same-matched-source GRU/SSM/attention where feasible, bigram/FIT prior, source-owner-aware wrong-query and zero-memory controls. Never describe length256 *characters* as 256 semantic tokens or long-document reasoning. No untouched TEST until prereg/weights are sealed. The primary quality gate must require a meaningful correct-address advantage **without next-token degradation**, not just large NLL improvement relative to a previously untrained horizon.

## Reproducible evidence and open gates

- Conversation-exported MLAG002/003 evidence bundle: 5 original-model-derived new SiLU checkpoints with full AdamW+random generator states and per-seed sample hashes, new `RESULTS.json`, full 20-model frozen lesion receipts and source FIT gradient receipts, and all runnable Python scripts. ZIP SHA256 `92610c5cb3acb190ec1c38278bb7a160f176b019865b76d1dec633baeff9587e`, approximately 5.76 MB. Raw copyrighted source text is not redistributed.
- Source model and 15 existing original/MLAG001 checkpoints were unchanged; all MLAG003 checkpoints were cold-reloaded locally and their metrics reproduced.
- Original/v2 checkpoints are not hosted as GitHub binary artifacts; genuine source manual unavailable to cold GitHub CI. **Independent repository replay and model-promotion gates are still OPEN.** GitHub CI only tests constructed operator diagnostics and exact-real Lean contracts; it does not independently rerun authentic training.
- Research limitations: same previously exposed DEV, only five independent seeds, SiLU-only intervention arm, no equally-compute retrained comparator, no protected TEST, no Mamba-3/RWKV-7 head-to-head, no measured production token cost, no confirmed cross-entity temporal reasoning.
- TAC-LANG-CURRENT-005 remains unfavorable on next-character likelihood: custom small SSM 11.384 PPL, TAC 15.014, FIT-only bigram 14.315 (different training objectives and uncontrolled FLOPs). More correct top-1 characters with worse PPL does **not** by itself diagnose ECE/calibration or tiny probability mass on wrong options: NLL scores probability of the **true** character, especially penalizing confident errors. A 10-bin ECE of 0.0204 doesn't resolve all distributional calibration deficiencies.

**Decision:** retain both memory banks for now, keep dual-horizon learning, do not prune aligned memory, do not claim projection solved the language trade-off. First qualify aligned memory's source-specific historical utility; then use independent real data and longer context to test the 256 horizon.
