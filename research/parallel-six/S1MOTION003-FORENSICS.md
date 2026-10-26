# S1MOTION003 original frozen models — forensic diagnostics (read-only)

**2026-10-26 · source-locked real robot FIT-gradient + original DEV correction decomposition · 0 optimizer calls · no new model qualification**. Original scientific FAIL 0/3 and unchanged source score ledger: [S1MOTION003-RESULTS.md](S1MOTION003-RESULTS.md), [S1MOTION003-ORIGINAL-LEDGER.json](S1MOTION003-ORIGINAL-LEDGER.json).

## Cold reproducibility receipts
- First independent [cold-replay run 40277878071](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40277878071) **FAILED** after verifying SHA and original values on four of six models, with an over-strict same-run **AUPRC numeric ranking mismatch on seed27003 TAC**. Original first-score run was SUCCESS and the first scores/checkpoints remained frozen and unchanged. The first forensics failure is preserved; its size/cause was not printed by that run.
- Corrective change to the **diagnostics script only** makes any later mismatch error print exact replay/original/delta; does not loosen thresholds, alter source numerics or neural model weights. Second read-only [run40278108953](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40278108953) **PASS** under the SAME original score thresholds, SHA-validating **all six original neural checkpoints**, FIT-only source normalization and all first five metric types. This second run created [artifact12631160332](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40278108953/artifacts/12631160332) with per-episode scores, FIT-gradient cosines and non-training correction decomposition.
- **Open reproducibility risk**: the earlier strict ranking mismatch remains unexplained; passing later is not a proof of bitwise or platform-stable replay, and can indicate borderline score-order sensitivity. No unearned deterministic-equality claim; investigate FP32 kernel/backend and near-tied probability ordering separately before formal numerical qualification.

## Authentic original DEV source residual decomposition
For each frozen model, let `b = 2*x[t] - x[t-1]`, `c = neural_forecast - b`, and `r = actual_x[t+1] - b`, all converted back into publisher joint coordinates. Exactly, real-number MSE difference is `E[c^2] - 2*E[c*r]`.

| Seed | Arm | Correction energy E[c²] | Signed alignment E[c·r] | MSE change versus velocity |
|:--|:--|--:|--:|--:|
| 27001 | TAC | 1.02367e-7 | −2.81364e-9 | **+1.07994e-7** |
| 27001 | GRU | 4.59211e-8 | +3.06204e-8 | −1.53197e-8 |
| 27002 | TAC | 1.80154e-7 | −1.28686e-8 | **+2.05891e-7** |
| 27002 | GRU | 3.17545e-8 | +2.53628e-8 | −1.89712e-8 |
| 27003 | TAC | 1.01097e-7 | +1.16105e-8 | **+7.78763e-8** |
| 27003 | GRU | 4.26718e-8 | +3.24257e-8 | −2.21796e-8 |

This is a **quantitative description of the observed weakness**, not a causal attribution. TAC's corrections were either anti-aligned (seed27001, 27002) or insufficiently aligned (27003) to beat velocity. GRU corrections were smaller and beneficial in all three first scores.

## FIT-only gradient conflict at original frozen checkpoints
A fixed genuine FIT batch of 8 windows from each original seed's preregistered sample plan was passed through the actual original frozen model; no optimizer updates, no protected scores. Report per-parameter-block gradient norms for individually weighted (1) next-state MSE, (2) 0.1× high-motion-proxy BCE, and (3) 0.01× predictive dispersion NLL. The full per-block original numeric results and no-training receipts are in the diagnostic replay artifact.

| Original TAC seed | encoder risk-to-point gradient norm | associative memory point-vs-risk cosine | associative memory point-vs-uncertainty cosine |
|:--|--:|--:|--:|
| 27001 | 5.05× | +0.194 | −0.521 |
| 27002 | 4.75× | **−0.828** | −0.570 |
| 27003 | 1.23× | **−0.648** | +0.893 |

Large auxiliary gradient magnitudes and negative cosines on SOME original seeds imply simultaneous conflicting optimization forces in this FIT snapshot. **They do not prove global optimizer step harm** under AdamW, do not isolate physical causes, and are not proof of why test outcome failed. The separate `reactive_head` receives zero gradient from point-only MSE by design, because the point head is elsewhere; it receives risk/dispersion gradients. That is intended graph wiring, not a discovered unused-parameter bug.

## Prospective design hypothesis (not tested, not tuned on scored DEV)
For S1MOTION004, keep the strong source-authorized velocity prior and the same single shared six-family parallel TAC; test a point-priority correction path with **causally separated auxiliary representation/gradient routing** against matched velocity-GRU/velocity-only controls under *fully measured compute* and identical authentic FIT exposures. Compare:
- observed-real-source signed residual alignment and physically meaningful next-state errors;
- point/uncertainty/risk parameter-gradient cosines and actual AdamW preconditioned step directional effect (EUCLIDEAN cosine alone insufficient);
- zero-correction, read-off, auxiliary-stop-gradient, and no-memory ablations, all with matched active parameter and source budgets;
- independent true actual event/success labels and device latency before adverse-response claims.

**Do not use opened Tape DEV 25–34 or sealed episodes35–49 to tune this study without pre-registration.** Prefer a newly licensed source cohort, fixed threshold/calibration and source-identity/clock authority gates. The current risk output remains strictly observational high-motion proxy, never permission to act. Original previous negative findings and all six registered active family names remain intact.
