# TAOTUNE004 → TAOFINAL005 — authentic TAC language optimizer results

**Date:** 2026-10-30. **Evidence tier: internal research, previously exposed publisher DEV, not independently qualified.** Complete original trial results and original failed gates preserved on the draft research branch, not promoted to `master`.

## Research question

Does a mixed orthogonal optimizer beat AdamW at matched 100,521-parameter TAC native language model weights, original genuine FIT publisher source, exact per-seed real next-byte targets and exposure, after an equal three-LR-per-mode search, at a short 64-update budget?

Optimizer treatment only: AdamW control, Muon, sign(momentum)-AdaMuon, NorMuon, and speculative pre-NS adaptive moment. For orthogonal arms, exactly `ffn_up.weight` (94×160=15,040 trainable entries) is selected and every other native recurrent/memory/concept/output/embedding parameter stays AdamW at LR0.001. AdamW control tunes full-model LR. All variants have exactly 100,521 trainable/deployed weights, original data/model source SHA checks, global gradient clipping at 1.0, WD0.01 and zero GPU use.

## Actual source and rigor

- Original genuine licensed PSF CPython 3.13.0: source SHA256 `911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4`; original model SHA256 `7179908fa3672879f3c2046ec4521f0b9f783d5c696aaba82c4c2cd0b6ebc9b1`, 14 publisher FIT documents and UTF8 supplement.
- Pre-registered [TAOTUNE004](../taotune004/FAIR_COMPARISON.json): first32 original seed1701 source-sampled steps, B4 ×256, 32,768 actual next-byte targets per candidate, 5 modes ×3 LR trials. Pure AdamW grid 0.0003/0.001/0.003; each orthogonal group's grid 0.001/0.005/0.020. Equal tuning count, unequal total optimizer-specific compute (disclosed).
- Preserved all 15 authentic full model+optimizer+Py/NumPy/Torch RNG Git checkpoints **before** a distinct runner read two frozen **previously exposed** original publisher tuning DEV owners, 512 next-byte labels. Selection uses only lower NLL (lower-LR tie-break).
- Pre-registered [TAOFINAL005](FAIR_COMPARISON.json): original 5 seeds (1701..1705) ×5 modes, 64 steps × B4 ×256 =65,536 actual FIT labels per seed/arm. Same source order and initialization within every seed, original fixed sample SHA per seed. Every full original model/optimizer/RNG checkpoint committed under `research/evidence/taofinal005/seed<id>/`.
- *Only after 25 original Git checkpoint hashes/model/optimizer/cold logits passed* in a **different GitHub runner/checkout**, original previously exposed `Doc/tutorial/introduction.rst` and `Doc/tutorial/venv.rst` were scored read-only, 512 genuine next-byte labels per arm per seed. They are disjoint from tuning owners within this experiment, **but are not independent blind evidence because the whole cohort was exposed in previous TAO003**.
- Primary old and new failures preserved, including TAO003 first invalid Git checkpoint push, a separate NorMuon cross-host bitwise replay discrepancy, and TAOTUNE004 first failed no-fit unit test that was corrected prospectively before FIT. No sealed TEST, Kaggle GPU, paid inference, merge, release, externally certified reward/goal or safety claim.

## Equal three-LR search: independent tuning results

Tuning runner [#41013466989](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41013466989) passed; fifteen genuine original FITs [#41013865166](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41013865166) passed. `research/evidence/taotune004/seed1701/TUNING_RESULTS.json` is the complete Git record, including all trial NLL/accuracy/Brier and selected original checkpoint SHA.

| Optimizer | selected LR | tuning DEV NLL ↓ | top1 ↑ | Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 0.003 | **3.490361** | 15.4297% | 0.948962 |
| Muon | 0.020 | 3.594493 | 16.0156% | **0.937357** |
| NorMuon | 0.020 | 3.607302 | **16.2109%** | 0.938047 |
| Pre-NS | 0.020 | 3.607211 | 15.6250% | 0.941091 |
| AdaMuon | 0.020 | 3.798769 | 13.6719% | 0.957762 |

**All selected the highest LR in their own grids.** Three trials cannot find a global optimum; this also limits any extrapolation to larger models, other FFN routes, schedules or time-to-quality.

## Five matched-seed report: independent original-checkout evidence

[Actions #41014891281](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41014891281) completed five parallel original seed jobs and separate independent report job **SUCCESS**, with actual final count/checkpoint gates. Full fixed rows, original source SHA, per-target report SHA, 25 numerical replay receipts, descriptive uncertainty, and all original git checkpoint paths are in `research/evidence/taofinal005/REPORT_RESULTS.json` and `FINAL_COMPARISON.json`.

| Optimizer | mean report DEV NLL ↓ | mean accuracy ↑ | mean Brier ↓ | NLL versus AdamW (absolute) | 95% paired-seed t interval, descriptive |
| --- | ---: | ---: | ---: | ---: | --- |
| NorMuon | **2.395149** | 37.1484% | 0.774324 | −0.060814 | [−0.190751, +0.069124] |
| Muon | 2.401649 | 37.3438% | **0.774220** | −0.054314 | [−0.174036, +0.065409] |
| Pre-NS | 2.424792 | 32.6563% | 0.785113 | −0.031171 | [−0.280223, +0.217881] |
| AdamW | 2.455963 | **37.6172%** | 0.781264 | 0 | [0, 0] |
| AdaMuon | 2.686249 | 30.0000% | 0.840539 | +0.230287 | [−0.002017, +0.462590] |

NorMuon mean NLL is 2.48% below tuned AdamW; Muon mean NLL is 2.21% below. Both win 3/5 paired seeds (Pre-NS 4/5 but highly variable with a large negative seed). NorMuon versus AdamW 95% paired interval **crosses zero**, so **no robust optimizer-superiority claim** is supported. AdamW has the highest top1 accuracy. All are evaluated on the same fixed 512 report targets per seed; this interval reflects seed variation only, *not uncertainty from unseen publishers or document populations*.

**Efficiency:** mean optimizer-only wall seconds for 64 genuine updates: AdamW 0.15896s; Muon 0.35459s; NorMuon 0.36024s; pre-NS 0.35922s; AdaMuon 0.35787s. Orthogonal updates are about 2.23–2.27× slower on the tiny routed FFN matrix. Whole training loops average 71.4–72.3s across variants, without a measured meaningful end-to-end time benefit; wall-clock is an efficiency metric, not a fairness equality condition. All training is CPU; GPU fusion/Triton, sharding and future-time inference cost are OPEN. The optimizer does not add a deployed phone inference state.

**Cross-host replay:** all 25 original model/optimizer/source/checkpoint identities passed, all cold logits numerically equivalent under the existing 3e-5 abs/1e-4 relative contract with unchanged top1, but 4/25 were not bitwise identical; maximum cross-host absolute difference 1.4305e-6. Both this and historical bitwise failure remain documented. Numerical replay is not falsely called bitwise proof or independent qualification.

## Decision

- **Keep AdamW as production/reference default**: it is simpler, cheaper per update, and the apparent NorMuon advantage is not established beyond five seeds on previously exposed tiny DEV with a boundary-limited LR grid.
- **Keep NorMuon as leading research candidate** on this specific 64-update NLL criterion, with Muon a close control. Preserve AdaMuon and pre-NS negative/mixed results; do not quietly drop them from scientific evidence.
- Next preregistered follow-up, *not run by this study*: wider/better orthogonal and AdamW LR search with an explicit common tuning budget, longer causal training curves and time-to-quality, fresh prospectively source-disjoint report documents (still internal unless independent verification), and scoped optimizer mathematical reference/PNDS policy. No TEST opening/model promotion without their respective gates.

No inference throughput/memory improvement is claimed from changing the optimizer. This is optimizer training methodology and its impact on genuine publisher next-byte prediction only.
