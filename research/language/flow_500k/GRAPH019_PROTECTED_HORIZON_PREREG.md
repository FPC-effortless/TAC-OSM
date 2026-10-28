# GRAPH019 — prospective protected-auxiliary native TAC five-seed training contract

Date: 2026-10-28. **Freeze before any GRAPH019 fitting.** System-1/language lane, stacked on GRAPH018. GRAPH017 results are NEGATIVE and immutable. No seventh model.

## One candidate, exactly frozen objective
Same `Graph017HorizonTAC500K` model (512,634 active learned parameters), same h1/4/16 readout distributions, same original 30 Vim9.1 FIT + six EXPOSED DEV publisher hashes and original seeds 1701–1705, exactly the same original 120 update × batch8 × context128 per-seed FIT source/offset schedule (original SHA digest required), AdamW lr=.001/wd default, clip global grad norm 1. No hyperparameter sweeps, no after-DEV tuning.

Difference from GRAPH017: compute g1,g4,g16 for each trainable tensor on the same causal window using `protected_backward` from GRAPH018; for each separate gradient tensor gh, project only the component opposing main g1, then aggregate `g1+0.15*g4'+0.15*g16'` (h4/h16 weights fixed). Before AdamW step, clip global norm 1. Record number of projected tensors per step and actual step time. No source label is permitted in a model's pre-prediction state or future-token hidden input. This is a **nonstandard optimizer update**, so it must not claim equal full-compute architecture parity with GRAPH017, much less original TAC/SSM. Document increased FLOP and memory cost.

## Frozen comparison/evaluation
Same **already exposed** authentic Vim DEV six physical source chapters and same exactly original fixed eight length129 windows/chapter, 6144/6000/5424 genuine observed targets per seed at h1/4/16. Report proper NLL/Brier/accuracy and per-source & per-seed results, including failed fits. Score original TAC old h1, GRAPH017 h1/4/16 and frozen FIT-only smoothed three-character h4/h16 target reference on identical real labels. GRAPH015 CMake and GRAPH016 Python corpora are EXPOSED and forbidden as fresh TEST. No sealed cohort opened.

Preregister h1 guard: mean GRAPH019 h1 NLL <= historical original TAC h1 2.537438 + .01, <=1/5 seed regression >.03. H4/H16 learned ability requires >=.01 nats advantage vs FIT-only proper horizon target 3-gram in >=4/5 seeds each; mixed or fail explicit. An h1 gain alone without h4/h16 gain is improvement in 1-step ability only on exposed DEV, not general intelligence or new skills. If no gains, preserve failure and stop, do not tune after seeing DEV.

## Mathematical precondition
Only Euclidean **first-order gradient** positivity is conditionally assured. Repeated finite AdamW updates, decay and gradient clipping may violate actual h1 NLL non-regression. Testing actual training is falsification, not proof of a desired outcome. Multi-horizon direct marginal logits are not joint trajectory forecasts. Source authorization and exact original result replay in Git/LFS remain promotion blockers.

## Evidence gates
Check source SHA36/36, original FIT sample digests, actual active parameter 512634, original full model/optimizer/RNG/sampler checkpoint bytes, independent no-fit cold score replay locally, protect previously opened cohort, record training cost, pass exact probabilities/tensor causality and negative records. Git/LFS actual model source + original checkpoint binaries are required before REPO_REPLAY=PASS, and new independently authorized real documents before any generalization claim. PNDS end-to-end IEEE/autograd/Lean refinement OPEN.
