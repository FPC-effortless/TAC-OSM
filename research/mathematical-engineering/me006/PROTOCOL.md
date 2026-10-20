# ME-006 — Verification contracts and future-conditioned CDL-PHR experiment
**Registered:** 2026-10-20, before new ME-006 real-data fitting or validation scoring.
**Status:** research only, no merge or scientific model promotion.

## Paired computational question
ME-005's original `cdl_gain` is computed against the **current observation embedding**, so it cannot establish the usefulness of a memory for actual future outcomes. The new causal-order surrogate measures fixed-variance future-target conditional likelihood improvement against the *same frozen ME-004 baseline*: `R = 0.5 * (||y - base||² - ||y - (base + retrieval_residual)||²)`. Future labels are used ONLY by external training/evaluation; inference consumes causal observation histories alone. This is a Gaussian relevance surrogate, **not** the exact original LM conditional-NLL CDL, and is not causal effect identification.

## Frozen source and arms
Reuse authenticated UCI MetroPT-3 TRAIN days 0–20 and previously accessed exploratory validation days 20–25 from ME-004. **Do not open** March 2–9 future evaluation, publisher later periods, or other reserved tests. Compare two equally future-supervised adapter arms on *both* original, hash-checked pretrained two-timescale TAC checkpoints, seeds 606001/606002:
- `direct`: encoder of latest observed input to residual forecast head (no fast association).
- `associative`: learned causal encoder with 16×16 bounded Hebbian association matrix `A_t`, `A_{t+1}=Π_{||A||F<=2}((1-λ) A_t + η · e_t e_{t-1}^T/16)`. The latest encoded query retrieves from `A_T`, then a zero-initialized residual predicts two real future horizons.
Both heads initially reproduce the authentic pretrained TAC forecast exactly. Frozen base has zero parameter updates.

Fixed **160 AdamW updates**, batch 64, LR=0.001, weight decay 1e-4, Huber beta=0.2, clip-grad max-norm 1; 2 model seeds × 2 arms = 4 fits. Same 12 physical sensor channels+mask, 32 history steps, same two forecast horizons 12/60 observed steps, same chronological examples. No score-driven retuning. Real-source checksum and original checkpoint hashes must match exactly; archive all four original weights and results even if the associative mechanism loses.

## Evaluation
Primary authentic future-target macro MAE averaged over all observed sensors/horizons; secondary fixed-variance Gaussian log-likelihood gain over same frozen original, paired seed comparisons, training CPU seconds, trainable parameter counts and memory norm contracts. The original validation set is historically **reused**, so even 2/2 favorable seeds do not qualify new scientific TEST improvement. No entity-identity discovery, verified real action-cost or counterfactual behavior claimed.

## Concurrent formal/operational contract
Companion `FPC-effortless/pnds-lean` draft PR #4 formalizes Boolean admission-gate conditions, no unauthorized E→P transitions, bounded convex recurrent state and fixed-target refinement. `src/tac_osm/verified_promotion.py` provides executable fail-closed admission with a required independent receipt verifier, PNDS utility/stability/integrity/cost predicates, authentic receipt IDs, context diversity and negative-outcome quarantine. Formal boolean receipt authenticity is an assumption about an external verifier; code does NOT prove external truth, real commercial safety or crash-safe persistence.

## Decision and next engineering
If associative retrieval fails a meaningful direct-control or base gain, **do not increase sophistication**, keep direct base/native TAC. If it wins, preregister a genuinely fresh independent real source/case before enabling it in the native agent. All proposals in the unified master plan remain tracked; none should be silently deleted. TAC-OSM remains its own predictive-executable architecture, not an SSM.
