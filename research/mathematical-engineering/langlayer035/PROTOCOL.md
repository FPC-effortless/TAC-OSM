# LANGLAYER035 — prospective authentic language fit, depth versus parent (internal)

Owner request 2026-10-29: train and benchmark the 500K layered
LANGLAYER034 source against its unchanged 500,122-parameter parent.

## Frozen BEFORE any fit

- Three active model arms: parent CONTEXT026 500,122;
  depth-2 LANGLAYER034 499,725; depth-3 LANGLAYER034 499,934.
  0.0794% maximum gap, no dummy padding, frozen embedding, borrowed
  external model or post-initialization capacity change.
- Five fresh training seeds 1701–1705; **320 optimizer updates per arm**,
  batch size eight, 129 original source bytes, 128 next-byte targets per
  window, 327,680 genuine FIT byte targets per trained model.
- Original immutable PSF CPython 3.13.0 publisher source and exact
  source-owned windows, with original 13 FIT documents + 1 UTF8 FIT
  supplement and four already-exposed DEV documents.
- AdamW learning rate 0.001, gradient norm cap 1, identical CPU
  environment, zero hyperparameter trials and zero protected TEST.
- The original DEV window set is 4 original documents × 8 windows ×
  128 next-byte targets = **4096 already-exposed DEV targets** per seed.
  Report NLL, perplexity, bits/byte, top-1/top-5, Brier and CPU time
  for all seeds; no champion selection on DEV.
- P3 **1.877306112** (legacy immutable baseline on same original DEV)
  is an unmatched, count-based practical reference; it has not received
  neural optimizer compute or the new matrix-run training protocol.
- Every arm/seed writes original model + optimizer + Python/NumPy/Torch
  RNG, exact sampled FIT digest, source identity, model parameter count,
  SHA-256 of the binary, cold load and identical scored DEV receipt.
- Real CPU usage and memory are measurements, **not** a matched FLOP
  allocation. No claim of formal FLOP-matched architecture superiority.
- Checkpoint archive is first GitHub Actions artifacts, retention
  90 days; long-term archival/independent replay beyond that is a
  separate task and must not be assumed complete.
- Existing original negative results, training artifacts and parent
  reference are never overwritten.

## Engineering / math status

Depth-2/3 structural forward/gradient tests passed in TAC PR #562.
Independently formulated PNDS serial residual tests and scoped Lean
recurrence are in draft PNDS #63 on top of complete-context PR #62.
The mathematical proof of the **entire composed** layered TAC graph,
general CED inverse nonsingularity correspondence, all IEEE/AD
guarantees and meaningful memory/long-context performance remain OPEN.
These fit runs are exploratory research, not mathematical certification.

This owner's express new instruction authorizes a narrowly scoped
**language-model ~500K internal training** exception to the older
universal 100k rule. The source validator is amended for only
LANGLAYER035, leaving all unrelated new fits bound to 100k.
This is NOT permission for production, secrets, protected evaluation,
GPU quota or paid providers, and does not waive source-trust gates.

## Rejection criteria

Any source/hash or split mismatch, forbidden future information,
nonmatching sampling digest, model-size drift, loss/gradient nonfinite
or disconnection, mismatched checkpoint cold replay, incorrect
DEV target count, missing optimizer/RNG or workflow failure must be
recorded as a FAIL. Count every attempted run, do not edit failed
metrics after inspecting DEV. Do not infer "better architecture"
from previously exposed DEV even if all five seeds improve.

## Next after this internal comparison

Ablate the depth block's recurrent path and concept path separately,
establish actual long-context benefit and fitted memory bytes. Run
matched parameter/data/exposure/compute controls for selective SSM,
GRU, Transformer and any published matched Mamba implementation.
Require a genuinely untouched evaluation partition for independent
generalization. Compare source-corrected experiments with the earlier
P3-n-gram + TAC hybrid without absorbing its counts into native TAC.
