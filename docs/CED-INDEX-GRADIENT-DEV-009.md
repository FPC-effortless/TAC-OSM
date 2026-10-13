# DEV-009 — First-stage sparse-index gradient correction

**Classification: DEVELOPMENT / NOT CONFIRMATORY.** This report records an intervention on the trainability failure observed in DEV-008. It is not a result on general learned persistence, real-world prediction, DeepSeek model replication or asymptotic complexity.

## Parent results that must not be overwritten

DEV-008 is preserved in `docs/CED-HARDENED-DEV-008-RESULTS.md` and [PR #164](https://github.com/FPC-effortless/TAC-OSM/pull/164). Three matched seeds: shared mean 43.40% pre-repair task accuracy vs sparse 13.19% on synthetic unique-equality cases; final sparse cross entropy as high as 667,885.875. A hard top-K index excluded gold and gave it constant -1e6 final logits, so final cross entropy alone could not correct early exclusion.

## Intervention

Expose the first query-decoder layer's unmasked global candidate scores as `QueryOnlyCED.last_first_index_scores`. These scores are differentiable and constructed exclusively from the same public query/state reference and already-retained candidate descriptors; they are **not** given gold or hidden outcomes at inference. The original hardened loop, state transitions, index, executor, verifier and delayed writes remain unchanged.

Two arms: **shared+indexCE** and **sparse+indexCE**. Both start from matching weights and train on identical examples. Both receive first-index cross-entropy against the supervised gold candidate during development. The shared arm always has all candidates accessible and applies final-route CE; the sparse arm applies final-route CE *only when gold is in the selected top-k*, otherwise first-index CE alone. The conditional loss is necessary to avoid the severe constant-mask pathology, but it means **these training gradients are not equivalent across arms**.

All supervised indices exist **only** in the offline trainer/evaluator. The deployed router still knows no hidden action, gold index or verifier output.

## Tests and measures

- Explicit unit test: an excluded gold candidate still supplies nonzero first-index query-weight gradient.
- Byte-identical starting parameters and identical data streams across arms, three seeds 43,59,71, 120 training steps each in the full development grid.
- Measure unmodified TAC-OSM `HardenedLoop` **pre-repair** exact decisions for N=8/32 candidates and temporal delay D=0/4, 24 evaluator episodes/cell/arm/seed (576 model decision episodes total).
- Separately measure first-stage **recall@4** on unseen synthetic unique-equality tasks for N=8/32, 96 trials/size/arm/seed (1,152 diagnostic recall cases).
- Report first index CE, final task CE, number of training steps where sparse gold was excluded, actual score-pair cost and cache scalars.
- CI must rerun all 11 targeted tests, enforce repository test-count integrity, run smoke, and save six checkpoints plus every evaluation row. A separate validator reloads checkpoints and reconstructs all scheduled task and recall data; replay is **not** re-training.
- The exact two-bit equality generator is solvable deterministically by a public/reference signature comparison; **100%-capable nonlearned control is available by construction**, so these learned scores should not be generalized.

## Decision rule

If sparse first-stage recall improves markedly but pre-repair decision accuracy remains poor, the issue has moved from addressing to downstream decoder/readout. If first-stage recall remains poor despite supervised index CE, assess whether the causal encoder's positional representation and its score function are expressive/trained enough, rather than concluding that sparse attention as a class is inferior.

If the sparse model improves, do not backfill this finding into DEV-008 and do not claim CED architecture superiority: the loss changed following observation of earlier failure, and the experiments are development data, not an untouched preregistered comparison.

## Next actual TAC-OSM question

A good retained-candidate reader is only a component. The next nontrivial experiment must operate on the **learned bounded finite memory** and noisy-alias/competing-write tasks from DEV-003–006, with fair shared/sparse/ordinary router controls and measured write/update/read/verification costs. That is where persistent memory and future prediction claims become meaningful.
