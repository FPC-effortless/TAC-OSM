# COLLECT018: local first-answer acquisition remains unresolved

All 18 original CPU fits completed 1,024 updates. No model passes either the registered one-key positive control or four-key binding gate. The 12 native phase/context models each score 0/96 on the four-key local primary first byte and 0/96 exact4. The standard GRU controls also fail the local positive control. This is a recipe/learnability null, not an identified phase-memory capacity limit.

| Arm | One-key first (mean) | Four-key first (mean) | Absent first (mean) | Original-text DEV NLL | Seconds/update |
|---|---:|---:|---:|---:|---:|
| phase_joint | 0.00% | 0.00% | 100.00% | 2.3635 | 0.7184 |
| phase_first | 0.00% | 0.00% | 100.00% | 2.5096 | 0.7642 |
| context_joint | 1.39% | 0.00% | 100.00% | 2.3569 | 0.7409 |
| context_first | 0.00% | 0.00% | 97.92% | 2.5282 | 0.7925 |
| gru_joint | 0.00% | 1.39% | 85.42% | 2.2533 | 0.0250 |
| gru_first | 0.00% | 0.35% | 98.61% | 2.3607 | 0.0263 |

First-byte-focused loss increased probability on the answer alphabet but worsened natural DEV loss in all 9 within-family/seed pairs. It did not improve native retrieval. Most present queries were rejected with`?`. An always-reject strategy passes absent queries without detecting absence; report false rejection, not absent accuracy alone. TRAIN has 20% absence versus roughly10% per present first-letter class, so the observed solution is consistent with a learned marginal prior. No causal attribution to this imbalance is claimed.

The original 18,432 full updates presented 9,437,184 input bytes (including repeated source and generated framing); six smoke fits add 48 updates. All 72 full and 12 smoke snapshots, AdamW/RNG, original artifact ZIPs, journals, provenance and raw256logits are retained. TEST remains unread, GPU spend 0, full training FLOPs unmeasured. CPU timings are descriptive shared-runner measurements. No architecture superiority, exact global retrieval or deployment claim.

## Graph and scope

The context arm derives phase/router/admission input from already-computed native context while keeping the original normalized value input. It adds compute but no weights/state. Both native arms retain the normalized preconditioned write, read-after mixing, FFN and SSA. Read/write phase codes remain tied: this is not a separate learned query projection or an exact-slot experiment.

The implemented memory gain is`g Pk/max(k^T Pk,1e-8)`, while the precision update has denominator`1+g k^T Pk`. This is not standard RLS, so batch-ridge equivalence of standard RLS does not certify this normalized memory write. No full Lean/source/IEEE/AD proof is claimed.

## Frozen primary and supplementary interpretation

All 41,472 original main-query rows were independently recomputed from 256 raw logits and source provenance, plus 7,776 stage-query rows. All seeds/budgets/cells are retained. One-key query swap is identity; gap 256 resamples key/quote cohorts and changes position, so the delay contrast does not isolate retention.

Phase/native interventions were declared before any full-score inspection. Training-family-on-DEV-quote localization was declared post partial null at f88c58dbae530c3d89e5ad2b24427babe2510050. It preserves all non-key bytes/RNG draws. Normalized-gradient diagnostics were declared after FP32 sum-probe failures; they never clear original failures. See machine-readable full validation for completion and exact failure labels.

## Next falsifiable repair

Prioritize a causal key/role frontend with TRAIN-only address/bank consistency supervision, several distinct queries forked from each identical prefix, and a successful local full 256 copy control. Keep inference free of oracle masks/identity tables; include shuffled-pair control and a loss/exposure-matched baseline. Separate acquisition from key-family generalization and presence from conditional value identity.

After acquisition passes, test post-write damage sentinels for residual-tiered admission against pre-write residual and admission-rate-matched random controls. Budget sentinel bytes with exact-store bytes. These are proposals, not validated fixes or novelty claims. Larger state, Triton work and a full CPU remain unmotivated by this null.

Original source graph 7b9af0d5dccc1f52c5055db850f8f8c28c3ae70c; execution 19f69b6ced3fa8baa4320f53e2b6dc2914b73308; original run 41561294525. Source draft PR702 and mathematical draft PR109 remain unmerged. Old C16/17 evidence is unchanged.

Post-result eight-present-letter ranking from original logits (exclude`?`, zero inference/fits) remains11.11–12.50% across four-key arm means, with no consistent correct-history advantage. Rejection calibration alone does not uncover useful four-key retrieval. This is a restricted diagnostic, never a replacement for the full256primary.

All18new normalized-loss FP32gradient probes pass at their declared thresholds. Dividing the scalar by4,608changes the effective absolute-error scale; it is a sensitivity diagnostic and does not erase original sum-loss probe failures. All original failure labels remain explicit.

Final key-family localization: all 18 models score 0/24 on one-key present first-byte retrieval using the TRAIN key family and the same DEV quotes/filler. Unseen key-family generalization alone cannot explain the local failure. Original primary/gates remain unchanged.

Final coverage: 41,472 main + 7,776 stage raw query scores independently recomputed; 12 native phase-intervention sets and 18 key-family localizations complete. All intervention normal-condition four-byte predictions match the original remote outputs. All 18 exposure/checkpoint/owner/causality audits finish; 15 retain FP32 sum-probe gradient misses (no FP64 failures), and 3 pass every original numerical criterion. No thresholds or original predictions were changed.
