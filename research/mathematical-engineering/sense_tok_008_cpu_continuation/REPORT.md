# SENSE-TOK-008 — 1,024-update CPU continuation

The original 101,008-parameter S005 model completed **896 additional AdamW updates**, reaching **1,024 total**, in 716.65 seconds of continuation (877.69 cumulative training seconds). The original step-128 weights, AdamW moments and RNG were restored; the copied parent checkpoint retains SHA256 `f027782d8fcb08a132cbdb92b212a66f0e907ee649ce5582bb2097d9330489a6`. This is a continuation of one seed, not another independently initialized fit.

The owner requested the longer run after approving exploratory CPU training. Before resumed optimization, source and recipe were committed at `628d517aa5fe9e326ff4bfd373b4399c9c81def0`. The fixed continuation schedule warms from the prior learning-rate floor toward 3e-4 over 16 updates, then cosines to 3e-5 over 880 updates. This is a planned warm restart, not an unchanged extension of the original 128-step schedule. No endpoint tuning or automatic retry occurred. The 45-minute CPU limit was respected.

## Actual endpoint results

| Exposed DEV condition | Bits per byte |
| --- | ---: |
| Step 128 adaptive | 5.461519271 |
| Step 1,024 adaptive | 4.706933685 |
| Step 1,024 weights, forced BYTE (posthoc) | 4.706933685 |
| Step 1,024 weights, forced TOKEN (posthoc) | 4.885671232 |

Continuation reduced exposed-DEV BPB by 0.754585586 (13.82%). Adaptive predictions do not beat the better forced route on this exposed DEV sample; useful conditional switching is not established. Forced modes are posthoc interventions on identical endpoint weights, not independently trained comparators. TOKEN accounts for 0/1530 (0.00%) endpoint DEV decisions. Code assignment counts are `{}`, entropy is undefined because there were no completed token events. At the endpoint all 1,530 DEV choices are BYTE and there are zero completed token IDs. Adaptive and forced-BYTE logits are bitwise identical on all six prefixes. The controller collapsed to BYTE on this exposed DEV sample; token reuse and semantic identity are not demonstrated. Assignment entropy being undefined is not evidence of single-ID codebook collapse. Raw local-byte information remains in the output pathway.

## Data and limits

The same publisher-matched CPython v3.13.5 fractions.py/statistics.py originals provide 101,852 FIT bytes and 397 full 256-byte windows; 220 tail bytes were omitted. The continuation adds 917,504 target-byte exposures, bringing the total to 1,048,576. Sampling with replacement visited 397 unique eligible windows, with 2–23 uses per observed window. Repeated exposure to this small corpus can overfit. Six previously exposed DEV prefixes total only 1,536 scored bytes; these are not an independent holdout. No sealed TEST, GPU, new seed, comparator fit, hyperparameter search or scientific promotion was used.

## Preservation and replay

Eight checkpoints (128,256,384,512,640,768,896,1024) retain model, optimizer/RNG, step and cumulative source-order state. Nine final AdamW parameter states report step 1,024; position-recency and token-code parameters report 888 updates. The 136 batches without completed token events produce no gradient for those two tensors, so AdamW skips their per-parameter updates. The audit verifies these counters exactly against the journal, rather than assuming all branch-specific parameters update on every batch. Two fresh Python processes reproduce start/final logits bitwise, recompute endpoint scores, verify checkpoint hashes, all 1,024 journal sample batches against their original bytes and deterministic sampling schedule, and pass future-prefix invariance on all six DEV prefixes. Every continuation learning rate matches the preregistered schedule. Audits perform zero optimizer updates. This verifies restored checkpoint continuity and local prediction replay; it does not claim bitwise equivalence to an independently rerun 896-step training trajectory or independent auditor qualification.

The full weights, AdamW/RNG states, both journals, receipts, prediction tensors and audits are in the evidence archive. The original SENSE-TOK-007 artifacts are preserved separately and unchanged. Local source suite: **1,311 passed**. Independent source/evaluator attestations, matched-compute trained comparators, complete adaptive probes and complete PNDS model/IEEE/AD correspondence remain OPEN. Architecture superiority and downstream capability improvements remain unproven.

Training entrypoint: `scripts/run_sense_tok_008_cpu_continuation.py`; read-only audit: `scripts/audit_sense_tok_008_cpu_continuation.py`. The one-continuation authorization is exhausted. No further optimization is implied by the replay recipe.
