# SENSE-TOK-009 — 4,096-update CPU continuation

The original 101,008-parameter S005 model completed **3,072 additional AdamW updates**, reaching **4,096 total**, in 1835.06 continuation seconds (2712.75 cumulative training seconds). The exact step-1,024 model, AdamW and RNG state were restored. Parent checkpoint SHA256: `5e167e312a31f1a44d21abd594eca869dbd1dc024acbf239e23c329780f60e8f`. This is a continuation of seed 1, not an independently initialized fit.

The user requested another longer run under the existing exploratory CPU scope. Source and recipe were committed before resumed optimization at `2a4ee93e7790d6de65fe743d531a8c8ddf3cc2ae`. The schedule is fixed: 16 warm-restart updates from the prior floor toward 3e-4, then cosine decay to 3e-5 over the remaining 3,056 updates. The architecture, corpus, seed, batch/history and next-byte cross-entropy loss (with the existing AdamW weight decay) remain unchanged. No DEV-based endpoint tuning occurred. The hard wall cap is 3,600 seconds.

## Endpoint results

| Exposed DEV condition | Bits per byte |
| --- | ---: |
| Step 1,024 adaptive | 4.706933685 |
| Step 4,096 adaptive | 4.134956476 |
| Step 4,096 weights, forced BYTE (posthoc) | 4.134956476 |
| Step 4,096 weights, forced TOKEN (posthoc) | 5.091408193 |

The exposed-DEV BPB change is 0.571977209 (12.15% reduction; negative would mean worsening). The controller remains collapsed to BYTE on this exposed DEV sample: all 1,530 decisions are BYTE and no token ID completes. Adaptive and forced-BYTE logits are bitwise identical. More training has not restored adaptive token use on these prefixes. Adaptive BPB does not beat the better forced route on this exposed sample; no conditional routing advantage is established. Forced modes are interventions on identical endpoint weights, not independently trained comparators. A worse forced-TOKEN score would not show that tokens are intrinsically inferior: these weights have received overwhelmingly BYTE-routed training in the later segments.

Code assignment counts: `{}`. Assignment entropy is undefined because no completed token event occurs; the absence of assignments is not evidence of single-ID codebook collapse. Repeated code IDs, when present, do not establish semantic tokens. Raw local-byte information remains available to the output head in both pathways.

| Exposed prefix | Start BPB | End BPB | End TOKEN decisions | Completed IDs |
| --- | ---: | ---: | ---: | ---: |
| ast.py | 4.496924 | 3.896948 | 0 | 0 |
| inspect.py | 4.446059 | 3.845390 | 0 | 0 |
| csv.py | 4.549694 | 3.843438 | 0 | 0 |
| difflib.py | 4.465693 | 3.731871 | 0 | 0 |
| textwrap.py | 5.077235 | 4.558375 | 0 | 0 |
| email/parser.py | 5.205997 | 4.933716 | 0 | 0 |

## Data and interpretation

The same publisher-matched CPython v3.13.5 fractions.py/statistics.py files contain 101,852 original FIT bytes: 397 full 256-byte windows with 220 tail bytes omitted. The continuation adds 3,145,728 target-byte exposures; all three training segments cumulatively expose 4,194,304 target bytes. Sampling with replacement visited 397 unique eligible windows, with 26–62 uses each. Repetition on this small corpus can overfit. Six previously exposed DEV prefixes score just 1,536 bytes; they are not an independent holdout. Near-duplicate boilerplate and common repository/domain overlap remain unresolved. No sealed TEST, GPU, new seed, trained comparator, hyperparameter search or checkpoint promotion was used.

This run tests whether additional optimization changes the existing routing collapse. It does not isolate the effects of extra updates from the planned learning-rate restart. The unchanged loss does not price inference compute or explicitly reward token reuse. Local loss reduction is not a demonstration of architecture superiority, semantic tokenizer learning or downstream capability.

## Replay and preservation

Seven original checkpoints at 1,024/1,536/2,048/2,560/3,072/3,584/4,096 retain model, AdamW/RNG, step and cumulative source-order hashes. Two fresh read-only Python processes reproduce start/end logits bitwise and recompute scores. The resumed start logits also match the original SENSE-TOK-008 endpoint tensor bitwise. They verify all checkpoint hashes, intermediate sample-order hashes, all 4,096 deterministic journal batches against source bytes, every new learning rate, and future-prefix invariance on all six prefixes. No optimizer step is used in audits. Branch-specific AdamW update counters are verified against batches with completed token events rather than incorrectly equating every parameter counter to the batch count. Nine final AdamW parameter counters are 4,096; position-recency and token-code counters are 1,455, including 567 updates during this continuation. Full per-parameter counters and changes are in the audit/summary.

The evidence archive preserves all seven checkpoints, both journals, publisher receipts, prediction tensors, two audit records, frozen recipe and hashes. Prior SENSE-TOK-007/008 originals remain unchanged and separately preserved. CPU wall time is measured efficiency evidence, not a cross-architecture compute-parity result. Local suite: 1,311 passed.

## Qualification and reproduction

The ordinary preflight and the postfit gate with actual measured final parameter/checkpoint receipts remain BLOCKED on independent signed outcome/source/ACL evidence. The immutable preregistration is preserved; actual measurements are appended in a separate final manifest. The phone planner selects CPU and explicitly does not authorize training. This execution uses the user's requested exploratory CPU continuation scope; no global gate was modified. Full independent source/evaluator qualification, complete adaptive probe integration, PNDS correspondence, matched-budget trained controls and independent evaluation remain OPEN. The bounded exploratory training is completed; scientific qualification is not achieved.

Runner: `scripts/run_sense_tok_009_long_cpu_continuation.py`. Read-only replay: `scripts/audit_sense_tok_009_long_cpu_continuation.py`. The registered 3,072-update continuation is exhausted. No automatic retry or further optimization is implied by the replay recipe.
