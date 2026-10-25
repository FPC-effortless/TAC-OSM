# ME103 proposed protocol: fixed equal continuation of the ME102 panel

**Status: proposal only. This is not preregistered and authorizes no training.** Source identity, custody of every original checkpoint and optimizer/RNG state, continuation support, data/profile parity, and current preflight eligibility must be validated before this protocol can be registered. If any original state cannot be recovered and verified, the proposed continuation is blocked; do not substitute a fresh fit or silently reconstruct state.

## Question and hypothesis

ME102 trained 15 models: three fixed recipes (`light_floor`, `strong_reconstruction`, `physical_score`) across seeds 102001–102005. All three recipes failed their all-seed full gate. This proposal tests the post hoc hypothesis that some failures reflect undertraining at the original 1,536-update budget, rather than an irreducible recipe limitation. It asks whether one fixed, equal continuation of every original model changes those results. This hypothesis is motivated by the observed failures and is exploratory; it is not an independent confirmation or a claim of architectural novelty.

## Arms and continuation

Continue **all 15** original ME102 models, with no seed, recipe, or checkpoint selection. For each model, restore its exact original model weights, optimizer state, Python/NumPy/PyTorch RNG states and data/augmentation sampler state, then perform exactly 1,536 additional optimizer updates. Each lineage therefore has 3,072 cumulative updates. Keep the ME102 optimizer, schedule, batch size, update order, initialization and every other training setting unchanged; do not reset optimizer moments or RNG streams. Preserve the original artifacts and write continuation checkpoints and receipts as new, linked artifacts.

There is one extension length and no interim scoring, early stopping, extra tuning, retries, or additional hypotheses after results are observed. A failure, interruption, or invalid update is recorded; do not replace that lineage with a new run. No model or recipe is selected for continued claims based on intermediate outcomes.

## Fixed data, masks, and profiling

Use the identical ME102 source version and checksum, FIT-only preprocessing/statistics, stored per-seed sample-order plans, batch composition, context cycle (8, 64, 256), and original augmentation and masking logic. Preserve the original clean/corrupted input proportions, eligible-origin rules, causal ages, genuine-current-label targets, and natural missingness. Do not add data, alter masks, refit preprocessing, or expose evaluation data to training.

Before any continuation, independently verify source and artifact hashes and establish that the restored optimizer and RNG/sampler states reproduce the expected next ME102 update inputs. Repeat the original graph/profile parity checks across the three recipes at contexts 8, 64, and 256, using the same profiling method and operator/shape/count signatures. Any mismatch blocks the affected panel before updates; it does not permit changing the recipe or profile contract.

## Evaluation and unchanged gates

After all 15 fixed continuations finish, score every model with the original ME102 scoring implementation and evaluation cohorts. Report all original and continued per-seed results side by side, including failures and gate changes. Apply the original full gate independently to each model, unchanged:

- Gap-cohort PM 90% interval coverage is 0.90–0.98 inclusive.
- Gap-cohort PM MAE is at most 1.05 times the same-index ME101 parent result.
- General-cohort daily PM MAE is at most 1.05 times the LOCF reference.
- Gap-cohort density NLL is at most 6.234504261016846.
- Gap-cohort PM interval score is at most 1.05 times the same-index ME101 parent result.

A recipe passes only if every one of its five continued seeds passes every gate. Publish gate results for all 15 models and all three recipes. Do not change thresholds, round values into a pass, pool seeds to erase failures, or describe a recipe as passing when any seed fails. Report CPU/process time as an efficiency observation, not a parity gate.

Only the already exposed 2015 cohorts may be scored: the gap cohort (143 cases) and general cohort (4,288 cases), with exact ME102 membership and scoring semantics. These are development evidence. Keep protected 2016 and 2017 cohorts closed and unscored. This continuation cannot establish independent qualification, generalization, or architectural superiority.

## Interpretation and reporting

The primary descriptive outcome is the unchanged full-gate status for each of the 15 continued lineages and the all-five-seed result for each recipe. Report all metrics, original-to-continued paired differences, update receipts, checkpoint hashes, runtime, and failures. A change after extra updates is evidence about these specific continued runs only; it does not establish that ME102 was undertrained in general or that additional updates caused a reliable capability gain. No favorable-seed selection, recipe winner claim, or post hoc threshold adjustment is allowed.

## Registration blockers

Before any preregistration or fit, validate this proposal against the exact ME102 source and original artifacts; verify complete restorable optimizer, RNG and sampler state for all 15 models; confirm the one-extension implementation preserves the stated 1,536 additional update budget and original optimizer settings; check current data rights and cohort hashes; and run the required machine-checkable preflight on a complete study manifest. This document alone does not satisfy those checks. Until they pass and a valid registration is separately approved, status remains **PROPOSED / NOT AUTHORIZED FOR TRAINING**.
