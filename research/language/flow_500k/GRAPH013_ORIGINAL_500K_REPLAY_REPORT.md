# GRAPH013 — original 500K TAC checkpoint restoration and frozen DEV replay

**28 October 2026.** Research only, System-1/language within existing six families. **No training; no protected TEST; no 500K SSM comparison; no architecture promotion.** Stacked on GRAPH012 #537.

## Problem
Five original WaveCED500K trained checkpoints are available in the researcher-owned tac_language_long003_replay.zip archive, including AdamW optimizer and Torch/NumPy/Python/sample RNG states. They are NOT hosted as binary files in this GitHub branch. The matching original five-seed **500K SSM checkpoint archive was not recovered in inspected materials**. Published prior SSM scores are not a substitute for original model bytes or a matched comparative evaluation.

## Authentic original source and results
- Independently pinned original archive SHA256: cde9a91be40a4e613975c0bd8caa4c19ac24e3d1d0e993db5cacfbd002fecc32.
- Original WaveCED500K source SHA256: acf80cb9051b7f2d2c3900d1676eed212890208cd5213088c6751f9cfd24392c; 499,290 active learned parameters.
- Verified source SHA256 for 30 authentic FIT and six previously examined DEV Vim 9.1 publisher chapters.
- Verified five original checkpoint byte hashes, original 120×8 per-seed FIT sampling schedule, stored complete optimizer and RNG metadata, and frozen scoring with restricted weights_only=True checkpoint loading.
- Recomputed 48 authentic *exposed* DEV windows, 128 real next-character targets per window, **6,144 per model per seed**. Maximum deviation from checkpoint's stored original NLL: **7.95e-8**, against original declared tolerance 1e-6.
- Executed four local source-only regression tests. GitHub source-only CI does not have checkpoint binaries and MUST NOT claim original clean-checkout score replay.

| Original 500K TAC seed | Replayed exposed DEV NLL |
| --- | ---: |
| 1701 | 2.584317724 |
| 1702 | 2.525547425 |
| 1703 | 2.518691142 |
| 1704 | 2.509428859 |
| 1705 | 2.549204588 |
| **Five-seed mean** | **2.537437948** |

Machine receipt: [GRAPH013_ORIGINAL_500K_LOCAL_REPLAY_RECEIPT.json](GRAPH013_ORIGINAL_500K_LOCAL_REPLAY_RECEIPT.json). No original GRAPH006–012 results were altered.

## Repeat locally with original lawful materials
Requires Python 3.13, Torch 2.10 CPU, NumPy 2.3.5, matching Vim 9.1 publisher files and the original verified ZIP archive.

~~~sh
cd research/language/flow_500k
python -m unittest -v test_graph013_archive_replay.py
python graph013_500k_archive_replay.py \
  --archive /path/to/tac_language_long003_replay.zip \
  --source-root /usr/share/vim/vim91/doc \
  --output original500k_replay_receipt.json
~~~

The script does not train, download or alter checkpoint bytes. It hard-rejects changed archive/source identity, altered real data, wrong sample schedule, absent optimizer/RNG, wrong model parameters and replay score drift.

## Open scientific gates
- LOCAL_FROZEN_REPLAY: **PASS**; the original five TAC checkpoints, previously exposed DEV only.
- REPO_COLD_REPLAY: **BLOCKED**; the ~84MB original archive/binaries are not in Git/LFS for independent clean checkout. A JSON receipt is not an original checkpoint.
- MATCHED_500K_SSM: **NOT RUN**; no authenticated original SSM checkpoint binaries or replay-ready source and training FLOP receipts located.
- FAIR_COMPUTE/OBJECTIVE: **UNVERIFIED**; comparable learned parameters and optimizer steps do not establish equal full FLOPs or tuning effort.
- GRAPH012_EQUAL_CACHE_500K: **NOT RUN**; no matched, frozen paired SSM and TAC neural prediction traces with identical common cache.
- INDEPENDENT_TEST: **NOT RUN**; historical Vim DEV and already-scored Python files are not fresh sealed TEST.
- ARCHITECTURE_SUPERIORITY/PRODUCT: **NOT ESTABLISHED**.

**Next:** restore original matched SSM checkpoints if available, or prospectively register and train genuinely matched 500K architectures with equal objectives, real data and training/tuning compute. Store both families' original binary checkpoints and replays in Git/LFS before qualifying results. Then export paired frozen next-byte predictions with GRAPH011 shared cache/query to GRAPH012 scorer and evaluate independently unseen genuine source, preserving all failures.
