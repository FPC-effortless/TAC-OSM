# TEMP022: preserve originals and replicate TEMP021 from Git

PR550 reports local artifact recovery and NLL calibration on exposed DEV. This
increment preserves that result; current session lacks access to that other
research environment's original archive/Vim corpus. Path/URL requested, not
an external-signer requirement. Actual preservation and cold scoring are
**NOT RUN/BLOCKED** here. No new NLL/accuracy result is asserted.

Implemented now:
- original ZIP8MiB shards, streaming parent SHA, numbered member hashes/sizes,
  byte-exact restoration, no pickle/model execution or archive extraction;
- finite CPU FP64 logit/scalar/label guards, finite original NLL comparison,
  saved RNG/finite-weight checks, before/after model state/gradient integrity,
  strict finite JSON in the existing TEMP021 runner;
- conditional Git-only replay requires committed original shards,36 original
  publisher document bytes and reviewed license, original protocol and original
  full result receipt. It records the replay router before any original run,
  verifies full parent ZIP hash and every source hash, then executes the same
  frozen graph/split/scalar recipe. Original report's historic BLOCKED gate is
  retained; any newer passed internal replication has its own receipt.

A NaN native NLL previously could bypass `abs(actual-expected)>tolerance` because
NaN comparisons are false; new negative regressions reject NaN/Inf explicitly.
This is a prospective guard fix, not evidence TEMP021 actually produced NaNs.
All valid-data formulas and original model/checkpoint bytes remain unchanged.
PNDS61 exact-real scalar math is unchanged; whole-model IEEE/AD remains OPEN.

When the already recovered archive is accessible, inspect its member metadata
and permissions first; do not publish unknown private assets or sealed TEST.
Preserve original archive bytes (full known parent SHA, no replacement ZIP):

```
python scripts/original_language_archive.py preserve \
  --archive /PATH/tac_language_long003_replay.zip \
  --destination research/replay-bundles/TEMP021/original-archive
```

Before independent Git execution, commit the following actual originals under
`research/replay-bundles/TEMP021/` (no invented placeholders):

- `original-archive/MANIFEST.json` and numbered binary parts;
- `source/` with only the36 original FIT/exposed-DEV chapters from the exact
  original protocol, unchanged publisher bytes;
- `VIM-LICENSE.txt` with the actual applicable publisher license;
- `SOURCE_RIGHTS.json`: publisher_url=`https://github.com/vim/vim`, original
  source_manifest_sha256, license name and license_sha256, and a documented
  redistribution_reviewed=true (ordinary internal review, not independent legal
  certification; include upstream version/source URL and review basis);
- `ORIGINAL_RESULTS.json`: exact original TEMP021 full receipt SHA
  4c197b5623b6dca74f7c8a25ba6931b7d83b25b1198a66898458860c672c34cd.

Git source/receipt membership and rights checks must pass before model execution.
The read-only GitHub workflow then restores archive bytes in a fresh temporary
directory and produces original-model replay, original receipt comparison and
separate status artifacts. Numerical profile Python3.13/Torch2.10.0+CPU/
NumPy2.3.5/two threads; expected comparisons frozen in PREREGISTRATION.md.
A successful engineering job with absent originals reports **BLOCKED**, not
scientific replay PASS. Even a separate-runner PASS remains internal replication,
not qualification, reward/safety certification, independent TEST or promotion.

No new neural model fit, GPU, source split tuning, new documents, TEST access,
checkpoint promotion, merge or deployment. Existing H64 and later language
negatives and TEMP021 adverse CAL chapter remain immutable. Offline inference
code gains no network dependency; preservation/replay networking is research CI.
