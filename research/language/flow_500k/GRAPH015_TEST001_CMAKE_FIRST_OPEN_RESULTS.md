# GRAPH015-TEST001 — first-open CMake3.31 external real-text evaluation

**28 October 2026. Frozen protocol commit e2dc905786d0b3a4299b0a8b175939a0326fb518 precedes first model scoring on the 8 CMake documents.** All eight CMake source identities and window indices were chosen and committed before opening any targets. They are now EXPOSED and cannot count as sealed TEST for later tuning. Training used original Vim9.1; the new Kitware CMake3.31 manual text was never included in those model FIT schedules.

Licensed original source: locally installed Debian `cmake-data` package, CMake 3.31.6, primary BSD-3-Clause. No publisher text redistributed. Eight frozen SHA256s are recorded in source-pinned `graph015_test001_cmake_frozen_replay.py`, using eight 129-byte windows per document with a 128-character context, 128 real next-byte labels each. Exactly **8,192 source next-byte targets per seed**, 40,960 per model across all five original frozen seeds, with eight documents as correlated source clusters (not 40,960 independent cases).

## New-source score (mean of five original model seeds, NLL ↓)

| Backbone | Native | + identical exact cache | + identical approx query |
| --- | ---: | ---: | ---: |
| New selective SSM 498288 | **2.958202** | **2.770574** | **2.716341** |
| Original TAC 499290 | 3.022614 | 2.812268 | 2.755120 |
| New bounded delta TAC 499290 | 3.032347 | 2.822100 | 2.765930 |

**Pre-registered delta improvement fails.** Delta minus old TAC native mean +0.009732 nat/byte; only 1/5 seed improvements, all eight physical document clusters unfavorable to delta at their five-seed averages. Two-level descriptive bootstrap using entire documents and seeds (10,000 fixed draws) gives 95% interval **[+0.000858, +0.017497]** for delta minus TAC native NLL. This is *descriptive*, and eight same-publisher technical manuals are not independent sites. The registered win required NLL ≤−0.010, ≥4/5 improving seeds, upper bound <−0.010 and no material guardrail regression; outcome is unambiguously NEGATIVE.

Native categorical Brier / top1 accuracy:
- SSM: 0.856873 / 29.436%.
- Original TAC: 0.861110 / 28.008%.
- New delta TAC: 0.863534 / 27.317%.

Exactly 1,322 exact cache-hit positions, 6,614 approx eligible positions, 256 no retrieval contexts per seed, identical to all backbones, all legal before target observation. Unlike exposed Vim DEV, the already frozen GRAPH011 approximate-query *helped all three* full backbones on the CMake documents. Do not use this result to tune a query gate or claim cross-domain universality.

## Independent-source reliability limits
- First opening after a prereg: **CONFIRMED LOCALLY** by Git commits/source log sequence. Whether this public material was ever exposed to an unrelated prior workflow is not independently audited; precise scientific stage is `FIRST_OPEN_NEW_SOURCE`, not externally attested `INDEPENDENT_GAIN`.
- All source files are genuine author-published docs, not synthetic. Classification and targets are authentic observed characters, but they do **not** test customer tasks, real-world autonomous decisions, long history >512, entity tracking or causal planning.
- Original TAC, new delta TAC, new SSM and GRAPH011 source-query checkpoints restored locally without fitting, with original train sample schedules and source ownership; model binaries are not Git hosted, therefore **REPO_REPLAY=BLOCKED** and independent clean-checkout original score replay absent.
- New SSM versus TAC is not strictly FLOP/tuning parity. Profiler counted 3.219e9 old/new TAC vs 2.931e9 SSM operators per step, with hundreds of uncounted op categories, not certifiable end-to-end FLOPs.
- No protected customer data, third-party authorization claims or model deployment.

## Original evidence preservation
The immutable full per-seed/per-source test scores and original local SHA are contained in downloaded evidence archive, with `GRAPH015_TEST001_FIRST_OPEN_CMAKE_RESULTS.json` SHA256 `72ec468d2525d43aab8f537116415066ca3dfec04a462039a9967072b3e66ea5`. The distributed reproduction script checks the frozen file SHA before reading publisher source. This report and source ledger contain no original CMake text.

No tuning on this first-open source from here forward. Further genuine confirmation requires *another* separately preregistered, independently collected source and all original replay gates.
