# Scientific Audit — TACOSM-SELECTIVE-001

## Disposition

**CONDITIONAL / bounded runtime-boundary evidence.**

## Benchmark validity

PASS within the exact equality-addressing control. H={8,64,256}, K={2,4}, five seeds, 100 queries/cell. The benchmark deliberately has multiple acceptable candidates in each queried equality bucket, preventing a hidden single-gold shortcut.

## Leakage

PASS for the registered control. The index receives only the public query/state representation; hidden reference bits remain evaluator-side. The experiment does not measure learned semantic addressing.

## Protocol and instrument

PASS. The successful artifact records contract fingerprint `7c7db37bef062e02`, measurement commit `ebb1dad73edd79c94bac97af6ac42c7d8729af1f`, and zero contract deviations. The indexed capability gate passed all six H×K cells.

## Statistical audit

QUALIFIED. The result is deterministic-looking across five seeds: exhaustive and indexed success are 1.0 in every cell. This is adequate for the registered mechanism gate but provides no uncertainty around a nontrivial effect because the endpoint is saturated. The work numbers are descriptive mechanism accounting, not wall-clock significance tests.

## Material observation

At H=256, the index reduces router candidates from 256 to 2 (K=2) or 4 (K=4), while success remains 1.0. Amortized candidate work falls from 256 to 6.56 or 8.56 after including one-time index construction over 100 queries. Wall-clock time is approximately unchanged in this tiny executor control, so there is no speedup claim.

## Nonclaims

- no semantic retrieval claim;
- no learned/neural addressing claim;
- no end-to-end compute-speedup claim;
- no hardware speedup claim;
- no C5 core scaling claim.

## Reuse

Future experiments may inherit the explicit retrieval boundary and amortized build accounting. They must use a nontrivial relevance relation to test learned or semantic addressing.