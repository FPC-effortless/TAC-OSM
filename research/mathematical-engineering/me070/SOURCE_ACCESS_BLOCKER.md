# ME070 original long-history source qualification — BLOCKED

**Date:** 2026-10-23. **Outcome:** `SOURCE_ACCESS=FAIL`, `SOURCE_IDENTITY_SHA256=UNKNOWN`, `TRUE_256_EVENT_CASE_SUPPORT=UNKNOWN`, `ME070_FULL_TRAINING=NOT_STARTED`.

## Actually executed

Read-only [GitHub Actions run 39920980632](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39920980632) attempted to retrieve original BPI Challenge 2018 XES bytes from 4TU's public dataset file endpoint cited by the published Verona dataset loader:

`https://data.4tu.nl/file/443451fd-d38a-4464-88b4-0fc641552632/cd4fd2b8-6c95-47ae-aad9-dc1a085db364`.

**Actual evidence:** all three `curl` connection attempts failed with `curl: (28) Failed to connect to data.4tu.nl port 443 after 30002 ms: Timeout was reached` (job 120305863016); the workflow exited 28. This is **network/source access failure**, not evidence that the source contains too few 256-event histories, and not proof of a licencing defect. No original source file was downloaded; no legal source hash, temporal eligible-cohort receipt, model fits or original ME070 checkpoint is reported.

The official source DOI is https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972. Third-party published descriptive totals of 43,809 cases, ~2.51m events and a 2,973-event longest trace do not certify **the number of disjoint, independent, chronological 256-event cases** or the actual byte identity. No unseen TEST labels have been used for model development.

## Fail-closed resolution criteria

Do not start ME070 fits. Qualify an original, **legally accessed, byte-hashed** XES source from the official publisher or an independently proven byte-identical authorized mirror. Run `scripts/me070_qualify_long_xes.py --input <original.xes.gz>` to produce the source hash, original lifecycle and trace-length statistics, 60/20/20 original case-start time split and excluded overlapping-case counts. Require >=50 distinct eligible FIT and >=25 independent eligible TEST cases with true >=256-event history and next-event targets, otherwise `STOP_NO_LONG_HISTORY_COHORT`. Preserve the result in this TAC-OSM repository (aggregates and hashes only, not raw traces/identities). If access continues to fail, assess other **authentic** longitudinal sources using the same gate. A different converted event-log variant is NOT assumed byte-identical.

The current `research/mathematical-engineering/me070/FAIR_COMPARISON.json` is explicitly **noncomparative diagnostic only** and does not license new training. Before any later fits, register original qualified real dataset checksum, 5 paired seeds, parity-qualified arms, matched actual compute/data/evaluation and in-repository original checkpoint storage, then rerun the machine preflight. A source-access timeout or CI rerun is not a completed experiment.

## Separate completed retrospective finding

The original ME069 Git-owned case-scores and ten fitted checkpoints remain preserved. [EXPOSED_ME069_DIAGNOSTICS.md](EXPOSED_ME069_DIAGNOSTICS.md) presents completed paired McNemar/interval-width tail diagnostics from run 39920681790; **these are post-hoc analyses on an already exposed TEST**, not fresh memory-performance results.

## Continued authenticated publisher metadata and delivery diagnostics, 2026-10-23

- Official Figshare [article 12688355](https://api.figshare.com/v2/articles/12688355) returned consistent title **BPI Challenge 2018** and DOI `10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972`, with **original file ID 24025820**, filename `BPI Challenge 2018.xes.gz`, **158,270,992 bytes**, official publisher MD5 `66e54ddaa6b50d81a0166766d6f7a5d4`. These are metadata facts; there is **no verified local source SHA256 or parsed cohort count**.
- [Actions run 39922875284](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39922875284) retrieved and validated this metadata but download returned HTTP **504**.
- [Actions run 39923219365](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39923219365) tried official publisher `ndownloader.figshare.com/files/24025820` and `figshare.com/ndownloader/files/24025820`; both responded HTTP **504**.
- [Actions run 39923515491](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39923515491) additionally tried the [documented Figshare API download route](https://info.figshare.com/user-guide/how-to-use-the-figshare-api/) `api.figshare.com/v2/file/download/24025820`, also HTTP **504**. Only checksum-confirmed original bytes may pass. These failures are recorded and the automatic publisher retry workflow has been switched to manual dispatch to avoid repeatedly wasting CI resources.
- **Do not confuse the verified *metadata* MD5 with a measured SHA256 of downloaded source bytes.** The published longest case of 2973 is not enough to accept long-history experiment. `SOURCE_ACCESS=BLOCKED`, `ME070_NEW_NEURAL_FITS=0`.

**Fallback original BPI2012 actual case eligibility:** [real checksum-verified qualification in Actions run 39923421549](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39923421549) found FIT=174 / DEV=15 / exposed TEST=61 original cases with a recorded next event after >=64 observations; FIT=4 / DEV=0 / TEST=0 after >=128 and **zero original cases** after >=256. Exact source SHA256 `5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3`. [Machine-readable aggregate receipt](EXPOSED_BPI2012_HISTORY_ELIGIBILITY.json) is Git-owned. BPI2012 cannot qualify ME070's independent 256-event memory comparison; its already exposed TEST is diagnostic only. The additional FIT-only n-gram practical reference study is distinct and must have original fitted state committed before scoring; it cannot establish TAC-specific memory benefits.
