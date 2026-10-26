# Enterprise prospective source candidate — October 27, 2026

**Scientific tier: ORIGINAL_PUBLIC_SOURCE_ACQUISITION, NOT TRAINING-QUALIFIED.** This document is an addendum to the [blocked Enterprise source and experiment protocol](enterprise-forecast-source-preflight-2026-10-26/PROTOCOL.md), not its activation. The existing [PROTOCOL.json](enterprise-forecast-source-preflight-2026-10-26/PROTOCOL.json) null thresholds and `activated_training_registration=false`, `fit_authorized=false`, `REPORT=NOT_ACQUIRED` remain unchanged.

## Actual source now acquired

- On **2026-10-26T23:21:54.161638Z**, the public NYC Open Data `erm2-nwe9` source returned **100 cases currently reporting `Open`** in a strictly bounded acquisition-time query requesting only `unique_key`, `created_date`, `agency`, `complaint_type`, and `status`. There is no requested `closed_date`, full address, geo coordinates, or case narrative.
- [GitHub Actions acquisition #40419202170](https://github.com/FPC-effortless/veritas/actions/runs/40419202170) executed the real publisher GET and saved [source response and cohort manifest, artifact #12697711013](https://github.com/FPC-effortless/veritas/actions/runs/40419202170/artifacts/12697711013) with up-to-90-day retention. Initial original response SHA-256: `751a0e826e29067f59935491f82a0378a45c15f9d4e76e3e72fb2ac432d7f568`; immutable sorted ID-list SHA-256: `b2f3c8e8a65e9672a146560eb4f152b948bad84782355c31927347eb7f794831`.
- Source code for immutable cohort and exact original ID follow-up is in [Veritas/Forge draft PR #423](https://github.com/FPC-effortless/veritas/pull/423). The code enforces a technical **seven-day minimum elapsed period before any fixed-ID follow-up request**, rejects injected/missing/duplicate memberships and tampered receipts, and **does not** impute missing cases as negative or create physical closure labels. Its current CI covers constructed malicious cases, not the future source status of these cases.
- **Earliest permitted follow-up by code:** 2026-11-02T23:21:54.161638Z (November 3 at 00:21:54 WAT). This is a no-early-source-read rule, **not** permission to automatically run follow-ups, fix a seven-day target observation tolerance, register a training dataset or open protected evaluation.
- Other candidate captures that happened during source-code evolution have different acquisition times and IDs. **Only the above initial named hash is this anchor; do not substitute a later newest-100 capture.** Repeated code commits now run tests only. Further acquisitions use manual dispatch, with no scheduler enabled.

## Historical archive recovery remains separate

NYC's official [Open Data 101](https://www.nyc.gov/opendata/get-started/user-guides/open-data-101) states that archives download **whole files**, not a selected safe subset. The archive version 187190 is indicated as done in verified metadata and has a publisher-defined ~3.48-billion-byte `refSize`, but case rows have not been acquired. Blind download could expose unnecessary address/narrative information; no unfiltered multi-gigabyte export should be used to claim historic field visibility. A single archived revision also cannot reconstruct enough longitudinal source snapshots for an as-of study by itself.

## Distinctions required by protocol

| Question | Verified / remaining state |
| --- | --- |
| Authentic public source at first observation | **Acquired**: 100 publisher-reported open cases, original bytes and hash, acquisition cutoff |
| Case IDs selected independently of later outcomes | **Acquired as a frozen candidate membership**; future case-level follow-up remains unobserved |
| Actual prior historical knowledge and column revision clock | **UNQUALIFIED**; source `created_date` is not the acquisition clock |
| Later source case status | **NOT OBSERVED** for the named seven-day horizon yet |
| Real publisher status as actual physical case resolution | **NOT INDEPENDENTLY VERIFIED** |
| Missingness, status revisions, reopenings, follow-up timing | **NOT YET MEASURED**; unobserved is UNKNOWN, not Closed/Open truth |
| Lawful rights, privacy, AI-use and archival preservation | **PENDING independent source-specific review** |
| Exact FIT/DEV/REPORT cohorts and embargo | **NOT FROZEN**; 100 raw candidates do not equal 100 scored independent REPORT cases |
| Case-level Brier, uncertainty and neural superiority | **NOT SCORED**; no trained checkpoints, no seed runs |
| Veritas independent outcome verifier | **NOT WIRED/NOT ATTESTED**; Forge source inspection is not Veritas scientific qualification |
| TAC original checkpoint clean-clone replay | **REPO_REPLAY=BLOCKED** |

## Research decision

Keep existing Enterprise six-family architecture; **do not start training** on this candidate until the source and scientific gates are satisfied and a separate signed prospective activation addendum freezes the missing PROTOCOL.json fields. Future observations may be useful as acquisition-time measurements for a *new prospective task*, but that is distinct from a retrospective comparison of archived field history. The current 100 candidates are an engineering/data-provenance milestone, not a trained world-model capability claim.

**Next actionable independent gate:** retain the original artifact beyond its expiration only through source-rights-approved storage, authorize a future *same-ID* publisher follow-up with actual capture-clock evidence, then audit completeness and legitimate label semantics separately. Independently verify full CI/review for the new Forge PR and never infer merge or release eligibility from an acquisition result.
