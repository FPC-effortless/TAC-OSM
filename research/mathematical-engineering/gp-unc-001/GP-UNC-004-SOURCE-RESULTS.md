# GP-UNC-004: original-source event-boundary audit

**Date:** 2026-10-27. **Evidence:** authentic historical UCI374 original-source read-only software integration with a pre-existing frozen classical checkpoint, plus constructed negative controls. **NOT** GP fitting, new architecture prediction evidence, independent TEST, or safety qualification.

## Exact inputs and replay identity

- Publisher: [UCI Appliances Energy Prediction (DOI 10.24432/C5VC8G)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), CC BY 4.0.
- Original publisher ZIP SHA256: `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a`.
- Separate source/model repository snapshot: [TAC commit `66cc1ffc1935bad67118962091a016e04c2eff81`](https://github.com/FPC-effortless/TAC-OSM/commit/66cc1ffc1935bad67118962091a016e04c2eff81), checked out by immutable hash in CI.
- Original already-trained CLASSICAL ridge checkpoint JSON SHA256: `cb756a734b026602bb13f6af11b08711e2029f78aad1e95bb469d55131bc6c82`. This is **not** a 100k TAC-OSM neural checkpoint; excluded from architectural parity claims.
- Source-aligned 10-minute row-clock indices: issues `[9867, 9931)`, one-hour-ahead labels `[9873, 9937)`. Both are in the previously exposed CAL partition; the exact existing source reader parses FIT/CAL/DEV to index 15788 and stops **before original protected TEST**. No original-source labels are released into issue-event payloads. The historical row clock is NOT an independently verified operational availability timestamp.

## Executed source and code checks

| Check | Measured outcome |
| --- | --- |
| Original source bytes pinned and verified | PASS |
| Original checkpoint unchanged + reference SHA verified | PASS |
| Frozen original source-aligned forecast requests | 64/64 preserved |
| Separate outcome-arrival events | 64/64 processed after issuance |
| Source-anchored future-target poison invariance | 64/64 passed |
| Full GP residual updates from authentic labels | **0** (disabled by design) |
| Protected original TEST numeric rows parsed | **0** |
| Promoted scientific/prediction/calibration claim | **NONE** |

CI: [original-source audit run 40485229043](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40485229043), artifact `gp-unc-004-original-uci-exposed-cal-no-fit`, result `PASS_ORIGINAL_SOURCE_READ_ONLY_EVENT_WIRING`. Original ZIP and saved checkpoint SHA values in the artifact exactly agree with the pinned identifiers. Original source-from-forecast data producer was re-evaluated with only the **future outcome** changed by a constructed perturbation, ensuring its upstream request and frozen model output remain identical. This is an anti-leakage diagnostic, **not** genuine capability data from a new independent cohort.

The separate streaming boundary `src/tac_osm/gp_event_boundary_nofit.py` also has 11 constructed tests of truth-exclusion at issue, time order, late label handling, same-clock exclusion, per-entity isolation, fixed checkpoint IDs and explicit queue/ID bounds. The streaming event-based interface does not need access to future truth to issue its forecasts. In this offline audit the original publisher archive is nevertheless pre-parsed into memory by the source loader: do **not** infer source IO-level nonaccess to future data or prospective deployment readiness.

## Status for real-world learning experiment

**BLOCKED / NOT RUN.** The GP posterior is deliberately untrained here; its mean remains the zero-prior correction. This audit demonstrates real original data→frozen base inference→truth-isolated forecast event plumbing only. No GP calibration, sharpness, NLL, MAE, business decision cost, physical control, or six-family model improvement can be inferred.

Before the FIRST qualified real-data GP experiment:
1. Obtain an independently authenticated, lawful source with genuine issue/observed/available-at evidence, owner/role separation and signed prefit verifier receipt.
2. Freeze source and checkpoint SHA, bona-fide real FIT/CAL/DEV/TEST split, six-family task targets and permitted adaptive-vs-frozen update policies. Previously exposed UCI CAL/DEV must not be recycled as independent TEST.
3. Implement source-specific output/head contracts and training models with genuinely useful 95k–105k parameters (matched within 1%), real-data exposure and matched training/tuning FLOPs. Classical pre-existing ridge serves a clearly identified diagnostic comparator, not padded neural parity.
4. Verify source-integrity negative controls at actual producer/encoder, full model checkpoint replay, independent scoring, calibration and task utility. Preserve prior LATFUSE019 negative real-source findings.
5. Require an independent final scorer/evaluator receipt, not a self-attested metrics artifact. Keep protected future TEST sealed until registered.

Related work: [TAC-OSM draft PR #491](https://github.com/FPC-effortless/TAC-OSM/pull/491); [PNDS proof draft #45](https://github.com/FPC-effortless/pnds-lean/pull/45); [experiment qualification tracker #492](https://github.com/FPC-effortless/TAC-OSM/issues/492).
