# DEV-025 execution status: blocked at exact diagonal integrity

**Incomplete scientific experiment. No DEV-025 performance or attribution
conclusion.** The first frozen cross-evaluation run failed its registered exact
diagonal checks. The independent full-grid audit was skipped. Partial outputs
must not be used to decide whether the DEV-024 long-regime discrepancy follows
trained models or sampled evaluation cases.

[Run38892429482](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38892429482)
used source `afcec5dbc03a6ec784216176bcae31375ca840fe`. The integrity job passed
all1,104 tests. DEV-023 short completed both grids and separate-process replay.
DEV-023 long, DEV-024 short and DEV-024 long stopped at diagonal mismatch.
The complete40-cell audit did not run; no complete audited artifact exists.
The original protocol and implementation remain unchanged.

## Diagnosed numerical mismatch

Local deterministic CPU reconstruction of DEV-024 long seed59 reproduced the
failure over288 H512 rows. Checkpoint and final parameter hashes matched the
registered manifest. Across26,116 differing floating leaves, maximum absolute
difference was1.1920928955078125e-6 in a logit. Every discrete field matched.
All derived metrics matched except primary Brier, differing by about5.13e-9.
Actions, service choices, physical costs and outcomes were unchanged in this
inspected cell. This establishes numerical replay sensitivity for that cell;
it does not identify a CPU/backend cause or characterize the whole cohort.

The engineering record is [DEV-025-REPLAY-DIAGNOSTIC.json](DEV-025-REPLAY-DIAGNOSTIC.json).
Reproduce with `scripts/diagnose_dev025_diagonal.py` and the original complete
DEV-024 archive. The record binds the exact input report, checkpoint, final
parameter hash, diagnostic source and runtime versions. Additional diagnosed
cells are saved separately where completed; no off-diagonal scientific
aggregate has been opened or claimed.

## Preserved failed-run evidence

| Raw artifact | ID | ZIP SHA256 |
|---|---:|---|
| DEV023 short |12042945306|45c33f2f11a8bef85f0b4b8f308b18f1f34cc7b90e5244ad0f2489ce50e84009|
| DEV024 long |12042935098|2a0472be153dfbbb2d528dbc357e5566c539b87f9b7673c08777700516cb4e60|
| DEV024 short |12042810691|22483788f72feb923046c134fa78295ea7322ecf259cab60790ba36d6438d7e8|
| DEV023 long |12042766065|ac413eb2696606757508583b64731d92037507efe106f78059c3eecfbc4703d8|

These are partial raw artifacts, not independently audited scientific results.
DEV-024's earlier complete audit remains valid and is a separate run.

## Next bounded experiment

[DEV-026 proposal](DEV-026-NUMERIC-REPLAY-PROPOSAL.md) defines a prospective
numerical comparator for archived diagonals only. Prediction, inspection,
service and realized-cost comparisons stay exact, as do new separate-process
replays. Five focused tests cover rounding, changed predictions, changed cost,
excess error and nonfinite evidence. The proposed comparator accepts the known
seed59 numerical case with exact decisions and physical costs. It has not been
used to declare DEV-025 successful. All20 original diagonals must be checked
before a successor's off-diagonal summary is opened.

The initial about30minute research batch stops with DEV-024 audited, DEV-025
blocked, and a tested prospective engineering fix ready for review. The ledger
preserves both findings and the failed attempt. No further matrix is launched.
Native TAC overall gates remain failed. Research branches stay draft/unmerged.

Final validation: all1,109 regression tests passed, plus lint and test-count
provenance. Two additional local diagonal reconstructions were stopped at the
batch boundary before producing records; they supply no evidence. Only the
completed seed59 diagnostic is characterized above.
