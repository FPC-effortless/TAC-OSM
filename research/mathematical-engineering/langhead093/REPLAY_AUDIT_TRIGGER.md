# LANGHEAD093 replay-only independent checkout verification trigger

Original CPU training matrix 25/25 completed in Actions run 41010339562.
That run's archival job verified all original 25 checkpoints and per-token new
developer source scores, then committed their original Git-backed artifacts at
commit 1e41dc12537d5581c66441e1df65212974be0fdc.

The bot-originated follow-up CI was ACTION_REQUIRED. This new source-only
human-connector commit triggers a fresh PR event and its replay-only phase.
Do not fit again: the presence of TRAINED_NEW_DEV_RESULTS.json makes the
workflow preflight select replay rather than fit.

Preserve the registered 3 CPython library source owners, negative arms,
original checkpoint bytes and the failed first preflight. Do not promote:
FLOPs not matched; all claims internal, two new independent-publisher
cohorts not established; protected TEST closed; full PNDS math OPEN.
