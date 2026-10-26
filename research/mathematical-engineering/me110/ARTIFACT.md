# ME110 original repository-owned artifact

Original five runs: `research/replay-bundles/ME110/typed_roles/seed-110001` through
`seed-110005`. Fifteen actual trained model/optimizer/RNG checkpoints at1024,
2048and4096updates; final checkpoint is `final.pt`. Fixed monitoring arrays and
receipts at0/1024/2048/4096; step0 is a parent-derived initialization view, not an
additional trained checkpoint. Every original step loss/routing/guard record is
preserved in both checkpoints and TRAINING_HISTORY.json.

All original bytes were Gitfrozen in6389521before REPORT scoring. Original
REPORTquantiles/metrics/gates, postfit declaration and summary committedc8e25cc.
SHA256SUMS binds the payloads, excluding itself. Verify from the bundle directory
with `sha256sum -c SHA256SUMS`.

From repo root with pinned Torch2.14.1+cpu/Numpy2.3.5 and PYTHONPATH=src:.:

    python scripts/me110_typed_training.py replay
    python scripts/me110_replay_audit.py --replay
    python scripts/me110_summary.py --replay

These commands restore originals, never train or select checkpoints. They read
only the declared FIT/exposed2015REPORTsource via the checksum-pinned original
archive. Protected2016–2017numeric rows remain closed. Actual reportforecast
replay uses relative1e-5/absolute1e-6 per scalar; source-reference precision checks
use their separately recorded FP32/FP64 contracts. Original checkpoints and gates
are immutable; do not waive a failed criterion.

Independent detached checkoutc8e25cc restored398790REPORTquantile scalars and
115200FITmonitor scalars, all20views,10fitted source forward/backward checks and
all original scores/gates/summary. No new fits or optimizer steps. Full1171local
postfit testsPASS. GitHub original-final replay40377943112PASS. Current source,
artifact hash, empirical and full-neural mathematical qualification are distinct.
The complete predictive gate remainsFAIL0/5.
