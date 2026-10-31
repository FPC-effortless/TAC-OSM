# LANGCAP105 trained-checkpoint completion audit

This is a read-only supplement to [training PR #683](https://github.com/FPC-effortless/TAC-OSM/pull/683).
The original execution is [run 41229365510](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41229365510),
commit `ba9220163c777a7d9b6f765378810408baf00b1f`.
The owner authorized this exploratory FIT-only program in chat on 2026-10-31.

The fixed cohort contains TAC associative stack, causal Transformer and stacked
GRU at 1M and 10M parameters, seeds 10301–10303. Each fit has exactly 2,560 AdamW
updates, batch 4, context 128, and 1,310,720 byte presentations. Workers receive
only the 560 authenticated original CPython FIT files. No DEV or TEST scoring
or independent scientific qualification is implied by completing training.

## Verification

`checkpoints.py` checks each original artifact against the frozen protocol and
source code hashes. It verifies the checkpoint SHA, exact deployed count,
finite FP32 weights, changes from seed initialization, every AdamW moment and
update counter, RNG restoration, full loss-series and exposure receipts, actual
prediction serialization, bounded future-byte invariance, entity isolation and
all parameter gradients. It also reproduces the first original loss from the
same source order and seeded initialization, without an optimizer update.

Run against an exact checkout of the original training commit, the downloaded
`LANGCAP105-authentic-FIT-only-inputs` artifact, and the original full-fit artifacts:

```bash
python research/langcap105_verification/checkpoints.py \
  --source-root ORIGINAL_CHECKOUT --fits DOWNLOADED_FITS \
  --fit-inputs DOWNLOADED_FIT_INPUTS --output CHECKPOINT_AUDIT.json
```

Use the original `torch==2.15.0+cpu` and NumPy. Default mode rejects a missing
fit. `--allow-partial` labels a partial audit explicitly and lists all missing
models; it cannot report an incomplete cohort as completed.

## Preserved limits

- Original code repeatedly overwrites `LATEST.pt`, then deletes it after
  success. Final model, AdamW and CPU RNG are retained; all intermediate
  milestone snapshots are **not** retained. The original protocol and worker
  have not been edited to conceal this discrepancy.
- This is source/weights/record verification, not independent external outcome
  attestation. First-loss and serialization replay do not establish exact
  full-trajectory optimizer replay.
- Per-step FIT loss curves are optimization diagnostics. No heldout quality,
  architecture superiority, full training-FLOPs or general intelligence claim
  follows. The 2,560-update owner pilot is distinct from the larger proposed
  LANGCAP103 training milestones.
- Original GitHub full-fit artifacts have 90-day retention. Source-control
  evidence files contain hashes and audit results, not the model binaries.

## Current evidence

The first nine 1M fits passed direct checkpoint and first-loss replay checks.
`results/1M_CHECKPOINT_REPLAY_AUDIT.json` is intentionally a partial report.
The original training run continues the nine 10M fits. A final all-18 audit is
required before declaring this cohort complete.

`results/AUDITOR_NEGATIVE_CONTROLS.json` records six deliberately corrupted
temporary copies rejected by this auditor: nonfinite weights, incorrect AdamW
step, nonfinite loss, false qualification, incorrect exposure and fresh initial
weights substituted for trained parameters. These are audit falsifiers, not
additional trained fits or scientific performance evidence.
