# LANGCAP105 trained-checkpoint completion audit

This is a read-only supplement to [training PR #683](https://github.com/FPC-effortless/TAC-OSM/pull/683).
The original execution is [run 41229365510](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41229365510),
commit `ba9220163c777a7d9b6f765378810408baf00b1f`.
The owner authorized this exploratory FIT-only program in chat on 2026-10-31.

**Completed 2026-10-31:** All 18 original full-fit jobs and the original cohort
audit succeeded. All 18 final checkpoints passed the tensor/source-order audit,
and all 36 FP32/cast-FP64 trained-source/equation comparisons passed. Verification
performed zero optimizer updates. This completes the fixed exploratory pilot;
heldout capability, independent scientific qualification and architecture
superiority remain unestablished.

The fixed cohort contains TAC associative stack, causal Transformer and stacked
GRU at 1M and 10M parameters, seeds 10301–10303. Each fit has exactly 2,560 AdamW
updates, batch 4, context 128, and 1,310,720 byte presentations. Workers receive
only the 560 authenticated original CPython FIT files. No DEV or TEST scoring
or independent scientific qualification is implied by completing training.

| Scale | Model | Actual trainable/deployed parameters | Completed seeds | Updates per fit |
| --- | --- | ---: | ---: | ---: |
| 1M | TAC associative stack | 1,000,028 | 3/3 | 2,560 |
| 1M | Causal Transformer | 999,972 | 3/3 | 2,560 |
| 1M | Stacked GRU | 997,680 | 3/3 | 2,560 |
| 10M | TAC associative stack | 9,999,922 | 3/3 | 2,560 |
| 10M | Causal Transformer | 9,999,880 | 3/3 | 2,560 |
| 10M | Stacked GRU | 9,987,024 | 3/3 | 2,560 |

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

`trained_equations.py` additionally reuses the source-hashed verification
helpers from commit `0b21f0feed5d6f89bb49218d8ca320cca6220850` (LANGCAP103 /
PNDS PR #103). It first checks that the helpers instantiate byte-identical
original model code. For each actually trained checkpoint it compares full
logits, state and every parameter gradient against the separate PNDS Python
equations at FP32 and cast FP64 on two nine-byte original FIT prefixes. Native
serial/parallel and carried-state checks, GRU carry, future-byte poisoning and
a deliberately wrong readout falsifier are included. These finite tests do
not establish global floating-point equivalence or a full mathematical release.

The read-only Actions workflow downloads original run artifacts with an
`actions: read` token, records their run/commit/digests, independently repeats
the checkpoint and equation checks, and invokes the original cohort auditor
once all eighteen artifacts are available. A green partial-audit job remains
explicitly **partial**, never a completed-training claim.

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
  `package_models.py` packages the verified final weights, optimizer/RNG,
  complete per-step loss receipts and audit reports in four standalone ZIPs;
  `results/BUNDLE_RECEIPTS.json` records their exact sizes and SHA256s.

## Current evidence

The first nine 1M fits passed direct checkpoint and first-loss replay checks.
`results/1M_CHECKPOINT_REPLAY_AUDIT.json` is intentionally a partial report.
The three 10M TAC fits then completed and passed the same checks; preserve that
stage in `results/TWELVE_CHECKPOINT_AUDIT.json`. All twelve trained models also
passed 24 FP32/FP64 source/equation cases in `results/TWELVE_TRAINED_GRAPH_AUDIT.json`.
All three 10M Transformers then completed and passed the same checks.
`results/FIFTEEN_CHECKPOINT_AUDIT.json` and `results/FIFTEEN_TRAINED_GRAPH_AUDIT.json`
preserve the fifteen-model / thirty-case milestone. The remaining three 10M GRUs
then completed. The original all-18 cohort audit passed at 19:37:56 UTC.

Final evidence (earlier partial snapshots remain unchanged):

- `results/ORIGINAL_FULL_FIT_AUDIT.json`: original successful cohort auditor.
- `results/FINAL_CHECKPOINT_AUDIT.json`: all 18 deployed tensors, full AdamW/RNG,
  source-order first-loss replay, exposure and loaded-model integrity checks.
- `results/FINAL_TRAINED_GRAPH_AUDIT.json`: all 36 source/equation cases.
- `results/TRAINING_ARTIFACTS.json`: original immutable run/commit/artifact
  identities, download-archive hashes and retention dates.
- `results/BUNDLE_RECEIPTS.json`: four verified downloadable model bundles.

Total original full-fit work: 46,080 optimizer updates and 23,592,960 FIT byte
presentations across 18 models. Six separate eight-update smoke checks are
retained as smoke evidence and are not counted as completed full fits.

`results/AUDITOR_NEGATIVE_CONTROLS.json` records six deliberately corrupted
temporary copies rejected by this auditor: nonfinite weights, incorrect AdamW
step, nonfinite loss, false qualification, incorrect exposure and fresh initial
weights substituted for trained parameters. These are audit falsifiers, not
additional trained fits or scientific performance evidence.
