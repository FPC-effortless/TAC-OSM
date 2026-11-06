# DECISION-SYS1-003 — bounded source acquisition and SSM reference qualification

Owner-requested continuation of draft #719 at a457fc24f179c7efbf2a3b357237794eb3af84d0. Draft, no fitting and no training admission. Original models/checkpoints, source auditor, failed studies and protected holdouts remain unchanged.

## Frozen source scope

Obtain ONLY original v3 train_operational_readouts.csv and train_tte.csv through exact publisher individual-file HTTPS routes. Redirects are constrained to the same files on api.researchdata.se/doris.snd.se. No ZIP, validation, test, credentials, raw-data upload, model scoring or optimization. Readouts must match historical SHA256 e01cb0bd87dfab4c9dbad215d51d81282fc0d413be96d6f819ea872fb7a3c715 before labels are requested. Outcome hash is a new receipt, not independent attestation. Run the unchanged #719 audit; report failures without changing thresholds. Existing TRAIN exposure prevents untouched-test claims.

## Resource bounds

CPU/source-only GitHub Actions, one job, 25-minute timeout; one attempt at each of two allowed publisher routes per file; 1.3 GB readout and 10 MB outcome caps, 900-second per-route acquisition cap. No GPU quota or paid provider. Only compact receipts exported. This source audit is NOT a model training/evaluation registration or permission to fit.

## Parallel engineering scope

Prepare a faithful, source-pinned Mamba-2 mathematical CPU reference and explicit official-backend qualification hook. Do not claim that a CPU reference is the optimized official kernel or a qualified speed baseline. Match actual parameters without dummy weights, report state capacity separately, and verify full-sequence/streaming outputs and gradients. Original native/GRU/Transformer remain unchanged. Full PNDS correspondence, official CUDA kernel comparison, independent review and fair training registration remain OPEN.

## Acceptance

Actual authenticated source audit receipt OR explicit endpoint failure; no proxy/mirror substitution. Executable SSM reference evidence must separately state implemented, numerically tested and official-kernel-qualified. Zero optimizer updates and no real-data model predictions. Never convert fixtures, counts or green CI into intelligence claims. Keep draft and unmerged.
