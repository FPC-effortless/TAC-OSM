# Original-source H64 historical-memory audit — internal result

**Measured on GitHub CPU; not independently qualified. The registered benefit gate failed for both seeds.**

[Successful source run and fresh-process replay](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40471346707) at `b4ccc28e488e832315b5e2d89dfa9e755b762b44`. Original UCI501 PRSA archive, CC-BY-4.0, 48 frozen May–June2014 FIT cases (24 matched pairs), same cases for seeds80701/80702. Four original checkpoints each retain 99,390 parameters; no optimizer, model or checkpoint changes. No protected TEST or GPU use.

## Historical-event Brier loss (lower is better)

| Predictor | Seed 80701 | Seed 80702 |
| --- | ---: | ---: |
| Native associative historical read | 0.370006 | 0.274253 |
| Frozen fast/slow historical-readout control | 0.350755 | 0.317005 |
| Same-checkpoint zero associative read (OOD lesion) | 0.251863 | 0.252963 |
| Balanced-cohort constant prior | 0.250000 | 0.250000 |
| Causal current-last threshold | 0.520833 | 0.520833 |

Native read loses to the constant prior by 48.00% / 9.70% relative Brier and to zero-read by 46.91% / 8.42%. Station-bootstrap loss-gain intervals versus both controls are entirely negative in both seeds. Against the frozen fast/slow readout, native is 5.49% worse in seed80701 and 13.49% better in seed80702; that mixed result cannot establish a consistent memory advantage. Beating the weak current-last threshold alone does not pass the preregistered all-baseline gate.

## Original +6h all-channel forecast normalized MAE

| Predictor | Seed 80701 | Seed 80702 |
| --- | ---: | ---: |
| Native | 0.681714 | 0.665292 |
| Frozen comparator | 0.676491 | 0.716426 |
| Causal persistence | 0.538771 | 0.538771 |

Both neural forecasts lose to persistence on this cohort. These are aggregate normalized errors, not per-channel calibration or operational utility.

## Trace and replay evidence

- 8,448 actual associative writer-step traces across the two seeds; captured target and selected control keys resolve to actual same-owner nonzero writes. All selected controls lie strictly between target and query.
- Maximum observed write-update reconstruction and native read-to-logit reconstruction errors: **0.0 FP32** in both seeds. This is local engineering correspondence, not a whole-model proof.
- Actual normalization/last/age prefix checks passed; raw labels and future feature rows were absent from both inference mappings. All original checkpoint and state_dict digests remained unchanged.
- Fresh-process output replay was byte-identical. Full result SHA256: `99e67e0e5b2391f6f6d1326dcc22626b640b2c66b9ed2f41b64afcea91988163`. Full trace archive: [Actions artifact](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40471346707/artifacts/12719300524); compact per-case identity/outcome receipts and summary are committed here.
- Historical original H64 native/nonassociative Brier receipts agree within 1e-6; historical reports and checkpoints are unchanged.

## Limits and preserved failure

The first CPU execution [40470836544](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40470836544) stopped on a CLI package-import bug before source observation parsing/checkpoint deserialization. `FAILED_RUN_40470836544.json` preserves it. The package-path fix and regression test changed no cases, metrics, control selection or thresholds.

The cohort is small, label-selected and exposed FIT, not an independent holdout. Whole-training FLOP parity was not established historically, so there is no architectural-superiority claim. The zero-read control is an OOD intervention. Privileged source write-keys themselves encode past observations; neither their scores nor cosine geometry establish uniquely stored content. A failed native linear readout does not prove memory contains no recoverable information.

Complete source-file-reload future poisoning remains **INCONCLUSIVE_NOT_IMPLEMENTED**. Whole-model PNDS Lean/Python/IEEE/AD equivalence remains **OPEN**. The measured gate failure motivates a separately preregistered legal-query/content-readout investigation; it does not authorize retraining, promotion or independent qualification.

Attribution: Chen, S. (2017), *Beijing Multi-Site Air Quality*, UCI Machine Learning Repository, [DOI 10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY 4.0.
