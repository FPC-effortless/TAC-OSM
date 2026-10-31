# LANGCAP105-EVAL01 — completed frozen DEV evaluation

All eighteen original LANGCAP105 checkpoints completed the post-training protocol. Every model/seed reduced DEV NLL relative to its own original initialization. Zero optimizer updates, checkpoint selection or protected TEST access occurred.

Each DEV measurement covers the same 4,646 complete 128-byte windows: 594,688 target bytes from eighteen original CPython files in seventeen held-out module/package groups. Means below include all three seeds. Lower NLL is better; accuracy is teacher-forced next-byte accuracy, not executable-program success.

| Scale | Frozen model | DEV NLL | DEV byte accuracy | Final-model actually-seen FIT NLL | Initial DEV NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| 1M | TAC | 1.741724 | 54.056% | 1.629568 | 5.535716 |
| 1M | Transformer | 2.275448 | 39.066% | 2.192945 | 5.710351 |
| 1M | GRU | 1.727076 | 55.491% | 1.634320 | 5.533460 |
| 10M | TAC | 1.617619 | 57.327% | 1.503027 | 5.532631 |
| 10M | Transformer | 2.082963 | 43.740% | 1.973925 | 5.769703 |
| 10M | GRU | 1.346757 | 63.603% | 1.235382 | 5.551685 |

The 1M TAC–GRU NLL difference is +0.014648, with descriptive paired seed/group interval [-0.001426, +0.030646]; this cohort does not resolve that small difference. At 10M, TAC–GRU is +0.270862 [0.251127, 0.291435], indicating lower recorded DEV loss for those GRU checkpoints. TAC–Transformer differences are negative at both scales. These intervals describe this one source project and three-seed pilot; they do not qualify a general architecture winner.

Fixed final-model DEV-minus-seen-FIT gaps range from approximately 0.083 to 0.115 NLL across the six configurations. This is distinct from changing-model training-tail loss. Source-group composition can contribute to the gap; it is not a unique diagnosis of memorization.

The unmatched FIT-exposure byte-frequency control recorded about 3.1657 DEV NLL / 28.665% byte accuracy. All frozen neural checkpoints recorded lower NLL. It remains a diagnostic lower bound; Transformer and GRU are the primary neural references.

Training full-FLOPs/tuning qualification, independent source/outcome/ACL assessment, near-duplicate audit and independent/out-of-repository TEST remain open. This is exposed same-project DEV held out from the original optimizer. The task uses 128-byte chunks and does not test long-term state carry, reasoning, code execution or agent work.

Preregistration: `48c41e27a59659bac03ba59547646fc26aa9cc3d` (protocol SHA256 `79d82775eeb346068bb8e38b2209f4976d2ed36920a643baa0f407581917447c`), before first DEV model scoring. Frozen evaluator source: `9866092ff7e44baf7de26a3cf0c277abd7316fe3`.

[Original successful scoring run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41238556668) · [Full summary JSON](metrics/SUMMARY.json) · [Original immutable evidence ZIP](ORIGINAL_EVIDENCE.zip) · [Artifact receipt](ARTIFACT_RECEIPT.json) · [Reauthenticated publisher receipt](SOURCE_RECEIPT.json)

`ORIGINAL_EVIDENCE.zip` retains all eighteen original RESULT files, completed mode reports, registration snapshot, run/artifact provenance and collector output byte-for-byte. Its SHA256 is `f5e8e460ba0c04bbd4c7d9eaea7f52b5e6fc2118a561ffa8df2d151280e36b5e`. The source corpus bytes are not duplicated in this results directory.

`validate.py` extracts the SHA-locked evidence safely, verifies every checkpoint/cohort, recomputes summaries and bootstrap intervals, compares them to the original and published values, and optionally checks original run success/head/artifact digest through read-only GitHub APIs. It loads no weights and performs no model predictions.

```bash
python research/langcap105_eval01_results/validate.py
```

The original evaluation workflow is triggered only by pushes changing evaluation-code paths. This separate publication directory and its no-prediction validation workflow do not rescore DEV or retrain models. Original training protocols, weights, earlier results and scientific gates remain unchanged.
