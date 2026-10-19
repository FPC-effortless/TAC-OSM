# SEQ010 — real SCANIA recurrent pilot results

**Status: COMPLETE original REAL-data train-internal validation pilot; registered 2-seed promotion hypothesis FAILED.**
Original successful, fixed-code GitHub Actions run:
https://github.com/FPC-effortless/TAC-OSM/actions/runs/39317978344
Original successful source commit: 1ef39eb80fa64b1de16f73c1effc62e860425a0d
Original artifact: seq010-original-real-recurrent-results, ID **12218742162**,
contains all four trained weights, original results.json and source.json.
Original failed preflight run **39317773019** remains preserved; it exposed
a nested-preprocessing-return software mistake, fixed before real fitting.
Source preflight and complete run both passed on the corrected commit.

## Real operational evidence

Authentic SCANIA Component X dataset v3, DOI 10.5878/bnh5-ka77,
downloaded from official publisher ZIP at
https://api.researchdata.se/dataset/2024-34/3/file/zip.
The original archive ZIP and selected TRAIN file were SHA256 hashed in
original run manifests. Publisher VALIDATION/TEST body and all published
repair outcome labels were NOT opened by this experiment.

- Original source TRAIN vehicles: **23,550**.
- Original authentic TRAIN readout records: **1,122,452**.
- Fixed SHA256 identity-partition TRAIN vehicle cohort: **18,758**.
- Fixed internal vehicle-heldout cohort: **4,792**.
- Actual TRAIN next-observation windows: **46,799**.
- One actual last transition per internal-heldout truck: **4,792**.
- Fixed first **24 real sensor columns** in publisher TRAIN schema.
- Every model evaluated identical observed predictor+target components.
  Minimum paired observed target count for each sensor: 4,752.
- Trained GRU / gated-readout architectures independently with
  exactly 200 updates at seeds 810101 and 810102.

## Original measured predictions

Lower signed-log MAE (macro over first 24 columns) is better.

| Model | Seed 810101 | Seed 810102 | Mean over seeds |
|---|---:|---:|---:|
| Persistence, no training | **0.04294285521** | **0.04294285521** | **0.04294285521** |
| Plain GRU residual (8,664 parameters) | 0.04571130720 | 0.04091200449 | 0.04331165585 |
| TAC-inspired gated readout (13,848 parameters) | 0.04601240724 | 0.04055574939 | 0.04328407832 |

- At seed 810101, GRU was **6.45% worse** and gated readout was
  **7.15% worse** than persistence.
- At seed 810102, GRU was **4.73% better** and gated readout was
  **5.56% better** than persistence.
- Mean recurrent accuracy **does not beat** persistence; the
  predefined gate that BOTH fixed seeds must beat persistence FAILED.
- Under seed 810101 the additional gate hurts relative to plain GRU;
  under seed 810102 it helps slightly. There is no reproducible
  independent gating benefit.
- The more parameter-rich gated readout is NOT native TAC-OSM,
  not a full CASM/CDL system, and not a plastic SSM or BDH.

Original checkpoint SHA256:
- gru-810101: d2da4a3b5bed7dd1d895c390f821a17cd63f3821a2d89f588805114a4f7f85a3
- tac_gated_readout-810101: bf6f4166f19f03ba9c4f5180024f564597b2911459e38822251283644807d0be
- gru-810102: 354ccd2142081e3e928cc6c30364b15036e78e30bf424eb8deb689970e970940
- tac_gated_readout-810102: f9ac097371b02e6792aee8d68b916636ffc18b078711ca456eaa1ba2c00d3070

## Scientific limitations

This is authentic operational *next-readout* prediction on an
internal vehicle disjoint slice of publisher TRAIN, NOT
publisher VALIDATION or publisher TEST performance. The original
SEQ009-S persistence/affine numbers were from different vehicles,
all sensor columns rather than the first 24, and a different
protocol; those numerical errors are not directly comparable.

Selection of the first 24 sensor columns may favor stable sensors
and does not encompass all equipment health indicators. This
diagonal last-value residual model predicts only the next
observed readout without knowing the future sampling interval.
Readout durations and counters may be anonymized/perturbed.

Only two training seeds were fit. No tuned LightGBM,
long-context baselines, learning-rate sweep, bootstrap-by-truck
confidence interval, SSM, independent native TAC runtime,
full-dataset sensor head, closed-loop action policy,
future-calendar generalization, dropout uncertainty or
censor-aware repair-risk outcome has yet been tested.

The checkpoint file and source results are present in the
original artifact. **Independent checkpoint restore/replay has
NOT yet been completed at the time of initial result archival.**
Any later completed audit must be appended with its exact run ID
and should not refit or reopen published TEST.

No unseen physical action effect, observed inspection cost,
economic benefit, causal utility, commercial readiness,
or architectural promotion is established.

## Decision: FAILED exploratory neural gate; next controls

Do not select only the successful seed 810102, tune on these
held-out vehicles without a new training-only tuning split,
or replace the unfavorable seed. Both seeds and all failures
must remain in the evidence ledger.

1. Independent recorded-only replay of all four original
   checkpoints/vehicle splits/observed component masks.
2. Separate real-data TRAIN-internal tuning experiments,
   with a reproducible training budget and stronger
   persistence-residual, multivariate/LightGBM,
   GRU and genuinely implemented native TAC model.
3. Separate censor-aware recorded-repair-risk experiment
   using real event times and censor limits, against
   logistic/LightGBM/survival and a correctly calibrated
   recurrent baseline.
4. Only after native recurrent model passes cost/uncertainty
   gates should learned entity binding, executable memory,
   or plastic SSM / BDH-style adaptations be evaluated,
   one component at a time.

**Architecture promotion: NONE.**
