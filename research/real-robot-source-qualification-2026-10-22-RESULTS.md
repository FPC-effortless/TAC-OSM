# Original real-robot source audit — verified evidence (2026-10-22)

**This is a publisher-source qualification, not a model fitting experiment.**

- Original publisher: BodenAI RW-RL-HIL-Dataset, CC-BY-4.0.
- Immutable actual HuggingFace commit: 9290f5a96dc91ea58256e81d9ce2dd9621890b51.
- Observed physical task directory (the actual file name, NOT the shortened
  dataset-card alias): scoop_cat_litter_into_trash_can.
- Original episode metadata: **505 genuine, contiguous episode identifiers**.
- Original first sample: episode 0, **942 genuine measured robot frames**.
- Actual original frame-level state/action: 14 finite robot joint observations
  and 14 finite executed action components, correctly timestamped, with
  per-frame human-vs-policy control authority.
- Source parser and source integrity software checks: **6 passed**.
- Original real robot source eligibility job: 
  https://github.com/FPC-effortless/TAC-OSM/actions/runs/39702153827
- Original hashes/aggregate receipt (no raw trajectory frames):
  https://github.com/FPC-effortless/TAC-OSM/actions/runs/39702153827/artifacts/12393101743
- Original artifact SHA256: 4c8b2c0ff4672441d60a43b4d89f0f2a822ce40c963f39376b17d534f4f4a06a.
- Source original parquet file hashes and original info/episodes SHA256
  are retained INSIDE the checksum audit receipt, not recopied without opening
  the artifact.
- Training runs: **0**; new model optimizer steps: **0**; held-out model
  predictions: **0**; matched architecture superiority: **NOT QUALIFIED**.

The initial [source audit run 39702025867](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39702025867)
correctly stopped with HTTP 404 before reading any physical state. The
publisher's task-folder name actually lacks the r1lite_ prefix used in
descriptive task labels. The exact original folder was verified in the
publisher's file browser and the source-only path was corrected. Reproduced
successful run 39702153827 proves actual access to the three unchanged original
source files. The CI source revision is now frozen; no reward, cost,
label or dataset split thresholds were changed to obtain the source pass.

**Scientific interpretation:** the real physical-data format is sufficient
to begin designing a bona fide within-episode temporal observation/action
prediction study. It is NOT sufficient to establish safe autonomous decisions,
causal action improvement or verified task success. The publisher explicitly
notes its HIL release has no scalar rewards, done, or episode success labels;
a recorded intervention reflects human operator judgment, not a measured
economic failure cost. This audit covered metadata for one of nine task
folders and the entire state/action sequence for ONE episode only. Full
training/heldout cohort acquisition and data-leakage tests remain blocked.

**Mandatory next gate before fitting:** completely register the live source
cohort and independent untouched physical episodes; verify no future-label
access; construct a fully trained parameter-, exposure-, actual FLOP- and
training-budget-matched recurrent world-model comparator; instrument real pilot parity
on identical original physical batches; supply minimum five predeclared seeds;
obtain a reproducible verified FAIR_COMPARISON.json that passes preflight.
No model training or performance claims are authorized by this result.


> **User-specified scientific correction (2026-10-22):** Matching CPU execution
> time is **not** compulsory. Match trainable parameters, genuine dataset and
> exposures, training/tuning budgets and evaluation conditions. Record CPU time
> as an efficiency finding, not a fairness PASS/FAIL gate. The standing
> `research/REAL-WORLD-EVIDENCE-RULES.md` amendment controls future runs.
