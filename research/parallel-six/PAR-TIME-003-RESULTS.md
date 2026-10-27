# PAR-TIME-002 / PAR-TIME-003 — genuine source preflight and original five-model development

**Scientific status: UNQUALIFIED; NO demonstrated prediction win and NO
matched learned-architecture superiority.** Source original UCI PRSA 2017
SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.

## PAR-TIME-002: Original physically recorded no-fit CPU source preflight

First true publisher-source profile: [GitHub Actions #40491271033](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40491271033)
PASS, genuine preflight artifact #12727918059.
Revised source-gap-stratified profile: [Actions #40491602789](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40491602789)
PASS. No training or original model checkpoint was created by either profile.

- 13,709 eligible March-April 2014 source training cases; 235
  genuine but *previously examined* May-June 2014 development cases.
- Source source-legal history: PM2.5 observed and >=12 of 13 physical
  channels observed. Among registered genuine training source cases,
  **513 histories contain a >1-hour authentic prior observation gap**,
  the other 13,196 are all unit-duration. The revised 4-case diagnostic
  samples two of each source-trace pattern, physical observed gaps 1–2h.
  Selection uses only prior observed source chronology, not targets.
- With exact shared weights and temporal projection initialized at zero,
  the time-aware and time-blind model outputs **agree exactly** on the
  authenticated batch and have identical genuine masked loss.
- Both models were run through forward, backward, clipped gradients and
  lr=0 AdamW math with NO weight updates.
- Active learned parameters: parent 96,303; temporal 96,639 (0.348%
  discrepancy). Supported Torch profiler matmul FLOP counts parent
  52,792,034; temporal 52,601,186 on original batch4 x history12.
  Supported-subset FLOP relative difference **0.3615%**.
- **117 unsupported operator-shape signature differences** remain;
  Torch profiler did not close all nonlinear/AD/optimizer mathematical
  operations. Therefore *complete training-FLOP parity is NOT proven*,
  and a fair two-neural-architecture performance claim remains BLOCKED.
  This cannot be excused by the supported matmul <1% figure alone.

## PAR-TIME-003: Actual genuine-source fitted time-aware model

Registered five seeds 271028–271032; exactly64 AdamW source updates ×batch4
per model, active96,639 parameters and source 12-event history. All 5
model+optimizer+RNG original checkpoints and individual sample orders
were SHA-frozen in repository BEFORE May-June 2014 source development
score results were written. Strong real nonlearned persistence was scored
on exactly the same 235 timestamp cases and 3,026 genuine physically
observed next-hour channels. There are 234 physically present next-hour
PM2.5 labels.

**Five-seed genuine source development outcomes (lower MAE is better):**

| Model seed | 13-channel normalized MAE | Raw PM2.5 µg/m³ MAE |
|---|---:|---:|
| 271028 | 0.24181160 | 9.808938 |
| 271029 | 0.24160868 | 9.809679 |
| 271030 | 0.24192917 | 9.824318 |
| 271031 | 0.24186429 | 9.809071 |
| 271032 | 0.24160515 | 9.801477 |
| Five-seed TAC mean | **0.24176378** | **9.810697** |
| Strong persistence | **0.24163699** | **9.808120** |

- Mean original normalized MAE is 0.0525% WORSE than persistence;
  two of five seeds beat persistence on the normalized aggregate.
- Mean original raw PM2.5 MAE is 0.0263% WORSE; one of five seeds
  beats persistence on that sensor channel.
- This is a small directional loss, but genuine negative development
  evidence; do not select the one favorable seed, alter scores, or
  claim real superiority. The time-aware model was trained in isolation:
  no same-budget time-blind, GRU, SSM, or Transformer was fitted here.
- The model emits point forecasts only in this task. No calibrated
  probability distribution, safety/policy inference or time-specific
  memory retrieval is qualified.
- This genuine 2014 period was already inspected by LATFUSE research:
  it is historical development ONLY, not an untouched independent TEST.
  The source reader stopped at 2014, before 2015/2016–17 numeric data.

### Interpretation and next falsifiable question

This actual 64-step result is not evidence that chronological position and
age encoding is harmful in general. The temporal projection initializes
to zero and may still be weakly learned. The parent core's prediction is
bounded near observed input by a small source-normalized delta scale.
We have not attributed the result to temporal features versus memory
interference, model initialization, optimization steps or source seasonal
distribution. Need **read-only frozen checkpoint FIT-only time-feature
lesions** first, then registered matched total-FLOP trained comparisons
on a genuinely independent source/site and longer source histories.
Do not repeatedly tune to previously viewed 2014 dev results.

Artifacts:
- [Real development JSON](https://github.com/FPC-effortless/TAC-OSM/blob/research/partime003-real-single-model-fit-2026-10-27/research/replay-bundles/PAR-TIME-003/DEV_RESULTS.json)
- [Five original checkpoint SHA + optimizer training receipt](https://github.com/FPC-effortless/TAC-OSM/blob/research/partime003-real-single-model-fit-2026-10-27/research/replay-bundles/PAR-TIME-003/TRAIN_RECEIPT.json)
- [Training sample/DEV timestamp plan committed BEFORE first optimizer update](https://github.com/FPC-effortless/TAC-OSM/blob/research/partime003-real-single-model-fit-2026-10-27/research/parallel-six/PAR-TIME-003-FROZEN-PLAN.json)
- Read-only cold replay CI in .github/workflows/partime003-original-replay.yml.
  Do not claim a pass until its independent run completes successfully.
