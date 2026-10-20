# ME-007 — Falsification of TAC forecasting advantage against fairer GRU controls

**Registered:** 2026-10-20 before training or opening a new future scoring cohort. **Purpose:** stop adding complexity until the improved native TAC's limited single-compressor results survive strong conventional controls.

## Verified prior state
ME004 real data native TAC fast/slow 9,368 parameters, original GRU32 6,360. ME004 two-timescale train 450 updates; ME006 direct 1,082-parameter future-trained readout further trained 160 updates, with original TAC weights frozen. ME006 direct mean MAE ~0.675213 on newly exposed March9–16 2020, while original TAC mean ~0.686020; associative memory lost to direct and was slower. None of that establishes a capacity or training-budget-matched advantage. PNDS verified admission is separate from prediction.

## Controls, train budget and checkpoints
On unchanged UCI MetroPT-3 publisher ZIP SHA256 `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`, using exactly the authentic TRAIN-only ME004 pipeline and original TRAIN-derived normalization (first 20 days), train a GRU hidden-41 with **9,249 trainable parameters** (vs native TAC 9,368, 1.27% difference), identical 12 real sensor readings + masks, 32 time steps and 12/60 future observation-horizon predictions.

Three preregistered initializations 240401, 240402, 240403; each trained for 1,000 fixed AdamW optimizer steps with snapshots at **step 450 and 1,000**, batch128, lr=.001, weight decay=.0001, Huber=.2, gradient clipping=1. Same underlying 450-step optimizer settings as ME004, then an additional fixed 550 steps. The 1,000 step variant is an approximate **training-time control**, NOT exact FLOP equality; record actual wall time, trainable parameters and data budget and do not claim exact compute matching.

Only the historically re-used original first 20–25 day validation split may be inspected as a software diagnostic; no model-selection or refits based on its results. Publisher source reading is truncated at day25. Do not score March 2–9 nor March9–16 again, or other time-stamped reserved/held-out spans for development.

## Preregistered independent time-shifted test
After source+checkpoint hashes and all six training snapshots are frozen, evaluate **2020-03-16 inclusive to 2020-03-23 exclusive** exactly once, with ZERO training/calibration. Compare original ME004 two-timescale TAC (seeds 240401/240402), both frozen ME006 direct adapters (seeds 606001/606002, paired with their specific original TAC) and the fixed GRU41 candidates at 450 and 1,000 steps for all three initializations. Add persistence. MAE is primary; report 12/60 horizon MAE, per-channel signed error, actual change magnitude and direction, composition of positive/negative target changes, wall time and model size. The original two TAC seeds cannot establish 3-seed reliability; distinguish paired-seed and all-seed summaries.

**Decision:** If the GRU41 450 or 1000 budgets match/beat the direct TAC adapter, the previous predictive superiority claim has failed a meaningful stronger control. If TAC wins, do not promote yet—one site and one future week, no independently measured repairs, costs, multiple sites, calibrated uncertainty or safety. No automatic merge. No adjusting models after viewing March16–23 results.

## Mathematical research policy
TAC-OSM remains a distinct learned predictive-executable architecture **not converted to SSM or GRU**. The comparison isolates baseline competition, not a new architecture. Preserve full unified registry: CDL/PHR, CASM, BDH, procedural/executable memory, entity-local read/write, oscillatory gates, VSA binding, PNDS Lean and action verification are not discarded by a forecasting result, but only promoted on their relevant real tasks.
