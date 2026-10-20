# ME-004 — Model-first TAC memory experiment (preregistered 2026-10-20)

## Goal
**Get a genuinely trained native TAC-OSM recurrence working on real operational observations** with minimal architectural invention. Native TAC is **not converted to SSM**. Reuse an established PyTorch GRU as the reference, and test only the identity-first and two-timescale update alternatives motivated by ME002. All other master-plan ideas remain registered but paused until the core predictive baseline performs.

## Source and leakage controls
Authentic UCI MetroPT-3 compressor telemetry, DOI 10.24432/C5VW3R; official ZIP SHA256 aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a. Exactly 12 named real signals. Source code stops reading before day 25. Day 0–20 = training; day 20–25 = **exploratory chronology-disjoint validation**. Day 25+ remains unopened by THIS experiment; earlier ME003 inspected aggregate counts within days 0–30, so the time-period is not scientifically pristine. This study does not access action logs, repair labels, or independent deployment outcomes. Source provenance is hashed; if ZIP hash differs, abort (do not silently use another publisher version).

## Frozen comparison (no architecture fishing)
- Persistence, no learned parameters, always predict delta zero.
- PyTorch GRU, hidden 32, history 32 real samples.
- Native TAC bounded identity-first recurrence, width 32, write-gate bias -2.5.
- Native TAC with an additional slow recurrent memory, width 32, slow gate bias -4.0.

All get the **same** 12 physical channels plus missingness mask, 32 past readouts; predict residual changes at 12 and 60 future observation steps. Output-head zero init. TRAIN-only normalization, TRAIN-only absolute residual change threshold at 90th percentile with 0.04 minimum. Reject input sequences spanning gaps >30 seconds; no future measurements provided as inputs. Fixed stride 12, PyTorch CPU, 450 AdamW updates, batch 128, LR=0.001, weight decay 1e-4, Huber beta=0.2, clip grad L2 to 1; seeds 240401 and 240402. Record complete parameter counts and runtime; not capacity-matched. Evaluate macro MAE in TRAIN-normalized units, per-horizon MAE, change-only magnitude error and class sign composition; do not claim reasoning from change sign alone. Optimizer budget/hyperparameters are fixed without tuning on validation.

**Decision rule:** Do not promote anything from this exploratory two-seed comparison. A credible candidate would need to improve over persistence on both seeds and over the GRU at acceptable compute, then pass a separate independent future real-data test with tuned/compute-matched controls and per-channel analysis. If every model loses, report failure and target the data contract and optimization before adding new architectural components.

## Scientific limits
Not an entity-identity discovery experiment, not executable memory or CASM attribution, not a semantic reasoning test, not causal repair effectiveness, not a commercial deployment. Previous real-data forecasting failures retained. ME002 gradients were software diagnostics, not predictive evidence. Record failed CI too. Never use synthetic values for a scientific success claim. Reserve all future periods, other repo checkpoint artifacts and sealed synthetic tests.

## Reproducibility
Standalone script: scripts/me004_native_real_forecast.py; original GitHub Actions workflow archives original results/checkpoints. Independent replay to follow using frozen original source/checkpoint artifacts, with **zero refits**, and original training workflow must be locked to its successful SHA before unrelated edits. No automatic merge.
