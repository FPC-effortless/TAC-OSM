# ME-006F — One-time frozen later-week real forecast test
**Preregistered on 2026-10-20 before opening the March 9–16 source period.**
**Test period:** 2020-03-09 00:00:00 inclusive to 2020-03-16 00:00:00 exclusive. This period is after the original ME004 TRAIN (first 20 days), reused validation (days 20–25), ME003 first-30-day audit and ME004F previous March 2–9 future-week exposure.

Restore exact six ME004 model checkpoints from original GitHub artifact `12235023980` and exact four frozen ME006 adapters from artifact `12244879044`, **without retraining or changing hyperparameters**. Verify every checkpoint SHA256 against the original result ledger before opening data. ME006 models: direct and associative, seeds 606001 and 606002, each with matching original TAC backbone seed 240401 or 240402. Baselines: frozen original TAC and persistence.

Data contract: original UCI MetroPT-3 publisher ZIP SHA256 `aab991a970e58210de853bb4d9412fdc5c79792dae3d8e1721a`, same 12 physical sensor channels, 32 history observations, paired 12/60 future observation-step horizons, original TRAIN-only means/standard deviations and train-only target-change thresholds; chronology and max 30-second gap integrity checks identical to ME004F. No readings at or after March 16 used. Zero gradient updates, model selection or source-specific calibration.

Registered primary metric: macro absolute forecast error in original TRAIN-normalized units, over both horizons/12 features. Secondary: real significant-change magnitude error, sign composition and per-horizon accuracy. Compare both model seeds with their exact original TAC baselines; the direct adapter should outperform the original model in **both** paired seeds for evidence of a generalized representational improvement; Hebbian associative arm must beat direct in both seeds to count as a retrieval benefit. A failure is preserved, not retuned.

Even a pass provides only **one-site limited out-of-time** evidence, not a model-wide superiority or independent commercial qualification. The ME004 source is a single compressor with no observed repair-action outcomes; do not claim counterfactual effects, entity semantics, Lean-verified learned behavior or production safety. Do not touch original ME004F already-opened March2–9 results or the reserved TAC-OSM synthetic test namespaces.

Model/predictive test code and protocol are retained in ME006 stacked draft research PR; no merge or promotion authorized.
