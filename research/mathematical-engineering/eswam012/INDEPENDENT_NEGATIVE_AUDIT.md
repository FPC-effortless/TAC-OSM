# ESWAM012 independent post-fit audit — authentic outcomes, science gate failed

Evidence: 10 original genuinely trained models (five arms × two seeds), original source SHA `9aa78102006dc8d8a64675cf64eba5264085cfb1c006b1f4d0e856d94e657aec`, 267 strict 256-frame multi-entity FIT anchor samples and 90 exposed DEV anchors, with 266 scored source-actor future cases per horizon, 37 source actor-cluster IDs. Both horizons are source-real original future x/y. Training models: 101702 native, 100788 GRU, 101370 attention, 101702 native-short, 101702 unscoped. Original checkpoints and all true future case scores under `research/replay-bundles/ESWAM012/`.

**Compulsory research gate failure:** The original fit script used a raw optimizer, step, and torch.save, rejected by independent `scripts/audit_training_budget.py`; it was executed in a separate training workflow that did not call the mandatory shared budget guard. The original genuine train results are exploratory, NOT a runtime-guard-qualified model-comparison certificate. Adding a manifest or new guarded code *after* this original fit cannot retroactively certify its checkpoints. Training FLOPs were not measured/matched, so the original results cannot establish equal-compute objective architecture superiority. Exposed DEV was used for scoring and should not be treated as untouched TEST. Do not merge/promote original models.

## Real mean forecast L2 errors, meters (average of original two seed fits)

| Model | +10 original frames | +30 original frames |
|---|---:|---:|
| Real measured last-velocity physical prior | 0.257792 | 1.957001 |
| Native parallel fast/slow/observed-associative memory (256 frames) | 0.247357 | 1.862645 |
| Native short-only eight real history frames | 0.236878 | 1.797508 |
| GRU 256-frame control | **0.229117** | **1.739160** |
| Two-layer causal attention 256-frame control | 0.232278 | 1.770261 |

## Independent same-actor paired bootstrap readout
Using the original identical per-case future labels and source actor IDs, bootstrap resampled each actor ID as an entire cluster, not overlapping source frame as independent. Each seed/horizon: 266 real actor-case predictions from 37 original actor tracks.

- Native minus eight-frame native: seed 86001 +0.009058m at +10, 95% CI [+0.002741,+0.015972]; +0.050079m at +30, CI [+0.022437,+0.080834]. Seed 86002 +0.011899m at +10, CI [+0.005821,+0.018902]; +0.080196m at +30, CI [+0.050069,+0.114206]. **256-frame native is robustly worse than eight-frame** for this setting.
- Native minus GRU: seed 86001 +0.021010m at +10, CI [+0.008562,+0.033180]; +0.135530m at +30, CI [+0.087747,+0.184224]. Seed 86002 +0.015469m at +10, CI [+0.004781,+0.026675]; +0.111442m at +30, CI [+0.061062,+0.158707]. GRU wins descriptively in all tests; parity of FLOPs unqualified.
- Original native 256-frame stationary-regime error at +10 measured present source speed below0.2 m/s is 0.087359/0.087365 meters across seeds, versus measured velocity prior 0.071570. Thus absolute stationary quality failed its promotion requirement.
- Unscoped observed-associative retrieval did not produce a stable identity-address advantage: its relative sign to scoped native varies between seeds. This is a negative/no-proof finding, not learned cross-actor identity protection evidence.

Root diagnosis constrained by evidence: indiscriminate full-history slow/associative evidence diluted short causal motion information, because native-short outperformed full. A prospective source-causal *conditional relevance gate*, bounded physically measured-motion correction and explicit real one-step acceleration input are testable improvements. They must not be promoted until trained with repository guarded optimizer/weight receipts and evaluated on original measured outcomes under frozen future research protocol.
