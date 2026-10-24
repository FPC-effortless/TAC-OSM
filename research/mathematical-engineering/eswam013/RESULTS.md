# ESWAM013 — final factual real-model results and qualified gates

**Fitted study:** original publisher INTERACTION TC_BGR_Intersection_VA (one physical site), archive SHA256 `9aa78102006dc8d8a64675cf64eba5264085cfb1c006b1f4d0e856d94e657aec`. Three FIT recordings 000/001/002 yielded 267 strictly source-aligned real 256-frame multiagent origins. Previously exposed original DEV recording003 yielded 90 origins (266 actor forecast cases per horizon across 37 original actor tracks). True future position original +10 and +30 source frames, measured physical units in meters. No simulated, synthetic or causal intervention training.

## Prospective guarded neural fits — 101,057 active/deployed params
Seeds 87001 and 87002 each fitted exactly 320 AdamW real-source optimizer steps batch8. Every step passed actual live whole-model parameter inventory, frozen preregistration, optimizer hook. Original checkpoints, complete optimizer/RNG states, SHA and live guard receipts, all original per-case real DEV x/y predictions and future labels Git frozen in `research/replay-bundles/ESWAM013/`. Original fitting workflow https://github.com/FPC-effortless/TAC-OSM/actions/runs/40109898068 **PASS**.

| Measure | seed87001 | seed87002 | measured real P1 |
|---|---:|---:|---:|
| +10 forecast Euclidean L2 meters | 0.236446 | 0.231552 | 0.257792 |
| +30 forecast Euclidean L2 meters | 1.792025 | 1.777038 | 1.957001 |
| +10 actor-cluster paired bootstrap 95% model−P1 | [-0.044246,-0.001937] | [-0.051471,-0.004922] | — |
| +30 actor-cluster paired bootstrap 95% model−P1 | [-0.312809,-0.052842] | [-0.305201,-0.084071] | — |
| +10 low-speed (<0.2m/s) model L2 | 0.074551 | 0.072574 | 0.071570 |
| +30 low-speed (<0.2m/s) model L2 | 0.755420 | 0.726184 | 0.719308 |

Mean candidate+10 0.233999m versus 0.257792m P1 (9.23% lower); +30 1.784532m versus 1.957001m P1 (8.81% lower). Each original seed/H horizon's grouped CI excludes zero for the **strong nonlearned P1** baseline; source development only. **Stationary quality still FAILS**, above the fixed physical prior in both seeds at both horizons. ESWAM013 is a research candidate, NOT a promoted general-purpose or robot model.

## Independent no-fit real replay
https://github.com/FPC-effortless/TAC-OSM/blob/research/eswam013-motion-relevance-native-2026-10-24/research/mathematical-engineering/eswam013/INDEPENDENT_ORIGINAL_SOURCE_REPLAY.json. Separate run restores exact two checkpoints from disk and original true future labels, independently re-verifies their original optimizer budget receipts, scores 532 source actor/horizon cases per seed with **zero extra optimization steps**, maximum numerical discrepancy 3.7253e-8m (87001), 5.9605e-8m (87002), below registered 5e-6. **PASS**, frozen.

## Mathematical full-model typed verification
Actual model source `src/tac_osm/variants/eswam013_motion_relevance.py`. Lean typed operator composition `research/mathematical-engineering/eswam013/WholeNativeGraph.lean` specifies all parts of **this window-fed observational predictor** from source features, parallel actor/time encoder, causal motion difference, fast and long-slow evidence, eight source-observed associative blocks, per-actor scoped memory, trust-gated fusion, shared actor exchange and physically bounded residual head. Lean kernel https://github.com/FPC-effortless/TAC-OSM/actions/runs/40110836448 **PASS**. Exact rational algebra lemmas prove 0 relevance returns fast and 1 relevance returns slow and 0 measured speed returns physical baseline; **abstract typed operators are NOT formal proof of real-number neural nonlinearities, floating-point execution, autograd or observed physical generalization**.

## Unpassed scientific gates
- Native model is **worse** than the fixed physical P1 in low-source-speed regimes; near-stationary gate failed.
- Prior original ESWAM012 256-history native was robustly worse than 8-frame and GRU, and original older competitive fits were **unguarded** and training FLOPs parity unverified; those historical results remain exploratory. This new run is a single-arm capability development, **not** equal-data, equal-parameter, equal-training-FLOPs verified head-to-head with GRU/attention.
- Only **one physical intersection**, and DEV was **previously exposed**. No unexposed independent physical site test or transfer qualification. Even a DEV gain is not proof of generalization.
- No logged/interventional do(action), real policy authority, decision cost, downstream task success or commercial validation.
- The candidate receives and encodes a *full real 256-frame window* each prediction. It does NOT demonstrate native online continuously retained hidden-state learning across arbitrary histories or autonomous source entity discovery. The per-actor keys and values are computed from genuine recent windows, rather than carried persistently across calls.
- No complete clinical/safety/robotics proof and nothing merged.

**Next testable mathematics:** replace linear near-zero kinetic relevance `||v||/(||v||+0.5)` with quadratic relevance `||v||²/(||v||²+0.25)`; it has zero first derivative at speed0 and may preserve observed stationary states, without an oracle. Preregister and re-fit under the same full guards, evaluate *all* outcomes including failures, and never use a repeated exposed DEV as independent TEST.
