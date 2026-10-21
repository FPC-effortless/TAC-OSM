# ME038 original trained models and evidence

Local research deliveries, not GitHub Actions training artifacts. No model refit,
promotion, merge or deployment. Full trained weights are preserved, not only
seeds or manifests; timed stopping is hardware-dependent on a new training run.

| Delivery | Bytes | Payload |
|---|---:|---|
| `/workspace/me038-native-inference-models.tar.gz` | 30,196,067 | 60 native physical-input models, validation probes, metadata and standalone verifier |
| `/workspace/me038-models-and-evidence.tar.gz` | 1,048,963,143 | All 1,008 parent/adaptation checkpoints, optimizer/RNG states, training traces, sources, exports and evidence |

Compact SHA-256:

`3bb753071dcc544919437aecf54eedd015da13e89aa61c6e15ae508d1e8c6d04`

Complete SHA-256:

`dd8c656b9ef47e9f25e4f8a737d496a28e3492dc566570d2f92548f924b2ffcb`

Complete source snapshot: `0e74ea85455fb96a20a1cbf60a2b49bf1af71b2f`.
All 4,212 indexed payloads passed hash verification before and
after isolated restoration at `/workspace/me038-restored`. All 1,008 original
checkpoints replayed 37,310,976 validation forecast scalars exactly without
refitting; 48 shared optimization prefixes reproduced 23,103 steps exactly.
The archive excludes circular archive/restoration receipts and this delivery page.
Unrelated old model trajectories are excluded; all 288 relevant AR-parent
endpoints and original parent trace/fit files are preserved where available.

The standalone compact extraction at `/workspace/me038-inference-restored`
verified every payload and loaded all 60 models with Torch only, reproducing
2,112,000 physical-input values bit exactly without importing the TAC package.
48 models are from the exposure lane and 12 from the CPU-budget lane; filenames
and INDEX.json distinguish them. All standard/raw export checks total 4,224,000
values. Training-only anchor buffers are removed without changing learned weights.
Caller input contract is raw Bx96xC; output BxHxC in original publisher units.

Restoration itself replays saved validation probes, not the entire final metric
matrix. Separately, the executed NumPy reduction audit independently recalculated
ALL 52 Tetouan selected-model/reference final scores over 1,479,493,632 forecast
values, within a prospectively registered float64 bound. That original receipt
is in metric-audit.json and final-metric-recalculation.log; it is distinct from
restoration.json's `test_metrics_recomputed: false`. No independent rerun of the
full legacy final test matrix is claimed.

Original public CSV/ZIP/publisher license/attribution bytes remain intact and
separate from derived trained models. See DATA.md and sources.json for collection
and license limits. State/parameter/CPU costs and both failed gates remain explicit.

[Execution/replay commands](EXECUTION.md) · [Measured outcomes](RESULTS.md) ·
[Native state equations](MATHEMATICS.md)
