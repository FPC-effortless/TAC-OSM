# ME-009: Original real historical reliability-gate fitting

**Code/model status:** Unpromoted research candidate. No full TAC-OSM action/controller claims.

Original first workflow [39383983906](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39383983906) FAILED before fitting because a dictionary loop variable shadowed the training function. Its failure trace is preserved. **No experimental scores came from that run.** The only code change was a variable rename; mathematical model, optimizer and data contract remained frozen.

Corrected original successful [training run 39384177806](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39384177806), original source SHA `9dbdc849a0feea91b54ba1c8dac06a0471939400`; original three-checkpoint archive `12246314308` (`me009-original-real-reliability`). Three SHA256 checkpoint digests independently inspected and verified against source ledger.

**Training data class:** authentic UCI MetroPT-3 from historically **reused days20–25** original validation records, repurposed strictly for small-gate calibration after frozen native TAC training days0–20. A train-only real qualification is **not** established from that cohort. Native TAC1000 original checkpoints from ME008, three seeds, 0 native base refits; three fixed 161-parameter gates each trained 220 AdamW steps on real targets.

| Frozen native base seed | Historical native MAE | Learned gated MAE after calibration | Quiet false-movement rate after calibration |
|---|---:|---:|---:|
| 240401 (gate 909001) | 0.25199207 | 0.24956297 | 2.00% |
| 240402 (gate 909002) | 0.26486093 | 0.26058267 | 2.22% |
| 240403 (gate 909003) | 0.24517209 | 0.24312049 | 2.82% |

Software contracts passed. All new gate weights freeze original first-20-day native recurrence. The model output is a convex blend between persistence (0 delta) and native forecast (full delta), but the formal scalar boundedness is **not** an error-reduction guarantee. Actual quiet false movement and future MAE must be measured on a new cohort.

**Decision:** Retain for one independent frozen March30–April6 test only; do not promote from previously exposed calibration samples. Every prior negative/positive result remains unchanged. Experiment follows model-first bottleneck correction rather than introducing more neuroplastic memory complexity.
