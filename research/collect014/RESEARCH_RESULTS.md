# COLLECT014–015 — completed repair and residual-memory research

Nine new fixed-budget fits completed, with three immutable COLLECT012 controls reused. All 12 models have 99,971 parameters. These are internal same-graph repairs, not a FLOPs-qualified architecture comparison.

| Setting | DEV NLL mean | Paired delta vs control | Mean largest-bank share | Balanced delayed retrieval | Complete maps |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 2.71844 | +0.00000 | 67.08% | 10.42% | 0.00% |
| balance | 2.71293 | -0.00551 | 29.17% | 12.50% | 0.00% |
| write_first | 2.70709 | -0.01135 | 70.33% | 11.11% | 0.00% |
| combined | 2.70502 | -0.01342 | 30.11% | 13.89% | 0.00% |

DEV units are nats/byte on the same 8,192 shared exposed-DEV targets, recomputed in one Torch 2.15 CPU runtime. Each fit completed 1,024 FP32 AdamW updates, batch 4/context 128, identical initial tensors and sampled minibatch indices per paired seed. All three repair settings improve DEV loss in every paired seed in this cohort. Combined is not best in every seed; these small gains do not establish general or compute-qualified architecture superiority.

Balancing makes bank utilization much more even; write-first alone does not. Both the current-value and historical phase-output contributions improve short-window language loss in frozen-checkpoint interventions. Those interventions retain native memory paths and are not separately trained alternatives. A trained current-value-only control remains necessary for stronger attribution.

The supplemental balanced probe is zero-shot and was not trained. Its best-constant present-answer baseline is 12.5%; combined achieves 13.89% at a 1024-byte delay, versus 14.58% with wrong history and 12.5% after reset. All settings have zero complete-map and absent-marker accuracy. Full-vocabulary top-1 present-answer accuracy is 0% for all three repairs in that cohort. This does not demonstrate useful query-dependent fact retrieval or identify an intrinsic storage-capacity limit.

The original primary probe accidentally restricted latest values to four letters (25% best-constant baseline). It remains preserved. The supplemental evenly balanced cohort and read-split protocol were declared after launch, before inspecting new full-fit results; they are explicitly additional analyses. Twelve histories per cohort are reused across models/seeds/delays, not independent repeats.

Original-text 512-byte continuation shows small, mixed carry benefits. Balance-only hurts two seeds relative to reset. All training windows started fresh at 128, so long-state use is evaluated beyond its trained state-age distribution. No robust long-horizon continuity or global-attention replacement is established. SSA still uses bounded query/key weighting; this is not a claim that every attention-like operation has been removed.

COLLECT015 separately tests the user’s residual-tiered hypothesis with ideal supplied phase addresses, local exact buffering, two decays and a bounded exact store. The 4,608 policy evaluations use 576 unique generated stream histories and 96 summary cells. At 16 facts with repeated distractors and 16 cache slots, residual class recall is 99.35% versus 77.86% for count-matched random; at 64 facts, it is 38.41% versus 45.41%. Complete maps remain zero at high load. The gate is conditionally useful but does not consistently outperform matched random. Count matching is an explicitly retrospective diagnostic control, not a deployable causal admission rule.

Counterexamples show a zero-error uncached fact can be overwritten later, a high pre-write error need not imply post-write failure, and absolute residual thresholds depend on value/state scaling. Exact-cache invalidation, future relevance, eviction and learned addressing remain distinct problems. Closely related residual-to-exact-cache prior art exists; see residual_tiered/README.md and its SR-TTT v2 correction reference. No broad novelty claim is made.

Execution: 9,216 new full-fit updates, 24 separately charged short-smoke updates, 3,072 reused earlier control updates; 45 new plus 15 reused full-fit model/AdamW/RNG checkpoints; 24 generation samples in this cohort (historical C12 samples remain in original control archives too). New recorded optimizer CPU time is 107.13 summed minutes, excluding setup/evaluation/audits. GPU quota used: 0.

All nine new trained-source audits pass checkpoint/journal/AdamW, causality, owner isolation and declared full-graph FP32/FP64 reference/surrogate-gradient fixtures. All sampled training exposures and initial identities match the paired originals. Original same-runtime replay is exact; cross-runtime replay uses 2e-5 tolerance. An old replay-field name caused an initial read-only audit adapter failure; its source and failure record are retained, and corrected checks reran without training or changed tolerances.

All three final internal single-model gates pass with architecture-superiority eligibility false. Protected TEST, independent qualification, full FLOPs parity, checkpoint promotion and full formal/IEEE/AD release remain open.

- Frozen fit source: `6e19362ab0028b7e857e4506657f2703403b451f`.
- [Successful training run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41532283413).
- [Draft PR #699](https://github.com/FPC-effortless/TAC-OSM/pull/699).
- Original controls: COLLECT012 fit commit `036d778255a416d3aae886acdbfeef8fb1b6a1ab`, run `41513912016`.
- Read-only scripts: `collect014/audit_completed.py`, `common_evaluation.py`, `supplemental_balanced.py`, `read_split.py`, `collect_completed.py` and `build_report.py`.
- Generated models, receipts, raw predictions/logits and full replay evidence accompany the persistent report; source/data remain in Git.
