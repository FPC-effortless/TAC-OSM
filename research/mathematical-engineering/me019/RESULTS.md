# ME019 — Verified ORIGINAL 32/128 genuine-action training results (not a fresh test)

**Verified successful original run:** [39443325909](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39443325909). [Artifact 12275359676](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39443325909/artifacts/12275359676), archive digest SHA256 `757800580c11dde45f625f096a4a182920f2020c53335f78da738cc1d7fde10c`. **Raw inner original `results/me019/results.json` SHA256 `2a10ae3205e866ed792f5ad145e3c3c775e95c39de675f421184de44e56455b6`.** All 18 checkpoint SHA256s independently checked against the original raw archived manifest. The initial attempted Actions run 39442442594 failed on TorchScript's inference tensor after backwards; no completed first-run model matrix was produced. The successful run used the identical trained recurrence and a `no_grad` inference-context fix, not a revised scientific hypothesis.

Source: genuine publisher KuaiRec ZIP SHA256 `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`; 96976 eligible genuine FIT anchors, source-only stride 9 → 10776 FIT training anchors, 422575 eligible genuine Aug01–Aug14 DEV anchors, stride 71 → 5952 inspected development anchors. Exact matched causal target action/outcome and newest32 input history across all 18 variants. Original trained user %16 == 0. No new future users/date outcomes opened. All arms exactly 135050 active parameters, exactly 300 updates ×128 real training target windows, same seeded target indices. Historical development labels already inspected in earlier research: **NOT independent generalization evidence**. Per-arm training CPU/time and contextual history lengths differ; **NOT a fully compute-fair head-to-head comparison**.

## Preregistered original measured results

**Brier lower is better.** Each row is seed 14001, 14002, 14003 respectively followed by across-seed mean.

| Arm | Context | Seed 14001 | Seed 14002 | Seed 14003 | Mean Brier | Actual training seconds 14001, 14002, 14003 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| native_chrono_anchor | 32 | 0.203136775 | 0.205151062 | 0.201749777 | **0.203345871** | 4.411, 4.651, 4.664 |
| native_chrono_anchor | 128 | 0.204670059 | 0.205711652 | 0.203290934 | **0.204557548** | 17.997, 18.921, 18.793 |
| gru_original | 32 | 0.203536038 | 0.205974606 | 0.203597637 | **0.204369427** | 4.449, 4.575, 4.815 |
| gru_original | 128 | 0.203039580 | 0.204568581 | 0.203423405 | **0.203677189** | 17.043, 17.986, 17.037 |
| gru_anchor | 32 | 0.203517909 | 0.203603898 | 0.204365281 | **0.203829029** | 4.455, 4.768, 4.955 |
| gru_anchor | 128 | 0.203231562 | 0.203603901 | 0.204365255 | **0.203733573** | 17.924, 18.568, 17.481 |

Real *causal prior32* baseline same DEV targets Brier **0.2032487097**. Native 32 does NOT beat this simple historical predictor on the measured reused DEV subset, despite outperforming GRU32 in mean Brier. Native128 degrades vs native32 in all three seeds. Original GRU128 improves vs original GRU32 in all three seeds. Anchored GRU differences in seeds 14002/14003 are negligible (at about floating precision), so its across-seed improvement is driven primarily by seed 14001.

Observed native H128/H32 median training time ratio approximately 4.04x, expected for 4x longer sequence. **Do not describe this as superlinear scaling; total time scaling approximately linearly with T**.

## Scientific status and guardrails

- **ME019 H1 fails**: the 128-event native model does not improve observed real development Brier in any of three seeds.
- **H2 fails**: native improvement exceeds original GRU improvement in none of the seeds.
- **H3 fails**: 128 native fails the original causal prior32 Brier baseline.
- **H4 passed**: all 18 fits completed, trained parameter counts/checkpoint SHA checks passed, no long-recurrence numerical training instability detected on the tested 300-step budget.
- Long-horizon **credit dilution**, **gate saturation**, **entity interference** and **poor learned relevance** are plausible explanations, but NONE have been measured in ME019 as the causal mechanism. Likewise, GRU improvement cannot be conclusively attributed to dynamic pruning or specific gate behavior by these Brier metrics alone.
- Mechanistic next diagnostic if once-frozen evaluation replicates: actual 128-step input-gradient lag and singular spectrum/jacobian local gains; real event-weight perturbation rather than synthetic success score; compare gate statistics and history ablations; parameter-free carefully controlled memory timescale changes only if diagnosed, with fair full CPU/FLOP/time training budgets.
- **No automatic ME020 selective gating/Titans/MIRAS/attention adoption** absent a verified bottleneck. Original native TAC is not a SSM and must remain its own mathematically defined architecture.
- Original 32-event engineered TAC checkpoint from ME015/016 remains the defensible frozen research default, without claiming it is already a production or commercial-qualified agent.

## Separate new-user boundary

Previously unopened %16==4 users and Aug14–Sep6 actions are protected by preregistered ME019F proposal. **Use the exact successful run 39443325909 with pinned original raw manifest SHA and ALL 18 checkpoint SHA proofs BEFORE parsing any new test response**. Never rerun model selection or retrain after the first new-user opening. The matched parameter/update source does NOT imply equal FLOPs/time across architectures and cannot establish a fully controlled architectural victory.
