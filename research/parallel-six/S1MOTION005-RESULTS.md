# S1MOTION005 — ORIGINAL real-source graduate research findings (2026-10-26)

**Research preregistered full gate FAIL, 2/3 source seed trials pass.** Train, source-only provenance, all 15 checkpoint freeze and first actual robot DEV scoring were completed successfully. Original source-linked evidence and failures MUST remain immutable. This successor is a candidate to the same six-family shared TAC System1 model, not a seventh family or qualified robotics policy.

## Data ethics, source and controls
- Real MIT-labelled publisher dataset [LeRobot ALOHA static screw driver](https://huggingface.co/datasets/lerobot/aloha_static_screw_driver) 50 authentic 14D bimanual joint/action episodes, 400 measured frames each at published 50Hz. Used explicitly pinned published v2 revision `0b873be59c9702f8d3de89022617121970db112f`, 25 independent FIT original Parquet SHA256 receipts [S1MOTION005-SOURCE-LOCK.json](S1MOTION005-SOURCE-LOCK.json). Current publisher card is marked MIT; historical revision card lacks its own license field, and this provenance caveat is explicit. Not a claim of independent robot-site outcomes.
- Source-only original attempts [run40282597080](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40282597080) FAILED because live HEAD v3 differs from pinned expected v2; [run40282810894](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40282810894) FAILED because old card had no license field. Historical v2 source-only FIT audit [run40282959710](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40282959710) PASSED after explicit revision and current-card evidence scoping. These provenance failures are preserved, not erased.
- Original [real fifteen-model fit + first DEV score run40283616062](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40283616062) **CI PASSED**, original head SHA `a45f22534fa19ee5cb2ecd82869a07de92b3629b`. All 15 original model/optimizer/RNG checkpoint SHA256 frozen before **ANY of the DEV episode file downloads**, original evidence [artifact12632762763](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40283616062/artifacts/12632762763). Only **25 FIT files** ep0–24 accessed before freeze; only **10 DEV files** ep25–34 opened afterward; **TEST ep35–49 never downloaded**. State x[t], x[t−1], previously recorded action u[t−1] allowed by clock; no future state/action, command intervention or video source.
- Original five arms (TAC shared, isolated, 0.1-budget projected; GRU shared, isolated), 3 registered seeds29001/29002/29003, 192 updates × batch8 × window32 genuine FIT histories each, equal original windows+loss scalar, paired same TAC weights/optimizer hyperparameters, 95,907 vs95,869 learned weights, native 664B streaming FP32 TAC memory excluding weights/metadata. Within-TAC arms each do the SAME two source-autograd gradient evaluations, but projection adds computation. **Fully equal measured training FLOPs across arms not established** and cross-TAC/GRU training time differs; do NOT claim a fully equal-compute scientific head-to-head comparison.
- Primary output actual next 14D measured joint state in source coordinate units; high measured next motion above FIT-only p90 is an **observational HIGH-MOTION PROXY** (not hazard, collision, success/failure, decision utility or action authorization). Actual DEV high-motion proxy prevalence **8.745%**. Original DEV 2,550 measured next-frame outcomes from 10 same-publisher episodes, NOT 2,550 independent robot clusters.

## First untouched-within-study DEV scores, source-coordinate 14-joint next-state MSE ×10^-6 (lower is better)

| Seed | Physical velocity | TAC shared | TAC isolated | **TAC projected** | GRU shared | GRU isolated |
|:--|--:|--:|--:|--:|--:|--:|
| 29001 | 3.559536 | 3.571564 | 3.463685 | 3.452590 | 3.513630 | 3.446837 |
| 29002 | 3.559536 | 3.522431 | 3.463921 | 3.446752 | 3.544115 | 3.450923 |
| 29003 | 3.559536 | 3.933416 | 3.449297 | 3.444715 | 3.503013 | 3.438335 |
| 3-seed means | 3.559536 | 3.675804 | 3.458967 | 3.448019 | 3.520252 | 3.445365 |

**Original predictive gain:** projected TAC mean **3.448019e-6** is **3.133% lower MSE** than causally available velocity, **6.197% lower** than full-shared TAC and **0.317% lower** than fully isolated TAC (much smaller gain). It beat velocity by >=1% and both TAC baselines **in all three original seeds**. Strong matched GRU-isolated mean 3.445365e-6 is slightly lower than projected TAC; unequal complete training compute and multitask proxy performance prevent architecture superiority claims.

## High-motion-proxy AUPRC (higher is better; still NOT real adverse events)

| Seed | TAC shared | TAC isolated | **TAC projected** | GRU shared | GRU isolated |
|:--|--:|--:|--:|--:|--:|
| 29001 | 0.325705 | 0.087207 | 0.173386 | 0.499865 | 0.082117 |
| 29002 | 0.228164 | 0.095569 | 0.238044 | 0.460894 | 0.086593 |
| 29003 | 0.241069 | 0.110321 | 0.232421 | 0.513617 | 0.070695 |
| Mean | 0.264979 | 0.097699 | 0.214617 | 0.491459 | 0.079802 |

TAC projected retained more proxy ranking information than full isolation on this task but **seed29001 AUPRC 0.173386 < 90% of TAC-shared 0.325705**. This is a meaningful violation of the preregistered event guard, not tuning noise to dismiss. Meanwhile the same seed's projected actual forecast MSE beat velocity and TAC-isolated, illustrating a genuine multiobjective tradeoff.

## Actual original per-seed gate ledger
G2: projected MSE <= .99×velocity AND <= both original TAC controls; G3: high-motion source MSE <=1.05×velocity; G4: proxy AUPRC >=0.9× shared.

| Seed | Velocity >=1% | Both TAC controls | High-motion <=5% harm | Proxy AUPRC >=90% | Full seed |
|:--|:--:|:--:|:--:|:--:|:--:|
| 29001 | PASS | PASS | PASS | FAIL | **FAIL** |
| 29002 | PASS | PASS | PASS | PASS | **PASS** |
| 29003 | PASS | PASS | PASS | PASS | **PASS** |

**Original full study scientific gate FAILED: 2 of 3 seed trials pass.** No original checkpoints modified. If a new model/threshold is designed after seeing these DEV scores, those scores cannot be presented as an unseen prospective gate.

## Why the optimizer evidence matters
- Source gradient projection kernel separates point and auxiliary gradients on shared native encoder/memory/readout. Opposing Euclidean auxiliary direction removed and remaining allowed auxiliary update norm <=0.1× point gradient norm (plus cap beta<=.25). Auxiliary head receives own supervised gradients; no new parameters. Ideal-SGD first-order point descent is only an algebraic local property.
- Original real FIT actual **AdamW applied parameter update** and **post-step point loss** were logged on EVERY original train step, not just raw gradient cosine. Example seed29001 TAC shared has 135/192 FIT minibatch point-loss improvement steps; isolated 180/192; projected 173/192. This is diagnostic, not guaranteed optimizer convergence or generalization. Original complete per-step gradient/AdamW evidence in artifact.
- Numerical and source-authorization checks passed. **Source-level PyTorch exact AD/refinement, IEEE accuracy, genuine risk law, action do(a) identifiability, independent robot task/site utility, full model compute and hardware latency are still OPEN**. Historical S1MOTION001 candy and S1MOTION003 tape and S1MOTION004 towel negative qualifications unchanged. Six families remain on a shared backbone.

## Next research constraint (NOT a retrained result)
The next task is to address **multiobjective event signal retention** without relying on opened Screw Driver DEV scores or treating a retrospective budget adjustment as a prospective success. Expand with authentic measured adverse/action costs, strict source/time provenance, independent task/robot, honest FLOPs, and full PNDS finite-precision and gradient-refinement. Previous TEST episodes35–49 remain sealed. Keep PR draft/unmerged.


## Post-hoc, source-respecting paired episode diagnostic (not original promotion gate)

Read only the **already frozen first DEV per-episode scores** from the original model artifact. For each actual DEV episode ID 25–34, retain exact same episode ID and the three original seed checkpoints across every comparator; resample 10 episode IDs **with replacement**, 50,000 replicates, numpy PCG64 generator seed **29005**, source-unit MSE averaged jointly across the three original seeds and the sampled episode IDs, paired relative improvement `100*(baseline_MSE - projected_MSE)/baseline_MSE`. Weights across episodes are uniform because exactly 255 authentic next-frame observations were scored per episode. This is conditional single-task source evidence, NOT inferential evidence about independent unseen robots or trials.

| Projected TAC gain compared with | Original mean MSE gain | Paired episode-bootstrap percentile 2.5–97.5% | DEV episode IDs with improvement (mean over 3 seeds) |
|:--|--:|--:|--:|
| Constant velocity | **+3.133%** | **+2.652% to +3.498%** | **10/10** |
| TAC fully isolated | **+0.317%** | **+0.256% to +0.379%** | **10/10** |
| TAC fully shared | **+6.197%** | **+5.549% to +6.796%** | **10/10** |
| GRU fully isolated | **−0.077%** | **−0.650% to +0.343%** | not a qualified advantage |

Bootstrap draws do NOT make the ten episodes independent robots, the three seeds independent tasks, or prior analyzed data novel. A 0.317% within-TAC projected-vs-isolated mean gain is small despite stable episode ordering, and a point forecast does not stand in for actual model intelligence, intervention outcomes, or risk calibration. Original classifier guard for seed29001 still **FAIL**; original complete qualification unchanged **2/3 seeds pass**.
