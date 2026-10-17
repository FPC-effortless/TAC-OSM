# DEV-041: bounded MTSK writer training against an identically initialized frozen random writer

**Status: audited development evidence; not confirmatory.** Original training source: `4ba2ea65cd3d4f3961277d80c49d7fb641f556fc`, registered protocol [research/dev041/protocol.json](../research/dev041/protocol.json). The archived run [39003550880](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39003550880) passed the complete three-seed replay and independent physical-result audit. The source/frozen training outcomes are not overwritten by this report.

## Primary: calibrated cost/entity, trainable versus frozen writer

| Seed | Learned bounded MTSK | Initial random writer frozen | Memoryless | Last-event calibration table | Learned failures | Frozen failures |
|---:|---:|---:|---:|---:|---:|---:|
| 4101 | 0.539714 | 0.870768 | 1.000000 | 0.518880 | 0 | 618 |
| 4102 | 0.539497 | 0.862847 | 1.000000 | 0.518880 | 11 | 413 |
| 4103 | 0.541124 | 0.834418 | 1.000000 | 0.518880 | 10 | 346 |
| Mean | **0.540111** | **0.856011** | **1.000000** | **0.518880** | **7/seed** | **459/seed** |

The paired **learned-minus-frozen** mean cost/entity is **−0.3158998843**, exploratory three-seed t(df=2) 95% interval **[−0.3654643892,−0.2663353793]**. Per-seed failure differences are −618, −402, −336. The **prespecified descriptive development criterion passes**, but n=3, known synthetic generator, fixed recipe and reused evaluator scenarios cannot establish population-level reliability, training-convergence superiority or complete TAC-OSM capability.

Each arm received a fixed **216 updates**, H512 histories, two episodes/update, uniformly supervised wait/service/current targets and identical paired training histories. The learned and frozen MTSK were **identically initialized**. The frozen arm held memory-module parameters fixed *before* training the encoder, query, adapter and head. Replaying a writer reset after training would not provide the same attribution test.

## What the comparisons do and do not show

The **trainable writer** achieved mature reissued wait accuracies of 78.65%, 97.92%, 82.29% across seeds, versus 77.08%, 50%, 50% for the frozen writer. Pending-state accuracies and physical costs should be read together: high service-query accuracy (100% for all arms) is trivial when a service intervention forces a zero outcome. The memoryless controller serviced every entity, scoring zero failures through over-service, not through competent state prediction.

An important **non-neural reference remains ahead of the neural winner**. The per-entity last-event table trained only on disjoint calibration observations achieved 0.518880 cost/entity with zero failures on the same 1,152 evaluation worlds per seed. Its inputs were public last-event kind/value/due status; it is not an inference from hidden oracle future labels at evaluation. The exact public-history cost reference was also 0.518880.

**Interpretation:** Under this fixed recipe, learning recurrent writer parameters was much better than freezing them *within bounded banked MTSK*. This does not establish that learned recurrent memory is necessary for the benchmark. The fixed writer has **4,258 trainable parameters** versus **5,414** in the trainable writer (because 1,156 memory parameters are frozen); exposure, initialize-once tensor values and architecture match, but remaining trainable-parameter count is inherently different. Training longer, tuning frozen-encoder/readout or changing random reservoirs might change the outcome.

## Evidence and numerical integrity

* Complete original Actions ZIP: [download audited archive](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39003550880/artifacts/12090820964).
* ZIP size: **2,572,084 bytes**, SHA-256 `b0dc2842537c50e2e76ecf0139b1519a032c2575af9181076897ccce0d5bd7c7`. Downloaded archive was independently integrity-checked and contains 13 entries (nine model checkpoints, three per-seed reports, one aggregate audit).
* Regression suite: **1,197 tests passed**. All three training jobs passed strict separate-process checkpoint/prediction/action replay. Independent audit regenerated 1,152 world conditions, counterfactual labels and physical costs per seed, checked frozen-writer bytes and the last-event fitted reference.
* Prior failed CI/preflight attempts and their logs remain accessible on GitHub; no failed-science evidence was discarded.

The separate audit reused the same simulator and registered rules. It does not replay the stochastic-gradient optimization independently or test unseen configurations. The two MTSK arms' architectures have equal 768-scalar recurrent state, but a frozen-parameter ablation is not a param-count-matched cross-architecture comparison.

## Next controlled question

The last-event table is a **sufficient summary for the tested synthetic cohort**. Register a harder, action-conditioned *order-sensitive* paired task, where (a) the last public event for every entity and final time/query match, (b) earlier schedule/reset **ordering** differs, (c) an active service would change the future and (d) an unchanged last-event controller must yield identical predictions for opposite targets. Validate the paired causal construction and physical simulator before any fresh training or head tuning. Compare full-history learned state, frozen random reservoir and explicit schema reference; retain unseen configuration and multi-seed follow-ups. Do not claim an architectural winner from this three-seed pilot.
