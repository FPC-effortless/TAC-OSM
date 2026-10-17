# DEV-052A — Frozen DEV-051 policy transfer to noisy public repairs

**Status: COMPLETED, independently audited adaptive development. NOT confirmation; NOT a trained recurrent PLM–OSM world model.**

* Protocol: [research/dev052/study-protocol.json](../research/dev052/study-protocol.json)
* Frozen DEV-051 source and checkpoints: original [successful run 39045522413](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39045522413), trained source `d514fb5bd34b2de3932925cd1d1beb3b480c85f3`.
* DEV-052A fixed source: `69e641e30ee0fa6c3f579486d6de3399981c22ba`. [Development evaluation and independent physical audit](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39047034949) **PASSED**, including **1,286 regression tests**.
* Veritas native runtime: pinned `b21d829b62d3cab2811e746267ca305e110ed004`.

## Scientific question and adaptive design

DEV-051 imitated hand-written inspect–verify–retry on two-entity, accurate-sensor worlds with guaranteed second repair. DEV-052A **froze** the exact DEV-051 learned BC and student-state aggregation heads (seeds 7101–7103, checksum validated) and the originally selected confidence gate (0.0). It evaluated new public worlds with:

* inspection **sensitivity/specificity 85%**; noisy readings deterministic per seed, world, entity and inspection count, but hidden state never sent directly to policy;
* original first-service outcome (~75% success), a **separately seeded 80% second-success chance**, and an additional guaranteed third attempt **only in this synthetic environment**;
* native paid inspect, isolate, service transitions, exclusive shared maintenance slot and final Veritas verifier, including blocked action and unfinished action checks;
* 64 new development/calibration worlds and 128 new evaluation worlds, disjoint from original DEV-051 worlds; same 128 evaluation worlds reused across model initializations. No model or threshold selected from DEV-052 evaluation.

This is an adaptively authored distribution-shift task after the DEV-051 result, not prospective independent evidence of generic neural inability. The old BC and DAgger policies had a two-service-attempt representation and were intentionally not given new third-service logic; the comparator was given the updated public action contract and known success probabilities. Thus the comparison establishes *frozen transfer* failure, not matched retraining or equal available action-space capacity.

## Results — 128 distinct evaluation worlds

| Policy | Cost per entity | Residual faults | Complete worlds | Blocked actions | Mean inspections/world |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explicit public Bayesian finite-horizon planner | **1.2875000** | **10** | **118** | **0** | **5.000000** |
| Old hand-written inspect–verify–retry | 1.3515625 | 29 | 100 | 0 | 2.968750 |
| Frozen DEV-051 behavior cloning | 1.3515625 | 29 | 100 | 0 | 2.968750 |
| Frozen DEV-051 student-state aggregation | 1.3515625 | 29 | 100 | 0 | 2.968750 |
| Frozen DEV-051 DAgger with its original 0.0 confidence gate | 1.3515625 | 29 | 100 | 0 | 2.968750 |
| Always service each entity three times | 3.6000000 | 0 | 128 | 0 | 0 |
| Wait without any actions | 2.0156250 | 129 | 36 | 0 | 0 |

The Bayesian planner reduced cost by approximately **4.74%** and residual faults by **65.52%** relative to the old frozen neural policies, with an average of 5 inspection calls per world versus 2.96875. It is a known-simulator, explicit public-information finite-horizon belief optimizer; it is **not** an exact-history oracle or globally certified optimal for all operating contexts. A low-cost strategy can still be unacceptable if its 10 residual faults breach safety requirements. The all-three-services controller has zero failures *because third repairs are guaranteed in this constructed simulator*, but it costs 3.6 per entity.

All three network seeds yielded identical actions and physical results, because they had already learned the same original rule. These outcomes reuse the same 128 worlds, so the aggregate log's 30 Bayes and 87 neural faults across three seeds must NOT be interpreted as independent 384-world replication. Original student-state aggregation and the confidence gate provided no transfer benefit over the old rule, and the gate never used fallback because it had been selected at 0.0 on DEV-051.

## Integrity

* Native model/source and original DEV-051 checkpoint hashes validated in each seed job and in an independent auditor.
* All three frozen transfer seed runs **passed**, plus a separate audit that reconstructed native paid tool calls, public sensor receipts, action traces, budget, blocked actions and final verifier outcomes for **every policy in every world**. No SGD was performed in DEV-052A.
* Split identity disjointness and unchanged weights enforced. The bounded synthetic third repair, repair stochasticity and sensor sampling rules were registered before execution.
* Regression suite: **1,286 tests passed**. Initial smoke on the same source passed before full seed evaluation.
* Original complete archive from successful Actions run includes all three raw per-seed reports and the independent audit analysis. SHA-256 archive verification and longer-lived GitHub release are handled separately; this report is not itself independent evidence of an immutable release.

## Falsification and what remains

**Result:** The existing learned action head was not economically or operationally superior to a public-information Bayesian decision rule under the adapted noisy-repair distribution. There was no demonstrated DEV-052A neural advantage. This does **not** isolate model representation as the cause: the transferred learner lacked third-attempt policy support, had never trained on noisy inspection receipts, and saw an altered action-consequence distribution.

**Next controlled DEV-052B:** train an entity-aware recurrent public-state encoder and action-conditioned predictor on *new disjoint noisy-repair histories*, with an appropriate third-service action interface, plus a policy head optimizing actual measured physical cost under a prespecified failure guard. Compare a properly tuned BC/DAgger continuation, a compact explicit Bayesian belief controller, a simple fixed-schema model, the same frozen DEV-051 heads and the all-service safety floor. Add OOD noise/repair-regime splits, unseen identities and multiple resource/repair constraints. The model must beat or match the public belief baseline in residual faults **and** cost, not merely improve imitation agreement.
