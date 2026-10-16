# DEV-018 — Audited memory attribution and horizon results

**Outcome:** completed adaptive-development experiment. **TAC success: FALSE.** This is not confirmatory research, not a full native CASM execution comparison, and not a superiority claim about all transformers or recurrent models.

## Frozen evidence and reproducibility

- Training and independent stdlib audit: [GitHub Actions run 38802569971](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38802569971), **SUCCESS**. Source head commit `6b319dee36291021bc486c51cd40136af4b08d4e`.
- [Complete audited artifact 12006605848](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38802569971/artifacts/12006605848) — 97,006,137 bytes, archived SHA-256 `e297a432e1b2eec708731adfb5547cf3908bbbfb910ad443df225b5624001e31`.
- All six training jobs (three seeds × short/long exposure) and separate independent audit succeeded. **60 checkpoint models, 86,400 held-out model-episode evaluations**; neural checkpoint replay and evaluator-side target, action, cost and sensing audits completed. Replaying frozen weights is not rerunning optimization.
- Inspectable primary source: [audit job](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38802569971), whose `AUDITED` JSON gives exact means/contrasts/gates, plus individual seed-job `SUMMARY` JSON. The complete per-seed.csv, accuracy-by-history-delay.csv, failure-service-frontier.csv and analysis.json reside in the archived evidence. [Export run 38802820475](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38802820475) printed a SHA-256-identified compressed text export, but those tables are **not yet committed as plaintext**.
- A subsequent [run 38802820237](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38802820237) only passed smoke and intentionally skipped large jobs because its PR head differed from the pinned training SHA. **Do not mistake this for a missing full run.**

## Primary endpoint

H512, pre-sensing pending/reissued-cancellation, at/after due, registered delays 32/128/512, balanced opposite-future pairs. Chance/majority baseline is 50%. Numbers below are three-seed means; all three seed values follow.

| Arm | Short primary | Long primary | Short selected cost/entity | Long selected cost/entity |
| --- | ---: | ---: | ---: | ---: |
| memoryless | 50.00% | 50.00% | 1.0000 | 1.0006 |
| gru768 | 50.00% | 50.00% | 0.9989 | 1.0003 |
| delta768 | 50.00% | 50.00% | 1.0000 | 1.0000 |
| schema41 learned | **96.88%** | **90.28%** | **0.5051** | **0.5105** |
| schema41 frozen GRUs | 94.79% | 87.50% | 0.5055 | 0.5090 |
| tac_joint | 50.00% | 50.00% | 0.9961 | 1.0000 |
| tac_encoder | 50.00% | 50.00% | 1.0023 | 1.0000 |
| tac_memory | 50.00% | 50.00% | 1.0145 | 0.9995 |
| tac_reader | 50.00% | 50.00% | 1.0132 | 0.9999 |
| transformer (small full-prefix reference) | 50.00% | 50.00% | 1.0000 | 1.0000 |

Per-seed primary accuracy (seeds 43, 59, 71, in that order):

| Arm | Short seed 43 / 59 / 71 | Long seed 43 / 59 / 71 |
| --- | --- | --- |
| schema41 | 94.79% / 95.83% / 100.00% | 87.50% / 94.79% / 88.54% |
| schema41 frozen | 91.67% / 92.71% / 100.00% | 85.42% / 88.54% / 88.54% |
| all eight other arms | 50.00% / 50.00% / 50.00% | 50.00% / 50.00% / 50.00% |

## Frozen interpretation

1. Reliability and cost success: **schema41 succeeds in both regimes**. The frozen-GRU schema control succeeds in short, but fails in long. Every other arm fails. The hard preset primary threshold was >=80% per seed alongside H512 failure <=5%, unnecessary service <=10%, cost <=0.80, and mean selected cost <=no-sensing.
2. Trained native TAC writer attribution: **FALSE**. Both joint-minus-encoder and memory-minus-reader primary contrasts are exactly 0.00 pp in both regimes across all three seeds; long TAC joint state-swap primary remains 50%. Native TAC full success: **FALSE**.
3. The structured inductive bias dominates on this bounded task, but learning GRU recurrence itself is not isolated as the explanation: learned-minus-frozen schema41 difference is only +2.08 pp short, +2.78 pp long, with three-seed exploratory 95% intervals including zero.
4. The controlled 512-history training recipe **does not rescue** native TAC critical prediction, nor the tested GRU, standard gated-delta equation baseline, or small full-prefix transformer. This is not proof that history length alone has no effect: short/long regimes match steps and pre-sensing events but differ in numbers of episodes, supervised labels and inspection receipts.
5. In the long regime, tac_joint selects **1.0000 cost/entity**, with ~51.53% unnecessary services and zero failures (mean), essentially the always-service policy. Schema41 selects ~0.5105 cost/entity, ~1.14% unnecessary service, ~0.48% failure. Low failure without controlling unnecessary interventions is not an efficiency or safety victory.
6. The small full-prefix transformer failing shows only that this *particular model/training recipe* did not learn the task; it cannot diagnose an inherent transformer limitation. Scalar budgets match GRU/delta/TAC recurrent states at 768 but not parameters, training FLOPs, optimization ease, or time. Schema41 explicitly receives public entity indexing and this structural prior is decisive.

## Research continuation

DEV-019 read-only forensics on DEV-017 established 0/768 native TAC predicted-class disagreements for balanced opposite-future pairs. DEV-018 independently reproduces 50% native prediction accuracy with different horizon/exposure controls. This narrows the failure, **not** its mechanism: it remains unknown whether state encoding, state overwrite/interference, learned readout, or optimization is responsible.

Next research should freeze original DEV-018 checkpoint hashes, instrument internal state / target-query read features / logits on identical opposite-past histories and controlled prefixes, and use truly disjoint training/test histories for any diagnostic probes. A positive held-out state probe with a failed original readout supports a reader bottleneck; a negative limited-capacity probe **does not prove information absence**. Preserve original DEV-018 endpoints, gates and artifacts without edits.

## Scientific caveats

- Synthetic, eight-entity maintenance domain; no general world-model or commercial deployment claim.
- Adaptive development, three seeds, exploratory uncertainty, no independent retraining on another organization or hardware.
- Held-out replay checks saved checkpoint/evaluation consistency; standard library audit recomputes public world targets and costs; neither is external scientific replication.
- DEV-018 uses native TAC `DeltaMemory` with common read interface, **not the complete integrated CASM verifier/repair loop**.
