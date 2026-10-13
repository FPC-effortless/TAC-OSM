# DEV-012 — Rehearsal writer/protection results: bit recovery improves, native TAC-OSM decisions do not

**Status: NEGATIVE FOR END-TO-END DECISION TRANSFER, POSITIVE AUXILIARY *DIAGNOSTIC* DEVELOPMENT SIGNAL. This is NOT CONFIRMATORY or a state-of-the-art claim.**

Evidence: [GitHub Actions run 38369210699](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38369210699) completed successfully. Eight mechanism tests passed, live suite count integrity validated (**972 collected tests**), smoke and full-grid saved-checkpoint evaluation replay passed. Full artifact **DEV-012-RAW-NOT-CONFIRMATORY** (ID 11798218707; archive SHA256 `c200d93bfb45b8947b454151704c3f84a49581debc21136a439abbd34862617a`). Archive includes **15 trained checkpoints and 2,400 per-model decision/bit evaluation episodes**; source/checkpoint hashes plus each raw condition/prediction were replayed. This is evaluation replay from checkpoints, **NOT** reproduction of optimizer trajectories, preregistration, or independent external validation.

## Experimental conditions and confounds

Five arms started with identical tensors and identical benchmark streams; seeds **43, 59, 71**, each **120 AdamW updates** with four opposite-past training episodes/update and existing native TAC-OSM learned PLM/conditional operator readout. No raw history is retained at query time: each writer still has **4×8×24 = 768 persistent scalars**. Rehearsal is an additional **training-time** supervision target, made from two already-observed prior address/value pairs with independent query noise; it increases training compute and uses historical addresses outside the runtime state for offline loss only. Learned protection adds **929 actively trainable gate parameters** (39,531 versus 38,602 in other arms), so parameter/FLOP matching is incomplete. The new auxiliary head exists in task-only too but is **not trained** without rehearsal.

Held-out: two replicates of 40 factorial cells (retention vs acquisition; H=32/128/512; competing writes D=4/16/32; near collisions fraction 0/1; noise .08/.25), 80 opposite-past pairs=160 decisions/seed/arm. Each of the 15 trained models saw exactly the same seeded evaluation pair observations. No inference forward receives evaluator gold target, target slot, future write, or task role. All reset-memory task/bit controls returned **exactly 50.00%**, as required by the counterfactual-pair construction.

## Exact held-out results (means across 3 independent training seeds)

| Writer training arm | Native TAC-OSM task accuracy | Auxiliary historical-bit recovery |
|---|---:|---:|
| **task_only** (always-update) | **51.46%** | 51.04% (untrained auxiliary head) |
| rehearsal (always-update) | 50.83% | **60.42%** |
| shuffled_rehearsal (null labels) | 49.38% | 48.96% |
| fixed_rehearsal (constant gate) | 52.50% | 60.63% |
| learned_rehearsal (learned gate) | 52.29% | 60.42% |
| full-history Euclidean reference | 85.63% | Not comparable on state/storage/read cost |
| zero-memory control | 50.00% | 50.00% |

Primary task accuracy by seed (43/59/71):
- task_only: 48.75 / 52.50 / 53.12%.
- rehearsal: 49.38 / 51.25 / 51.88%.
- shuffled_rehearsal: 48.75 / 48.75 / 50.62%.
- fixed_rehearsal: 57.50 / 50.00 / 50.00%.
- learned_rehearsal: 57.50 / 47.50 / 51.88%.

Secondary bit-recovery accuracy by seed (43/59/71):
- rehearsal: 59.38 / 61.88 / 60.00%.
- shuffled: 51.25 / 50.00 / 45.62%.
- fixed: 60.00 / 61.25 / 60.62%.
- learned: 60.00 / 60.62 / 60.62%.

### Paired seed-level uncertainty (three seeds, exploratory 95% Student-t, df=2)

- Rehearsal **bit recovery minus shuffled**: +11.46 percentage points, **[+3.64,+19.27]**. Distinct content-supervision signal in this synthetic development family, not a general capability result.
- Rehearsal **bit recovery minus task_only**: +9.38 pp, [+2.26,+16.49]; task-only head is untrained, hence this is not a capacity-matched reader comparison.
- Rehearsal **native task minus task_only**: **−0.63 pp, [−3.31,+2.06]**. No evidence that the improved auxiliary retrieval carries into native decisions.
- Fixed-gate rehearsal native task minus task_only: +1.04 pp, **[−15.56,+17.64]**. A single seed drives its apparent best result.
- Learned-gate rehearsal native task minus task_only: +0.83 pp, **[−16.82,+18.49]**.
- Learned-gate minus fixed-gate native task: −0.21 pp, [−5.66,+5.24]; adaptive memory protection benefit not established.

Do **not** infer independent episodes are independent training replications. The observed positive bit score is *diagnostic*: all writers remain far below the **85.63%** full-history Euclidean reference, and that reference uses O(H+D) external storage/address scans. It is not a state-capacity-matched efficiency baseline.

## Role, interference, and H=512 stress

| Arm | Retention task | Acquisition task | H=512 task | H=512 bit recovery |
|---|---:|---:|---:|---:|
| Task only | 52.50% | 50.42% | 51.04% | 51.04% |
| Rehearsal | 50.42% | 51.25% | 51.04% | 54.17% |
| Shuffled rehearsal | 50.42% | 48.33% | 50.00% | 48.96% |
| Fixed-gate rehearsal | 53.75% | 51.25% | 50.00% | 53.12% |
| Learned-gate rehearsal | 49.58% | 55.00% | 50.00% | 54.17% |

For real rehearsal, diagnostic bit recovery at retention/acquisition = **56.25% / 64.58%**. The older fact remains notably harder than acquiring a newer write. No arm has demonstrated reliable long-horizon task use: H=512 native task stays approximately at chance.

Zero-memory results are exactly 50% for each arm; swapping each counterfactual pair's recurrent state produces bit-recovery accuracies around 39–40% for the real rehearsal variants, as expected when a weak learned signal inverts under past-bit swapping. These controls support the presence of *some* historical information at the auxiliary head, **not** robust persistent memory.

## Interpretation: answer to DEV-011's bottleneck

1. **Real historical-bit supervision trains a decodable readout** from the bounded recurrent memory; independently shuffled supervision does not. This is an actual development diagnostic, but it may reside in the new auxiliary read head and altered representation rather than translate to the original execution system.
2. **The original TAC-OSM decision path remains near chance.** Adding a training-only auxiliary historical recall head does **not** demonstrate improved end-to-end TAC-OSM decisions.
3. **The existing learned protection gate does not beat constant-gated protection.** Slow/fixed updates are an appropriate comparator, and fresh acquisitions must not be sacrificed to retain older writes.
4. **Long retention is unresolved.** The writer processes every event O(H), and H=512 accuracy is weak. Neither state capacity nor compute claims have changed. CED attention was intentionally not a factor in DEV-012: earlier DEV-010 had already isolated its limitations.

## Next architectural research gate (DEV-013, design only)

Instead of enlarging CED, test **whether the auxiliary historical-bit head can be coupled to the original CASM decision pathway without bypassing structural execution**. Freeze/save the same trained writers; use a **diagnostic shared readout alignment** or an intervention that routes only memory-derived features to the existing execution mechanism. Compare trained writer vs frozen untrained writer at matched head supervision. At the same time, examine a capacity-matched *writer-level* alternative inspired by Gated DeltaNet-2's decoupled erase/write or SchemaMem's schema-indexed recurrence, with explicit write/amortized/update cost and acquisition checks. These are future experiments, not results of this PR.

No earlier baseline, previous DEV-003–011 failure report or archived result has been changed. This PR is deliberately stacked/draft and not ready for merge.
