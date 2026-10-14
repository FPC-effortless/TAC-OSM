# DEV-015 — Actual finite PLM-to-CASM end-to-end training: small, inconclusive gains; no robust retention

**Classification: REPLAY-VALIDATED ADAPTIVE DEVELOPMENT ONLY. NEGATIVE for an established improvement from end-to-end *writer* learning or long-horizon retrieval.** This does not demonstrate SOTA, a true world model, learned verifier/repair, natural-environment dynamics, or end-to-end compute advantage.

## Evidence, provenance and integrity

Full [GitHub Actions run 38492915543](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38492915543) **succeeded**: eight new mechanism/gradient/no-oracle tests passed, live suite cardinality integrity **996**, smoke and full-grid independent saved-checkpoint evaluation replay completed.

Exact full artifact: `DEV-015-RAW-NOT-CONFIRMATORY`, GitHub Actions artifact ID **11862671552**, ZIP SHA-256 **e9f66a639177992f7ab70eb0d2e743f4d2955932b649505e1f10705eca2efba3**. Includes **15 trained state dict checkpoints** and `report.json` with source/checkpoint SHA-256, per-episode raw logits-derived classifications, bit posteriors, opposite-past targets, coordinate shifts, null/swap controls, exact Euclidean baseline, operator call counts and stratified summaries.

Validator loaded all 15 checkpoints, authenticated training/parent parameter hashes and exact source files and regenerated **4,800 model-episode decisions** across in-family and signed-permuted address observations (2,400 original, 2,400 transformed). Replay **does not reproduce SGD training**, certify independent scientific replication, or make 4,800 partner decisions independent seed replications.

Parent DEV-013 `aligned_aux` weight archive ID 11836120110, ZIP SHA `880d56e177b78d44961a8fc75e6e0a4151ebe6eeb7a55ce81afaa6c358172d65` and all 15 parent checkpoint hashes were validated. Original DEV-003–014 negative/mixed reports and experiments remain unchanged.

## Precisely what was measured

Five `PLMCASMBridge` arms per model seed 43/59/71, **96 training steps**, 4 opposite-past examples/step, matching data and **byte-identical initial parameters** for each seed. Every arm has **69,941 total model parameters** and exactly 4×8×24 = **768 persistent recurrent state scalars**. The learned state is constructed by the actual TAC-OSM recurrent DeltaMemory streaming update from only public observed noisy addresses/payload bits. The bit reader is the trained parent native CASM historical-bit head; its posterior feeds *linear* query/routed projections followed by a second, real TAC-OSM `ConditionalExecutor` and action answer head.

No gold historical bit, evaluator task identity, hidden target slot, future write, or true answer reaches inference; **there is no teacher-bit arm** (unlike the explicitly privileged DEV-014 control).

Arms:
- **frozen_task:** original trained PLM writer AND historical reader frozen; train action adapter only (**29,735 trainable parameters**).
- **reader_aux:** freeze original PLM writer, train historical reader and action adapter with task CE + offline true historical-bit CE (**65,171 trainable parameters**).
- **joint_task:** train writer, reader and action from task CE alone (**68,467 trainable parameters**).
- **joint_aux:** train writer+reader+action with task CE + offline true bit CE (**68,467 trainable parameters**).
- **joint_shuffled:** same all-trainable route and extra auxiliary CE but independently random bit labels (**68,467 trainable parameters**).

**Cost fairness caveats:** offline true-bit labels are extra supervision, not self-supervised online retrieval; gradient and training work differ by arm even when total parameters match. Every model also contains a duplicated conditional operator bank (bit read + action), so it is **not the original native TAC-OSM `predict`** and not parameter/FLOP-matched to DEV-013. Its recurrent state remains bounded but still ingests all H+D observations at O(H+D) work.

Evaluation at 3 independent model seeds × 5 arms × 80 *opposite-past pairs* per coordinate family × 2 partners × 2 coordinate families = **4,800 evaluated model episodes**. Conditions H=32/128/512, D=4/16/32, near interference 0/1, noise .08/.25, retention/acquisition. The signed/permuted-coordinate transformation preserves Euclidean distances and the target but is a **representation shift on the same synthetic cases**, not an independent task family.

## Held-out native-action-adapter accuracy

| Arm | Seed 43 | Seed 59 | Seed 71 | **Mean original axes** | **Mean signed/permuted axes** |
|---|---:|---:|---:|---:|---:|
| frozen_task | 51.25% | 59.38% | 55.62% | **55.42%** | 56.88% |
| reader_aux | 51.25% | 65.62% | 60.00% | **58.96%** | 56.25% |
| joint_task | 51.25% | 63.12% | 59.38% | **57.92%** | 57.29% |
| joint_aux | 53.75% | 63.75% | 55.62% | **57.71%** | 55.83% |
| joint_shuffled | 50.00% | 54.37% | 53.75% | **52.71%** | 54.38% |
| no-bit neutral control | 50.00% | 50.00% | 50.00% | **50.00%** | 50.00% |
| full-history Euclidean | — | — | — | **86.25%** | **86.25%** |

Full-history Euclidean uses O(H+D) external address/value storage and scans all records. It is NOT fixed-memory or compute-matched. Identical Euclidean accuracy after signed coordinate change validates the task-preserving transformation; this does **not** prove model invariance, and no arm consistently benefits from the shift across seeds.

### Pre-selected primary paired seed-level contrasts (Student-t 95%, df=2; exploratory)

| Comparison, native action accuracy | Mean difference | 95% CI |
|---|---:|---:|
| reader_aux − frozen_task | **+3.54 percentage points** | [-4.43, +11.51] |
| joint_task − frozen_task | +2.50 pp | [-2.88, +7.88] |
| joint_aux − frozen_task | +2.29 pp | [-3.16, +7.74] |
| joint_aux − reader_aux | **-1.25 pp** | [-9.89, +7.39] |
| joint_aux − joint_task | -0.21 pp | [-8.18, +7.76] |
| joint_aux − joint_shuffled | +5.00 pp | [-4.70, +14.70] |

**All intervals include zero.** There is **no robust evidence** that end-to-end training the recurrent writer is better than merely fine-tuning its reader, no demonstrable benefit of historical-bit supervision beyond shuffled controls on *native learned actions*, and no valid superiority claim against the true frozen baseline. The strongest seed is 59; seed 43 is near chance for almost all arms.

### Historical bit retrieval, retention and H=512 — original axes

| Arm | Historical-bit accuracy | Retention action | Acquisition action | H=512 action |
|---|---:|---:|---:|---:|
| frozen_task | 60.00% | 52.08% | 58.75% | 52.08% |
| reader_aux | 60.42% | 54.17% | **63.75%** | 53.12% |
| joint_task | **62.50%** | 53.33% | 62.50% | 53.12% |
| joint_aux | 60.00% | 53.75% | 61.67% | 53.12% |
| joint_shuffled | 58.12% | 52.08% | 53.33% | 51.04% |

The older retention condition remains distinctly weaker than recent acquisition. **No tested variant shows meaningful performance at H=512** (near 50–53%). Even under favorable readout fine-tuning, native decisions remain far below the full-history Euclidean reference.

The signed/permuted test does not establish representation robustness: e.g. reader_aux task 58.96%→56.25%, joint_aux 57.71%→55.83%, while frozen_task 55.42%→56.88%. Those are paired outcomes from the **same** synthetic cases and few seeds; do not promote any delta as independent generalization.

### Information quality and conditional operator dependence

Raw saved posteriors support the following **post hoc** Brier scores for the true historical bit (lower is better; constant 0.5 prediction = 0.25):

- frozen_task: **0.27267**
- reader_aux: **0.25242**
- joint_task: **0.28454**
- joint_aux: **0.25604**
- joint_shuffled: **0.24100** (mostly less confident, but also lower classification accuracy).

This is an instructive **calibration-vs-information confound**: the shuffled-control arm can earn a good Brier score by predicting nearer to 0.5 while performing worse at classifying the history and selecting an action. Do not equate calibration improvement with memory quality.

In-family no-CASM *non-retrained* ablation averages 50.0–50.8% across arms, whereas learned CASM actions reach 52.7–59.0%. This confirms behavior depends on the conditional operators inside these trained checkpoints but is **not** a fair separately trained non-CASM baseline. Mean operator sample calls per evaluated opposite-past 2-episode batch vary 2.91–4.35 across arms, and no benchmarked wall-time/FLOP claim is supported.

## Scientific decision and next experiment

**Outcome: NEGATIVE for proof that training the writer solves TAC-OSM's long-range retrieval or native decision problem.** DEV-015 demonstrates a genuine differentiable task pathway from action through TAC-OSM's finite writer, with reproducible gradients and saved decisions, but it does not establish a robust improvement over frozen/reader-only or shuffled controls. Improvement in newly trained action adapters cannot be promoted to the original hardened agent or native `CASMAlignedModel.predict`.

**Next falsification priorities rather than scaling CED:**

1. Introduce a genuinely **different temporal generative process** with structured state changes, delayed counterfactual intervention and action-dependent observations. Freeze the task and evaluator before training; keep matched finite-state budget and test pre-action decisions, not repair-adjusted outcomes.
2. Study non-oracle **write-address retention** and interference using task-blind contrastive/key-consistency objectives and query-aware state access; require gains on older retention **without** sacrificing recent acquisitions or increasing hidden state.
3. Compare with a strong fixed-capacity schema-indexed recurrent writer and modern gated-delta baselines, including optimized training and measured effective work; TAC-OSM already implements independent learned erase/write controls, so new gating names alone are not a novel contribution.
4. Only if information recovery improves and held-out native CASM decisions improve should the learned PLM be promoted into the full TAC-OSM hardened planner/verification/repair loop. A green CI run or 3-seed synthetic difference is insufficient.

All prior DEV-003–014 results and original TAC-OSM architecture remain preserved; this PR should remain draft and unmerged until independent review and nontrivial generalization evidence.
