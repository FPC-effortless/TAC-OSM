# PREDICTIVE-WORLD-DEV-006 — action-conditioned future prediction

**Status: DEVELOPMENT DESIGN ONLY, NOT A CONFIRMATORY RESULT.**
**Parent:** PR #161, branch `research/retention-protection-dev-005-2026-10-12`.
**Scientific question:** Does action-conditioned prediction of a future *latent target*, given finite learned memory, transfer to improved held-out decisions under noisy addressing and competing writes, compared with training the decision alone?

## Why this experiment

DEV-003 achieved only 54.86% task accuracy versus 86.81% Euclidean; DEV-004 isolated likely retention/interference; DEV-005 did not show a benefit from a per-expert protection gate. A world-prediction objective is **not** an established remedy. V-JEPA 2-AC, DreamerV3 and TD-MPC2 provide external precedents for action-conditioned latent dynamics and planning, not evidence for TAC-OSM's architecture.

This is a **limited prediction-objective experiment**, not JEPA pretraining or a world model of real environments. The target is a fixed, low-dimensional synthetic transition, so successful prediction does not establish open-world understanding. There is no learned evaluator, deployed planning, or demonstrated economic compute scaling.

## Causal data construction

Use the existing DEV-005 `make_pair` and `tensors` without changing that benchmark. Each pair contains the **same present query and current bit** and a **single different historical payload**; the hidden bit is revealed only to the offline trainer/evaluator. Pair partners receive the **same action**. For target bit `b`, present bit `c`, and action `a ∈ {-1,+1}`, define `m=2b-1`, `q=2c-1` and the fixed world transition:

```
z_next(b,c,a) = [0.7*m + 0.4*a*q,
                 0.8*q + 0.4*a*m,
                 0.6*a*m + 0.25*q]
```

The model sees only `addresses, values, query, current_bit, action`. At inference it cannot see `b`, `z_next`, task role, target slot, action optimum, evaluation pair id, or label. The future target is computed from the evaluator's past bit **after** sampling the action. The formula is not given to the model. In the paired examples, no-memory prediction has irreducible uncertainty for the terms involving `m`; without action conditioning it also cannot represent both counterfactual outcomes.

This is a **designed latent** (not a frozen learned encoder). It is intentionally transparent to allow causal diagnostics and must not be described as a JEPA representation.

## Training arms

Every arm starts with byte-identical weights and trains on the **same shuffled paired examples**, optimizer, budget and architecture:

- `task_only`: cross-entropy on the existing binary decision; world head exists but receives no loss.
- `future_only`: action-conditioned latent MSE; task head exists but receives no loss.
- `joint`: both losses with frozen weights `1.0 CE + 1.0 MSE`.
- `joint_shuffled`: same two losses but action-conditioned target is swapped **between different training examples**, destroying correspondence of future outcomes to the correct history/action. This is an auxiliary-label quality control, not an oracle arm.

All arms use the inherited finite 768-scalar `ProtectionModel(mode='always_update')` memory to isolate the prediction objective from the unsuccessful protection gate. No pre-trained checkpoints are transplanted. The newly attached future head is identically initialized in all four arms. The unselected heads are retained for parameter-count equality, but active-gradient compute is **not** matched; report both separately. No checkpoint selection from evaluation outcomes.

## Frozen development grid

Training seeds `[43,59,71]`, independent evaluation seeds `[241,257,269]`; 120 training steps/seed/arm, batch size 8 (complete counterfactual pairs), AdamW learning rate 0.003, clip 1.0. Train alternates history 32/128, new writes 4/16, near competitors 0.0/1.0, noise 0.08/0.25 and retention/acquisition. Evaluation uses histories 32/128/512, writes 4/16/32, near fractions 0.0/1.0, noise levels 0.08/0.25, roles retention/acquisition, 2 pairs per cell/seed. Compute the evaluator labels separately from model-visible inputs. This is expensive with dense writes; smoke runs are explicitly out-of-contract.

## Endpoints and interpretation

**Primary development endpoint:** held-out task exact accuracy, matched across arms and separated into retention/acquisition and H/D cells. **Secondary:** future latent MSE, versus public-only conditional-mean baseline; counterfactual response rates (both past-bit and action interventions); model-state sensitivity; gradients to memory write/read/operator routes; per-step actual write and operator-call counts; training curves; seed-level paired gaps/intervals. Evaluate `task_only` against `joint` first, then `future_only`, then `joint_shuffled`. No claim of benefit unless all seeds support a meaningful improvement and uncertainty is disclosed. An MSE improvement without task improvement is a world-head diagnostic, not a memory or planning improvement.

**Validity gates:** equal initialization, identical data fingerprints across arms, different train/eval fingerprints, no model input gold fields, exact counterfactual paired labels, action effect on target, non-degenerate no-memory target, trained weight integrity, finite loss and nonzero gradient along the intended memory pathway, and preserved raw rows sufficient for independent checkpoint replay. Detect trivial future heads that use only action and present bit. Persist all negative outcomes. If any gate fails, reject the measurement, not just the interpretation.

## Decision rules

- **Joint increases held-out task accuracy over task-only across seeds AND counterfactual state sensitivity improves:** develop a fresh, prospectively registered follow-up with stronger latent representation and external domains. This development run itself never establishes a general capability.
- **Future MSE improves but task accuracy does not:** report predictive fitting without decision transfer; do not add the objective to the core architecture as a demonstrated improvement.
- **Joint harms accuracy or has mixed/uncertain effects:** preserve the negative outcome; investigate gradient interference or objective balancing rather than selecting an advantageous seed.
- **Reset, input-leakage, or counterfactual gates fail:** declare instrument invalid, regardless of scores.

This experiment does **not** train a computation-outcome/value-of-compute prediction objective yet. That is a separate second axis requiring prospective targets built from independently executed counterfactual compute policies and strictly charged total work. Calling a task-dependent future head "prediction of computation" would be misleading.

## Literature (precedent, not inherited claims)

- V-JEPA 2 (2025), https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/
- DreamerV3 (Nature 2025), https://doi.org/10.1038/s41586-025-08744-2
- TD-MPC2 (ICLR 2024), https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html
