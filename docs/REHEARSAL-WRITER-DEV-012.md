# DEV-012 — Writer rehearsal and protection on TAC-OSM fixed recurrent memory

**Classification: DEVELOPMENT-ONLY / NOT A CONFIRMATORY OR STATE-OF-THE-ART RESULT.** This experiment follows DEV-011's negative finding that pretrained and untrained writers supplied comparably weak recoverable historical information (~55–57% through an independently supervised small keyed decoder) relative to full-history Euclidean retrieval. All prior DEV-003–011 source/results and frozen baselines remain unchanged.

## Central hypothesis and architecture

Can direct supervision of *previously observed* address/value writes train TAC-OSM's existing finite learned recurrent memory to preserve facts under competing writes, without removing its ability to acquire recent facts?

We DO NOT replace TAC-OSM with an LLM, unconstrained external memory, CED decoder, or oracle retrieval. The existing learned **DeltaMemory writer [4 experts, 8 value channels, 24 keys] = 768 state scalars** is unchanged. The native `ProtectionModel.predict()` path still performs learned query, expert selection, conditional structural operator computation and binary decision. The additional **auxiliary training-only bit head** reads fixed recurrent state at a noisy historical address; it introduces trainable parameters but no extra *persistent* capacity. The `rehearsal_logits` route is diagnostic, not a replacement for the original decision.

## Five initially identical writer arms

Every arm starts with byte-identical parameter tensors and uses the *same* synthetic training episodes / optimizer hyperparameters / number of updates:
- `task_only`: always-update writer; original task classification CE only; rehearsal head untrained.
- `rehearsal`: always-update writer; task CE plus historical-bit rehearsal CE.
- `shuffled_rehearsal`: always-update writer; same loss structure but auxiliary labels sampled independently from the observed payload, testing generic regularization vs meaningful rehearsal.
- `fixed_rehearsal`: existing constant-gated protection (sigmoid(2)); true rehearsal CE.
- `learned_rehearsal`: existing learned per-expert protection (before each write, from public state/input); true rehearsal CE. Has extra *active* gate parameters/compute vs other arms, so it is **not** parameter/compute matched.

This uses existing `ProtectedMemory` write protection as the candidate; not faithful implementations of Gated DeltaNet-2 or SchemaMem. Their peer-reviewed and preprint ideas motivate future alternatives but confer no performance on TAC-OSM.

## Permitted training information and temporal constraints

At each training step, two opposite-past episode pairs are generated with retention and acquisition roles. Each episode supplies an ordered history of public noisy **addresses** and observed binary **payloads** to the existing causal writer. The original answer target is evaluator-only and is used solely in the task CE.

Training-only **rehearsal queries** are independently jittered addresses from two already visible events per episode: one earlier history item and one newer write. Their payload bits are used only in auxiliary classification loss. Timestamp choices do **not** use the evaluator's future query target slot, role identifier, or gold answer. The model's forward pass never receives the target position or either auxiliary label. This is offline hindsight supervision on actually observed facts; it incurs two additional historical association reads per training episode and thus cannot be claimed as free online learning.

The shuffled arm draws independent Bernoulli bits from a separate seeded RNG; it does NOT invert labels or shuffle paired opposites in a way that accidentally teaches the true bit.

## Protocol and measurement

Train 3 seeds 43,59,71; 120 AdamW steps, LR .003 and gradient clipping 1.0, four training episodes per update per arm. Train H=32/128, competing D=4/16, near-address fraction 0/1, noise .08/.25, with retention and acquisition present in each step. The setting remains a developmental adaptation, not prospective independent preregistration.

Evaluate on the same synthetic opposite-past stream across **all five** arms with held-out generator seeds (241,257,269); two replicates of the 40 DEV-010 conditions, including H=512/D=32. This is 80 opposite-past pairs = 160 episodes per model per seed, 2,400 held-out model-episode predictions total.

Primary metric **native TAC-OSM task decision accuracy** (before repair), by seed, historical retention versus recent acquisition, history H=512, and near-address interference. Report zero-state/reset and pair swap controls. Secondary: auxiliary head's recovered bit and reset/swap, Euclidean full-history comparator (charge O(H+D) work), actual state difference under a past-payload flip, gradient to writer and protection gates, active parameter counts, and final task/auxiliary losses.

Critical **evaluation separation**: no gold timestamp enters query-time inference; zero-state controls should score exactly 50% on opposite-past pairs. Train/eval visible-input fingerprints are disjoint. The full-history comparator is a nonlearned reference with unbounded external storage and is NOT comparable on persistent capacity/cost.

## Interpretation and minimum gates

- Meaningful rehearsal must beat **both** task-only **and** independently shuffled-label rehearsal across matched seeds; otherwise it is not evidence of better memory learning.
- A protection benefit must improve earlier retention while preserving fresh acquisition, not simply freeze all writes.
- If rehearsal improves only the auxiliary diagnostic but leaves native conditional-execution decisions at chance, it is **not** a TAC-OSM end-to-end performance improvement.
- If learned protection does no better than a constant gate, learned targeted protection is **not** established.
- If all arms remain near 50% at H=512, no long-horizon memory capability has been demonstrated. Keep full negative data.
- With three seeds, do not treat 2,400 partner episodes as independent evidence of architecture superiority. Use seed-level paired contrasts and uncertainty; report separate per-condition metrics.
- Because the auxiliary loss modifies training budget and computational cost, **there is no wall-time / efficiency superiority claim** even when predictive accuracy rises.
- **Replay** verifies checkpoint/source hashes and all held-out inference rows from the saved weights, not training SGD, protected architecture or independent preregistration.

## Literature context

- Gated DeltaNet-2 (2026): https://arxiv.org/abs/2605.22791
- SchemaMem (2026): https://arxiv.org/abs/2609.33436
- Prior experiment: `docs/FROZEN-PLM-PROBES-DEV-011.md`

Follow-up DEV-013, *only after results audit*: construct a budget/capacity-matched differentiable write novelty/rewrite gate or schema-indexed recurrence with proper interference and acquisition controls, then test in the complete hardened verifier/repair loop. Do not silently promote this development intervention to an architectural breakthrough.
