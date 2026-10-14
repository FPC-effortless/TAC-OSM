# DEV-015 — Train the TAC-OSM finite-PLM memory-to-CASM interface end to end

**Status: ADAPTIVE DEVELOPMENT ONLY; not state of the art, not full hardened-loop integration, not external scientific confirmation.**

## Why this experiment is necessary

DEV-014 established a narrow diagnostic: the inherited TAC-OSM `ConditionalExecutor` with correctly supplied **oracle** historical bit achieved 100% on the deliberately simple XOR rule, while the *frozen learned PLM bit posterior* achieved ~60% bit accuracy and the additional trained CASM adapter ~57% task accuracy. **Oracle results are not real-memory results.** DEV-014 could not establish whether training gradients from actual task decisions into the PLM writer and bit reader would improve non-oracle retrieval or action accuracy.

DEV-015 connects the real finite learned memory to a CASM action translator **in one differentiable task forward**. Unlike DEV-014, no arm receives teacher bits in forward or evaluation. The **only recurrent state** remains inherited 4×8×24=**768 scalars**, updated through observed address/bit events. The model computes its historical-bit posterior from that state, then passes it and the public current bit through linear translators into the actual inherited TAC-OSM conditional operator bank and action head.

**Architectural limitation:** this is still a new additional two-stage supervised action interface, **NOT** the exact unmodified native `CASMAlignedModel.predict`, not a closed-loop PLM/CDL/CASM/verifier/repair model, and not a DeepSeek/CED result. Its bit-reader and action CASM each execute a conditional operator path, adding parameters and runtime work. The original TAC-OSM decision from DEV-013 remains a separate comparison, not a matched baseline. The teacher-bit oracle is excluded from DEV-015 entirely.

## Immutable parent

Download the exact DEV-013 15-checkpoint GitHub Actions archive ID **11836120110**, ZIP SHA256 `880d56e177b78d44961a8fc75e6e0a4151ebe6eeb7a55ce81afaa6c358172d65`. Use the `aligned_aux` 768-scalar recurrent writer and associated historical-bit head at each seed 43/59/71. Verify all parent checkpoint file hashes and the loaded parameter hashes. **Never silently initialize a substitute.**

## Five matched model arms (same architecture and initial tensors)

All five arms share exactly the same tensor values and parameter count per seed; the optimizer is AdamW with LR .002, weight decay .0001 and clip norm 1.

- `frozen_task`: freeze the entire historical memory model (writer and bit reader), train only newly added CASM action adapter with native task labels. This is the closest DEV-014-like non-oracle control.
- `reader_aux`: freeze the entire recurrent memory writer/key/gates, train the reader and CASM adapter with native task CE **plus** true historical-bit CE as an offline supervised auxiliary target. The trained memory container itself is unchanged.
- `joint_task`: train recurrent writer, historical-bit reader, and action adapter using **only the native task CE**, allowing end-to-end action gradients to the PLM.
- `joint_aux`: train the full writer+reader+action with native task CE plus true historical-bit CE. This tests whether structured auxiliary labels solve an optimization bottleneck.
- `joint_shuffled`: same full gradient routes and two loss terms as joint_aux, but labels for historical-bit CE are independently sampled random bits. A positive difference against joint_aux supports the value of **actual association supervision** rather than mere extra-loss training.

Bit CE labels are derived by the offline trainer as `task_target XOR public_current_bit`. This is additional explicit **teacher supervision during TRAINING**, not a hidden bit in inference. It makes DEV-015 different from an unsupervised online agent; do not claim autonomous self-supervised improvement. At inference **no gold bit, hidden slot, task identity or correct action** crosses into forward. The same native `ConditionalExecutor` executes actual sparse top-1 operator selection for all arms. Parameter count is held fixed but number of **trainable** parameters and gradient compute differ by arm; cost fairness is incomplete.

Training: 3 independently initialized seeded parent weights 43/59/71, **96** matched training steps/arm, 4 counterfactual samples each; H=32/128, competing writes D=4/16, near-address distractors, noise .08/.25, retained vs acquired task roles. This is exploratory, not preregistered.

## Held-out and coordinate shift

Use deterministic disjoint `dev014-eval` procedural case seeds with 2 repetitions of 40 conditions H=32/128/512, D=4/16/32, near distractor fraction 0/1, noisy addresses .08/.25, retention/acquisition. Every evaluation pair shares query/current/present addresses, differs in one historical payload and has opposite target answers.

**Two evaluation families** for each identical episode:
1. **Original axis representation** (as generated);
2. **Signed/permuted coordinate representation**, generated from a deterministic seed and applied uniformly to every address and query coordinate. This is a new coordinate-system stress of the *same* underlying task, not an unrelated domain. It exactly preserves Euclidean distances and gold target. This helps detect reliance on address-coordinate conventions rather than transferable relational memory.

80 opposite-past pairs/family/arm/seed × 2 family variants × 2 partner episodes × 5 arms × 3 seeds = **4,800 model-episode inferences**, saved in report and evaluated from 15 checkpoints. The signed/permuted cases share information with original cases, so **do not count them as independent external observations**.

Primary: paired-seed mean **actual learned CASM action accuracy** for in-family and transformed coordinates. Separately report bit-posterior classification accuracy, retention vs acquisition, H=512 stress, counterfactual swap, no-CASM *non-retrained* ablation, and exact full-history Euclidean reference including number of address records scanned. Null historical bit p=[0.5,0.5] must yield **50.00%** on both coordinate families for all arms.

## Measured/charged costs

- Exactly 768 persistent recurrent PLM scalar elements per episode; the writer still consumes **every** observation O(H+D).
- The bridge contains a full learned historical-bit reader (including a native CASM pass) and a separate conditional action executor. Its model parameters and execution cost exceed the unmodified native model; we will save per-arm parameter counts and operator sample calls.
- `reader_aux`, `joint_aux` and `joint_shuffled` use extra offline teacher bit supervision; they are not same-budget comparisons with task-only training.
- Raw Euclidean reference scans every event and requires external O(H+D) storage; it does not satisfy the fixed-memory constraint.
- There are **no** direct future-state/environment dynamics predictions, no online truth-free write rewards, no cached inference speed measurements, no real-world benchmark, and no end-to-end sublinear addressing.

## Required interpretation before any architectural promotion

1. A positive `joint_task` versus `frozen_task` at **same model capacity** suggests value of allowing action gradients into the recurrent memory; nevertheless the training cost differs.
2. A positive `joint_aux` versus **both** `joint_task` and `joint_shuffled` suggests explicit historical-bit supervision contributes information. If only the bit metric rises but task decisions remain weak, mark **negative task transfer**.
3. A `reader_aux` result similar to joint_aux implies memory read/decoder retraining, not writer fine-tuning, explained improvement. Only an improvement beyond `reader_aux` can support learned-writer adaptation.
4. If signed-coordinate stress collapses even while exact Euclidean retrieval stays stable, do not claim learned representation invariance or outside-family generalization.
5. If H=512 or retention stays at chance, no robust long-term-memory claim. If acquisition rises by forgetting old information, this is not a balanced improvement.
6. Seed-level 95% t intervals have only 3 independent writer initializations; treat evidence as tentative even if nominal improvement appears. Preselected observations come from a familiar synthetic generator family, so no independent confirmatory result is possible.
7. The full report must include all negative seeds, reset leakage checks, exact file/source SHA-256, all 15 checkpoint states and **replay every evaluation row**. Replay does **not** reproduce the training SGD path.

Do not merge this stacked exploratory PR or rewrite previous DEV-003–014 results as positive TAC-OSM scientific advances. A future test must evaluate a genuinely different temporal process and the full hardened agent cycle.
