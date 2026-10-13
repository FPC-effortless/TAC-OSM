# DEV-013 — Align historical memory supervision with TAC-OSM's actual CASM readout

**DEVELOPMENT ONLY. NO CLAIM OF A STATE-OF-THE-ART ARCHITECTURE, LONG-TERM MEMORY, OR END-TO-END SOTA PERFORMANCE.** Stack on draft PR #168, and preserve every earlier failure report and immutable benchmark artifact.

## Motivation / explicitly tested failure explanation

DEV-012 found *real auxiliary* historical-bit recovery around 60.4%, but the original TAC-OSM decision remained ~50.8%. The auxiliary reconstruction head in DEV-012 used a separate expert-memory projection and MLP. The native prediction read path used learned top-2 expert relevance routing, conditional operator selection, 3-depth halt behavior and a separate task answer head. Therefore historical information available to the auxiliary head might never influence the native CASM representation.

**Hypothesis, not fact:** giving the *same original CASM feature path* historical-bit supervision can transfer some of the retrieved signal to the native decision. This cannot be inferred from existing data because information loss in finite recurrent state, insufficient training, noisy synthetic aliases or the XOR answer transformation might also limit transfer.

## Intervention and causal boundaries

The new `CASMAlignedModel` is an extension of DEV-012's existing `RehearsalWriterModel`. It preserves the original `ProtectedMemory(mode='always_update')` writer, trained noisy addresses, 4×8×24=**768** state scalars, learned expert keys, learned top-2 read selection, existing `ConditionalExecutor` with actual sparse operator dispatch and halt, and original `answer` classifier. `casm_decision_features` expresses the native routing and computation, returns (post-CASM, query) features, and is explicitly tested against the previous `predict` output with identical initial weights. No system verifier/repair or OSM planner is trained or invoked; this is an integrated **learned finite-PLM + native CASM** development test, not the fully hardened loop.

`aligned_historical_logits` runs the same relevance-router+CASM computation with a neutral current bit (=0), then predicts the observed historical payload using a training-only 2-class head attached to post-CASM features. Unlike DEV-012, that auxiliary loss can backpropagate through the exact modules used by the original decision. No additional persistent history or recurrent scalars; auxiliary training uses extra CASM calls, and is not free. Auxiliary inference and the head do not replace the native `predict`. The evaluator additionally records a **diagnostic-only** algebraic XOR of recovered bit and visible current bit to expose whether the remaining failure is in the native answer mapping. That XOR is **not** an actual TAC-OSM decision and must never be promoted as such.

## Five identical-parameter arms

All model instances begin from **byte-identical tensors** per seed, with the same architecture and same total trainable parameter counts. Per-arm labels:
- `task_only`: original task cross-entropy only; both auxiliary heads remain unused.
- `separate_aux`: DEV-012's isolated historical-bit head receives real rehearsal supervision.
- `aligned_aux`: real historical-bit supervision on the **native CASM output** shared with `predict`; intentionally trains native routing and operators.
- `aligned_shuffled`: same CASM-side supervision and computation but independently sampled null labels, controlling auxiliary-loss regularization.
- `aligned_detached`: real historical-bit labels and same aligned head, but stop-gradient after CASM so auxiliary training **cannot change** memory writer/routing/operators. Compare with `aligned_aux` to distinguish learned head calibration from actual shared representation updates.

Training uses DEV-012's **unchanged** 4-opposite-past examples/step: balanced retention and acquisition, histories 32/128, competing writes 4/16, near collision fraction 0/1, noise .08/.25. The two additional training-only noisy rehearsal addresses refer to actually observed events at fixed training-only indices, **not** the evaluator's hidden target index; no label/target/role enters the forward read. All arms use the same 120 AdamW updates, learning rate .003, batch stream and gradient clipping, at 3 seeds 43/59/71. Training and evaluation have separate namespaces with fingerprint overlap checks.

## Evaluation / interpretation

Two held-out repetitions of the 40 registered DEV-012 factorial cells (including H=512, D=32, retention/acquisition, noisy collisions); each mode/seed evaluates **80 opposite-past pairs = 160 episode decisions**; **15 checkpoints and 2,400 episode predictions total**. The **primary** endpoint is the original `predict` task accuracy **before repair**, stratified by seed, target role and H=512. Report auxiliary bit accuracy separately. All reset-memory task/bit accuracy must be exactly 50%; swapping counterfactual states should disrupt memory-dependent predictions. Also report full-history Euclidean with O(H+D) external memory/search cost, and diagnostic XOR accuracy **separately from actual prediction**, where it is allowed to exceed actual CASM quality without constituting native progress.

The extra CASM historical-supervision calls raise training cost (two additional CASM forward/backward passes per training episode in `aligned_aux`). The stop-gradient arm still pays additional *forward* passes but does not propagate auxiliary gradients to writer/router/compute. Query-time inference still processes all arriving memory writes O(H) and all four expert reads: no compute/memory efficiency superiority claim. No CED-specific changes occur in this experiment; CED's previous failure should not be obscured.

### Acceptance and failure gates

1. `aligned_aux` must materially exceed **both** `separate_aux` and `aligned_shuffled` on **native decisions**, not merely on auxiliary bit classification. Include independent seed-level paired uncertainty; the study is adaptive development and not confirmatory.
2. If `aligned_aux` exceeds `aligned_detached`, that supports shared-feature gradient transfer; if not, training the aligned auxiliary head alone may explain improvement.
3. If `diagnostic_xor_accuracy` exceeds native task accuracy, report the action-transformation/conditional-execution gap, **not** an end-to-end TAC-OSM improvement.
4. Improvement in retention must not erase acquisition; evaluate each role and H=512 separately.
5. If all native arms remain approximately 50%, preserve the **negative transfer result**. Do not claim CED, world-model prediction or new scalable memory capability. The next step should isolate route-vs-CASM-vs-answer submodules or train the native XOR mapping using suitable controlled supervision without changing the benchmark after seeing results.
6. Replay validates saved source hashes, 15 checkpoints, all per-example predictions and reset/counterfactual controls. It is **not training replay** and no independent holdout is preregistered.

This study exists to develop **TAC-OSM's distinct persistent-memory / relevance / conditional execution architecture**, not transform it into another language model.
