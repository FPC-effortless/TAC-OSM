# LANGLAYER036 — frozen trained-depth diagnostic (NO NEW FIT)

Parent: LANG035 training draft PR #563, layered source PR #562, companion PNDS #63.

**Preregistered question:** Why did stacking causal residual/concept
layers at ~500K parameters fail to improve exposed-DEV NLL at 320
source-matched updates? Is the extra path attenuated, ineffective, or
limited by insufficient exposure or unseen document generalization?

No optimizer operations, changed trained weights, re-training,
DEV-based model selection, protected TEST, or GPU spending.

## Original source and checkpoint proof

All 15 completed 320-update 500K CPython3.13.0 source-trained checkpoints
come from immutable GitHub Actions #40785939365. A runner downloads
each original ZIP and checks the SHA256 in its original trained receipt,
the sample-order SHA, the source model key geometry, and original
4 documents x 8 exposed DEV windows x 128 targets (4096).
Any absence or mismatch fails. The original P3 gap remains negative.

## Frozen interventions

Each added block is tested separately on the 4096 source-verified but
previously exposed DEV targets. Report lesioned NLL MINUS original NLL.
Positive means removal hurts the checkpoint, negative means removal
helps that checkpoint. No intervention trains parameters.

1. Entire block skip: output equal to block input.
2. Recurrent read: sigmoid of read gate forced effectively zero, while
   current-token concept, memory write schedule and other paths remain.
3. Concept-to-state projection zero; recurrence and other operations remain.
4. Concept history seed write-rate zero; instantaneous concepts remain.
5. Uniform graph: source query projection zero, yielding uniform
   source softmax, while the other concept and residual ops remain.

The interventions are frozen functional sensitivities, not a proof
that a mechanism caused the historical learning gain.

Additional source probe: learned scalar residual gate; residual output
RMS divided by layer-input RMS; cosine(input, output); full parameter
gradient RMS from one training-exposed FIT batch, without any weight
updates. Nonzero gradient alone is not an information-utility guarantee.

## Information capacity vs training exposure

We evaluate the first FOUR exact source-owner/offset batches previously
presented in training for each seed (32 x 129 source bytes, 4096 targets)
versus the same original four disjoint exposed DEV documents (4096
targets). Same FIT sample indices across all three architectures for
each seed. The raw difference in NLL is only a possible generalization
or distribution-gap signal, since document cohorts differ. Never
assert classic overfit merely from this difference.

## Next prospective experiment, depending on results

If recurrence contributes but layer benefit is null, test native memory
readout to logits or ablate memory on longer authentic contexts.
If upper residual gates are small or saturation is diagnosed, test
initialization / normalization one change at a time. If gains are
training-only, test source exposure and regularization with new frozen
controls. Only after this phase train a genuine depth-matched
feedforward-only comparator at matched parameters and data/sampling
and optimizer budgets; record unequal FLOPs rather than claiming full
compute parity. Verify every failed seed. Long-context skill and
whole-model Lean/IEEE/AD equivalence remain OPEN.

No automatic merge, protected TEST opening, scientific superiority
or checkpoint promotion.
