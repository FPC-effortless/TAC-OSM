FLOCK013: diagnose local learning and repair zero-gain router credit

Question: does the zero-gain write branch suppress the straight-through router
gradient? What do the saved failed phase/context checkpoints reveal about
alignment, state use, chunk boundaries and the path from recorded value to loss?

This is an explicitly NO-FIT engineering diagnostic requested after the completed
single-use trial. It does not extend that trial or its synthetic fitting
exception. Zero optimizers, zero optimizer updates, no new trained weights,
no held-out queries, no protected TEST, no GPU, no promotion and no merge.

The current hypothesis is a source-level adjoint defect, NOT an established
explanation of all failed learning. Legacy write_step uses where(gain!=0,
candidate,previous), which makes the gain derivative zero in unselected banks.
The hard router otherwise declares gain = one_hot(argmax(p)) + p-stopgrad(p).
Those two operations together remove the unselected WRITE-credit term intended
by that surrogate. Router gradients through reads can still exist; it is false
to say the entire router is disconnected.

The candidate preserves the exact forward freeze (including signed-zero bits)
and carries the smooth candidate adjoint through zero gain. The observation
write mask remains multiplied into gain, so masked observations cannot learn
a write. The repair is a declared straight-through adjoint; hard argmax has no
ordinary derivative. Precision-state analysis is on the valid symmetric
positive-definite domain. This is not arbitrary-state IEEE or full-graph proof.

Frozen test plan

1. Independently compute memory and precision right derivatives at zero gain,
   plus the two-head softmax-chain derivative for an unselected bank. Preserve
   legacy failures. Check candidate FP64 finite differences, FP32/64 exact
   forward freezes, positive-gain source/gradient equality and mask behavior.
2. Load only initial/final checkpoints from the completed run 41623750060,
   verifying six frozen file hashes. Observe the three original models and two
   parameter-identical AD repairs. Do not instantiate an optimizer.
3. Inspect actual first-answer gradients, routing, per-layer variation and
   value-source influence on the eight already published fixed FIT examples.
   Test whole-vs-prefix/query evaluation, future suffix invariance, frozen query
   writes, phase reset and total state reset. These interventions are diagnostic
   measurements, not candidate-selection or held-out performance claims.
4. Require bitwise equality of original/repaired checkpoint forward logits,
   unchanged live parameter counts and unchanged checkpoint weights. The repair
   is expected to change adjoints but cannot improve existing inference without
   training. Learning improvement is OPEN.

Bounds: five model constructors, eight checkpoint loads, at most 100 neural
forwards and five neural backwards; 290 process CPU seconds and 300 wall seconds.
Kernel finite-difference/autograd calls are reported separately and create no
learned model. GitHub Actions is selected by the unchanged phone planner; its
receipt does not authorize training. Failure and partial artifacts are retained.

Original source: c4666aed949e408a58dd9b1b99f825da415feb3f
Original results: https://github.com/FPC-effortless/TAC-OSM/actions/runs/41623750060
Original draft PR: https://github.com/FPC-effortless/TAC-OSM/pull/704
All original model source, scientific runner, fit evidence and failures remain
unchanged. Whole-model Lean, source/IEEE/AD completeness, genuine-data capability
and independent qualification remain OPEN. Paired PNDS records must preserve
these limits; a kernel proof cannot establish learned retrieval.
