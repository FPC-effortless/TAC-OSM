# DEV-032: frozen failure diagnosis and integration engineering

Registered engineering work; no recipe selection or architectural winner.
DEV-031 remains incomplete: ten non-GRU checkpoint replays passed, while the
GRU stopped after its saved216 checkpoint with a non-finite clipping norm.
That exception alone cannot distinguish non-finite gradient elements from
finite gradients whose float32 norm calculation overflowed.

The [frozen protocol](../research/dev032/protocol.json) requires the original
source `4a8728ef3f08e01d9c542374cf1e722bf8701240`, original216 checkpoint hash,
and exact recorded runtime. The diagnostic wraps the original clipping call,
records the exception, gradients, pre-update parameters, optimizer and RNG,
and rethrows. It changes neither clipping arguments nor training acceptance.
Norm observations include CPU float32 and float64 calculations. A captured
failure completes the diagnosis, not the original training trial. No causal
claim that CUDA caused the failure follows from a CUDA reproduction.

The Test6 engineering harness connects preserved MTSK streaming memory,
learned CDL relevance routing, neural CASM execution, OSM latent prospective
planning, action, public receipt, verifier, bounded executed repair, and
verified memory commit. CDL is not a bit translator; OSM is not the verifier.
An action is chosen before its receipt. Hidden future targets enter only the
evaluator after the action sequence is fixed. A second service is charged
again. Rejected commits preserve the post-observation memory state; this does
not mean observation updates themselves are rejected.

This is a new single-entity public-query engineering task, with eight H32
cases and seven inference arms at identical random initialization. It is not
the DEV-031 all-entity benchmark. Branch-forcing fixtures test that repair
actually executes and incurs costs; they do not establish useful learned
verification. Public receipt acceptance is not a hidden correctness label.

Implementation boundaries remain explicit:

* MTSK is integrated here; masked-local TAC is not yet integrated.
* Neural ConditionalExecutor is integrated; original Boolean graph CASM is not.
* Learned CDL routing is integrated; exact conditional-NLL teacher is not.
* Uniform routing changes the number of active experts; it needs a matched-k
  control before scientific attribution.
* The MLP alternative is untrained. Inference ablations do not replace matched
  retraining, parameter accounting or physical compute measurements.
* Untrained OSM latent predictions establish a connected computation path,
  not learned future-state forecasting.

The [scientific integration plan](TEST6-SCIENTIFIC-PROTOCOL-PROPOSAL.md) is
separate. No scientific test set is accessed by this work.

## Recorded engineering results

At frozen source `4b9da75cf318271bb00e062e3c90469058ff370a`, all56 ordinary
rollouts completed,40 executed repair, and the two explicit branch fixtures
passed. The [raw report](DEV-032-TEST6-ENGINEERING.json) retains every action,
receipt and cost. A separate policy-independent physical simulator replay
verified99 actions including the fixtures, with exact receipt and cost
agreement: [physical audit](DEV-032-TEST6-PHYSICAL-AUDIT.json). Deliberately
changing a reported cost caused audit rejection. This audit does not rerun
neural predictions and does not establish a scientific performance benefit.
The complete local suite passed:1143 tests. No training or scientific test
evaluation occurred in this engineering preflight.

## Frozen GRU diagnostic

Kaggle notebook version2 completed the diagnostic at source
`de2e56d8f30849bbe67c4845badf77ba31cd2a87`. Resuming the original216 checkpoint
with the original source and exact recorded runtime reproduced the clipping
exception at attempted update286. The forward loss was finite,0.56964409,
and every pre-update parameter was finite, but1,802,668 gradient elements
were non-finite. This is not solely overflow of a norm applied to otherwise
finite gradients. The [raw gradient report](DEV-032-GRU-GRADIENT-HEALTH.json),
[diagnostic status](DEV-032-GRU-DIAGNOSTIC-STATUS.json) and
[snapshot inspection](DEV-032-GRU-SNAPSHOT-AUDIT.json) retain the evidence.
The snapshot hash and non-finite counts were independently checked on CPU.
Raw gradient/parameter/optimizer/RNG tensors remain in the private Kaggle
version2 output; no trained checkpoint is added to Git.

The original trial remains failed. This observation does not establish that
the GPU backend caused the problem, or that the architecture cannot learn.
An additional registered single-batch comparison will recompute the captured
pre-update model and identical fingerprinted training batch on CPU, CUDA
cuDNN, and CUDA unfused execution in the same runtime. It performs no
optimizer steps and does not evaluate validation or test accuracy.

The account owner explicitly authorized transient reuse of the existing
Kaggle credential after a rotation request. Authentication stays outside the
repository and notebook package. Revocation has not been verified, and this
work must not be described as completed credential rotation.
