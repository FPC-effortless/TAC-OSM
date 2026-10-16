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

The account owner explicitly authorized transient reuse of the existing
Kaggle credential after a rotation request. Authentication stays outside the
repository and notebook package. Revocation has not been verified, and this
work must not be described as completed credential rotation.
