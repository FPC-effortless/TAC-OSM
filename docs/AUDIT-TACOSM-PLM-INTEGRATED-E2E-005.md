# TACOSM-PLM-INTEGRATED-E2E-005 audit

## Purpose

This is the first post-017B multimodal integration lane. It is a functional
integration experiment rather than an optimisation sweep.

The chain is:
multimodal observation -> shared representation -> persistent entity state ->
query-conditioned read -> CASM -> action -> environment outcome -> verifier ->
post-action state update.

## Deliberate constraints

Entity addressing is explicit and deterministic. Operator dispatch is fixed by
the public query operator. These remove two known diagnosis bottlenecks from
the first integration test. They do not establish learned semantic addressing
or learned operator discovery.

There is no payload-bit auxiliary loss and no entity-classification auxiliary
loss. The pre-action learning signal is final action supervision only. The
verifier receives the environment outcome after action and has its own
post-action training signal.

## Integrity gates

The benchmark generator is independent of the runner and is fingerprinted by
SHA-256. The contract is also fingerprinted. The result validator recomputes
both hashes from the checked-out source.

Held-out compositions are excluded from training. Every evaluation query is
cross-modal and q1/q2 target distinct entities. Training and evaluation use
separate RNG namespaces, and semantic episode keys must have zero overlap.

All control arms reuse the exact same generated evaluation episode objects.
The query interface exposes only memory, entity, bit indices, and the public
operator; the answer and outcome cannot enter the pre-action call.

A gradient-surface probe verifies that the action path reaches the text,
image, and audio encoders, shared fusion, persistent write path, and CASM
decoder. A repository-wide regression test checks directly invoked local
scripts in GitHub Actions workflows so the missing-script failure seen in
017B cannot silently recur.

## Interpretation

A result is scientifically readable only after every integrity gate and the
independent validator pass. A primary-threshold failure is a finding about
this fixed functional configuration, not a proof that the architecture is
impossible. Optimisation and learned routing/addressing changes remain a
separate later lane.
## Pre-measurement correction: structural target separation

The first implementation allowed the outer training target to exist inside the model's episode-level forward method, even though query() did not accept it. That was rejected as unnecessarily permissive. The model no longer has a label-bearing episode-level forward method. The experiment harness now computes action loss outside the model, and computes the synthetic environment outcome only after the action has been produced. The model API contains no answer/target field.

This is an integrity correction made before any admissible E2E-005 measurement; no result is retained from the earlier implementation.
## Scientific scope correction discovered during measurement (no run modification)

The registered q1/q2 protocol deliberately targets `entities[0]` and `entities[1]`, which prevents q1's post-action verifier/update from causally affecting q2. Therefore E2E-005 tests multimodal encoding into persistent entity-addressed storage and later retrieval, but it does **not** test action-conditioned persistence across a feedback boundary. This limitation was discovered after the confirmatory run was authorized and does not modify the registered run. Any result must not be described as evidence for full PNDS feedback persistence. A subsequent functional lane will use the same target entity with disjoint query content so q1 feedback can affect q2 state without making q2's answer a copy of q1's answer.
## Post-measurement instrumentation correction

The first confirmatory measurement completed its full computation, but the independent
validator stopped on a stale access to the pre-registration threshold API. The
measurement artifact itself contains the complete result and all integrity fields.
The validator was corrected to read the registered threshold from the machine-readable
contract, with an explicit primary-endpoint name consistency check. This correction
does not alter the model, benchmark generator, contract, seeds, training schedule,
controls, or statistical rule. A second confirmatory run is authorized solely to
re-execute the fixed validator and reproduce the registered measurement under the
unchanged scientific design.

