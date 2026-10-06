# PLM Top-Down Assembly / Bottom-Up Adaptive Timescale — 001

Status: preregistered. Successor to the bounded null distributed-bank transfer lane.

## Scientific question

The preceding TDBU sequence produced two important results:

1. multiple persistent timescales improved finite history-conditioned capability
   over a single timescale; and
2. a fixed eight-trace distributed bank did not materially improve transfer over
   the three-timescale state, despite near-complete analytical representation of
   the held-out temporal kernels.

That leaves a sharper mechanism question:

**can the temporal dynamics themselves be learned end-to-end from experience,
rather than fixed in advance?**

## Top-down assembly

The complete loop is unchanged:

environment
-> observation
-> persistent state update/read
-> conventional tanh MLP policy
-> fixed binary executor
-> environment outcome
-> independent verifier.

The only novel adaptive mechanism is three trainable exponential decay
parameters. They initialize at the parent fixed scales (0.50, 0.90, 0.98) and
are updated only from training-set loss.

The current decision observation is always 0.0. No pair identity, temporal-filter
identity, gold action, verifier output, or target descriptor enters the policy.

## Bottom-up ladder

`no_state -> one_fixed -> three_fixed -> three_adaptive`.

The primary comparison is three_adaptive versus three_fixed. The one-timescale
arm is retained as a structural reduction; it is not the primary statistical
comparison.

All arms use the same 86-parameter conventional policy. One_fixed pads its
single state scalar with zeros to the common four-feature policy input.
Three_fixed and three_adaptive expose three state scalars directly.

The adaptive arm has three additional scalar trainable decay parameters. These
are the mechanism under test, not extra policy capacity.

## New held-out benchmark

Training temporal-filter constants:

`0.30, 0.50, 0.68, 0.82, 0.90, 0.96, 0.985`.

Adaptive-lane held-out constants:

`0.41, 0.57, 0.76, 0.875, 0.925, 0.991, 0.994`.

The held-out constants are disjoint from training and from the previous
distributed-bank transfer test set. The current observation remains 0.0 and
each evaluation history has an exact-negation partner with the opposite correct
action.

## Adaptive-state training

For each training history, the state recurrence is:

`s_t = alpha * s_(t-1) + (1-alpha) * x_t`.

The derivative used for end-to-end decay learning is:

`d s_t / d alpha = s_(t-1) + alpha * d s_(t-1)/d alpha - x_t`.

The three decay parameters are represented as
`alpha=sigmoid(logit)`, initialized from the registered parent scales.

Policy learning rate is 0.05. Decay learning rate is 0.02. Both schedules are
fixed for all seeds and H values.

No validation-set or test-set tuning is permitted.

## Decision

At H=256, the primary positive criterion requires adaptive minus fixed-three
success > 0.10 and a seed-bootstrap 95% CI excluding zero, plus adaptive above
no_state.

A null primary result leaves the persistent-state finding intact but does not
establish that learned temporal dynamics improve transfer.

## Confirmatory result and disposition

The registered exact-head run completed successfully at commit
`20622cf36d03fc7dc14fde781b6d67595eaa37de`. Workflow run
`37412279742` passed contract validation, unified research preflight, focused
tests, the full **844-test** repository suite, measurement, post-run invariant
validation, independent recomputation, and artifact upload. Artifact
`11389722597` has digest
`sha256:5bc9d5a31069596b7a32f75b43d8764bc80213c7da99179905a33fd1f8e0fe0f`.

At H=256, three_adaptive success was 0.85, 0.87, 0.81, 0.88, 0.90 across
the five seeds, mean **0.862**. Three_fixed success was 0.87, 0.87, 0.81,
0.87, 0.88, mean **0.860**. The preregistered paired difference was
**+0.002**, with seed-bootstrap 95% CI **[-0.010, 0.012]**. The +0.10 primary
materiality criterion therefore failed and the interval includes zero.

The no-state control remained at **0.500** mean with action gap 0.0. Both
stateful arms had action gap **1.0** on every seed. For the frozen adaptive
checkpoint, reset-to-zero success averaged **0.500** and cross-episode state
shuffle averaged **0.138**, versus normal adaptive success **0.862**.

The final adaptive decay constants at H=256 were:

| seed | fast | medium | slow |
|---|---:|---:|---:|
| 0 | 0.4424 | 0.8988 | 0.9793 |
| 1 | 0.4286 | 0.9037 | 0.9802 |
| 2 | 0.4485 | 0.9045 | 0.9800 |
| 3 | 0.4322 | 0.9039 | 0.9800 |
| 4 | 0.4153 | 0.8949 | 0.9805 |

The learned representation therefore moved the fastest timescale away from its
initial 0.50 in a consistent direction while leaving the slower two scales near
0.90 and 0.98. This is a descriptive optimization outcome, not a capability
improvement claim.

## Scientific boundaries

The experiment establishes only that, on this finite new transfer benchmark,
end-to-end learning of three decay parameters did not materially improve over
the parent fixed three-timescale representation. It does not prove that the
fixed scales are globally optimal, nor that adaptive temporal dynamics cannot
help on other temporal distributions.

The broader TDBU evidence now has a narrower interpretation: persistent temporal
state is repeatedly useful, while adding more fixed scales or learning the three
decays has not yet produced a capability gain. The remaining bottleneck is more
likely to be the mapping from persistent representation to decision or the
statistics/noise of the finite benchmark than raw temporal-state coverage.


## Scientific boundaries

This experiment cannot establish scale-free or power-law memory, biological
equivalence, optimal temporal learning, general intelligence, language/image/audio
competence, asymptotic scaling, or hardware efficiency.

The adaptive decay values are descriptive outcomes. They must not be treated as
evidence of biological timescales merely because the experiment is motivated by
multi-timescale adaptation.

## No-leakage requirements

Train/test filter identities remain separated. The new test constants are never
used in training or decay initialization. Every smoke run is explicitly
non-evidentiary. The full G0-G10/P0-P7 governance protocol is mandatory.
