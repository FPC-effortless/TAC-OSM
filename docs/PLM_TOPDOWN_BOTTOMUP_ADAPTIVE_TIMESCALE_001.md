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
