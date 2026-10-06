# PLM Top-Down Assembly / Bottom-Up Temporal-Scale Transfer — 001

Status: preregistered. This lane is a successor to
`TACOSM-PLM-TDBU-MTSK-001`.

## Scientific question

The MTSK-001 result established a bounded finite finding that multiple distinct
timescales can improve history-conditioned decision capability over one
timescale on its registered synthetic benchmark. The bottom-up reduction also
showed that two timescales matched the full three-timescale MTSK arm at H=256.
That leaves the more informative question:

**does the usefulness of multiple timescales transfer to temporal relations whose
filter constants were not present during training?**

This experiment therefore changes the benchmark distribution, not the surrounding
decision loop.

## Top-down assembly

The complete loop remains:

environment
-> observation
-> persistent state update/read
-> conventional policy/router
-> fixed binary executor
-> environment outcome
-> independent verifier.

Only the persistent temporal representation is novel: a fixed distributed bank
of exponential traces spanning eight decay constants.

The policy remains a conventional two-action tanh MLP. No task identity, pair
identity, temporal-filter identity, gold action, verifier output, or target
descriptor enters the state update or policy input.

## Transfer benchmark

Training and test use disjoint deterministic temporal-filter constant sets.

Training filters are:

`0.30, 0.50, 0.68, 0.82, 0.90, 0.96, 0.985`.

Test filters are:

`0.40, 0.58, 0.74, 0.86, 0.93, 0.989, 0.993`.

The test set is unseen at the temporal-constant level. The final two constants
also extend beyond the maximum training alpha and therefore form the registered
extrapolation subset.

For each pair, the correct action compares two hidden exponential-filter
outputs. The current observation at decision time is fixed at 0.0. Every test
history is paired with its exact negation, forcing identical current input with
opposite correct actions.

No test temporal constant is used for training, tuning, checkpoint selection,
or architecture selection.

## Bottom-up ladder

The state mechanism is reduced as:

`no_state -> one_timescale -> two_timescale -> three_timescale -> distributed_eight`.

The primary comparison is distributed_eight versus three_timescale on the
held-out test-filter endpoint. The two-timescale arm is retained because MTSK-001
showed that two scales already matched three in-domain.

All policy arms use exactly 86 trainable parameters. For one/two/three timescales, the policy input is padded to four features; distributed_eight uses eight state features plus the current observation and therefore a narrower hidden layer. The primary distributed_eight versus three_timescale comparison is exactly parameter matched.

## Capacity and state accounting

The policy parameter count is:

- no_state: 86;
- one_timescale: 82;
- two_timescale: 86;
- three_timescale: 86;
- distributed_eight: 86.

Thus the primary distributed-eight versus three-timescale comparison is exactly
parameter matched. Distributed_eight carries eight persistent trace scalars
and therefore performs more state-update work; no efficiency claim is implied.

For K=2,3,8 the hidden widths are chosen so the policy remains at 86 trainable
parameters. The K=1 arm uses the closest simple lower-capacity conventional
policy.

## Representability gate

Before confirmation, the runner checks that the distributed eight-trace bank
can approximate every held-out test exponential kernel at H=256 with a
pre-registered analytical least-squares R^2 threshold of 0.99. This check uses
only the committed temporal-filter constants and deterministic impulse kernels;
it does not train the policy and cannot select checkpoints.

## Causal interventions

After distributed_eight training, the frozen checkpoint is evaluated normally,
after replacing its temporal state with zeros, and after shuffling state vectors
between evaluation episodes.

A large drop from normal performance localizes dependence on the learned use of
persistent temporal state rather than merely on the existence of the state
machinery.

## Decision and boundaries

The primary threshold is a mean distributed_eight minus three_timescale
difference greater than 0.10 with a seed-bootstrap 95% interval excluding zero,
plus distributed_eight outperforming no_state.

A positive result would support only the finite transfer claim defined above.
It would not establish scale-free or power-law memory. Current biological
literature is motivation rather than evidence for this architecture: recent
PRX Life work reports scale-free bacterial adaptation across multiple
timescales, while a 2025 Nature study reports distinct computational benefits
from multiple reinforcement-learning discount timescales. Neither result
validates this machine architecture. 

## No-leakage requirements

The temporal-filter identity must never appear in model inputs. Train and test
RNG streams are disjoint. All temporal constants are fixed in the contract
before confirmation. Test constants are generated only by the test generator and
are not exposed to training.

No test run may be used as a validation run. Any smoke run is explicitly labeled
non-evidentiary.

## Confirmatory result and disposition

The registered exact-head confirmatory run completed successfully on commit
`2e8092ecfc16bdc557f6a53ef7c411eb2c3e2913`. Workflow run
`37411440322` passed contract validation, unified research preflight, focused
tests, the full 832-test repository suite, measurement, post-run invariant
validation, independent recomputation, and artifact upload. Artifact
`11389521777` has digest
`sha256:7ad3974e4e86c3aca74486e350602bbe8a528448c10ca63dc98b3909c2fdf5f5`.

At H=256, distributed_eight exact success was **0.8533** mean versus
**0.8450** for three_timescale. The preregistered difference was **+0.0083**,
with seed-bootstrap 95% CI **[-0.0017, 0.0183]**. The registered +0.10 materiality
criterion therefore failed and the interval includes zero.

The state mechanism itself remained active: distributed_eight had action gap
1.0 on all five H=256 seeds, while no_state had 0.0. Distributed_eight reset
success averaged **0.500** and shuffle success **0.1467**, versus normal success
0.8533. No_state averaged 0.500.

The transfer decomposition is informative. At H=256, interpolation success was
0.8712 for distributed_eight versus 0.8644 for three_timescale; extrapolation
success was 0.8361 versus 0.8262. These are secondary descriptive results, not
new decision criteria.

The distributed eight-trace bank passed the pre-run representability gate for
all seven held-out temporal filters, with minimum impulse-kernel R²
**0.99819** against the registered 0.99 threshold.

## Scientific interpretation

The preregistered claim that broader fixed distributed temporal coverage would
materially improve transfer over a three-timescale state is **not supported**.

The evidence instead supports the narrower result that persistent temporal state
can remain useful on this held-out temporal-filter benchmark: all stateful arms
formed a history-conditioned action gap, and the stateful arms substantially
outperformed no_state. Because the primary experiment did not preregister paired
stateful-versus-stateful hypothesis tests beyond distributed_eight vs
three_timescale, descriptive differences among one/two/three/eight scales are
not promoted as separate confirmatory claims.

The combination of near-perfect representability and the null primary effect
also shifts the research bottleneck away from raw temporal-basis coverage and
toward **scale placement and learned readout/credit assignment**.

## Successor rule

If distributed_eight wins, the next question is whether the improvement survives
randomized scale placement and learned decay parameters under the same transfer
benchmark.

If distributed_eight ties three_timescale, the next question is whether the
transfer bottleneck is scale placement, policy capacity, or benchmark hardness.

If distributed_eight loses, the current multi-timescale claim remains bounded to
the original MTSK benchmark and no distributed-memory generalization is made.

The universal G0-G10/P0-P7 governance protocol remains mandatory.
