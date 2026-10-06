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

## Successor rule

If distributed_eight wins, the next question is whether the improvement survives
randomized scale placement and learned decay parameters under the same transfer
benchmark.

If distributed_eight ties three_timescale, the next question is whether the
transfer bottleneck is scale placement, policy capacity, or benchmark hardness.

If distributed_eight loses, the current multi-timescale claim remains bounded to
the original MTSK benchmark and no distributed-memory generalization is made.

The universal G0-G10/P0-P7 governance protocol remains mandatory.
