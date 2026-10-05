# Information-capacity bounds for G-CASM active evidence

These functions are protocol infrastructure, not empirical evidence.

For a deterministic observation channel with at most q distinct signatures over M
candidate hypotheses, the expected compatible-set size under a uniform prior is

  E[|H(E)|] = sum_i n_i^2 / M >= M/q,

where n_i are bucket sizes. Equality requires an exactly balanced partition when
the integer constraint permits it.

For a binary K-bit evidence channel, q <= 2^K, so E[|H(E)|] >= M/2^K.
The exact integer optimum is provided by balanced_collision_floor(M,q).

For a terminal candidate budget B, a B-per-bucket selector can return at most B
candidates from each of q buckets. Consequently

  P(success) <= min(1, qB/M).

This is a universal counting ceiling, not a Shannon theorem and not a statement
about any particular router.

For the current synthetic workload:

- K=4 gives q<=16, so M=512/B=8 has a terminal ceiling of 0.25.
- K=6 gives q<=64, so M=512/B=8 has ceiling 1.0.
- K=12 gives q<=4096, so M=512/B=8 also has ceiling 1.0.

These ceilings do not guarantee achievability. Uneven buckets, action cost,
probe-selection errors, and downstream execution can all lower realized
performance.

The functions are intended to be reused by future experiment contracts and
audits so that a result can be compared against the mathematical ceiling
before a new router intervention is introduced.