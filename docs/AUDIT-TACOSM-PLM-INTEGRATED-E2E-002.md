# Audit — TACOSM-PLM-INTEGRATED-E2E-002

## Pre-measurement disposition

Status: READY FOR CONFIRMATORY EXECUTION

### Benchmark validity

The task uses the same 12 complementary latent bits and eight canonical held-out
operator/index compositions as E2E-001. Training and evaluation compositions are
strictly disjoint, including commutative reversals.

Text contains bits 0-3, image contains bits 4-7, and audio contains bits 8-11.
Every held-out query crosses modality partitions.

### Leakage

Explicit write uses the observation entity identity, which is structural
metadata present before action.

Explicit read uses the query entity identity, which is part of the public query.

No answer, target payload, evaluation correctness, or future outcome is provided
to the model before action.

### Controls

The same generated evaluation episodes are used across arms within each seed.
No post-result seed or hyperparameter selection is allowed.

### Interpretation

The E2E-001 0.5200 result is a frozen comparator only. E2E-002 is not permitted
to re-fit that baseline or tune against its held-out outcomes.

The primary endpoint is explicit-both q2 accuracy; explicit-write and explicit-read
are diagnostic arms rather than a ranked contest.

## Instrument boundary

A successful explicit address intervention does not prove learned semantic
addressing. It only demonstrates that the learned address path can be replaced
by the registered structural address without changing the rest of the chain.
