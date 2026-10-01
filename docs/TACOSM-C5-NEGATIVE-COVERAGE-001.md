# TACOSM-C5-NEGATIVE-COVERAGE-001

Status: PREREGISTERED — result pending.

## Purpose

The previous budget experiment showed a real learned signal but no reliable
gain from extending training from 32 to 512 epochs. This experiment holds the
512-epoch budget fixed and changes only the negative-sample objective.

The measured path remains continuous exhaustive semantic state retrieval.

## Registered protocol

- M=64 persistent state items.
- H=64 task anchor.
- 5 seeds (0..4).
- 512 epochs for every arm.
- 100 held-out noisy queries per seed.
- dual linear 10 -> 8 encoder.
- learning rate 0.02; margin 0.25.
- training codes = CODEBOOK[16:64] (48).
- evaluation codes = CODEBOOK[:16] (16).
- A1 H-invariant query/state stream.
- continuous state scoring only; binary indexing excluded.

## Arms

### Single negative

The current deterministic one-negative training schedule.

### Mean 8 negatives

Eight deterministic negatives are scored for each positive update and the
margin gradients are averaged before the single optimizer update.

### Hardest 8 negatives

Eight deterministic negatives are scored for each positive update. Only the
highest-scoring negative (the smallest positive-minus-negative margin) receives
the gradient.

Every arm makes exactly one optimizer update per positive code per epoch, so
the number of updates remains fixed at 24,576. The multi-negative arms pay more
training-time negative evaluations; that overhead is measured rather than
hidden.

## Primary endpoint

Target-state Top-1 recall under continuous exhaustive state scoring.

Secondary:

- mean target rank;
- fixed optimizer update count;
- negative evaluations per update;
- continuous inference arithmetic;
- one-time state embedding arithmetic.

## Decision rule

- >=0.20 absolute improvement and >=0.50 mean recall for an alternative:
  material evidence that negative coverage/objective design is a limiting factor;
- >=0.10 absolute improvement for at least one alternative, but no alternative
  meets the stronger criterion: partial evidence that the objective matters;
- all improvements <0.10: no material evidence that negative coverage alone
  removes the current representation boundary.

These are diagnostic rules, not claims of optimality.

## Cost

Continuous inference remains:

- 80 MACs/query for query encoding;
- 512 MACs/query for 64 cached 8-dimensional state scores;
- 592 MACs/query total;
- 5,120 MACs for one-time state embedding.

Training negative evaluation count is:

- 1 per update for the single-negative arm;
- 8 per update for each 8-negative arm.

## Interpretation boundary

A positive result would identify the training objective as a meaningful
limiting factor. It would not establish learned selective state retrieval,
sublinear learned addressing, natural-language reasoning, or broad C5.

The binary index is deliberately excluded until continuous retrieval is
strong enough to justify a selective boundary.

## Reproduction

Measurement command:

`python scripts/run_c5_negative_coverage_001.py`

Contract:

`contracts/TACOSM-C5-NEGATIVE-COVERAGE-001.json`
