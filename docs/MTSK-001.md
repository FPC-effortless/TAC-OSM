# MTSK-001 — preregistration

Status: PRE-REGISTERED; NOT YET MEASURED

## Question

Does a three-timescale adaptive state mechanism improve retention of an early
slow fact over repeated overwriting observations while preserving responsiveness
to a current fast fact, compared with a parameter-matched single effective
timescale?

## Why this experiment is separate

The existing PLM persistence result (TACOSM-PLM-TEMPORAL-PERSISTENCE-002) is a
valid single-scale persistence result. Its state mechanism is
ExplicitEntityState; it does not contain MTSK. MTSK must therefore be tested
as a new mechanism rather than inferred from persistence.

## Experimental arms

Both arms use the same multimodal encoders, shared representation, FixedCASM,
hidden width 60, state-bank tensor shape [batch, entity, 3, 20], optimizer,
learning rate, weight decay, training schedule, seeds, and evaluation episodes.

The single-timescale arm has one effective learned decay and one effective
input gate shared across all three state banks. It has the same parameter
counts as MTSK.

The MTSK arm has three input-conditioned write gates and three learned
retention coefficients constrained to fast <= medium <= slow. The banks are
otherwise the same width as the single-timescale control.

## Task

At time 0 the entity contains a slow binary fact and a fast binary fact.
Subsequent observations repeatedly overwrite the state with new observations.
The slow fact disappears from all observations after t0 and is replaced with a
mask token. The fast fact remains visible in text and changes with the current
observation.

At the end of each sequence the model answers two queries through the same
FixedCASM path:

1. slow_bit XOR 0
2. final_fast_bit XOR 0

The image and audio modalities contain nuisance bits only; they never encode
the slow or fast target bits.

Evaluation uses delays 1, 2, 4, 8, 16, and 32, with exactly 200 episodes per
delay and 50 examples in each slow/fast target combination.

## Primary endpoint

MTSK slow-query accuracy minus single-timescale slow-query accuracy at delay
32.

Materiality threshold: +0.15.

The hierarchical paired-bootstrap 95% lower bound must be > 0. The fast-query
difference at delay 32 must be >= -0.05 as a guardrail.

There is no checkpoint or seed selection.

## Integrity gates

The measurement is invalid if any of the following fail:

- slow target masking after t0;
- image/audio exclusion of slow and fast target bits;
- exact paired evaluation sequence identity across arms;
- zero train/evaluation sequence-key overlap;
- distinct evaluation fingerprints across seeds;
- exact 50/50/50/50 slow/fast cell balance at each delay;
- equal parameter counts;
- monotonic MTSK retention ordering;
- artifact consistency.

## Interpretation

A positive result licenses only a bounded state-mechanism finding on this
synthetic multimodal task. A null result means the registered task did not
show an MTSK advantage. Neither outcome establishes or refutes general memory,
semantic memory, PLM intelligence, or scaling.
