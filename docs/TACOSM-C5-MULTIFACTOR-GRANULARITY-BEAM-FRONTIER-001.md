# TACOSM-C5-MULTIFACTOR-GRANULARITY-BEAM-FRONTIER-001

Status: PRE-REGISTERED.

## Purpose

The previous three-factor factor-size-8 beam frontier showed that beam 4
remains near 14% reranking but misses the 0.90 capability floor at M=128 and
M=256, while beams 5 and 6 recover the floor only at approximately 25% and 42%
of M. This experiment changes the addressing geometry rather than widening the
same factor-size-8 beam again.

The intervention is three-factor addressing with factor size 16 and beams
4, 5, and 6.

## Registered arms

- three_factor_16_beam4
- three_factor_16_beam5
- three_factor_16_beam6

All use three factors, factor size 16, K=32, H=256, M={128,256,512}, and
seeds 10-19.

## Primary capability constraint

For each arm and M:

target_recall_retention_ratio = selective target recall / exhaustive target recall

The pre-registered capability floor is 0.90 at every M.

## Primary computation diagnostic

states_scored_over_M is the mean number of persistent-state embeddings
subjected to continuous reranking divided by M.

The final K=32 shortlist is not substituted for this measure.

## Frontier-improvement criterion

The current robust two-factor reference is the fixed
factor_size=16, factor_beam=6 configuration from
TACOSM-C5-FRONTIER-ROBUSTNESS-002, with mean
states_scored_over_M = 0.1542864583 on seeds 10-19.

An eligible fixed arm is a frontier improvement only if its mean
states_scored_over_M across all three M levels is strictly below that
pre-registered reference.

## Interpretation

The experiment distinguishes two mechanisms.

1. If a factor-size-16 arm meets the 0.90 floor and improves the 0.1542864583
   reference, finer cells have increased coverage efficiency without restoring
   the occupancy problem exposed by factor-size 8.

2. If factor-size-16 arms remain recall-limited even as beam increases, the
   bottleneck is localized to factorized representation/admission rather than
   cell occupancy alone.

No universal optimum, semantic retrieval guarantee, asymptotic theorem, or
hardware speedup is licensed.

## Held constant

The teacher representation, task stream, state population, state-distinct
executor, codebook fitting policy, K budget, seeds, M levels, evaluation
length, pooled aggregation, and capability floor are unchanged from the
previous beam-frontier measurement.

## Integrity

This experiment must be measured only after the contract and smoke path pass.
A commit carrying the full-run token dispatches the already registered full
measurement; the token is deliberately absent from the registration commit.

## Pre-run prediction addendum (non-binding)

This addendum records a falsifiable prediction made before the registered full
measurement. It does not alter the contract, primary capability floor,
frontier-improvement criterion, registered arms, or verdict rule.

**Hypothesis:** retention at M=128 and M=256 depends primarily on B/F rather
than on F itself.

For F=16, d=3, and B=4/5/6 (B/F = 0.25/0.3125/0.375), the prediction is that
all three arms remain below 0.90 retention at both M=128 and M=256, with
roughly 0.6-0.8 retention.

Under the product-cell occupancy model, the scored fraction has floor

    scored/M ≈ (B/F)^3

giving approximately 1.6%, 3.0%, and 5.3% for beams 4, 5, and 6 respectively.
The observed two-factor beam-6 fractions make this a lower-bound heuristic,
not a registered outcome.

The specific falsifier for the B/F-only retention hypothesis is any F=16 arm
that reaches at least 0.90 retention at both M=128 and M=256. Such an outcome
would refute this pre-run hypothesis without changing the registered primary
verdict rule, which still requires the capability floor at every registered M.

Secondary analysis may report the exhaustive-reference capability at each M
and seed-bootstrap uncertainty around retention. These analyses are
diagnostic only and cannot convert a primary failure into a pass.

## Reproduction

Smoke:

python scripts/run_c5_multifactor_granularity_beam_frontier_001.py --smoke

Full:

python scripts/run_c5_multifactor_granularity_beam_frontier_001.py

Artifact:

artifacts/TACOSM-C5-MULTIFACTOR-GRANULARITY-BEAM-FRONTIER-001.json
