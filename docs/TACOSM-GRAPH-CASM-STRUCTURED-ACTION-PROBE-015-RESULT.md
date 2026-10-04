# G-CASM-015 — Result and Scientific Disposition

**Status: VALID CONFIRMATORY EVIDENCE**

**Authoritative run:** 37223896895  
**Head:** 387a053b1477ea69bc2db624fa35b04c999e6805  
**Generator:** c31554413301e3c9d3e6b3f8c8c6be572a74a748  
**Repository gate:** passed  
**Focused 015 tests:** passed  
**Full measurement:** passed  
**Artifact:** 11310799340  
**Artifact digest:** sha256:90a3bf6c325932d235e707ad7f4e0d8e08b04acdb0b5d669dbca9b7e62687cef

## 1. Primary endpoint

The preregistered primary endpoint is deterministic one-step information gain
per expected environment acquisition work, with a seed-level paired
activation-trace minus scalar difference.

| M | Scalar I/work | Trace I/work | Paired delta | Seed bootstrap 95% CI |
|---:|---:|---:|---:|---:|
| 32 | 0.059117 | 0.208355 | +0.149238 | [0.146433, 0.152436] |
| 64 | 0.059238 | 0.232366 | +0.173128 | [0.171643, 0.174613] |
| 128 | 0.059342 | 0.247933 | +0.188592 | [0.187414, 0.189668] |
| 256 | 0.059289 | 0.255560 | +0.196271 | [0.195900, 0.196651] |
| 512 | 0.059154 | 0.257297 | +0.198143 | [0.197841, 0.198407] |

All registered M levels therefore satisfy the preregistered positive criterion.

The activation-trace channel provides approximately 3.5–4.35x the
environment-facing information efficiency of the scalar channel across the
registered population sizes.

## 2. Information content

The scalar channel remains essentially one bit of information at every M.

The activation trace produces:

| M | Trace information H(E) | Expected remaining candidates | Effective bits log2(M/E[|H'|]) |
|---:|---:|---:|---:|
| 32 | 4.773 | 1.250 | 4.678 |
| 64 | 5.316 | 1.794 | 5.157 |
| 128 | 5.666 | 2.850 | 5.489 |
| 256 | 5.844 | 4.866 | 5.717 |
| 512 | 5.893 | 9.154 | 5.806 |

The six-bit trace therefore uses nearly its full nominal 64-signature alphabet
at the larger population sizes.

At M=512 the theoretical collision/partition lower bound for a six-bit
deterministic channel is:

    M / 2^6 = 512 / 64 = 8.

The measured expected compatible population is 9.154. The residual factor is
approximately 1.144x above that lower bound.

This is the crucial result: changing the observation channel, rather than the
routing algorithm, increases the available distinguishing information enough to
move the one-step M=512 problem from an irreducible K=4 regime to a regime
close to an eight-candidate terminal execution budget.

## 3. Total computational cost

The experiment also exposes the next bottleneck.

The primary endpoint intentionally isolates environment-facing acquisition cost.
The preregistered secondary accounting charges candidate-cache preparation and
deterministic selector scanning.

| M | Scalar total information/work | Trace total information/work | Trace / scalar |
|---:|---:|---:|---:|
| 32 | 0.0007167 | 0.0011185 | 1.56x |
| 64 | 0.0005244 | 0.0007243 | 1.38x |
| 128 | 0.0003412 | 0.0004202 | 1.23x |
| 256 | 0.0002009 | 0.0002268 | 1.13x |
| 512 | 0.0001102 | 0.0001171 | 1.06x |

The trace advantage survives total-cost accounting in this implementation, but
the margin collapses with M. At M=512 the richer channel is only about 6%
better per total accounted work.

This is not a negative result for structured evidence. It identifies the next
computational bottleneck: the exact selector currently scans the candidate
population for every legal action, and the trace channel multiplies that scan
by six evidence bits.

## 4. Integrity

The authoritative run passed:

- full contract validation;
- repository test suite;
- focused 015 tests;
- full measurement;
- train/evaluation structural disjointness;
- train/evaluation truth-table disjointness;
- static per-seed candidate library;
- target-identity-blind action selection;
- no target evidence before action commitment;
- fixed generator commit;
- explicit raw per-seed results.

Smoke results are excluded from the scientific disposition.

## 5. Scientific interpretation

G-CASM-015 validates a stronger form of the information-bottleneck diagnosis:

**The scalar observation channel was the dominant information limitation.**

A single structured execution trace changes the channel capacity enough that
one action can nearly identify a target among 512 hypotheses.

However:

**Structured information does not automatically produce computational
efficiency.**

At large M, the exact candidate-side planner is itself expensive. This means
the PLM problem now has two distinct requirements:

    high-information observation
    +
    low-cost prediction/indexing of which observations are useful.

The next intervention should therefore not merely add more evidence bits.
It should make structured evidence computationally addressable.

## 6. Disposition

**G-CASM-015 is valid confirmatory evidence.**

It supports the claim that, on the registered finite-domain workload, an
instrumented six-bit activation trace is substantially more information-efficient
than a scalar output observation.

It does not establish that internal traces are available in black-box
environments, that a learned probe policy is superior, or that the mechanism
scales asymptotically.

## 7. Next experiment

G-CASM-017 is preregistered as an exact evidence-indexing intervention.

The goal is to retain the information advantage of structured evidence while
replacing exhaustive candidate scanning with an exact indexed representation of
candidate evidence signatures.

The hypothesis is that an evidence index can recover most of the trace
channel's total-cost advantage at M=512 without changing the observation
channel or introducing learned target-specific routing.

That is the appropriate bridge from:

    structured evidence

to:

    computationally efficient structured evidence

and then, only after that, to a learned CDL probe policy.


## Derived information-to-execution bound (post-measurement analysis)

A deterministic structured observation channel with q possible evidence leaves
can support at most qB distinct successful terminal candidates when the
terminal budget is B. Under a uniform M-hypothesis prior:

    E[success] <= min(1, qB/M).

This is a population expectation bound and is not used as the preregistered
primary endpoint.

The scalar K=4 / B=8 / M=512 regime has qB/M = 0.25. G-CASM-014 measured
23.125% at this point.

The six-bit activation-trace channel has q = 64, giving qB/M = 1 at M=512
with B=8. Thus the richer channel moves the necessary information-budget
condition from a four-fold deficit to parity with the terminal execution
budget. The remaining issue is how balanced the 64 evidence leaves actually
are and how cheaply their candidate-side prediction can be indexed.
