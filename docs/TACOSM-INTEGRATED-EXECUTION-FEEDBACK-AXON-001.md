# TACOSM-INTEGRATED-EXECUTION-FEEDBACK-AXON-001

## Purpose

This experiment connects the current successor routing loop to a post-execution
verifier and then carries verified transition experience into PST, StructMeans,
AXON, REGM-like durable experience, SSA operator retrieval, and SECA composition.

The experiment is intentionally dependency-free and uses the existing exact
executor. It is a prototype integration study, not a broad C5 claim.

## Scientific boundary

The route decision is made before the outcome exists. Only after execution does
the environment expose the outcome and the execution-ground-truth verifier
produce an accept/reject signal. On rejection, the verifier-side task record
provides the correct target as a repair label. That label is never visible to
route() before the decision.

## C5 arms

1. Outcome-only: existing successor RepresentationEnergyRouter update.
2. Verifier-feedback: VerifierDrivenEnergyRouter, which uses post-execution
   verified positives and the actually rejected candidate as a hard negative.

The runner reports target top-1 recall, top-8 admission recall, target rank, and
verifier acceptance.

## Procedural loop

Verified transition records are learned with PST. StructMeans groups recurring
transition structure. AXON consolidates recurring transition laws into
parameterized operators. REGM stores only verified records and rebuilds PST from
durable experience. SSA routes a bounded operator shortlist using predicted next
state. SECA proposes novel two-operator compositions and verifies them against
an independent reference implementation.

## Interpretation rules

A positive result requires the raw endpoint, not the implementation intention,
to cross the registered threshold. Failure of any layer remains a result and
does not license a stronger architectural claim.

The experiment does not establish:
- asymptotic sublinear addressing;
- hardware wall-clock speedup;
- universal reasoning;
- broad-domain continual learning;
- human-preference alignment.


## Rerun marker
This branch is the synchronized measurement branch for the registered run; no protocol fields or experiment parameters are changed.
