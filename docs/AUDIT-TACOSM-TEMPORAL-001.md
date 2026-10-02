# Scientific Audit — TACOSM-TEMPORAL-001

## Disposition

**CONDITIONAL / bounded mechanism evidence.**

## Benchmark validity

PASS within the registered synthetic causal-boundary control. H=64 is fixed; k={1,2,4,8,16,32}; five seeds; 100 probes per seed/arm/delay. The task is synthetic and the state vector is explicit. It does not represent semantic memory.

## Leakage

PASS for the registered boundary. The router receives the query address and persistent state read; acceptable actions, target action, hidden reference, outcome and verifier result remain evaluator-side.

## Protocol

PASS. The successful result uses measurement commit `ebb1dad73edd79c94bac97af6ac42c7d8729af1f` and contract fingerprint `765ac1e41631d975`. Amendments A1 and A2 were applied before the successful result; earlier attempts produced no result artifact.

## Causal/instrument audit

PASS. The pre-read boundary gate passed all 30 seed×delay cells. Carry retained readable state; reset removed it; corruption changed the stored state at the read boundary. The router candidate population remained 64 at the read boundary.

## Statistical audit

QUALIFIED. The artifact retains per-seed values and five-seed means/spreads, so the primary table is auditable. The experiment does not register a paired inferential test, so the result should be reported as a bounded mechanism comparison rather than a general statistical claim. In particular, `carry=1.0` across all 30 cells is a deterministic outcome of this synthetic mechanism under the tested seeds; it is not evidence about an arbitrary persistent architecture.

## Material observation

Carry decision success is 1.0000 at every tested delay through k=32; reset is 0.0100 at k=1..16 and 0.0140 at k=32; corruption is 0.0000 throughout. The causal boundary gate passed before reading the primary endpoint.

## Nonclaims

- no semantic long-horizon memory claim;
- no learned persistent representation claim;
- no PLM intelligence claim;
- no indefinite temporal extrapolation;
- no asymptotic scaling claim.

## Reuse

Future experiments may inherit the explicit causal write/read boundary and intervention vocabulary. They must not inherit the result as evidence for semantic memory or broad PLM capability.