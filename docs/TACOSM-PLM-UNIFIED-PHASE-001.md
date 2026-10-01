# TACOSM-PLM-UNIFIED-PHASE-001

## Status

RUN READY. CI produces the authoritative artifact. This protocol is not overwritten after measurement.

## Lineage

This phase is stacked directly from PR #61:
- parent branch: research/c5-composition-budget-audit-001
- parent commit: e3c59d1c630d810205c7e38b82ecb60f4a8410af

The phase continues C5 into unified PLM without re-selecting a C5 representation winner.

## Question

Can the fused system preserve useful execution selectivity and verified state transitions as history H and persistent population M increase, while explicitly separating addressing cost from execution cost?

## Registered system

S_t
-> hierarchical address
-> CDL relevance
-> CASM operator selection
-> adaptive execution
-> action/outcome
-> verification/repair
-> verified state commit
-> slow/fast learning and consolidation

## Workload

Synthetic XOR, XNOR, AND and OR relational episodes.

History H:
64, 256, 1024, 4096, 16384

Persistent population M:
H + 1

Seeds:
0, 1, 2, 3, 4

Key dimension:
12

Query noise:
0.10

Admission cap:
16

Execution budget:
4

Slow/fast learning rates:
0.01 / 0.10

Bounded state capacity:
256

The bounded-state arm is a capacity analogue, not an implementation of a named SSM.

## Metrics

Relevance:
- dense target rank;
- dense Top-1;
- local target rank (censored at admitted-set size + 1 when target is not admitted);
- local Top-1.

Addressing:
- raw bucket population;
- admitted candidates;
- index probes;
- full-scan fallback;
- target admission recall.

Execution:
- attempted operator steps;
- adaptive halt depth;
- operator cost.

Loop:
- unified success;
- verified commit;
- repair rate;
- total cost proxy.

State:
- active state size;
- experience writes;
- wrong verified writes under injected verifier error.

## Cost accounting

address_proxy = index_lookups + candidates_scored

execution_cost = operator steps actually attempted

total_cost_proxy = address_proxy + execution_cost

These are research proxies. A subsequent compute-backed phase must add wall-clock, memory bandwidth and I/O.

## Controls

Anti-leakage: target descriptors are evaluator-owned and are not router inputs.

State authority: experience is tentative until verification passes.

C5 dependency: prior C5 measurements are consumed as fixed evidence. No C5 representation winner or LSH operating point is selected here.

Negative controls:
- dense ranking;
- bounded-state truncation;
- deterministic false-acceptance stress;
- operator lifecycle mechanism probe.

## Decision rules

1. Report every registered H and seed.
2. Keep raw address population separate from the admission cap.
3. Keep addressing, execution and verification costs separate.
4. Treat finite-range scaling measurements as descriptive.
5. Do not claim semantic-language competence.
6. Do not treat the bounded-state analogue as a named SSM result.
7. Do not promote lifecycle mechanisms from plumbing to capability without held-out evidence.
8. A downstream verifier or compute intervention cannot mask an upstream representation failure.

## Authoritative command

python scripts/run_plm_unified_phase_001.py

CI uploads:
artifacts/TACOSM-PLM-UNIFIED-PHASE-001.json

## Next phases

P2: verified state dynamics.

P3: operator population scaling.

P4: continual specialist learning and capacity lifecycle.

P5: strong Transformer/SSM/scan comparisons.
