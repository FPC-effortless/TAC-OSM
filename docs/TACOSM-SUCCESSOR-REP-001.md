# TACOSM-SUCCESSOR-REP-001

**Status: PREREGISTERED — no capability result yet.**

## Purpose

Test whether Successor Architecture v1 can represent and learn the intended
query-to-candidate relevance relation after the C5-003 identifiability defect
has been removed.

This is a successor diagnostic. It does not modify, replace, or reinterpret
C5-001, C5-002, or C5-003.

## Architecture under test

```
Persistent State
  -> StateAddressor
  -> query representation z_q
  -> candidate representation z_i
  -> energy E(z_q,z_i)
  -> Top-K
  -> explicit executable graph
  -> exact Boolean execution
```

The executable topology is supplied as program structure. The router does not
infer hidden true edges.

## Primary questions

1. Is the required relevance relation representable by the successor router?
2. Can learned routing recover that relation from outcome-derived positives and
   hard negatives?
3. Does routing quality remain stable when unrelated state slots are added?
4. What is the routing computation cost as candidate population N changes?

## Preconditions

The run is invalid unless:

- repository CI passes the pre-model gates;
- the constructive representability test passes;
- exact executor tests pass;
- frozen legacy reward tests pass;
- baseline test-count provenance is current;
- C5 evidence files are unchanged.

## Conditions

A. Constructive analytic witness — representability only.

B. Random initialization + learned outcome ranking.

C. Random initialization + no learning.

D. Oracle-selected candidate + exact execution.

E. Learned-selected candidate + exact execution.

The analytic witness is not a trained capability result.

## Dataset

Use fixed seeds and the existing relational, lookup, and replay task
generators. Candidate action identity must not be supplied to the router as a
feature.

For the first diagnostic:

- 5 fixed seeds;
- 128 training episodes per seed;
- 128 held-out episodes per seed;
- 8 candidates;
- exact executor;
- Top-K = 1;
- fixed dimension 8;
- latent dimension 8.

## Primary metrics

- Top-1 target rank;
- Top-K target recall;
- selected-energy;
- hard-negative margin;
- episode success;
- exact execution success;
- candidate coverage;
- router MACs;
- state slots inspected;
- selected graph active nodes/edges.

## Scaling extension

Only after the fixed diagnostic passes, evaluate:

[
Nin{8,16,32,64,128}
]

and state-slot populations:

[
Hin{8,32,128,512,2048}.
]

Report:

[
C_{m route}(N)
=
d_zd_q+Nd_zd_c+Nd_z
]

and separately report state-addressing work. Do not combine these into a
single "sublinear" claim.

## Falsifiers

The successor architecture fails this diagnostic if any of the following
holds:

- constructive representability test fails;
- learned routing does not exceed the frozen no-learning baseline on held-out
  Top-1 rank/recall;
- performance depends on candidate action index;
- unrelated state-slot growth changes the addressed semantic representation;
- exact executor output differs from the declared executable topology;
- routing MACs or state work are not reproducible from recorded dimensions.

## Non-claims

Passing this diagnostic does not establish:

- general intelligence;
- compositional generalization;
- sublinear retrieval;
- economic superiority over Transformers;
- learned sparse execution;
- long-horizon planning;
- PLM validity.

Those require separate experiments.

## Evidence boundary

All results from this document are successor evidence (L2/L3 depending on the
specific claim). Historical C5 results remain frozen evidence and must not be
recomputed into a successor aggregate.
