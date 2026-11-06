# Integrated E2E benchmark correction — generator defect

## Disposition

**E2E-001: VOID for benchmark validity.**  
**E2E-002: VOID for benchmark validity; no scientific interpretation.**

A static audit of the shared episode generator found that the query-construction
helper accepted an `entity` argument but returned `entities[0]` unconditionally.
Consequently q1 and q2 queried the same persistent entity, despite the
registered protocol stating that they should target different entities.

The relevant defect was:

```python
def make_query(combo, entity):
    ...
    return entities[0], i, j, op, y
```

The intended construction is:

```python
return entity, i, j, op, y
```

with:

- q1 targeting `entities[0]`;
- q2 targeting `entities[1]`.

## Why this is material

This is not a cosmetic discrepancy. The experiment's temporal state question
depends on a sequence of distinct query entities. With the defect, q1 and q2
operate on the same state slot.

That changes the causal graph around the post-action update:

`q1 -> outcome -> verified write -> q2`

because q1's post-action write can now directly modify the same entity that q2
reads. Under the registered design, q1 and q2 target different entities and
that direct same-slot pathway does not exist.

Therefore the previously measured E2E-001 mean of 0.5200 and its controls cannot
be interpreted as evidence about the registered temporal/persistent-state
question.

## E2E-002 consequence

E2E-002 imported the same generator through
`scripts/run_integrated_e2e_001.py`. Its address-path interventions therefore
ran on the same malformed query-target protocol.

The E2E-002 run `37063799293` failed later in the evaluator and produced no
artifact, so there is no measured scientific result to preserve. Its
pre-registration is nevertheless marked superseded because its benchmark was
not the registered task.

## Evidence status

The earlier E2E-001 artifact remains available for provenance, but its numeric
results are **not evidence** for the registered claim and must not be used to
select or tune the corrected experiments.

The prior E2E-001 Layer-3 negative entry is superseded by this audit finding.

## Corrective experiment

A new experiment ID is required.

The corrected experiment must:

1. implement an independent benchmark generator whose q1/q2 entity identities
   are asserted to be distinct before measurement;
2. reproduce the exact registered held-out composition set;
3. retain the same five seeds and evaluation length;
4. keep the address-path diagnosis as a preregistered intervention if that is
   still the scientific question;
5. execute the full benchmark only after the benchmark-integrity test verifies
   q1 entity != q2 entity for every generated episode.

No result from E2E-001 or E2E-002 may be used to choose a hyperparameter,
threshold, seed, or arm in the corrected experiment.
