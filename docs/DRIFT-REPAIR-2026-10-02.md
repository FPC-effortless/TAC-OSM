# TAC-OSM drift repair — 2026-10-02

This note records documentation/evidence corrections made on branch
research/unified-native-model-v1. Historical measurement artifacts are not
rewritten.

## Corrections

1. The evidence register previously said that the temporal persistence
measurement had not been run. That was stale. TACOSM-TEMPORAL-001 has a valid
artifact and a successful bounded result: carry remained at 1.0 through
k=32 while reset removed readable state. The register now records this as an
L2 bounded mechanism result.

2. The evidence map previously listed persistent write and temporal transition
as unimplemented. The code now has explicit write infrastructure, verified-only
knowledge commit, invalidation/retraction and a temporal state path. These are
L1 implementation surfaces; they remain distinct from causal learning claims.

3. The evidence map previously treated the last two loop transitions as an
empty gap. It now separates verified persistence learning from selective
asymptotic computation and records the unified multimodal benchmark as a new
research boundary.

4. Product-key sparsity language is kept bounded. The current evidence supports
capability/computation frontiers and fixed-fraction sparsity; it does not
establish asymptotic sublinear state computation.

5. The unified multimodal pilot executed before this branch's contract is
explicitly excluded from confirmatory inference. Its results may only expose
implementation defects. No threshold or architecture decision in the
registered contract is based on the pilot.

## Rule retained

Historical artifacts and contracts remain immutable evidence. A correction to
prose changes the evidence map, not the experimental result. A scientific
status changes only when the measurement or its registered interpretation
changes.
