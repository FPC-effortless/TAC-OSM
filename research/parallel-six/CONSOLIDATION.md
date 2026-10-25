# Consolidation: six active TAC models on one parallel/stream architecture

2026-10-25. **Research-only, unmerged, no new fitted model or claimed gain.**

The old [six-family adapters](../SIX_VARIANT_ARCHITECTURE_AND_EVIDENCE.md) are brought intact from PR #358 onto the *v1.7 parallel graph branch* (source: TAC #389), instead of rewriting either historic code line. This defines the canonical prospective six families: System1, Physical (including ES-WAM), Enterprise, Temporal, Edge-Lite, Counterfactual. PNDS is the mathematical engineering authority and **not** model seven.

## Architectural interface

Input-only embeddings and causal prefix context -> provisional fast/slow 88D features -> normalized 8D key/value -> precomputed (memory-independent) gate schedule -> right-affine 8x8 associative matrix scan -> **read-before-write** read -> read-refined representations -> task-specific head -> next observation/decision *advisory* output. Trained task-specific head implementations are not implied by the shared core. Original native DIAGKEY004 has a different nonlinear sequential state transition. Reusing its weights in a new fully parallel graph changes semantics; it is not lossless original-source conversion.

**Two modes of one state update:** chronological O(T) work/O(log T) depth balanced prefix scan for batch sequences; stepwise rank-one memory update and fixed (88+64) FP32 state = 608 B/sample plus event count for live inference. These are algebraically compatible under the source-conditioned known gate schedule, **not** a proof that dynamic entity admission or future-feedback corrections are scan associative. Gradients and numerical finite precision need their own tests.

## What "six only" means

Exactly six **active products/research families**, a single shared graph implementation with task-specific adapters. The pre-existing 17 historical source-pinned *versions* are no longer counted as active families; their source/evidence/negative claims remain unchanged in the archival mathematical register. Additional unindexed legacy research remains in its original paths, PRs and foundation archive-gaps ledger. No renaming/removal/retraining of original checkpoints, test results, branches, or mathematical proofs.

Active manifest is `active-variants.json` in both repos. Ingested `research/SIX_VARIANT_EXPERIMENT_INDEX.json` from PR #358 remains historical (not six fitted successes). Family source adapters are engineering contracts; the new `src/tac_osm/variants/parallel_six.py` wires all six to the single experimental shared core *by configuration*, but refuses to certify any task-specific head or real-world skill. The physical continuous pilot is one input/output example only, not evidence for other output semantics.

## Open proof and empirical gates

- PNDS: per-family full typed observation→output mathematical specification, Lean definitions/proofs and Python reference numerical gradients/source equivalence; current PNDS exact-real scan theorem covers shared algebra only.
- Explicit real evidence source stamps, missingness, provenance, action timing, identity admission/retention and causal correctness for each family. Dynamic eviction/correction must be scheduled or sequential as necessary; no fake O(log T) credit.
- Original genuine data with future outcomes and held-out external cohorts, actual task-specific learned heads, equal authentic train/eval splits, exactly matched data exposure, parameter counts 95k–105k with ≤1% comparator mismatch, measured matched training FLOPs/tuning compute; CPU wall time is measured but **not** an equality requirement.
- Independent site task utility and real device latency verification. Counterfactual action effects **not** identified by logged conditional correlations alone. Edge hardware benchmarks, bitemporal revisions and executable policy qualification are separate work.
- Preserve v1.7 **negative ALOHA DEV**: multi-timescale and longer-retention alternatives were worse, despite mechanically longer survival. Read ablations show no robust beneficial retained information; gate remains closed.

**Do not merge based on architecture plumbing alone.** All original PRs #358/#389 and PNDS #7 remain draft evidence until experimental and formal gates justify promotion.
