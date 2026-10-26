# TAC-OSM Research Observatory — architecture comprehension & interpretation

**Status: no-fit engineering tools.** This suite extends, rather than
replaces, the anti-cheat gates introduced in PR #417. It neither changes model
weights nor makes claims about intelligence, state-of-the-art performance,
generalization, formal correctness, or real-world autonomy.

## Why these tools exist

For each actually instantiated model and authentic FIT/DEV episode, we want
to answer:

- What modules and trained parameters exist, and what is actually executed?
- What information moves through fast, slow and associative memory over time?
- Which measured output is locally sensitive to each state/input pathway?
- Which objectives compete for gradients in shared representation blocks?
- Can two results legally be compared, and which subgroups account for a
  descriptive difference?
- Which observations justify diagnosis, and which require more evidence?

Existing guards catch selected leakage and invalid training metadata; these
tools illuminate what the actual checkpoint computes without adding a new
architectural variant.

## Modules and interfaces

| Tool | Function | Result | What it **does not** establish |
|---|---|---|---|
| Architecture Atlas | architecture_inventory | live trainable/deployed counts, module types, leaf-module invocation order and unused modules per phase | Invocation order is NOT dependency graph; modules may be used in a different phase |
| Memory Tomography | memory_tomography | time-indexed fast/slow/matrix norms and changes, patched-state readout differences, read-only candidate test | Zero patch is out-of-distribution; effect is NOT causal real-world importance or learned long-term recall |
| Dependency Microscope | input_dependency_map | autograd gradient and connectivity between current context/candidates/memory and outcome/future heads | A locally zero gradient does NOT mean universally unused |
| Gradient Conflict Atlas | gradient_conflict_atlas | per-shared-module pairwise loss gradient norms/cosines and missing shared pathways | Gradients alone don't establish training improvement or recommend reweighting |
| Entity Isolation | entity_isolation_probe | batched-vs-single entity state/output differences; flags cross-entity contamination inside model | Does not prove source entity IDs or joins are correct |
| Runtime Microscope | profile_native_runtime | per-host replay+query median/p95/min/max milliseconds, state bytes, torch threads | No matched model speed comparison, safety latency guarantee or hardware normalization |
| Evidence Comparator | compare_evidence | exact paired sample match, hashes, resource and seed checks, per-group error changes, blocked/compatible status | Matching self-reported receipts is NOT independent validation, significance, capability or architectural superiority |
| Interpretation Rules | interpret_snapshot | narrow investigation hypotheses when probes reveal suspected issues | Heuristic hypotheses, not causal diagnoses |
| CLI | scripts/run_research_observatory.py | reproducible JSON reports with no raw observations or weights | Never auto-opens TEST, authenticates publisher or trains |

### Native model audited

The directly supported model is
src/tac_osm/predictive_state_agent.py: its native Belief carries
fast[B,32], slow[B,32], memory[B,8,8]. The existing real-robot wrapper
src/tac_osm/real_robot_predictive_state.py contains this native core but
also additional physical correction heads; the current observatory CLI is
**not a whole-wrapper audit**. Other System 1, Enterprise, Temporal,
Physical, Edge-Lite and Counterfactual variants require actual, separately
declared adapters, not copies of a generic success label.

The native input history is B×T×C and candidates B×A×D. The probe never
fabricates labels or executes verify() without genuine verified feedback.
The observed "next_observation" head directly reads fast, slow and candidate
actions, not the associative matrix. The outcome head additionally reads
the matrix through action-conditioned recall. The dependency microscope
can inspect whether source behavior matches these intended pathways.

## Commands

    python -m pytest -q tests/test_research_observatory.py
    python scripts/run_research_observatory.py --fixture --entity-isolation --profile

Fixture data and weights are constructed/untrained and can validate only
the instrumentation, never real training behavior.

Use a **source-authorized FIT/DEV** checkpoint and a frozen input export
containing exactly history and candidates:

    python scripts/run_research_observatory.py \
      --input-json fit_episode_inputs.json \
      --expected-input-sha256 PREPINNED_INPUT_DIGEST \
      --checkpoint pretrained_native_state_dict.pt \
      --expected-checkpoint-sha256 PREPINNED_CHECKPOINT_DIGEST \
      --run-id STUDY_NATIVE_SOURCE_FIT_AUDIT \
      --output diagnostics.json

Optional --entity-isolation checks B>=2 independently sourced entities for batched versus individual agreement. Optional --profile measures local end-to-end runtime (including full history replay), records thread count and state bytes, and **must never** serve as an equal-hardware efficiency ranking without a separate matched protocol.

The input digest is a caller-provided integrity assertion, not evidence of
independent publisher authenticity. Keep the original data source, original
publisher hash, source licence, episode continuity, observation cutoff,
sample sequence and checkpoint provenance in the separate evidence ledger.
The CLI only loads PyTorch state_dicts using weights_only=True and strict
parameter matching. It does not run or fine-tune an optimizer.

Run loss conflict analysis in the *existing* authorized FIT/DEV training
runner, after evaluating genuine independently sourced loss terms, prior
to optimizer step and without modifying any gradients:

    from tac_osm.research_observatory import gradient_conflict_atlas
    diagnostics = gradient_conflict_atlas(model, {
        "real_next_state": physical_loss,
        "verified_takeover": takeover_loss,
    })

These genuine terms must come from the exact matched original cohort with
correct missingness/availability masks. A scalar created from synthetic
targets or an arbitrary parameter norm is NOT equivalent to a real loss
conflict study.

Inspect two separately source-pinned per-example evaluation receipts:

    python scripts/run_research_observatory.py --compare \
      left_evaluation_receipt.json right_evaluation_receipt.json

Minimal receipt schema has run_id, checkpoint_sha256, source_sha256,
FIT/DEV/TEST split SHA256s, feature/target/evaluation contracts SHA256s,
training sample order, optimizer and tuning policy hashes, evaluation
metric/direction, seed, examples_seen, optimizer_steps, actual
trainable/deployed parameters and training_flops, plus observations:
list of id_sha256, value, group and horizon.

Comparison FAILS when source/splits/contracts/sample order/seed/steps differ,
when any sample or its group/horizon mismatches, or when active parameter
counts or training FLOPs differ by more than 1%. A compatible pair remains
**NOT eligible for architecture superiority** on its own: 2+ seeds,
genuine dataset rights/identity, an independent run audit, preregistration,
matching budgets, heldout independence, actual score reproduction, and
appropriate statistical confidence are external requirements.

Do not change the matching thresholds in response to results, repeatedly
inspect a sealed TEST while developing the model, or count any fixture
result toward capability promotion.

## Use alongside the existing anti-cheat framework

Run src/tac_osm/research_anticheat.py for direct forbidden-data poisoning,
cross-partition leakage, source time cut, activation NaNs and noninterference.
An architecture inventory / local Jacobian is NOT an anti-cheat substitute;
a tensor can encode forbidden information under an allowed name. The
first-divergence tools must be applied to the true producer-to-head path.

## Interpretation rubric

- Healthy prediction and memory sensitivity does **not** prove state is
  correct; paired long histories with genuinely delayed outcomes are needed.
- No associative-memory effect at a particular output might be expected if
  the task/head never reads it; inspect local autograd and source expression.
- An inactive expected module on one phase might activate in verify(); use
  phase-specific traces instead of globally labeling it dead.
- Negative loss-gradient cosine indicates immediate optimization tension
  on a shared subset of parameters, but not whether one objective is wrong.
- A paired metric delta without complete provenance and equal-resource
  controls is descriptive only, not an architecture ranking.
- Every full mathematical claim belongs in source-pinned PNDS Lean + Python
  proof obligations; runtime tracing cannot replace finite-precision and
  AD refinement.

## Remaining open work

Explicit real-data adapters for every registered research variant; causal
source-to-output lineage for every tensor producer; per-step actual retention
of verifiable facts; per-module FLOP/CPU measurement with synchronized runs;
independent source data auditing; complete forward/backward/precision
refinement in PNDS; and robust multi-seed, real-world long-horizon
decision-cost qualification. All are OPEN, not implicit passes.
