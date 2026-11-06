# TAC-OSM Research Diagnostics: model-agnostic contract

Status: no-fit instrumentation architecture; **NOT six-variant qualification**.

## Principle

Research diagnostics must NEVER assume a particular TAC-OSM model version,
memory taxonomy, layer width, action dimensionality, prediction head, sequence
layout or state type. A new model has to declare its exact observable contract,
rather than silently inheriting the historical native model's shape.

Core tools:

- src/tac_osm/research_torch_basics.py — architecture module inventory,
  gradient conflict atlas with arbitrary objective names, layer names and
  trainable parameters.
- src/tac_osm/research_probe.py — universal ResearchProbeAdapter protocol,
  named-state tomography, per-head lesion differences, side-effect detection,
  optional candidate-order invariance, explicit dependency graph, optional
  batch isolation, and timing. Native \`fast/slow/memory\` and \`Belief\` do
  **not** occur anywhere in this core.
- src/tac_osm/research_anticheat.py — reusable data boundary, source target
  poisoning, partition checks, gradient/activation traces and leaf/functional
  first-divergence probes. Independent of model names and tensor widths.
- src/tac_osm/research_interpretation.py — version-independent hypotheses based
  on arbitrary state lesions, head names, gradient conflicts and missing
  diagnostic coverage; no fast/slow/matrix name assumptions.
- src/tac_osm/research_evidence.py — paired receipt comparer independent
  of model, architecture and target name.
- src/tac_osm/research_probe_adapters.py — the explicitly *native-specific*
  adapter for historical PredictiveStateAgent, containing every
  input/state/action/output assumption. This is a plugin for the core, not
  a universal default or an authorization for all six research families.
- src/tac_osm/robot_anticheat.py — the *physical-real-robot-specific*
  source window boundary; this is not a universal source format.

## Adapter requirements

\`ResearchProbeAdapter\` carries named and validated callbacks:

1. \`validate(model, decision_time_sample)\`: enforce correct model family,
   authentic source boundary contract, no target fields, legal shape/order
   and finite inputs. The core cannot inspect tensor semantics.
2. \`steps(sample), batch_size(sample), start(model,sample),
   advance(model,state,sample,t)\`: causal replay or zero-step stateless
   execution. Fixed-size state is not assumed.
3. \`state_views(state)\`: a named dict of tensors, possibly empty; zero
   editing by \`patch_state(state,name,replacement)\` must be pure.
4. \`forecast(model,state,sample)\`: named map of float tensors. It must not
   modify input state OR mutate model weights/buffers.
5. \`batch_item(sample,i)\` (optional): lawful source-based slicing by
   entity, NOT a fabricated sample; unsupported => INCONCLUSIVE.
6. \`gradient_graph(model,state,sample)\` (optional): returns named outputs and
   differentiable named leaf inputs at an actual replay state; unsupported
   => INCONCLUSIVE, not proof of independence.
7. \`permute_candidates(sample)\` (optional): produce a candidate-only
   reorder and explicit output restoration callback; unsupported =>
   INCONCLUSIVE, never test another dimension by accident.

### How to use

    from tac_osm.research_probe import inspect_variant
    from tac_osm.research_probe_adapters import native_predictive_adapter

    report = inspect_variant(
        your_authorized_model, your_authorized_decision_time_sample,
        native_predictive_adapter, isolation=True, profile=False)

For any DIFFERENT variant, implement a named, reviewed adapter with the
model's real causal source boundary, state ownership, numerical precision,
buffers, candidate semantics, output heads and runtime. Do not use the native
adapter unchanged just because the variant also has something called
"memory". Importing adapters dynamically from untrusted checkpoint data
is forbidden.

Use \`probe_memory\`, \`probe_dependencies\`, \`probe_entity_isolation\`,
\`probe_runtime\`, \`architecture_inventory\`, \`gradient_conflict_atlas\`,
and generic anti-cheat primitives programmatically in the source-authorized
FIT/DEV runner. The CLI \`scripts/run_research_observatory.py\` is presently
the **native adapter's safe checkpoint ingestion**; it delegates to the same
generic core. It deliberately does NOT guess an untrusted checkpoint's
architecture and does not dynamically import arbitrary Python modules from
metadata. Exposing other checkpoint formats requires a trusted adapter-specific
constructor and an explicitly verified source-pinned state dictionary.

## Interpretation and assurance

A missing optional adapter callback is **INCONCLUSIVE**. The aggregate
\`integrity_status\` may remain inconclusive despite successful probes. Runtime
tensor traces are not exhaustive dataflow proofs; tensors may hide future data
under an innocuous feature name. For each actual run, keep all source-boundary
and negative-control evidence in \`AuditLedger\` with all required checks.
Independent training-data source authentication and exact real-world
prediction-time cutoffs remain mandatory.

A state component zero patch can create physically invalid hidden states.
Its measured output delta is a local numerical sensitivity, **NOT causal
world dynamics**. Local Jacobian connectivity, batch isolation and parameter
gradients do not verify intelligence, retention reliability, deployment
safety or comparative superiority.

Check physical/model version provenance, genuine longitudinal input history,
FIT/DEV only, and parameter/data/compute parity according to existing
binding TAC-OSM research rules; never train in no-fit diagnostic CI.

## Automated CI regression matrix

The existing observatory GitHub Actions gate now runs
\`tests/test_research_probe_agnostic.py\` together with
\`tests/test_research_observatory.py\` on every PR and every push to main/master.
Constructed falsifiers exercise three sharply different structures:

| Fixture | State contract | Forecast contract |
|---|---|---|
| Native PredictiveStateAgent | fast, slow, matrix; recurrent | outcome and next-observation heads; candidate permutation |
| Alternate GRUCell episode model | one named \`cell_state\`; recurrent | score and reconstruction; different dimensionality |
| Stateless feedforward model | empty named state, zero replay steps | single risk output; no action candidates |

Further adversarial fixtures intentionally violate cross-entity independence,
prediction purity, adapter identity, source field allowlist and conditional
probe availability. These tests validate the **instrumentation's breadth**,
not the performance of the six future architecture families. Explicit
System 1 / Enterprise / Temporal / Physical / Edge-Lite / Counterfactual
adapters, real checkpoint CI, PNDS input-to-output proofs, and real-world
performance qualification are OPEN until each is implemented and verified.
