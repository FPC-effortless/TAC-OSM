# TAC-OSM Unified Research Governance

Standing repository-wide research contract for historical, active, proposed, corrective, and future lanes.

## Continuity before novelty
New ideas are additive unless an explicit evidence-backed decision closes or supersedes an existing lane.
An agent must not silently abandon, overwrite, or reinterpret unfinished work. Every lane retains its question, evidence, blockers, unfinished experiments, invalidated runs, successor links, and next action.

## Lane lifecycle
PROPOSED -> PREREGISTERED -> IMPLEMENTING -> RUNNABLE -> RUNNING -> MEASURED -> REPRODUCING -> CLOSED
Alternative states: ON_HOLD and INVALIDATED. ON_HOLD requires a named blocker. INVALIDATED requires a preserved failure record and, when appropriate, a corrective successor.

## Anti-abandonment
Before implementing a new idea, inspect the lane registry, roadmap, evidence map/register, active PRs, contracts, and current blockers.
When a new idea arrives: register it; map it to an existing lane or create a new lane; record dependencies/conflicts; choose the next runnable blocker; leave other open lanes visible.
Priority may change only through a decision record containing old priority, new priority, rationale, deferred work, and affected lanes. Priority change is not scientific closure.

## Historical evidence
Research completed before this policy is LEGACY.
Legacy evidence is not retroactively declared false because the current gate did not exist. Audit it as LEGACY-AUDITED, LEGACY-BOUND, or LEGACY-UNVERIFIED.
LEGACY-UNVERIFIED evidence cannot silently become stronger evidence or sole support for a new system-level claim.

## Universal gates
Every confirmatory experiment uses G0-G10 before execution: evidence/necessity; contract/provenance; security/supply chain; benchmark integrity; interface and supervision leakage; representability; model-state integrity; degeneracy/discrimination; oracle/control validity; ablation/capacity/compute integrity; statistical readiness.
After execution it uses P0-P7: artifact/provenance integrity; observed-runtime leakage audit; benchmark/invariant re-check; independent metric recomputation; preregistered statistical decision; ablation completeness; failure/anomaly classification; scientific disposition.
Passing the gates makes a run eligible for evidence. It does not imply that the hypothesis is true.

## Security
Use least-privilege CI permissions, immutable third-party commits, pinned runtime/dependency versions, declared artifact paths, no artifact overwrite, path-traversal and unexpected-symlink checks where applicable, and secret scanning of logs/config/artifacts.
Measurement code must not acquire hidden labels, evaluator-private state, or mutable remote data during confirmation.
Generated agent code is untrusted until tests, diff inspection, and the run gates accept it.

## Leakage
Interface leakage: no component receives target, answer, gold, evaluator truth, future state/evidence, oracle information, or metadata that uniquely encodes them.
Supervision leakage: every learned intervention documents what reaches gradients, rewards, teachers, and auxiliary losses, and whether the same information exists at deployment.
An interface-safe gold-derived signal is still answer supervision and cannot be described as gold-free.
Post-run leakage auditing inspects observed runtime inputs where serializable; source inspection alone is insufficient.

## Benchmark integrity
Every benchmark has immutable version identity and generator provenance, disjoint splits, independent RNG streams, class/support checks, post-randomization truth derivation, candidate-order/count invariance, nuisance-variable checks, causal-variable checks, known-cheater tests, and exact generated-output evaluation.
For memory/persistence claims, paired histories with identical current observations but different optimal actions are mandatory, together with current-input-only controls.
Benchmark defects are instrumentation failures, not scientific negatives.

## Verification
Verification is independently audited. Each lane specifies verifier inputs, acceptance/rejection semantics, tolerance/range behavior, trace semantics, repair bounds, and the relation between environment success and verifier success.
Executed traces contain only executed values. Verified state commit and unconditional write remain separable when state updates are studied.

## Linting
Every research change passes syntax/compile checks, repository tests, contract validation, contract/script completeness, lane-registry checks, benchmark/schema checks, interface/import checks, static leakage checks, result-envelope checks, and frozen-result protection.
Linting is an admissibility boundary, not cosmetic formatting.

## Ablation
Every new architectural mechanism requires removal, matched-capacity, matched-compute or explicit work accounting, randomized/shuffled, and oracle/ceiling controls where meaningful.
Interactions are required when the claim depends on mechanism synergy. Loss, router accuracy, teacher agreement, information gain, and verifier rates are diagnostics, not capability proofs by themselves.

## Statistics
Before confirmation freeze primary unit, seeds, levels, endpoint, uncertainty method, multiple-comparison handling, materiality threshold, and decision rule. Failed or missing arms are explicit and never silently filtered.

## Invalid runs
Use one mutually exclusive disposition: PROTOCOL_INVALID, SECURITY_INVALID, LEAKAGE_INVALID, BENCHMARK_INVALID, REPRESENTABILITY_INVALID, MODEL_STATE_INVALID, DEGENERATE_INVALID, IMPLEMENTATION_INVALID, UNDERPOWERED, SCIENTIFIC_NULL, SCIENTIFIC_POSITIVE.
Implementation and benchmark defects are never relabeled as scientific negatives. Invalid artifacts remain visible and immutable; successors record exactly what failed and why.

## Lane closure
A lane may be CLOSED only when its registered question is answered, mandatory arms are resolved, post-run audits are complete, evidence layer and exclusions are recorded, and remaining blockers are closed or explicitly transferred to named successors.
Silence, inactivity, a newer idea, or green CI is not closure.

## Machine-readable continuity
Each lane retains: lane_id, title, status, question, hypothesis, priority, depends_on, blocked_by, parent_lane, successor_lanes, existing_evidence, not_inherited, active_experiments, next_action, last_updated.
This record exists so a new coding agent cannot erase unfinished research from effective context.

## Universal future rule
Any new research branch, experiment script, benchmark, contract, measurement output, architectural block, or evaluation surface must reference an existing lane or register a new lane before confirmation.
A new measurement script with neither a contract/lane registration nor an explicit named exemption is a test failure.

All TAC-OSM, PLM, G-CASM, CASM, CDL, C5, active-evidence, multimodal, bacterial-inspired, and future lanes follow the same policy.
## Default architecture research method

New architectural mechanisms follow `docs/TOP_DOWN_BOTTOM_UP_RESEARCH_METHOD.md`:
assemble the complete decision loop around one novel block with standard
surrounding primitives, then isolate the block with preregistered bottom-up
ablations and causal interventions. A new mechanism cannot be promoted from an
isolated component metric or from an integrated run whose surrounding controls
are absent.
