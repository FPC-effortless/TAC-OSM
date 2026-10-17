# DEV-049–050: reliability gates and Veritas research boundary

DEV-049A asks whether a fixed empirical reliability constraint and an executed fallback can improve fresh-seed reliability. DEV-050A asks whether the evidence–action–verification loop survives partial observations, a shared resource and uncertain repair outcomes in newly authored public Veritas worlds. Neither study establishes integrated TAC-OSM superiority.

## What is implemented

DEV-049A trains forecast-only, joint predictive/decision and failure-penalized objectives from matched initializations, with 216 AdamW updates per arm and ten fresh seeds. It retains the existing public-event encoder and entity-local 200-scalar memory. Calibration chooses a cost threshold and a separate zero-observed-failure threshold. A disjoint certification cohort accepts the latter once; rejection locks service-all before evaluation. All fallback services incur their actual simulator cost. The failure penalty is a smooth surrogate, not hard-constrained optimization.

DEV-050A uses Veritas at `b21d829b62d3cab2811e746267ca305e110ed004`. Its native `OperationalRuntime`, hidden transition preconditions, action events and seven-dimensional verifier execute and assess two-entity maintenance worlds. The TAC adapter supplies a charged current-state sensor, whose observable results are recorded separately from the harness-only native trace. Policies receive only the public task/records and observable action callback. The worlds are authored for this study; no canonical or sealed Veritas benchmark is accessed.

The shared slot prevents simultaneous isolation. Service requires isolation and ordered attempt numbers. First repair succeeds with probability .75 when generating a world; its outcome is then deterministic for replay. A second attempt succeeds by construction. High-confidence stale records can be wrong. Inspection, isolation, service and residual failure cost .1, .2, 1 and 4 respectively. Assumed accurate sensing and an eventually effective repair are limitations of the generator.

The six hand-written controls are WAIT, service without isolation, stale-record service, service-all once, inspect/service once and inspect/verify/retry. This isolates the benefit of verifying an uncertain intervention against the same inspect-once controller. It does not compare trained TAC-OSM against a native Veritas expert or establish general transfer.

## Claims that remain hypotheses

* Abstention is a request for another action, not a free safe outcome. Waiting, inspection, service and review must have explicit admissibility, information value, cost and coverage.
* Action-level distillation and contrastive scoring can supervise useful decisions. They do not guarantee calibration, prevent saturation or enforce reliability. DEV-049 does not label its failure penalty as ActionOPD.
* An SSM can maintain bounded state along a sequence. Preserving separate states for E entities still uses O(E) memory; a single fixed shared state introduces interference or capacity trade-offs. A constant-total-memory superiority claim needs a separate accuracy/retention/scaling experiment.
* Bacterial and retinal mechanisms motivate ablations; they do not establish a neural architecture or safety property.
* GEPA reflection may propose development candidates. This study uses a finite, prespecified threshold grid, not an LLM reflection loop or GEPA implementation. No optimizer may revise a recipe after inspecting its confirmatory outcomes.
* A weighted verifier reward cannot compensate for an individual violated invariant, a remaining fault or missing coverage.

## How Veritas can help next

The native verifier provides separate outcome, state, constraints, side-effects, process, efficiency and evidence results. Blocked actions do not mutate the state or count as successful service. Trajectory-wide invariants retain a violation even after the final state is repaired. These are useful gates for simulator-supervised action learning and adversarial tests.

Veritas also contains `foundry/tracing.py`, `trace_store.py`, `training_adapters.py` and `training_product.py`: reusable trace, replay and framework-neutral SFT/RL/preference/VOPSD export machinery. They are candidates for the next integration, not features exercised by this adapter. Exported training states must contain only public observations available before the selected action; hidden consequences belong in evaluator supervision. A training export is not evidence of learning or transfer.

The next research gate is to train a policy on student-encountered public states, compare it with the inspect/verify/retry reference and an explicit belief-state planner, vary sensor error and repair effectiveness, and measure failures and total computation on an independently frozen domain shift. A learned model must earn its memory, sensing and verification costs.

## Qualification

These are development experiments and public engineering controls. Checkpoint inference and physical replay audit stored decisions; they do not independently repeat every optimization step. Ten seeds provide variability evidence, not a zero-risk guarantee. Veritas itself distinguishes implementation, scientific, frontier and release qualification; using its runtime confers none of these qualifications on TAC-OSM.

The first DEV-050 development execution used source `af3cacb`. Review then identified that attempt numbers needed explicit transition preconditions. Source `c8e3ec8` adds an exploit regression and enforces first-before-second/no-repeated-first ordering. The original evidence is retained; the full study was rerun and every aggregate result remained unchanged. This correction followed a public development run and is disclosed rather than described as untouched confirmation.
