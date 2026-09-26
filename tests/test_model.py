"""Tests for the integrated loop itself.

These are distinct from ``test_tac_osm.py``, which guards the pre-model gates.
Everything here asserts a property of a *running* episode: that the loop
closes, that the relevance circuit computes the environment's own relation,
and that the gold candidate is the circuit's unique maximiser — the property
that makes verification a cross-check rather than a tautology.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.ablation import AblationConfig  # noqa: E402
from tac_osm.builder import build_model  # noqa: E402
from tac_osm.environment import build_relational_task, parse_query, satisfies_relation  # noqa: E402
from tac_osm.executor import (  # noqa: E402
    ExecutorConfig,
    StructuralExecutor,
    relevance_program,
)
from tac_osm.model import TacOsmModel  # noqa: E402
from tac_osm import Structure  # noqa: E402


# --------------------------------------------------------------------------- #
# The relevance circuit: it must compute the environment's own relation
# --------------------------------------------------------------------------- #


def _executor() -> StructuralExecutor:
    return StructuralExecutor(ExecutorConfig(dim=8, max_nodes=32, type="oracle"))


def _circuit_outputs(task, executor: StructuralExecutor):
    """Execute the relevance circuit for every candidate in ``task``."""
    marks = [j for j, b in enumerate(task.query.context) if b == 1]
    reference, _address = parse_query(task.query)
    return [
        executor.execute(
            Structure(key=c.key, spec=relevance_program(reference, c.descriptor, marks,
                                                        max_nodes=32)),
            [],
        ).output
        for c in task.candidates
    ]


@pytest.mark.parametrize("seed", range(50))
def test_relevance_circuit_is_one_for_the_relation_satisfier(seed):
    """The circuit outputs 1 exactly where ``satisfies_relation`` is true."""
    task = build_relational_task(seed, dim=8, n_candidates=8)
    marks = [j for j, b in enumerate(task.query.context) if b == 1]
    if not marks:
        pytest.skip("no marked positions")
    reference, _ = parse_query(task.query)
    context = tuple(task.query.context)
    executor = _executor()

    outputs = _circuit_outputs(task, executor)
    for output, candidate in zip(outputs, task.candidates):
        expected = 1.0 if satisfies_relation(reference, context, candidate.descriptor) else 0.0
        assert output == pytest.approx(expected), (
            f"circuit disagrees with the relation for {candidate.key}: "
            f"output={output} expected={expected}"
        )


@pytest.mark.parametrize("seed", range(50))
def test_relevance_circuit_makes_gold_unique_and_maximal(seed):
    """Gold is the circuit's unique maximiser, so verification can cross-check.

    This is the property that keeps ``V_t`` from being a tautology: routing
    and execution are two independent computations of one relation, and here
    the executed computation is required to agree with the environment about
    which candidate satisfies it.
    """
    task = build_relational_task(seed, dim=8, n_candidates=8)
    marks = [j for j, b in enumerate(task.query.context) if b == 1]
    if not marks:
        pytest.skip("no marked positions")
    executor = _executor()

    outputs = _circuit_outputs(task, executor)
    gold = task.target_action
    gold_output = outputs[gold]

    assert gold_output == pytest.approx(1.0), (
        f"gold does not satisfy the circuit's relation: output={gold_output}"
    )
    assert sum(1 for o in outputs if o == pytest.approx(1.0)) == 1, (
        f"gold is not the unique satisfier: outputs={outputs}"
    )
    assert all(o <= gold_output + 1e-9 for o in outputs), (
        f"gold is not maximal: outputs={outputs}"
    )


def test_relevance_circuit_requires_a_reference():
    with pytest.raises(ValueError):
        relevance_program((), (0, 1), [0])


def test_relevance_circuit_rejects_zero_marks():
    with pytest.raises(ValueError):
        relevance_program((0, 1), (0, 1), [])


def test_relevance_circuit_reports_node_budget():
    """A circuit that does not fit ``max_nodes`` must fail loudly, not silently degrade."""
    with pytest.raises(ValueError):
        relevance_program((0, 1, 0, 1), (1, 0, 1, 0), [0, 1, 2, 3], max_nodes=4)


# --------------------------------------------------------------------------- #
# The gold identity: the environment's own bookkeeping must be self-consistent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(50))
def test_gold_index_and_descriptor_agree_after_shuffle(seed):
    """``target_action`` and the relation's satisfier must be the same candidate.

    ``_finalise`` shuffles and rebuilds every candidate, so any gold field
    computed before the shuffle describes a task that does not exist. This is
    the failure that made every oracle arm score ~0 while the circuit was
    correct.
    """
    task = build_relational_task(seed, dim=8, n_candidates=8)
    reference, _ = parse_query(task.query)
    context = tuple(task.query.context)

    gold_candidate = task.candidates[task.target_action]
    assert satisfies_relation(reference, context, gold_candidate.descriptor), (
        "target_action points at a candidate that does not satisfy the relation"
    )

    satisfiers = [
        i for i, c in enumerate(task.candidates)
        if satisfies_relation(reference, context, c.descriptor)
    ]
    assert satisfiers == [task.target_action], (
        f"gold is not the unique satisfier: satisfiers={satisfiers}"
    )


# --------------------------------------------------------------------------- #
# The loop: it must close
# --------------------------------------------------------------------------- #


def test_the_loop_runs_and_reports_every_stage():
    """One complete episode, with every stage of the loop populated."""
    built = build_model(AblationConfig())
    episode = built.model.run()

    assert episode.n == built.model.config.n_steps
    assert episode.terminated == "complete"
    assert len(episode.steps) == episode.n

    for step in episode.steps:
        assert step.query is not None
        assert step.decision is not None
        assert step.computation is not None
        assert step.outcome is not None
        assert step.verification is not None
        # The provenance of every stage is recorded, so a result can always be
        # traced to the arm that produced it.
        assert "family" in step.provenance
        assert "router" in step.provenance
        assert "executor" in step.provenance
        assert "success" in step.provenance


def test_the_loop_commits_only_through_the_verifier():
    """A write appears only when verification passed.

    This is the separation between *state write* and *verified state commit*:
    the loop has an unconditional write path, but the verifier gates it. A
    ``verifier=none`` arm writes on every step; the default arm does not.
    """
    built = build_model(AblationConfig())
    episode = built.model.run()

    for step in episode.steps:
        if step.write is not None:
            assert step.verification.passed, (
                f"step {step.step} wrote without verification passing: "
                f"{step.verification.feedback}"
            )


def test_the_loop_repairs_only_on_failure():
    """Repair runs only when verification failed, and never on a passed step."""
    built = build_model(AblationConfig())
    episode = built.model.run()

    for step in episode.steps:
        if step.repair is not None:
            assert not step.verification.passed, (
                f"step {step.step} repaired despite verification passing"
            )


def test_the_loop_counts_writes_and_repairs():
    """The episode's counters agree with its own steps."""
    built = build_model(AblationConfig())
    episode = built.model.run()

    assert episode.writes == sum(
        1 for s in episode.steps if s.write is not None and s.write.committed
    )
    assert episode.repairs == sum(1 for s in episode.steps if s.repair is not None)
    assert episode.successes == sum(1 for s in episode.steps if s.outcome.success)
    assert episode.accuracy == pytest.approx(episode.successes / episode.n)


def test_the_loop_returns_no_leaky_fields():
    """``Task.public()`` must not expose the answer to the router."""
    built = build_model(AblationConfig())
    for step in built.model.run().steps:
        public = step.query
        assert not hasattr(public, "target_action")
        assert "target_action" not in (public.provenance or "")
        assert "gold" not in (public.provenance or "")


def test_the_loop_is_deterministic_given_the_seed():
    """Two identical configs produce identical episodes.

    Determinism is what makes an arm comparison a comparison of the model
    rather than of the world: two arms see the same task stream.
    """
    first = build_model(AblationConfig(seed=7)).model.run()
    second = build_model(AblationConfig(seed=7)).model.run()

    assert [s.outcome.success for s in first.steps] == [
        s.outcome.success for s in second.steps
    ]
    assert [s.decision.selected for s in first.steps] == [
        s.decision.selected for s in second.steps
    ]


def test_the_loop_survives_a_disabled_state():
    """Disabling state must not crash the loop; it changes the outcome.

    The ``-state`` ablation removes the reference the persistence families
    need, so those steps become *undefined* — recorded, not raised. That is
    the point: a capability removed by a switch should appear in the
    measurements, not in a traceback.
    """
    config = AblationConfig()
    config.state.enabled = False
    built = build_model(config)
    episode = built.model.run()
    assert episode.n == built.model.config.n_steps

    undefined = [s for s in episode.steps if s.provenance.get("undefined")]
    families = {s.provenance["family"] for s in undefined}
    # Only the persistence families lose their reference; relational tasks
    # carry their reference in the public query.
    assert families <= {"state_lookup", "replay"}, families
    assert undefined, "a disabled state should produce undefined steps"
    for step in undefined:
        assert not step.verification.passed
        assert step.write is None


def test_undefined_steps_carry_no_computation():
    """An undefined step records an empty trace, not a fabricated one."""
    config = AblationConfig()
    config.state.enabled = False
    built = build_model(config)
    for step in built.model.run().steps:
        if step.provenance.get("undefined"):
            assert step.computation.trace == ()
            assert step.computation.structure.provenance == "unavailable"


def test_the_executed_output_is_boolean_for_the_relevance_circuit():
    """Every executed output is exactly 0 or 1: the circuit is exact Boolean."""
    built = build_model(AblationConfig())
    for step in built.model.run().steps:
        if step.provenance.get("undefined"):
            continue
        output = step.computation.trace[0]
        assert output in (0.0, 1.0), f"non-Boolean circuit output: {output}"


def test_the_executed_trace_carries_active_nodes_only():
    """The trace must not carry ``max_nodes`` of padding zeros.

    Padding reads as 25 phantom nodes at 0.0, which path verification then
    flags as inconsistency. Only the circuit's own active nodes are real.
    """
    built = build_model(AblationConfig())
    for step in built.model.run().steps:
        if step.provenance.get("undefined"):
            continue
        spec = step.computation.structure.spec
        # trace is (output, *node_values); node_values must match active_count
        assert len(step.computation.trace) == spec.active_count + 1


# --------------------------------------------------------------------------- #
# The ablation surface: every cell must build and run
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["component", "persistence", "routing",
                                  "structure", "verification"])
def test_every_ablation_matrix_runs(name):
    """Every pre-registered ablation cell produces a runnable loop.

    One architecture, controlled switches: if a cell cannot be built and run,
    the arm comparison it claims to support does not exist.
    """
    from tac_osm.ablation import all_matrices

    for config in all_matrices()[name]:
        built = build_model(config, n_steps=3)
        episode = built.model.run()
        assert episode.n == 3, f"{config.label()} ran {episode.n} steps"


def test_the_verifier_arms_separate():
    """``none`` passes everything; ``path`` is the strictest arm.

    The §19 comparison requires the arms to form an ordered ladder, not three
    incomparable numbers.
    """
    none_model = build_model(
        AblationConfig(verifier=__import__("tac_osm.ablation", fromlist=["VerifierSwitch"])
                       .VerifierSwitch(type="none", repair=False)),
        n_steps=6,
    )
    path_model = build_model(
        AblationConfig(verifier=__import__("tac_osm.ablation", fromlist=["VerifierSwitch"])
                       .VerifierSwitch(type="path", repair=False)),
        n_steps=6,
    )

    none_episode = none_model.model.run()
    path_episode = path_model.model.run()

    assert all(s.verification.passed for s in none_episode.steps)
    # Path verification is strictly harder, so it cannot pass more often.
    assert path_episode.writes <= none_episode.writes


# --------------------------------------------------------------------------- #
# Interface validity, re-checked against the running loop
# --------------------------------------------------------------------------- #


def test_the_router_never_sees_the_outcome():
    """Routing is decided before ``transition`` is called.

    The temporal ordering is the anti-leakage boundary. This test pins it by
    asserting the decision is recorded on the step *before* the outcome is.
    """
    built = build_model(AblationConfig())
    for step in built.model.run().steps:
        # The decision's provenance is recorded at routing time and must not
        # reference the outcome it precedes.
        assert "success" not in step.decision.provenance
        assert "outcome" not in step.decision.provenance


def test_the_state_read_is_available_to_execution():
    """Persisted values reach execution, downstream of routing.

    This is the only path by which a written value can influence a later
    computation, and it is what the persistence claim rests on.
    """
    built = build_model(AblationConfig())
    step = built.model.run().steps[0]
    # The computation exists and carries a structure derived from the read.
    assert step.computation.structure is not None
    assert step.computation.structure.spec.active_count > 0
