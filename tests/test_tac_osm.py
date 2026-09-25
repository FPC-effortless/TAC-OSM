"""Tests for the TAC-OSM integration boundary.

These are the pre-model gates of ``docs/ABLATION_PLAN.md`` §2. They are the
only tests that must pass *before* any model is trained, because a failure
here invalidates every learned-arm measurement that follows.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.ablation import (  # noqa: E402
    AblationConfig,
    all_matrices,
    component_matrix,
    freeze,
    persistence_interventions,
    routing_arms,
    structure_arms,
    verification_arms,
)
from tac_osm.leakage import audit_router_inputs  # noqa: E402
from tac_osm.representability import representable  # noqa: E402


# --------------------------------------------------------------------------- #
# G1 — interface validity
# --------------------------------------------------------------------------- #


def test_router_inputs_must_not_leak_gold():
    """A candidate carrying gold_index is a contamination, not an oversight."""
    from tac_osm import Candidate, Query

    q = Query(text="q")
    bad = Candidate(key="c", descriptor=(0, 1), action=0, provenance="synthetic")
    # Simulate a leaky dataclass by constructing one with a forbidden field.
    leaked = type("Leaky", (), {"key": "c", "descriptor": (0, 1), "gold_index": 0})()
    with pytest.raises(AssertionError, match="leakage"):
        audit_router_inputs(query=q, candidates=[leaked, bad])


def test_router_inputs_clean():
    from tac_osm import Candidate, Query

    report = audit_router_inputs(
        query=Query(text="q", context=(1, 0)),
        candidates=[Candidate(key="c", descriptor=(0, 1))],
    )
    assert report["pass"] is True


def test_query_carries_no_answer():
    from tac_osm import Query

    assert "answer" not in {f.name for f in __import__("dataclasses").fields(Query)}


# --------------------------------------------------------------------------- #
# G2 — representability
# --------------------------------------------------------------------------- #


def _episode(gold: int, n: int = 4) -> object:
    return type("Ep", (), {"gold": gold, "n": n})()


def _features_aligned(episode) -> list[list[float]]:
    """A basis where the gold row is separable: the f989430-corrected shape."""
    n = getattr(episode, "n", 4)
    gold = getattr(episode, "gold", 0)
    return [[1.0 if i == gold else 0.0] for i in range(n)]


def _features_collapsed(episode) -> list[list[float]]:
    """The dd8f63c failure: identical rows for every candidate.

    Analytically ideal weights score exactly 0 for every candidate — a
    constant in its own null space — so the relation is unrepresentable.
    """
    n = getattr(episode, "n", 4)
    return [[0.5] for _ in range(n)]


def _score(weights, features):
    return [sum(w * f for w, f in zip(weights, row)) for row in features]


def test_representability_passes_on_correct_basis():
    episodes = [_episode(gold=g, n=4) for g in range(4)]
    report = representable(
        score_fn=_score,
        feature_fn=_features_aligned,
        gold_fn=lambda e: e.gold,
        episodes=episodes,
    )
    assert report["pass"] is True
    assert report["min_margin"] is not None and report["min_margin"] > 0


def test_representability_fails_on_null_space_basis():
    """This is the gate that would have caught dd8f63c before four commits."""
    episodes = [_episode(gold=g, n=4) for g in range(4)]
    report = representable(
        score_fn=_score,
        feature_fn=_features_collapsed,
        gold_fn=lambda e: e.gold,
        episodes=episodes,
    )
    assert report["pass"] is False
    assert "null space" in report["detail"]


def test_representability_rejects_too_few_episodes():
    report = representable(
        score_fn=_score,
        feature_fn=_features_aligned,
        gold_fn=lambda e: e.gold,
        episodes=[],
    )
    assert report["pass"] is False


# --------------------------------------------------------------------------- #
# G3 — benchmark validity, expressed on the ablation surface
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("matrix", all_matrices().keys())
def test_every_ablation_cell_has_a_label(matrix):
    for cell in all_matrices()[matrix]:
        assert cell.label(), cell
        assert cell.describe()["label"] == cell.label()


def test_component_matrix_removes_exactly_one_mechanism():
    """Adjacent rows differ in one switch only."""
    rows = component_matrix()
    full = rows[-1]
    for row in rows[:-1]:
        diffs = 0
        diffs += row.state.enabled != full.state.enabled
        diffs += row.router.type != full.router.type
        diffs += row.structure.type != full.structure.type
        diffs += (row.verifier.type != full.verifier.type) or (
            row.verifier.repair != full.verifier.repair
        )
        assert diffs == 1, (row.name, diffs)


def test_component_matrix_covers_all_mechanisms():
    names = {row.name for row in component_matrix()}
    assert names == {
        "baseline",
        "plus_state",
        "plus_routing",
        "plus_structure",
        "plus_verifier",
        "full",
    }


@pytest.mark.parametrize("index", range(len(component_matrix())))
def test_labels_distinguish_cells(index):
    """Two cells must never share a label, or results cannot be told apart."""
    rows = component_matrix()
    labels = [c.label() for c in rows]
    assert labels.count(labels[index]) == 1, labels


def test_persistence_interventions_include_corruption_dose_response():
    """Stage 3b-r2 measured 0.8667 -> 0.3500 -> 0.1000 -> 0.0000.

    A dose-response ladder is what a positive persistence result looks like;
    the matrix must contain it as a first-class arm.
    """
    rates = [
        cell.state.corruption_rate
        for cell in persistence_interventions()
        if cell.state.intervention == "corrupted"
    ]
    assert rates == [0.25, 0.5, 1.0]


def test_routing_arms_include_the_static_control():
    """Stage 2b: a fixed similarity scorer captured ~71% of headroom for free.

    A routing result that does not beat static is not a routing result, so
    static must always be present.
    """
    types = {cell.router.type for cell in routing_arms()}
    assert types == {"learned", "static", "random", "oracle", "full_context"}


def test_verification_arms_include_the_path_comparison():
    """No source repository has run path-vs-final verification."""
    types = {
        (cell.verifier.type, cell.verifier.repair) for cell in verification_arms()
    }
    assert ("final", False) in types
    assert ("path", False) in types
    assert ("path", True) in types


def test_history_levels_are_frozen_before_use():
    """A parity margin chosen after seeing results invalidates the claim."""
    cell = AblationConfig()
    assert cell.history.levels == tuple(sorted(set(cell.history.levels)))
    assert cell.history.parity_margin is None, (
        "the parity margin must be declared before the confirmatory run, "
        "not defaulted"
    )


def test_freeze_records_provenance():
    cell = freeze(AblationConfig(), commit="abc123", experiment_id="exp-001")
    assert cell.provenance["commit"] == "abc123"
    assert cell.provenance["experiment_id"] == "exp-001"
    assert cell.provenance["config"]["label"] == cell.label()


def test_structure_arms_include_flattened():
    """Flattened compute tests whether structural routing beats dense compute."""
    types = {cell.structure.type for cell in structure_arms()}
    assert types == {"learned", "random", "oracle", "flattened"}


def test_no_ablation_cell_silently_resets_provenance():
    base = AblationConfig()
    base.provenance = {"commit": "base"}
    derived = [c for c in component_matrix() if c is not base]
    for cell in derived:
        assert cell.provenance == {}, cell.name
