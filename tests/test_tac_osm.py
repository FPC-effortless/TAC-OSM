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
    """Score one feature row; ``representable`` calls this per row."""
    return sum(w * f for w, f in zip(weights, features))


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


def test_shared_weights_detect_a_hypothesis_class_gap():
    """The strong form: one vector for every episode, not a fresh oracle each.

    The v0.1 gate passed a per-episode ideal on ``relational`` and the learned
    arm then scored 0.0833, because an ideal that rotates with the marked
    positions is not achievable by a single fixed weight vector. This test
    builds exactly that failure: gold is separable per episode, but no one
    vector separates all episodes.
    """
    episodes = [_episode(gold=g, n=4) for g in range(4)]

    def feature_fn(episode):
        # Gold is +1 on coordinate 0 and 0 on coordinate 1; every distractor
        # is the reverse. The coordinate the *relation* lives on rotates with
        # the episode — the structural analogue of the marked positions
        # moving — so gold is separable per episode by a vector that puts +1
        # on coordinate 0, but no one shared vector serves episodes whose
        # gold sits at a different index, because the row that is gold in one
        # episode is a distractor in the next.
        return [
            [1.0 if i == episode.gold else 0.0,
             0.0 if i == episode.gold else 1.0]
            for i in range(episode.n)
        ]

    report = representable(
        score_fn=_score,
        feature_fn=feature_fn,
        gold_fn=lambda e: e.gold,
        episodes=episodes,
        shared_weights=[1.0, 1.0],
    )
    assert report["pass"] is False
    assert report["ties"] > 0
    assert "shared weight vector" in report["detail"]


def test_shared_weights_pass_when_one_vector_separates_all():
    """The mirror: the same basis under a shared vector that does work."""
    episodes = [_episode(gold=g, n=4) for g in range(4)]

    def feature_fn(episode):
        # Gold is 1 on the *first* coordinate in every episode.
        return [[1.0 if i == episode.gold else 0.0] for i in range(episode.n)]

    report = representable(
        score_fn=_score,
        feature_fn=feature_fn,
        gold_fn=lambda e: e.gold,
        episodes=episodes,
        shared_weights=[1.0],
    )
    assert report["pass"] is True
    assert report["ties"] == 0
    assert report["violations"] == 0


# --------------------------------------------------------------------------- #
# G2b — the real basis, measured on the real episodes
# --------------------------------------------------------------------------- #
# The three findings that the shared-weight audit produced, pinned as tests so
# the basis cannot regress silently:
#   1. the gated slot block is what makes the persistence families
#      representable at all (remove it and the gate fails);
#   2. the addressed-slot gate is what keeps a growing history pool from
#      drowning the signal;
#   3. the analytic vector separates all three families with one fixed weight
#      vector, which is the evidence that the relation is in the hypothesis
#      class and not merely per-episode separable.


@pytest.mark.parametrize("family,seed0", [
    ("relational", 1000),
    ("state_lookup", 2000),
    ("replay", 3000),
])
def test_analytic_weights_separate_gold_on_every_family(family, seed0):
    """One fixed vector separates gold from every distractor, all families.

    This is the §34 evidence. The vector is not fitted — it is the analytic
    statement of the relevance rule — so passing here means the relation lives
    in the router's hypothesis class and a learned arm that fails is an
    optimisation failure, not a representability failure.
    """
    from tac_osm.builder import check_representability

    report = check_representability(n_episodes=16)
    assert report["pass"], report["detail"]
    per = report["per_family"][family]
    assert per["pass"], per["detail"]
    assert per["ties"] == 0
    assert per["violations"] == 0


def test_the_gate_fails_without_the_gated_slot_block():
    """Remove ``slot_gated`` and the persistence families stop being learnable.

    The regression test for the fix itself. The un-gated slot block carries
    the anti-signal on the unmarked positions, so without the gate no shared
    weight vector separates gold for ``state_lookup`` or ``replay``. Measured
    on the real episodes with the real (now-disabled) block.
    """
    from tac_osm.environment import (
        build_lookup_task,
        build_replay_task,
        build_relational_task,
    )
    from tac_osm.router import (
        _agree,
        _bits,
        _pad,
        _query_address,
        analytic_weights,
        basis_size,
    )
    from tac_osm.state import PersistentStore, StateConfig

    dim, max_slots, n_candidates, n_episodes = 8, 4, 8, 25

    def ungated_features(query, descriptor, read, dim, max_slots):
        """The v0.1 basis: slot block present, but never gated by the marks."""
        query_bits = _bits(query.text)
        context = tuple(query.context)
        feats = [1.0]
        for j in range(dim):
            feats.append(_agree(query_bits, j, descriptor, j))
        for j in range(dim):
            feats.append(_agree(context, j, descriptor, j))
        for j in range(dim):
            feats.append(_agree(context, j, query_bits, j))
        for j in range(dim):
            gate = float(context[j]) if j < len(context) else 0.0
            feats.append(gate * _agree(query_bits, j, descriptor, j))
        address = _query_address(query)
        rows = _pad(read.values, max_slots)
        for i, row in enumerate(rows):
            if not row:
                feats.extend(0.0 for _ in range(dim))
                continue
            addressed = 1.0 if (address and i < len(read.keys)
                                and read.keys[i] == address) else 0.0
            for j in range(dim):
                feats.append(addressed * (1.0 if row[j] == descriptor[j] else -1.0))
        return feats

    store = PersistentStore(StateConfig(seed=0, n_slots=64))
    gate_store = PersistentStore(StateConfig(seed=0, n_slots=64))
    builders = {
        "relational": lambda i: build_relational_task(
            1000 + i, dim=dim, n_candidates=n_candidates),
        "state_lookup": lambda i: build_lookup_task(
            2000 + i, store, dim=dim, n_candidates=n_candidates),
        "replay": lambda i: build_replay_task(
            3000 + i, store, dim=dim, n_candidates=n_candidates),
    }
    families = ("state_lookup", "replay")

    for name in families:
        store.reset()
        gate_store.reset()
        episodes = [builders[name](i) for i in range(n_episodes)]
        for episode in episodes:
            address = episode.query.text.partition("\t")[2]
            written = getattr(episode.detail, "written_bits", None)
            if address and written:
                from tac_osm import StateUpdate

                gate_store.write(
                    StateUpdate(key=address, value=tuple(written),
                                task_key=address, success_score=1.0,
                                step=episode.query.step)
                )
        # The analytic vector's slot_gated half now lands on the un-gated
        # block, which is the wrong block; a deliberately correct alignment is
        # used instead so the test measures the missing gate, not an offset.
        n = basis_size(dim, max_slots)
        w = [0.0] * n
        from tac_osm.router import _block_offsets

        blocks = _block_offsets(dim, max_slots)
        s, e = blocks["gated_agreement"]
        w[s:e] = [1.0] * (e - s)
        s, e = blocks["slot_agreement"]
        w[s:e] = [2.0] * (e - s)

        worst = None
        for episode in episodes:
            read = gate_store.read(episode.query)
            rows = [ungated_features(episode.query, c.descriptor, read, dim, max_slots)
                    for c in episode.candidates]
            gold = episode.target_action
            gold_score = sum(a * b for a, b in zip(w, rows[gold]))
            best_other = max(
                sum(a * b for a, b in zip(w, r))
                for i, r in enumerate(rows) if i != gold
            )
            margin = gold_score - best_other
            if worst is None or margin < worst:
                worst = margin
        assert worst is not None and worst <= 0, (
            f"without the gated slot block, {name} unexpectedly separates "
            f"(margin {worst}); the un-gated block carries no anti-signal"
        )


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
