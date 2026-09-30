"""Tests for TACOSM-REP-003 semantic program routing."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm import Candidate, Structure
from tac_osm.explicit_executor import ExplicitGraphExecutor
from tac_osm.graph_program_router import (
    GraphProgramRouter,
    GraphProgramRouterConfig,
    semantic_match_rank,
)
from tac_osm.semantic_topology import (
    build_semantic_task,
    program_for_candidate,
    semantic_query,
    semantic_signature,
)
from tac_osm.state import PersistentStore
from tac_osm.state import StateConfig
from tac_osm.topology_tasks import TOPOLOGY_EDGE_UNIVERSE


def test_semantic_query_is_not_an_edge_mask():
    task = build_semantic_task(2)
    query_width = len(task.query.text.partition("\t")[0].split())
    assert query_width == 5
    assert query_width != len(TOPOLOGY_EDGE_UNIVERSE)


def test_target_semantic_signature_is_unique_in_candidate_pool():
    task = build_semantic_task(5)
    target = semantic_signature(task.target_edges)
    signatures = [semantic_signature(c.executable_edges) for c in task.candidates]
    assert signatures.count(target) == 1


def test_semantic_signature_tracks_transitive_output_dependency():
    task = build_semantic_task(7)
    for candidate in task.candidates:
        signature = semantic_signature(candidate.executable_edges)
        assert sum(signature[:3]) == 1
        assert sum(signature[3:]) == 1


def test_graph_router_ignores_action_index():
    task = build_semantic_task(11)
    router = GraphProgramRouter()
    before = router.score(task.public(), task.candidates)
    swapped = tuple(
        Candidate(
            key=c.key,
            descriptor=c.descriptor,
            action=100 - i,
            provenance=c.provenance,
            executable_edges=c.executable_edges,
        )
        for i, c in enumerate(task.candidates)
    )
    after = router.score(task.public(), swapped)
    assert before == after


def test_analytic_semantic_matcher_is_exact():
    for seed in range(5):
        task = build_semantic_task(100 + seed)
        rank, margin = semantic_match_rank(task.public(), task.candidates)
        assert rank == 1
        assert margin == 4.0


def test_graph_router_topology_is_visible_without_action():
    task = build_semantic_task(17)
    router = GraphProgramRouter(GraphProgramRouterConfig(seed=3))
    observations = [router.topology_observation(c) for c in task.candidates]
    assert len(set(observations)) == len(task.candidates)


def test_selected_program_executes_its_declared_topology():
    task = build_semantic_task(23)
    candidate = task.candidates[task.target_action]
    program = program_for_candidate(candidate, input_values=task.input_values)
    execution = ExplicitGraphExecutor().execute_with_work(
        Structure(key=candidate.key, spec=program),
        [],
    )
    expected = tuple(edge in candidate.executable_edges for edge in TOPOLOGY_EDGE_UNIVERSE)
    assert tuple(bool(g) for g in execution.result.gates) == expected


def test_semantic_query_round_trips_from_target_signature():
    task = build_semantic_task(31)
    signature = semantic_signature(task.target_edges)
    assert semantic_query(signature, step=task.query.step).text == task.query.text
