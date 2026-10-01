"""Controlled topology-identifiability tasks for TACOSM-REP-002.

Every candidate shares the same node-local descriptor and candidate substrate,
but carries its own explicit executable edge set. The public query names the
desired edge-set mask. This creates the missing information boundary without
exposing an environment-side gold field.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, Query
from .executor import Edge, INPUT, NOT, Node, Program

__all__ = [
    "TOPOLOGY_EDGE_UNIVERSE",
    "TopologyTask",
    "build_topology_task",
    "edge_mask",
    "program_for_candidate",
]


# Three inputs feed node 3; node 4 can read any input or node 3.
TOPOLOGY_EDGE_UNIVERSE: tuple[tuple[int, int, int], ...] = (
    (0, 3, 0),
    (1, 3, 0),
    (2, 3, 0),
    (0, 4, 0),
    (1, 4, 0),
    (2, 4, 0),
    (3, 4, 0),
)


@dataclass(frozen=True)
class TopologyTask:
    query: Query
    target_action: int
    candidates: tuple[Candidate, ...]
    input_values: tuple[int, ...]
    target_edges: tuple[tuple[int, int, int], ...]
    family: str = "topology_match"

    def public(self) -> Query:
        return self.query


def edge_mask(
    edges: Sequence[tuple[int, int, int]],
    universe: Sequence[tuple[int, int, int]] = TOPOLOGY_EDGE_UNIVERSE,
) -> tuple[int, ...]:
    """Encode an executable edge set as a fixed-width binary mask."""
    edge_set = set(edges)
    return tuple(1 if edge in edge_set else 0 for edge in universe)


def _all_topologies() -> tuple[tuple[tuple[int, int, int], ...], ...]:
    left = TOPOLOGY_EDGE_UNIVERSE[:3]
    right = TOPOLOGY_EDGE_UNIVERSE[3:]
    return tuple((a, b) for a in left for b in right)


def build_topology_task(
    seed: int,
    *,
    n_candidates: int = 8,
    step: int = 0,
) -> TopologyTask:
    """Create one program-selection problem with explicit candidate topology.

    All candidates have the same descriptor. Only executable_edges differs.
    Candidate action indices are assigned before presentation and contain no
    relation to target selection.
    """
    if n_candidates < 2 or n_candidates > len(_all_topologies()):
        raise ValueError(f"n_candidates must be in [2, {len(_all_topologies())}]")

    topologies = list(_all_topologies())
    rng = random.Random(seed)
    target_edges = rng.choice(topologies)
    distractors = [edges for edges in topologies if edges != target_edges]
    rng.shuffle(distractors)
    chosen = [target_edges, *distractors[: n_candidates - 1]]
    rng.shuffle(chosen)

    gold_position = chosen.index(target_edges)
    candidates = [
        Candidate(
            key=f"topo_{seed:07d}_{i}",
            descriptor=(0,) * 8,
            action=i,
            provenance="topology_match",
            executable_edges=tuple(edges),
        )
        for i, edges in enumerate(chosen)
    ]

    # The query exposes the desired topology, not the candidate index.
    mask = edge_mask(target_edges)
    query = Query(
        text=" ".join(str(bit) for bit in mask) + "\t",
        context=(),
        step=step,
        provenance="topology_match",
    )
    return TopologyTask(
        query=query,
        target_action=gold_position,
        candidates=tuple(candidates),
        input_values=tuple(rng.randrange(2) for _ in range(3)),
        target_edges=tuple(target_edges),
    )


def program_for_candidate(
    candidate: Candidate,
    *,
    input_values: Sequence[int] = (),
) -> Program:
    """Materialise a candidate-owned executable program."""
    if not candidate.executable_edges:
        raise ValueError("candidate has no executable edge set")
    if any(edge not in TOPOLOGY_EDGE_UNIVERSE for edge in candidate.executable_edges):
        raise ValueError("candidate executable edge is outside the fixed substrate")

    candidate_edges = tuple(Edge(*edge) for edge in TOPOLOGY_EDGE_UNIVERSE)
    true_edges = tuple(Edge(*edge) for edge in candidate.executable_edges)
    nodes = (
        Node(index=0, op=INPUT, depth=0, arity=0),
        Node(index=1, op=INPUT, depth=0, arity=0),
        Node(index=2, op=INPUT, depth=0, arity=0),
        Node(index=3, op=NOT, depth=1, arity=1),
        Node(index=4, op=NOT, depth=2, arity=1),
    )
    return Program(
        nodes=nodes,
        candidate_edges=candidate_edges,
        true_edges=true_edges,
        inputs=(0, 1, 2),
        input_values=tuple(int(x) for x in input_values),
        output=4,
        active_count=5,
    )
