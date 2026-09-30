"""Semantic program-selection tasks for TACOSM-REP-003.

The query describes an output dependency semantically (source input + path
depth), not the exact candidate edge mask. Candidates carry explicit
executable topology. The task therefore tests whether a learner can use
candidate structure compositionally rather than merely compare identical
edge masks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, Query
from .executor import Edge, INPUT, NOT, Node, Program
from .topology_tasks import TOPOLOGY_EDGE_UNIVERSE

SEMANTIC_SIGNATURE_WIDTH = 5


@dataclass(frozen=True)
class SemanticTask:
    query: Query
    target_action: int
    candidates: tuple[Candidate, ...]
    input_values: tuple[int, ...]
    target_edges: tuple[tuple[int, int, int], ...]
    family: str = "semantic_program_match"

    def public(self) -> Query:
        return self.query


def semantic_signature(
    edges: Sequence[tuple[int, int, int]],
) -> tuple[int, ...]:
    """Return [source_0, source_1, source_2, depth_1, depth_2] as one-hot."""
    edge_set = set(edges)
    into_node3 = [edge for edge in edge_set if edge[1] == 3]
    into_node4 = [edge for edge in edge_set if edge[1] == 4]
    if len(into_node3) != 1 or len(into_node4) != 1:
        raise ValueError("semantic program must have one edge into node 3 and node 4")

    src3 = into_node3[0][0]
    src4 = into_node4[0][0]
    if src3 not in (0, 1, 2):
        raise ValueError("node 3 must read one of the three inputs")

    if src4 in (0, 1, 2):
        source = src4
        depth = 1
    elif src4 == 3:
        source = src3
        depth = 2
    else:
        raise ValueError("node 4 has an unsupported source")

    out = [0, 0, 0, 0, 0]
    out[source] = 1
    out[3 + (depth - 1)] = 1
    return tuple(out)


def semantic_query(signature: Sequence[int], *, step: int = 0) -> Query:
    if tuple(signature) == tuple([0] * SEMANTIC_SIGNATURE_WIDTH):
        raise ValueError("semantic signature cannot be all zeros")
    if len(signature) != SEMANTIC_SIGNATURE_WIDTH:
        raise ValueError("semantic signature width mismatch")
    return Query(
        text=" ".join(str(int(x)) for x in signature) + "\t",
        context=(),
        step=step,
        provenance="semantic_program_match",
    )


def _all_topologies() -> tuple[tuple[tuple[int, int, int], ...], ...]:
    left = TOPOLOGY_EDGE_UNIVERSE[:3]
    right = TOPOLOGY_EDGE_UNIVERSE[3:]
    return tuple((a, b) for a in left for b in right)


def build_semantic_task(
    seed: int,
    *,
    n_candidates: int = 8,
    step: int = 0,
) -> SemanticTask:
    """Construct a semantic dependency-selection problem."""
    topologies = list(_all_topologies())
    if n_candidates < 2 or n_candidates > len(topologies):
        raise ValueError(f"n_candidates must be in [2, {len(topologies)}]")

    rng = random.Random(seed)
    target_edges = rng.choice(topologies)
    distractors = [edges for edges in topologies if edges != target_edges]
    rng.shuffle(distractors)
    chosen = [target_edges, *distractors[: n_candidates - 1]]
    rng.shuffle(chosen)
    target_action = chosen.index(target_edges)

    candidates = tuple(
        Candidate(
            key=f"sem_{seed:07d}_{i}",
            descriptor=(0,) * 8,
            action=i,
            provenance="semantic_program_match",
            executable_edges=tuple(edges),
        )
        for i, edges in enumerate(chosen)
    )
    signature = semantic_signature(target_edges)
    return SemanticTask(
        query=semantic_query(signature, step=step),
        target_action=target_action,
        candidates=candidates,
        input_values=tuple(rng.randrange(2) for _ in range(3)),
        target_edges=tuple(target_edges),
    )


def program_for_candidate(
    candidate: Candidate,
    *,
    input_values: Sequence[int] = (),
) -> Program:
    if not candidate.executable_edges:
        raise ValueError("candidate has no executable edge set")
    if any(edge not in TOPOLOGY_EDGE_UNIVERSE for edge in candidate.executable_edges):
        raise ValueError("candidate edge is outside the fixed substrate")
    return Program(
        nodes=(
            Node(index=0, op=INPUT, depth=0, arity=0),
            Node(index=1, op=INPUT, depth=0, arity=0),
            Node(index=2, op=INPUT, depth=0, arity=0),
            Node(index=3, op=NOT, depth=1, arity=1),
            Node(index=4, op=NOT, depth=2, arity=1),
        ),
        candidate_edges=tuple(Edge(*edge) for edge in TOPOLOGY_EDGE_UNIVERSE),
        true_edges=tuple(Edge(*edge) for edge in candidate.executable_edges),
        inputs=(0, 1, 2),
        input_values=tuple(int(x) for x in input_values),
        output=4,
        active_count=5,
    )
