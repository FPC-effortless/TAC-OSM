"""Persistent semantic routing tasks for TACOSM-REP-005.

The semantic requirement is stored in temporal persistent state at a public
address. The decision query contains only that address. Candidate programs
carry explicit executable topology.

The experiment therefore tests:
    world write -> causal delay -> addressed state read -> program routing
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, Query, StateUpdate
from .executor import Edge, INPUT, NOT, Node, Program
from .semantic_topology import (
    SEMANTIC_SIGNATURE_WIDTH,
    _all_topologies,
    semantic_signature,
)
from .topology_tasks import TOPOLOGY_EDGE_UNIVERSE


DEFAULT_ADDRESS = "goal:semantic"


@dataclass(frozen=True)
class PersistentSemanticTask:
    query: Query
    target_action: int
    candidates: tuple[Candidate, ...]
    target_signature: tuple[int, ...]
    target_edges: tuple[tuple[int, int, int], ...]
    input_values: tuple[int, ...]
    write: StateUpdate
    write_step: int
    read_step: int
    delay: int
    family: str = "persistent_semantic_program_match"


def build_persistent_semantic_task(
    seed: int,
    *,
    write_step: int,
    delay: int = 1,
    n_candidates: int = 8,
    address: str = DEFAULT_ADDRESS,
) -> PersistentSemanticTask:
    if delay < 1:
        raise ValueError("delay must be >= 1")
    if n_candidates < 2 or n_candidates > 8:
        raise ValueError("n_candidates must be in [2, 8]")
    if write_step < 0:
        raise ValueError("write_step must be >= 0")

    rng = random.Random(seed)
    topologies = list(_all_topologies())
    target_edges = rng.choice(topologies)
    target_signature = semantic_signature(target_edges)
    distractors = [
        edges
        for edges in topologies
        if edges != target_edges and semantic_signature(edges) != target_signature
    ]
    rng.shuffle(distractors)
    chosen = [target_edges, *distractors[: n_candidates - 1]]
    rng.shuffle(chosen)
    target_action = chosen.index(target_edges)

    candidates = tuple(
        Candidate(
            key=f"psem_{seed:07d}_{i}",
            descriptor=(0,) * 8,
            action=i,
            provenance="persistent_semantic_program_match",
            executable_edges=tuple(edges),
        )
        for i, edges in enumerate(chosen)
    )

    read_step = write_step + delay
    query = Query(
        text="\t" + address,
        context=(),
        step=read_step,
        provenance="persistent_semantic_program_match",
    )
    write = StateUpdate(
        key=address,
        value=target_signature,
        step=write_step,
        success_score=1.0,
    )
    return PersistentSemanticTask(
        query=query,
        target_action=target_action,
        candidates=candidates,
        target_signature=target_signature,
        target_edges=tuple(target_edges),
        input_values=tuple(rng.randrange(2) for _ in range(3)),
        write=write,
        write_step=write_step,
        read_step=read_step,
        delay=delay,
    )


def program_for_candidate(
    candidate: Candidate,
    *,
    input_values: Sequence[int] = (),
) -> Program:
    if not candidate.executable_edges:
        raise ValueError("candidate has no explicit executable topology")
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
