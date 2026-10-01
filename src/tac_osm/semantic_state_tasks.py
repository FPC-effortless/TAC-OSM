"""REP-006 synthetic task: semantic lookup over opaque persistent states."""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import Candidate, Query, StateUpdate
from .semantic_topology import semantic_signature, _all_topologies
from .temporal import TemporalPersistentState
from .topology_tasks import TOPOLOGY_EDGE_UNIVERSE


STATE_ADDRESS_COUNT = 8
STATE_ADDRESS_PREFIX = "mem-"
SIGNATURES = tuple(
    sorted(
        {
            semantic_signature(edges)
            for edges in _all_topologies()
        }
    )
)


@dataclass(frozen=True)
class SemanticStateTask:
    query: Query
    target_signature: tuple[int, ...]
    target_address: str
    state_updates: tuple[StateUpdate, ...]
    candidates: tuple[Candidate, ...]
    target_action: int
    write_step: int
    read_step: int
    delay: int = 1

    def stage(self, state: TemporalPersistentState) -> None:
        if state.current_step != self.write_step:
            raise RuntimeError(
                f"state clock {state.current_step} != write step {self.write_step}"
            )
        for update in self.state_updates:
            state.stage_world_write(update, delay=self.delay)
        state.advance_to(self.read_step)


def _random_opaque_address(rng: random.Random, used: set[str]) -> str:
    while True:
        value = f"{STATE_ADDRESS_PREFIX}{rng.getrandbits(48):012x}"
        if value not in used:
            used.add(value)
            return value


def _candidate_pool(
    rng: random.Random,
    target_signature: tuple[int, ...],
    n_candidates: int,
) -> tuple[tuple[Candidate, ...], int]:
    topologies = list(_all_topologies())
    target_edges = next(
        edges for edges in topologies if semantic_signature(edges) == target_signature
    )
    distractors = [
        edges
        for edges in topologies
        if semantic_signature(edges) != target_signature
    ]
    rng.shuffle(distractors)
    chosen = [target_edges, *distractors[: n_candidates - 1]]
    rng.shuffle(chosen)
    target_action = chosen.index(target_edges)
    return (
        tuple(
            Candidate(
                key=f"stateprog_{i}",
                descriptor=(0,) * 8,
                action=i,
                provenance="persistent_semantic_state",
                executable_edges=tuple(edges),
            )
            for i, edges in enumerate(chosen)
        ),
        target_action,
    )


def build_semantic_state_task(
    seed: int,
    *,
    write_step: int,
    delay: int = 1,
    n_states: int = STATE_ADDRESS_COUNT,
    n_candidates: int = 8,
) -> SemanticStateTask:
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    if n_states < 2 or n_states > 32:
        raise ValueError("n_states must be in [2, 32]")
    if n_candidates < 2 or n_candidates > 8:
        raise ValueError("n_candidates must be in [2, 8]")

    rng = random.Random(seed)
    target_signature = rng.choice(SIGNATURES)
    addresses: list[str] = []
    used: set[str] = set()
    for _ in range(n_states):
        addresses.append(_random_opaque_address(rng, used))

    target_index = rng.randrange(n_states)
    target_address = addresses[target_index]

    distractor_signatures = [s for s in SIGNATURES if s != target_signature]
    state_values = [None] * n_states
    state_values[target_index] = target_signature
    for i in range(n_states):
        if i == target_index:
            continue
        state_values[i] = rng.choice(distractor_signatures)

    updates = tuple(
        StateUpdate(
            key=addresses[i],
            value=tuple(state_values[i]),
            step=write_step,
            success_score=1.0,
        )
        for i in range(n_states)
    )
    candidates, target_action = _candidate_pool(
        rng, target_signature, n_candidates
    )
    read_step = write_step + delay
    query = Query(
        text=" ".join(str(int(x)) for x in target_signature),
        context=(),
        step=read_step,
        provenance="persistent_semantic_addressing",
    )
    return SemanticStateTask(
        query=query,
        target_signature=target_signature,
        target_address=target_address,
        state_updates=updates,
        candidates=candidates,
        target_action=target_action,
        write_step=write_step,
        read_step=read_step,
        delay=delay,
    )
