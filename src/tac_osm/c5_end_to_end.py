"""End-to-end C5 control combining persistent state addressing and selective execution.

State values are 10-bit codes stored behind persistent opaque addresses. Queries
have one bit of deterministic noise. The state index recovers the exact code,
then an exact candidate index returns four candidate programs sharing that code.
The correct output aggregates all four relevant programs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, Query, StateUpdate
from .addressing import ContentAddressIndex
from .hamming_state_index import HammingStateIndex
from .noisy_state_tasks import CODEBOOK, STATE_BITS
from .temporal import TemporalPersistentState

M = 64
QUERY_CODES = tuple(CODEBOOK[:16])
MATCH_CONTEXT = (1,) * STATE_BITS
WORK_UNITS_PER_CANDIDATE = 12


@dataclass(frozen=True)
class EndToEndTask:
    query: Query
    state_updates: tuple[StateUpdate, ...]
    candidates: tuple[Candidate, ...]
    target_address: str
    target_value: tuple[int, ...]
    expected_output: int
    relevant_indices: frozenset[int]
    step: int


class EndToEndExecutor:
    """Fixed candidate program whose work is counted per executed program."""

    def execute_candidate(self, candidate: Candidate) -> tuple[int, int]:
        if not candidate.executable_edges:
            raise ValueError("candidate program must expose executable_edges")
        contribution = 1 + sum(
            src + dst + port for src, dst, port in candidate.executable_edges
        )
        return contribution, WORK_UNITS_PER_CANDIDATE

    def execute_population(
        self, candidates: Sequence[Candidate]
    ) -> tuple[int, int, int]:
        output = 0
        work = 0
        invocations = 0
        for candidate in candidates:
            contribution, units = self.execute_candidate(candidate)
            output += contribution
            work += units
            invocations += 1
        return output, work, invocations


def _candidate_program(index: int) -> tuple[tuple[int, int, int], ...]:
    variant = index % 4
    return ((0, 3, 0), (variant, 4, 0))


def build_state(seed: int) -> tuple[StateUpdate, ...]:
    rng = random.Random(seed)
    addresses = [f"e2e-state-{seed}-{i:03d}" for i in range(M)]
    shuffled = list(CODEBOOK[:M])
    rng.shuffle(shuffled)
    return tuple(
        StateUpdate(
            key=addresses[i],
            value=tuple(shuffled[i]),
            step=0,
            success_score=1.0,
        )
        for i in range(M)
    )


def build_population(seed: int, h: int) -> tuple[Candidate, ...]:
    if h not in (64, 128, 256):
        raise ValueError("H must be 64, 128, or 256")
    rng = random.Random(seed * 1009 + h)
    candidates: list[Candidate] = []
    for code_id, code in enumerate(CODEBOOK[: h // 4]):
        del code_id
        for variant in range(4):
            index = len(candidates)
            candidates.append(
                Candidate(
                    key=f"e2e-c{seed}-{index}",
                    descriptor=tuple(code),
                    action=index,
                    provenance="c5_end_to_end_population",
                    executable_edges=_candidate_program(index + variant),
                )
            )
    rng.shuffle(candidates)
    return tuple(
        Candidate(
            key=c.key,
            descriptor=c.descriptor,
            action=i,
            provenance=c.provenance,
            executable_edges=c.executable_edges,
        )
        for i, c in enumerate(candidates)
    )


def _query_for_target(
    target: Sequence[int], step: int, flip: int
) -> Query:
    bits = list(target)
    bits[int(flip) % STATE_BITS] ^= 1
    return Query(
        text=" ".join(str(int(x)) for x in bits),
        context=MATCH_CONTEXT,
        step=step,
        provenance="c5_end_to_end_query",
    )


def build_task(seed: int, h: int, step: int) -> EndToEndTask:
    state_updates = build_state(seed)
    rng = random.Random(seed * 3001 + h * 97 + step * 7919 + 17)
    target_code = tuple(rng.choice(QUERY_CODES))
    flip = (seed + step * 3 + h) % STATE_BITS
    query = _query_for_target(target_code, step, flip)
    target_index = next(
        i for i, update in enumerate(state_updates)
        if update.value == target_code
    )
    target_address = state_updates[target_index].key

    candidates = build_population(seed, h)
    relevant = frozenset(
        i for i, candidate in enumerate(candidates)
        if tuple(candidate.descriptor) == target_code
    )
    if len(relevant) != 4:
        raise AssertionError(
            "registered end-to-end task must have exactly four relevant programs"
        )

    expected, work, calls = EndToEndExecutor().execute_population(
        tuple(candidates[i] for i in sorted(relevant))
    )
    assert calls == 4
    assert work == 4 * WORK_UNITS_PER_CANDIDATE

    return EndToEndTask(
        query=query,
        state_updates=state_updates,
        candidates=candidates,
        target_address=target_address,
        target_value=target_code,
        expected_output=expected,
        relevant_indices=relevant,
        step=step,
    )


def prepare_state(task: EndToEndTask) -> TemporalPersistentState:
    state = TemporalPersistentState()
    for update in task.state_updates:
        state.stage_world_write(update, delay=1)
    state.advance_to(1)
    return state


def full_state_scan(
    task: EndToEndTask,
    state: TemporalPersistentState,
) -> tuple[str, tuple[int, ...], int]:
    del state
    bits = tuple(int(x) for x in task.query.text.split())
    best_address: str | None = None
    best_value: tuple[int, ...] | None = None
    best_distance = STATE_BITS + 1
    operations = 0
    for update in task.state_updates:
        value = tuple(update.value)
        distance = sum(a != b for a, b in zip(value, bits))
        operations += STATE_BITS
        if distance < best_distance:
            best_distance = distance
            best_address = update.key
            best_value = value
    if best_address is None or best_value is None:
        raise RuntimeError("state full scan found no item")
    return best_address, best_value, operations


def build_state_index(
    state: TemporalPersistentState,
) -> HammingStateIndex:
    index = HammingStateIndex(bits=STATE_BITS, radius=2, shortlist_k=1)
    index.build(state)
    return index


def build_candidate_index(
    candidates: Sequence[Candidate],
) -> ContentAddressIndex:
    return ContentAddressIndex.build(
        candidates,
        context=MATCH_CONTEXT,
    )
