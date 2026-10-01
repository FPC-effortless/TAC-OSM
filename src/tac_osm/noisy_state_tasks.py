"""Noisy persistent-state retrieval task for REP-009."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations

from . import Query, StateUpdate
from .temporal import TemporalPersistentState

STATE_BITS = 10
MIN_HAMMING_DISTANCE = 3
MAX_POOL = 64


def codebook(bits: int = STATE_BITS, minimum_distance: int = MIN_HAMMING_DISTANCE) -> tuple[tuple[int, ...], ...]:
    codes: list[tuple[int, ...]] = []
    for value in range(1 << bits):
        bits_value = tuple((value >> (bits - 1 - i)) & 1 for i in range(bits))
        if all(sum(a != b for a, b in zip(bits_value, other)) >= minimum_distance for other in codes):
            codes.append(bits_value)
        if len(codes) >= MAX_POOL:
            break
    if len(codes) < MAX_POOL:
        raise RuntimeError("fixed codebook is smaller than MAX_POOL")
    return tuple(codes)


CODEBOOK = codebook()


@dataclass(frozen=True)
class NoisyStateQuery:
    query: Query
    target_address: str
    target_value: tuple[int, ...]


@dataclass(frozen=True)
class NoisyStatePool:
    updates: tuple[StateUpdate, ...]
    write_step: int
    read_step: int
    delay: int = 1

    def stage(self, state: TemporalPersistentState) -> None:
        if state.current_step > self.write_step:
            raise RuntimeError("state clock is past pool write step")
        if state.current_step < self.write_step:
            state.advance_to(self.write_step)
        for update in self.updates:
            state.stage_world_write(update, delay=self.delay)
        state.advance_to(self.read_step)


def noisy_query(value: tuple[int, ...], *, address: str = "", step: int = 1, flip: int = 0) -> Query:
    if not 0 <= flip < len(value):
        raise IndexError(flip)
    bits = list(value)
    bits[flip] ^= 1
    return Query(
        text=" ".join(str(int(x)) for x in bits),
        context=(),
        step=step,
        provenance="noisy_state_retrieval",
    )


def build_pool(seed: int, *, n_states: int, write_step: int = 0, delay: int = 1) -> NoisyStatePool:
    if n_states < 2 or n_states > MAX_POOL:
        raise ValueError(f"n_states must be in [2, {MAX_POOL}]")
    rng = random.Random(seed)
    selected = rng.sample(CODEBOOK, n_states)
    addresses: list[str] = []
    used: set[str] = set()
    for _ in range(n_states):
        while True:
            address = f"opaque-{rng.getrandbits(56):014x}"
            if address not in used:
                used.add(address)
                addresses.append(address)
                break
    updates = tuple(
        StateUpdate(key=addresses[i], value=tuple(selected[i]), step=write_step, success_score=1.0)
        for i in range(n_states)
    )
    return NoisyStatePool(
        updates=updates,
        write_step=write_step,
        read_step=write_step + delay,
        delay=delay,
    )


def build_query_for_target(pool: NoisyStatePool, target_index: int, *, step: int | None = None, flip: int | None = None) -> NoisyStateQuery:
    if not 0 <= target_index < len(pool.updates):
        raise IndexError(target_index)
    update = pool.updates[target_index]
    target = tuple(update.value)
    actual_step = pool.read_step if step is None else step
    flip_index = ((target_index * 3) + pool.write_step * 5) % len(target) if flip is None else int(flip) % len(target)
    query = noisy_query(target, step=actual_step, flip=flip_index)
    return NoisyStateQuery(query=query, target_address=update.key, target_value=target)


def prepare_state(pool: NoisyStatePool) -> TemporalPersistentState:
    state = TemporalPersistentState()
    pool.stage(state)
    return state