"""Temporal persistence primitives for the hardened TAC-OSM path.

The legacy v0 task builder materialises a target into persistent state while
constructing the current task. That is a keyed lookup, but it does not prove
that a write at time t survives an intervening sequence of transitions.

This module makes the temporal boundary explicit:

    world write at t
        -> advance through k decision boundaries
        -> read at t + k

The state container stores vectors by address, but availability is governed by
an explicit commit step. A read before available_step fails closed.
Nothing here is a scientific result; it is the runtime contract needed before
a persistence claim can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from . import Query, StateRead, StateUpdate, StateWrite


@dataclass(frozen=True)
class TemporalWrite:
    """A world event that becomes readable only after a causal delay."""

    update: StateUpdate
    written_step: int
    available_step: int

    def __post_init__(self) -> None:
        if self.written_step < 0:
            raise ValueError("written_step must be >= 0")
        if self.available_step < self.written_step:
            raise ValueError("available_step must be >= written_step")


@dataclass
class TemporalPersistentState:
    """Addressed persistent state with an explicit temporal availability rule.

    Addressing, readout, and representation are separate:
      * stage_world_write creates a causal world event.
      * advance_to commits due events.
      * resolve_address identifies the one named memory item.
      * read returns only the resolved item, never the whole pool.
    """

    current_step: int = 0
    _pending: list[TemporalWrite] = field(default_factory=list)
    _values: dict[str, tuple[int, ...]] = field(default_factory=dict)
    _write_steps: dict[str, int] = field(default_factory=dict)

    def stage_world_write(self, update: StateUpdate, *, delay: int) -> TemporalWrite:
        if delay < 0:
            raise ValueError("delay must be >= 0")
        event = TemporalWrite(
            update=update,
            written_step=self.current_step,
            available_step=self.current_step + delay,
        )
        self._pending.append(event)
        return event

    def advance_to(self, step: int) -> None:
        """Advance the causal clock and commit writes whose delay has elapsed."""
        if step < self.current_step:
            raise ValueError(
                f"cannot move time backwards: {step} < {self.current_step}"
            )
        self.current_step = step
        still_pending: list[TemporalWrite] = []
        for event in self._pending:
            if event.available_step <= step:
                self._values[event.update.key] = tuple(event.update.value)
                self._write_steps[event.update.key] = event.written_step
            else:
                still_pending.append(event)
        self._pending = still_pending

    def resolve_address(self, query: Query) -> str:
        """Resolve the query's address without scanning unrelated memory."""
        _, _, address = query.text.partition("\t")
        return address

    def read(self, query: Query) -> StateRead:
        address = self.resolve_address(query)
        if not address or address not in self._values:
            return StateRead(keys=(), values=(), slot_used=())
        value = self._values[address]
        return StateRead(keys=(address,), values=(value,), slot_used=(True,))

    def write(self, update: StateUpdate) -> StateWrite:
        """Explicit programmatic write; takes effect at the current boundary."""
        if self.current_step < update.step:
            self.advance_to(update.step)
        self._values[update.key] = tuple(update.value)
        self._write_steps[update.key] = self.current_step
        return StateWrite(
            committed=True,
            key=update.key,
            reason="explicit_commit",
            step=self.current_step,
        )

    def available(self, key: str) -> bool:
        return key in self._values

    def write_step(self, key: str) -> int | None:
        return self._write_steps.get(key)

    def pending(self) -> tuple[TemporalWrite, ...]:
        return tuple(self._pending)


@dataclass(frozen=True)
class TemporalProbe:
    """A test/control specification for a write/read separation."""

    key: str
    write_step: int
    read_step: int
    delay: int

    def __post_init__(self) -> None:
        if self.write_step < 0 or self.read_step < 0:
            raise ValueError("steps must be >= 0")
        if self.read_step < self.write_step:
            raise ValueError("read_step must be >= write_step")
        if self.delay != self.read_step - self.write_step:
            raise ValueError("delay must equal read_step - write_step")


def build_temporal_probe(
    state: TemporalPersistentState,
    *,
    key: str,
    value: Iterable[int],
    write_step: int,
    delay: int,
) -> TemporalProbe:
    """Stage a write and return the exact boundary at which it becomes readable."""
    if write_step < state.current_step:
        raise ValueError("write_step must be >= current state step")
    state.advance_to(write_step)
    state.stage_world_write(
        StateUpdate(key=key, value=tuple(value), step=write_step),
        delay=delay,
    )
    return TemporalProbe(
        key=key,
        write_step=write_step,
        read_step=write_step + delay,
        delay=delay,
    )
