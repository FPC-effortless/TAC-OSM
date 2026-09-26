"""``S_t`` — persistent state, adapted from the minimal ``tac_sie`` shape.

Adapted from ``TAC-transformer/tac_sie/types.py::IdentityState`` (``face2d9``)
— the deliberately minimal ``memory_keys`` / ``memory_values`` / ``slot_used``
shape — rather than the ~30-field ``tac_transformer/model.py`` version, for
the reason recorded in ``provenance/COMPONENTS.md``: the minimal shape forces
role separation, and an ablation surface with 60 config knobs is
uninterpretable.

``write`` is new. No source repository implements a persistent write; it is
the research target, not an import. Under the verified-only commit rule it is
reached only through the verifier.

## The intervention vocabulary

``StateIntervention`` is adapted from the TAC carry / reset / shuffle / corrupt
probes — the measurement style the ledger records as "state is causally
load-bearing, not correlational". A persistence result is not a persistence
result without a dose-response ladder, and the ladder is implemented here so
that every arm sees the same interventions.

* ``persistent`` — the control. Information survives.
* ``shuffled`` — all values preserved, their association to keys destroyed.
  Diagnoses *addressing*: does the loop use the key, or the value's position?
* ``reset`` — state cleared at the read boundary. This is the arm that
  isolates persistence from everything else, because it keeps the router,
  executor, and verifier intact while destroying only memory.
* ``corrupted`` — each value bit flipped at rate ``corruption_rate``. The
  dose-response ladder from Stage 3b-r2 (0.8667 → 0.3500 → 0.1000 → 0.0000)
  is what a positive persistence result looks like, and it is produced by
  varying one number.
* ``random`` — replaced by fresh random vectors at every read. Preserves
  *capacity* (the store is full) while destroying *content*, separating
  "there is memory" from "the memory holds the right thing".
* ``wrong_key`` — every read returns a *different* key's value. The
  ``dd8f63c`` diagnostic: it isolates whether the loop is reading by key or
  by slot position. It is the same test that caught the representability
  failure, applied here to persistence.

Interventions are applied at **read** time, not write time. Writing is what
the loop is being tested on, so writes are always honest; the intervention
corrupts the *readback*, which is where a persistence claim is either
supported or broken.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import PersistentState, Query, StateRead, StateUpdate, StateWrite

__all__ = [
    "StateConfig",
    "Slot",
    "PersistentStore",
    "INTERVENTIONS",
]

INTERVENTIONS = (
    "persistent",
    "shuffled",
    "reset",
    "corrupted",
    "random",
    "wrong_key",
)


@dataclass
class StateConfig:
    """Switch surface for persistence, mirroring ``ablation.StateSwitch``.

    Kept separate from ``ablation.StateSwitch`` deliberately: this is the
    *implementation* config and that is the *experiment* config. The builder
    copies one onto the other, so an experiment cell determines the store but
    the store never reaches back into the experiment config.
    """

    enabled: bool = True
    write: bool = True
    intervention: str = "persistent"
    corruption_rate: float = 0.0
    n_slots: int = 256
    seed: int = 0

    def __post_init__(self) -> None:
        if self.intervention not in INTERVENTIONS:
            raise ValueError(
                f"unknown intervention {self.intervention!r}; "
                f"expected one of {INTERVENTIONS}"
            )
        if not 0.0 <= self.corruption_rate <= 1.0:
            raise ValueError(
                f"corruption_rate must be in [0, 1], got {self.corruption_rate}"
            )
        if self.n_slots < 1:
            raise ValueError(f"need at least one slot, got {self.n_slots}")


@dataclass
class Slot:
    """One persistent slot: the minimal ``IdentityState`` triple, unfrozen.

    Frozen dataclasses are used throughout the loop values because they cross
    the interface boundary; a slot is *internal* state and must be mutable, so
    it is a plain dataclass.
    """

    key: str = ""
    value: tuple[int, ...] = ()
    used: bool = False
    written_at: int = -1


class PersistentStore:
    """``S_t`` — a deliberately boring key-addressed store.

    This is intentionally not a learned memory. The first question the loop
    must answer is whether information written at step ``t`` survives a
    boundary and affects a decision at step ``t + k``; adding a learned reader
    at the same time would confound the answer. ``read`` returns *candidates*
    addressable by the query, so a router learns to select among them rather
    than to retrieve.
    """

    def __init__(self, config: StateConfig | None = None) -> None:
        self.config = config if config is not None else StateConfig()
        self._slots: list[Slot] = [Slot() for _ in range(self.config.n_slots)]
        self._index: dict[str, int] = {}
        self._rng = random.Random(self.config.seed)
        self._reads = 0
        self._writes = 0

    # -- addressing -------------------------------------------------------- #

    def _locate(self, key: str) -> int | None:
        if not self.config.enabled:
            return None
        idx = self._index.get(key)
        if idx is None:
            return None
        if not self._slots[idx].used:
            return None
        return idx

    def _candidate_pool(self) -> list[Slot]:
        """The used slots, as seen under the configured intervention."""
        if not self.config.enabled:
            return []

        used = [s for s in self._slots if s.used]
        if not used:
            return []

        mode = self.config.intervention
        if mode == "persistent":
            return list(used)

        if mode == "reset":
            # The boundary is crossed here: everything written before is gone.
            self._slots = [Slot() for _ in range(self.config.n_slots)]
            self._index.clear()
            return []

        if mode == "shuffled":
            pool = list(used)
            self._rng.shuffle(pool)
            return [
                Slot(key=s.key, value=s.value, used=True, written_at=s.written_at)
                for s in pool
            ]

        if mode == "random":
            return [
                Slot(
                    key=s.key,
                    value=tuple(self._rng.randrange(2) for _ in s.value),
                    used=True,
                    written_at=s.written_at,
                )
                for s in used
            ]

        if mode == "corrupted":
            return [_corrupt(self._rng, s, self.config.corruption_rate) for s in used]

        if mode == "wrong_key":
            # Return a *different* key's value. If the loop still succeeds it
            # is reading by slot position, not by key — the dd8f63c test
            # applied to persistence.
            pool = list(used)
            self._rng.shuffle(pool)
            rotated = pool[1:] + pool[:1]
            return [
                Slot(key=rotated[i].key, value=s.value, used=True, written_at=s.written_at)
                for i, s in enumerate(pool)
            ]

        raise ValueError(f"unreachable intervention: {mode!r}")

    # -- the protocol ------------------------------------------------------ #

    def read(self, query: Query) -> StateRead:
        """Return candidate structures addressable by ``query``.

        The slot the query addresses, if present, is placed first; the rest of
        the pool follows. Ordering is stable under the ``persistent``
        intervention and deliberately scrambled under ``shuffled`` and
        ``wrong_key``, which is what makes those arms diagnostic.
        """
        pool = self._candidate_pool()
        if not pool:
            return StateRead(keys=(), values=(), slot_used=())

        bits, address = _parse_query(query)

        selected: list[Slot] = []
        if address:
            idx = self._locate(address)
            if idx is not None:
                selected.append(self._slots[idx])
        for slot in pool:
            if slot.key == address:
                continue
            selected.append(slot)

        # Values are returned as plain bit rows. The interface declares
        # ``values: tuple[tuple[int, ...], ...]`` and nothing downstream may
        # reach back into a Slot, because ``Slot`` is an implementation
        # detail of *this* store: an ablation swaps the store, and a caller
        # that touched slot internals would break silently instead of loudly.
        keys = tuple(s.key for s in selected)
        values = tuple(s.value for s in selected)
        used = tuple(s.used for s in selected)
        self._reads += 1
        return StateRead(keys=keys, values=values, slot_used=used)

    def write(self, update: StateUpdate) -> StateWrite:
        """Propose a persistent update. Returns whether it committed.

        Refusals are reported rather than raised, so a disabled-write ablation
        produces a *measurable* degradation instead of a crash. A write is
        refused when writes are disabled, when the value is empty, or when the
        store is full — v0 defines no eviction policy, and silently evicting
        would confound a persistence measurement.
        """
        if not self.config.enabled or not self.config.write:
            return StateWrite(
                committed=False, key=update.key, reason="write_disabled", step=update.step
            )

        if not update.value:
            return StateWrite(
                committed=False, key=update.key, reason="empty_value", step=update.step
            )

        if update.key in self._index:
            idx = self._index[update.key]
            self._slots[idx].value = tuple(update.value)
            self._slots[idx].written_at = update.step
            self._writes += 1
            return StateWrite(
                committed=True, key=update.key, reason="overwrite", step=update.step
            )

        free = next((i for i, s in enumerate(self._slots) if not s.used), None)
        if free is None:
            return StateWrite(
                committed=False, key=update.key, reason="store_full", step=update.step
            )

        self._slots[free] = Slot(
            key=update.key,
            value=tuple(update.value),
            used=True,
            written_at=update.step,
        )
        self._index[update.key] = free
        self._writes += 1
        return StateWrite(committed=True, key=update.key, reason="allocated", step=update.step)

    # -- diagnostics ------------------------------------------------------- #

    def reset(self) -> None:
        """Clear all slots. Used by the loop to open an episode boundary."""
        self._slots = [Slot() for _ in range(self.config.n_slots)]
        self._index.clear()
        self._reads = 0
        self._writes = 0

    @property
    def occupancy(self) -> float:
        used = sum(1 for s in self._slots if s.used)
        return used / len(self._slots)

    @property
    def reads(self) -> int:
        return self._reads

    @property
    def writes(self) -> int:
        return self._writes

    def snapshot(self) -> tuple[Slot, ...]:
        """A copy of the store, for a leakage audit or a reproducibility check."""
        return tuple(
            Slot(key=s.key, value=s.value, used=s.used, written_at=s.written_at)
            for s in self._slots
        )


def _corrupt(rng: random.Random, slot: Slot, rate: float) -> Slot:
    """Flip each bit of ``slot.value`` with probability ``rate``.

    Applied independently per bit, so ``rate`` is a genuine dose: at 0.0 the
    value is untouched and at 1.0 every bit is complemented.
    """
    if rate <= 0.0:
        return Slot(key=slot.key, value=slot.value, used=True, written_at=slot.written_at)
    corrupted = tuple(1 - b if rng.random() < rate else b for b in slot.value)
    return Slot(key=slot.key, value=corrupted, used=True, written_at=slot.written_at)


def _parse_query(query: Query) -> tuple[tuple[int, ...], str]:
    """Split a query into (public bits, state address).

    Mirrors ``environment.parse_query``. Duplicated rather than imported so
    that ``state`` does not depend on ``environment``: the store must stay
    substitutable, and a back-edge to the world module would make the
    dependency point the wrong way.
    """
    bits_part, _, address = query.text.partition("\t")
    bits = tuple(int(b) for b in bits_part.split()) if bits_part.strip() else ()
    return bits, address
