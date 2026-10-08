"""Multi-timescale persistent dynamics.

The mechanism is intentionally generic: it does not encode "iron", "ribosome",
or fly-specific anatomy. It implements the transferable computational idea:
fast and slow internal traces integrate experience and can prime future routing.

Authoritative world facts remain separate from dynamic experience memory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .interfaces import MemoryContext


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = True
    n_timescales: int = 4
    decays: tuple[float, ...] = (0.20, 0.60, 0.90, 0.985)
    dim: int = 8
    n_modules: int = 8
    local_enabled: bool = True
    global_enabled: bool = True
    priming_enabled: bool = True
    intervention: str = "persistent"
    corruption_rate: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_timescales < 1:
            raise ValueError("n_timescales must be >= 1")
        if len(self.decays) != self.n_timescales:
            raise ValueError("decays length must equal n_timescales")
        if any(not 0.0 < d < 1.0 for d in self.decays):
            raise ValueError("all decay factors must be in (0,1)")
        if self.dim < 1 or self.n_modules < 1:
            raise ValueError("dim and n_modules must be positive")
        if self.intervention not in {
            "persistent", "reset", "shuffled", "random", "corrupted", "wrong_key"
        }:
            raise ValueError(f"unknown memory intervention {self.intervention!r}")
        if not 0.0 <= self.corruption_rate <= 1.0:
            raise ValueError("corruption_rate must be in [0,1]")


class MultiScaleMemory:
    """Swappable dynamic memory with explicit time constants and local state."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config if config is not None else MemoryConfig()
        self._rng = random.Random(self.config.seed)
        self._global = [
            [0.0 for _ in range(self.config.dim)]
            for _ in range(self.config.n_timescales)
        ]
        self._local = {
            m: [
                [0.0 for _ in range(self.config.dim)]
                for _ in range(self.config.n_timescales)
            ]
            for m in range(self.config.n_modules)
        }
        self._priming = [0.0 for _ in range(self.config.dim)]
        self._volatility = 0.0
        self._surprise = 0.0
        self._steps = 0

    def context(self, module_ids: Sequence[int] = ()) -> MemoryContext:
        if not self.config.enabled:
            return MemoryContext((), (), (), 0.0, 0.0)

        selected = tuple(int(m) for m in module_ids if 0 <= int(m) < self.config.n_modules)
        local_items: list[tuple[str, tuple[tuple[float, ...], ...]]] = []
        if self.config.local_enabled:
            for module_id in selected:
                states = self._local[module_id]
                local_items.append(
                    (str(module_id), tuple(tuple(row) for row in states))
                )

        global_states = (
            tuple(tuple(row) for row in self._global)
            if self.config.global_enabled else ()
        )
        return MemoryContext(
            global_states=global_states,
            local_states=tuple(local_items),
            priming=tuple(self._priming) if self.config.priming_enabled else (),
            volatility=self._volatility,
            surprise=self._surprise,
        )

    def observe(
        self,
        *,
        module_ids: Sequence[int],
        signal: Sequence[float],
        reward: float,
        surprise: float,
        success: bool,
    ) -> None:
        if not self.config.enabled:
            return
        x = _fit(signal, self.config.dim)
        credit = _clip(float(reward), -1.0, 1.0)

        if self.config.global_enabled:
            for k, decay in enumerate(self.config.decays):
                row = self._global[k]
                gain = 1.0 - decay
                for j, value in enumerate(x):
                    row[j] = decay * row[j] + gain * value * (0.5 + 0.5 * credit)

        if self.config.local_enabled:
            for module_id in module_ids:
                if module_id not in self._local:
                    continue
                for k, decay in enumerate(self.config.decays):
                    row = self._local[module_id][k]
                    gain = 1.0 - decay
                    for j, value in enumerate(x):
                        row[j] = decay * row[j] + gain * value * (0.5 + 0.5 * credit)

        s = _clip(float(surprise), 0.0, 1.0)
        self._surprise = 0.85 * self._surprise + 0.15 * s
        self._volatility = 0.90 * self._volatility + 0.10 * abs(credit)

        if self.config.priming_enabled:
            # Priming changes slowly and decays even without new experience.
            for j, value in enumerate(x):
                self._priming[j] = (
                    0.97 * self._priming[j]
                    + 0.03 * value * (1.0 if not success else 0.25)
                )

        self._steps += 1

        # Interventions are read-time in the original TAC store. For dynamic
        # memory, the equivalent experiment is a deterministic state transform
        # after observation. It never changes the incoming task or outcome.
        if self.config.intervention == "reset":
            self.reset()
        elif self.config.intervention == "shuffled":
            self._shuffle_axes()
        elif self.config.intervention == "random":
            self._randomize_dynamic()
        elif self.config.intervention == "corrupted":
            self._corrupt_dynamic()
        elif self.config.intervention == "wrong_key":
            self._rotate_local_module_state()

    def reset(self) -> None:
        self._global = [
            [0.0 for _ in range(self.config.dim)]
            for _ in range(self.config.n_timescales)
        ]
        self._local = {
            m: [
                [0.0 for _ in range(self.config.dim)]
                for _ in range(self.config.n_timescales)
            ]
            for m in range(self.config.n_modules)
        }
        self._priming = [0.0 for _ in range(self.config.dim)]
        self._volatility = 0.0
        self._surprise = 0.0

    def _shuffle_axes(self) -> None:
        for rows in [self._global, *self._local.values()]:
            for row in rows:
                self._rng.shuffle(row)

    def _randomize_dynamic(self) -> None:
        for rows in [self._global, *self._local.values()]:
            for row in rows:
                for j in range(len(row)):
                    row[j] = self._rng.uniform(-1.0, 1.0)

    def _corrupt_dynamic(self) -> None:
        rate = self.config.corruption_rate
        for rows in [self._global, *self._local.values()]:
            for row in rows:
                for j in range(len(row)):
                    if self._rng.random() < rate:
                        row[j] = -row[j]

    def _rotate_local_module_state(self) -> None:
        if len(self._local) < 2:
            return
        ids = sorted(self._local)
        payload = [self._local[i] for i in ids]
        payload = payload[1:] + payload[:1]
        self._local = {i: payload[pos] for pos, i in enumerate(ids)}

    def inspect(self) -> dict[str, object]:
        return {
            "enabled": self.config.enabled,
            "n_timescales": self.config.n_timescales,
            "decays": self.config.decays,
            "steps": self._steps,
            "volatility": self._volatility,
            "surprise": self._surprise,
            "priming_norm": sum(v * v for v in self._priming) ** 0.5,
            "global_norms": [
                round(sum(v * v for v in row) ** 0.5, 6) for row in self._global
            ],
            "local_norms": {
                str(m): [round(sum(v * v for v in row) ** 0.5, 6) for row in rows]
                for m, rows in self._local.items()
            },
        }


def _fit(values: Sequence[float], dim: int) -> tuple[float, ...]:
    return tuple(float(values[i]) if i < len(values) else 0.0 for i in range(dim))


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
