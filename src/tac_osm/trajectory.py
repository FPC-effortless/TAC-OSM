"""First-class trajectory records for the hardened loop.

A trajectory is the inspectable artifact shared by debugging, evaluation,
regression, provenance and future learning. Every boundary crossing is
recorded without exposing target fields to the router-facing record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrajectoryStep:
    episode_id: str
    step: int
    state_before: Mapping[str, Any]
    query: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    selected: int | None
    computation: Mapping[str, Any]
    action: int | None
    observation: Mapping[str, Any]
    verification: Mapping[str, Any]
    repair: Mapping[str, Any] | None
    state_after: Mapping[str, Any]
    learning: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    episode_id: str
    steps: list[TrajectoryStep] = field(default_factory=list)

    def append(self, step: TrajectoryStep) -> None:
        if self.steps and step.step != self.steps[-1].step + 1:
            raise ValueError(
                f"trajectory step must be contiguous: got {step.step} "
                f"after {self.steps[-1].step}"
            )
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "steps": [step.to_dict() for step in self.steps],
        }

    def __len__(self) -> int:
        return len(self.steps)
