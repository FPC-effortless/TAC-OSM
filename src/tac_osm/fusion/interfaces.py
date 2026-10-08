"""Stable contracts and trace records for the fused TAC-OSM model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .. import Candidate, Outcome, Query, StateRead


@dataclass(frozen=True)
class MemoryContext:
    """Read-only dynamic memory exposed to routing and local computation."""

    global_states: tuple[tuple[float, ...], ...]
    local_states: tuple[tuple[str, tuple[tuple[float, ...], ...]], ...]
    priming: tuple[float, ...]
    volatility: float
    surprise: float


@dataclass(frozen=True)
class RegimeContext:
    """Learned history summary. No task target is permitted here."""

    embedding: tuple[float, ...]
    predicted_success: float
    volatility: float
    novelty: float
    confidence: float


@dataclass(frozen=True)
class CoordinatorDecision:
    """Which local circuits participate in the current computation."""

    selected_modules: tuple[int, ...]
    candidate_module_ids: tuple[int, ...]
    scores: tuple[float, ...]
    routing_work: int
    provenance: str


@dataclass(frozen=True)
class CandidateScore:
    """A module's score for one candidate."""

    module_id: int
    candidate_index: int
    score: float
    features: tuple[float, ...]


@dataclass(frozen=True)
class ModuleExecution:
    """One selected module's execution trace."""

    module_id: int
    candidate_scores: tuple[float, ...]
    selected_candidate_score: float
    trace: tuple[float, ...]
    halting_steps: int
    provenance: str


@dataclass(frozen=True)
class FusionStep:
    """Complete inspectable fused step."""

    step: int
    family: str
    query: Query
    state_read: StateRead
    memory: MemoryContext
    regime: RegimeContext
    coordination: CoordinatorDecision
    module_executions: tuple[ModuleExecution, ...]
    selected_candidate: int
    outcome: Outcome
    verification_passed: bool
    verification_feedback: str
    repair: str | None
    wrote_experience: bool
    provenance: dict[str, Any] = field(default_factory=dict)


class DynamicMemory(Protocol):
    def context(self, module_ids: Sequence[int] = ()) -> MemoryContext:
        ...

    def observe(
        self,
        *,
        module_ids: Sequence[int],
        signal: Sequence[float],
        reward: float,
        surprise: float,
        success: bool,
    ) -> None:
        ...

    def inspect(self) -> dict[str, Any]:
        ...


class RegimeEstimator(Protocol):
    def context(self, features: Sequence[float]) -> RegimeContext:
        ...

    def update(
        self,
        *,
        features: Sequence[float],
        outcome: Outcome,
        reward: float,
    ) -> None:
        ...

    def inspect(self) -> dict[str, Any]:
        ...


class Coordinator(Protocol):
    def route(
        self,
        *,
        query: Query,
        state_read: StateRead,
        memory: MemoryContext,
        regime: RegimeContext,
    ) -> CoordinatorDecision:
        ...

    def update(self, *, decision: CoordinatorDecision, reward: float) -> None:
        ...

    def inspect(self) -> dict[str, Any]:
        ...


class SpecialistModule(Protocol):
    module_id: int

    def score_candidates(
        self,
        *,
        query: Query,
        state_read: StateRead,
        memory: MemoryContext,
        regime: RegimeContext,
        candidates: Sequence[Candidate],
        adaptive: bool,
    ) -> ModuleExecution:
        ...

    def update(
        self,
        *,
        execution: ModuleExecution,
        candidates: Sequence[Candidate],
        selected: int,
        reward: float,
    ) -> None:
        ...

    def inspect(self) -> dict[str, Any]:
        ...
