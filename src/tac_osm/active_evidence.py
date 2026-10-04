"""First-class active-evidence interfaces for the PLM runtime.

This module adds a probe stage without changing the existing terminal loop.
A probe is an information-producing action; its evidence is typed, costed and
provenance-addressed so it can become persistent computational experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from . import Candidate, PersistentState, Query


@dataclass(frozen=True)
class ProbeAction:
    """A request to acquire information before terminal computation."""

    kind: str
    parameters: tuple[Any, ...] = ()
    expected_schema: str = ""
    budget_units: float = 0.0
    provenance: str = "declared"

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("probe action kind must be non-empty")
        if self.budget_units < 0.0:
            raise ValueError("probe action budget_units must be non-negative")


@dataclass(frozen=True)
class EvidencePacket:
    """Structured evidence produced by executing one probe action."""

    action: ProbeAction
    schema: str
    payload: Any
    cost_units: float
    confidence: float = 1.0
    provenance: Mapping[str, Any] = field(default_factory=dict)
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.schema:
            raise ValueError("evidence schema must be non-empty")
        if self.cost_units <= 0.0:
            raise ValueError("evidence cost_units must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("evidence confidence must be in [0, 1]")


@runtime_checkable
class ProbePolicy(Protocol):
    """CDL-level policy for selecting information-seeking actions."""

    def choose(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        evidence: Sequence[EvidencePacket],
    ) -> ProbeAction | None:
        """Return the next probe, or None when no probe is requested."""
        ...


@runtime_checkable
class ProbeEnvironment(Protocol):
    """Environment extension that can execute non-terminal information probes."""

    def probe(
        self,
        state: PersistentState,
        action: ProbeAction,
    ) -> EvidencePacket:
        """Execute the action and return structured evidence."""
        ...


@runtime_checkable
class EvidenceSufficiency(Protocol):
    """VRS-level pre-terminal stopping rule."""

    def sufficient(
        self,
        query: Query,
        state: PersistentState,
        evidence: Sequence[EvidencePacket],
    ) -> bool:
        """Return whether current evidence is sufficient for terminal computation."""
        ...


@dataclass(frozen=True)
class ActiveEvidenceConfig:
    """Traversal controls for the optional pre-routing probe stage."""

    enabled: bool = False
    max_probes: int = 0

    def __post_init__(self) -> None:
        if self.max_probes < 0:
            raise ValueError("max_probes must be non-negative")
        if not self.enabled and self.max_probes != 0:
            raise ValueError(
                "disabled active evidence must use max_probes=0"
            )


@runtime_checkable
class EvidenceCompiler(Protocol):
    """Compile public evidence into a router-visible query representation.

    Compilation is deliberately separate from probing. The compiler must not
    introduce target identity, evaluator labels, or future target evidence.
    """

    def compile(
        self,
        query: Query,
        state: PersistentState,
        evidence: Sequence[EvidencePacket],
    ) -> Query:
        """Return a public query representation after evidence acquisition."""
        ...


class IdentityEvidenceCompiler:
    """Control compiler: leave the public query unchanged."""

    def compile(
        self,
        query: Query,
        state: PersistentState,
        evidence: Sequence[EvidencePacket],
    ) -> Query:
        return query


class NoProbePolicy:
    """Backward-compatible inactive probe policy."""

    def choose(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        evidence: Sequence[EvidencePacket],
    ) -> ProbeAction | None:
        return None


class NeverSufficient:
    """Explicit control that never ends probing on its own."""

    def sufficient(
        self,
        query: Query,
        state: PersistentState,
        evidence: Sequence[EvidencePacket],
    ) -> bool:
        return False


class AlwaysSufficient:
    """Explicit control that preserves the historical immediate-routing path."""

    def sufficient(
        self,
        query: Query,
        state: PersistentState,
        evidence: Sequence[EvidencePacket],
    ) -> bool:
        return True


__all__ = [
    "ProbeAction",
    "ActiveEvidenceConfig",
    "EvidencePacket",
    "ProbePolicy",
    "ProbeEnvironment",
    "EvidenceSufficiency",
    "EvidenceCompiler",
    "IdentityEvidenceCompiler",
    "NoProbePolicy",
    "NeverSufficient",
    "AlwaysSufficient",
]
