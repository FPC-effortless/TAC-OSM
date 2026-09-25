"""TAC-OSM v0 — the integration boundary.

Five protocols defining the persistent structural computation loop:

    S_t → R_t → C_t → A_t → O_t → V_t → S_{t+1}

Nothing crosses this boundary without an adapter, and every adapter carries
the exclusions recorded in ``provenance/COMPONENTS.md``.

The interfaces are deliberately narrower than their source implementations.
An ablation surface is only interpretable if each mechanism can be swapped
without rewriting the model, so these protocols are the integration boundary
in both directions: the source code is not imported directly, and the model
does not know which implementation it is running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

__all__ = [
    # loop values
    "Query",
    "Candidate",
    "Structure",
    "Computation",
    "Outcome",
    "StateRead",
    "StateWrite",
    "StateUpdate",
    "RoutingDecision",
    "ExecutionResult",
    "VerificationResult",
    "RepairResult",
    "Step",
    # the five interfaces
    "PersistentState",
    "RelevanceRouter",
    "StructuralExecutor",
    "Verifier",
    "RepairController",
    # the environment
    "Environment",
]


# --------------------------------------------------------------------------- #
# Loop values
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Query:
    """A request against persistent state.

    Carries only information the router is permitted to see. The anti-leakage
    boundary (``PNDS_MASTER_BENCHMARK.md`` §7) forbids target, answer,
    outcome, gold structure id, oracle mask, and correct action.
    """

    text: str
    context: tuple[int, ...] = ()
    step: int = 0
    provenance: str = "synthetic"


@dataclass(frozen=True)
class Candidate:
    """One selectable persistent structure.

    ``provenance`` is the field that lets an integration repo keep research
    dependencies without collapsing their origins: a candidate knows whether
    it came from a CASM DAG, a CDL memory, or an oracle.
    """

    key: str
    descriptor: tuple[int, ...]
    action: int = 0
    provenance: str = "synthetic"


@dataclass(frozen=True)
class Structure:
    """A computational structure selected for execution."""

    key: str
    spec: Any = None
    provenance: str = "synthetic"


@dataclass(frozen=True)
class Computation:
    """The executed computation, presented to the verifier."""

    structure: Structure
    action: int
    trace: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Outcome:
    """Observed environment result, exposed only *after* the action.

    The temporal ordering is the whole point: routing cannot see it, so no
    router can learn to leak it.
    """

    success: bool
    value: float = 0.0
    feedback: str = ""
    detail: Any = None


@dataclass(frozen=True)
class StateRead:
    """What persistence returned for a query."""

    keys: tuple[str, ...]
    values: tuple[tuple[int, ...], ...]
    slot_used: tuple[bool, ...]


@dataclass(frozen=True)
class StateWrite:
    """Result of a persistent write attempt."""

    committed: bool
    key: str
    reason: str = ""
    step: int = 0


@dataclass(frozen=True)
class StateUpdate:
    """A proposed change to persistent state.

    Under the verified-only commit rule this is a *proposal*, not a write.
    ``task_key`` exists so a repair can reuse a retained procedure, per
    ``ProceduralMemoryStore``.
    """

    key: str
    value: tuple[int, ...]
    task_key: str = ""
    success_score: float = 1.0
    step: int = 0


@dataclass(frozen=True)
class RoutingDecision:
    """One router decision over a candidate set.

    ``scores`` is exposed for diagnostics and teacher-agreement measurement.
    It is deliberately *not* the primary metric: Stage B distilled a router
    that agreed with its teacher more often (72.92% vs 67.08%) while scoring
    worse on the task (73.75% vs 77.92%). Teacher agreement is not capability.
    """

    selected: int
    scores: tuple[float, ...]
    provenance: str = "learned"


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a selected structure."""

    output: float
    gates: tuple[float, ...] = ()
    node_values: tuple[float, ...] = ()
    provenance: str = "executed"


@dataclass(frozen=True)
class VerificationResult:
    """Verifier verdict on a computation.

    Shape mirrors ``tac_transformer/repair_controller.py`` so the existing
    repair controller can be adapted with no semantic translation.
    """

    passed: bool
    feedback: str = ""


@dataclass(frozen=True)
class RepairResult:
    """A repair attempt."""

    output_text: str
    passed: bool = False
    feedback: str = ""
    patch: str | None = None


@dataclass(frozen=True)
class Step:
    """One complete traversal of the loop, ``S_t → ... → S_{t+1}``.

    Carrying the whole trajectory on one object is what makes the path-vs-final
    verification comparison possible without re-running anything.
    """

    step: int
    query: Query
    decision: RoutingDecision
    computation: Computation
    outcome: Outcome
    verification: VerificationResult
    write: StateWrite | None = None
    repair: RepairResult | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The five interfaces
# --------------------------------------------------------------------------- #


@runtime_checkable
class PersistentState(Protocol):
    """``S_t`` — persistent state.

    Adapted from the minimal ``tac_sie/types.py::IdentityState`` shape
    (``memory_keys`` / ``memory_values`` / ``slot_used``) rather than the
    ~30-field ``tac_transformer`` version: the minimal shape forces role
    separation and keeps the ablation surface interpretable.

    ``write`` is new. No source repository implements a persistent write;
    it is the research target, not an import. Under the verified-only commit
    rule it is only reached through the verifier.
    """

    def read(self, query: Query) -> StateRead:
        """Return candidate structures addressable by ``query``."""
        ...

    def write(self, update: StateUpdate) -> StateWrite:
        """Propose a persistent update. Returns whether it committed."""
        ...


@runtime_checkable
class RelevanceRouter(Protocol):
    """``R_t`` — relevance routing.

    Two implementations are expected to coexist. ``CDLTeacher`` supplies
    supervision at ``O(N · C_LM)``; a learned router supplies the runtime
    path. The teacher is never the runtime router — its cost is
    structurally incompatible with the thesis that executed computation
    scale with the relevant subset rather than total history.
    """

    def route(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
    ) -> RoutingDecision:
        """Select one candidate from ``candidates`` for ``query``."""
        ...


@runtime_checkable
class StructuralExecutor(Protocol):
    """``C_t`` → ``A_t`` — structural execution.

    Adapted from ``casm_v01/phase1_dag/model.py``. The executor contract is
    preserved, including its sharpest constraint: routing depends only on
    structural state, and **runtime values never enter the router**. That
    boundary is what makes execution an addressable resource.
    """

    def execute(self, structure: Structure, inputs: Sequence[float]) -> ExecutionResult:
        """Execute ``structure`` over ``inputs`` and return its output."""
        ...


@runtime_checkable
class Verifier(Protocol):
    """``V_t`` — verification of an executed computation.

    Path-aware verification is a first-class variant: ``Computation`` carries
    a trace so ``V_path(z_1..z_n, y)`` can be compared against
    ``V_final(y)``. No source repository has run that comparison; the master
    benchmark requires it (§19).
    """

    def verify(self, computation: Computation, outcome: Outcome) -> VerificationResult:
        """Judge whether ``computation`` is consistent with ``outcome``."""
        ...


@runtime_checkable
class RepairController(Protocol):
    """Repair driven by verification failure.

    ``tac_transformer/repair_controller.py::VerifierGuidedRepairController``
    is already architecture-neutral — ``run()`` takes injected ``verifier``
    and ``repair`` callables — so this protocol is its shape, and it needs no
    adapter. Its claim exclusions travel with it: the REAL017 lineage is
    do-not-cite until audited. Only the loop shape transfers.
    """

    def repair(
        self, computation: Computation, verification: VerificationResult
    ) -> RepairResult:
        """Produce a repaired computation given a failed verification."""
        ...


@runtime_checkable
class Environment(Protocol):
    """Source of ``O_t``.

    New to this repository: none of the three source labs supplies an
    environment that closes the loop. The environment is the one component
    that must be written here rather than adapted.

    The temporal contract is load-bearing: ``transition`` is the only place
    ``Outcome`` is produced, and it is called *after* routing, so no router
    can observe the outcome it is selecting for.
    """

    def transition(
        self, state: PersistentState, action: int, query: Query
    ) -> Outcome:
        """Apply ``action`` and reveal the outcome."""
        ...

    def success(self, query: Query, outcome: Outcome) -> bool:
        """Score the outcome. The router never sees this."""
        ...
