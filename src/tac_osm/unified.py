"""Unified persistent structural world model (UPSWM) reference core.

The core is dependency-free. Numerical encoders/executors are injected so
scientific backends can be changed without changing the causal interfaces.
Historical TAC-OSM components remain separate evidence; this module defines the
new integrated architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class Budget:
    proposal_k: int = 8
    route_k: int = 1
    planning_depth: int = 2
    max_execution_work: int = 4096
    max_memory_items: int = 4096
    max_repair_attempts: int = 3

    def __post_init__(self):
        if min(
            self.proposal_k,
            self.route_k,
            self.planning_depth,
            self.max_execution_work,
            self.max_memory_items,
            self.max_repair_attempts,
        ) < 1:
            raise ValueError("all budgets must be positive")


@dataclass(frozen=True)
class Observation:
    modality: str
    payload: tuple[float, ...]
    stream_id: str
    step: int
    provenance: str = ""


@dataclass(frozen=True)
class PredictiveState:
    latent: tuple[float, ...]
    sufficient_for: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class Structure:
    key: str
    vector: tuple[float, ...]
    operator: str
    support: int = 0
    provenance: str = ""


@dataclass(frozen=True)
class ProposalSet:
    candidates: tuple[Structure, ...]
    recall_upper_bound: float
    work_units: int


@dataclass(frozen=True)
class Plan:
    candidates: tuple[Structure, ...]
    predicted_next: tuple[tuple[float, ...], ...]
    selected: int
    depth: int
    objective: float = 0.0


@dataclass(frozen=True)
class Execution:
    action_key: str
    output: tuple[float, ...]
    trace: tuple[tuple[float, ...], ...]
    work_units: int


@dataclass(frozen=True)
class Verification:
    valid: bool
    confidence: float
    failed_constraint: str = ""
    counterexample: str = ""
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryWrite:
    operation: str
    key: str
    value: tuple[float, ...] = ()
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class Experience:
    state: PredictiveState
    action_key: str
    next_latent: tuple[float, ...]
    observation: tuple[float, ...]
    verification: Verification
    step: int
    modality: str


@dataclass(frozen=True)
class IdentifiabilityReport:
    identifiable: bool
    collisions: int
    tested_pairs: int
    reason: str = ""


@dataclass(frozen=True)
class RepresentationValidationReport:
    representable: bool
    identifiable: bool
    noncollapsed: bool
    predictive: bool
    action_sufficient: bool
    details: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(
            (
                self.representable,
                self.identifiable,
                self.noncollapsed,
                self.predictive,
                self.action_sufficient,
            )
        )


class StructureDiscoverer(Protocol):
    def discover(
        self,
        observation: Observation,
        knowledge: "PersistentKnowledge",
    ) -> Sequence[Structure]:
        ...


class Addressor(Protocol):
    def address(
        self,
        state: PredictiveState,
        knowledge: "PersistentKnowledge",
        budget: Budget,
    ) -> ProposalSet:
        ...


class Predictor(Protocol):
    def predict(
        self,
        state: PredictiveState,
        structure: Structure,
        depth: int = 1,
    ) -> Sequence[float]:
        ...


class Planner(Protocol):
    def plan(
        self,
        state: PredictiveState,
        proposals: ProposalSet,
        predictor: Predictor,
        budget: Budget,
    ) -> Plan:
        ...


class Executor(Protocol):
    def execute(self, plan: Plan, state: PredictiveState) -> Execution:
        ...


class Verifier(Protocol):
    def verify(
        self,
        state: PredictiveState,
        execution: Execution,
        observation: Observation,
    ) -> Verification:
        ...


class RepairPolicy(Protocol):
    def repair(
        self,
        state: PredictiveState,
        plan: Plan,
        verification: Verification,
        attempts_used: int,
        budget: Budget,
    ) -> Plan | None:
        ...


class WritePolicy(Protocol):
    def decide(
        self,
        state: PredictiveState,
        execution: Execution,
        verification: Verification,
        budget: Budget,
    ) -> Sequence[MemoryWrite]:
        ...


class RetractionPolicy(Protocol):
    def decide(
        self,
        key: str,
        contradictions: int,
        confidence: float,
    ) -> MemoryWrite | None:
        ...


class RepresentationValidator(Protocol):
    def validate(
        self,
        observations: Sequence[Observation],
    ) -> RepresentationValidationReport:
        ...


@dataclass
class PersistentKnowledge:
    experiences: list[Experience] = field(default_factory=list)
    structures: dict[str, Structure] = field(default_factory=dict)
    invalidated: set[str] = field(default_factory=set)
    contradictions: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> "PersistentKnowledge":
        return PersistentKnowledge(
            list(self.experiences),
            dict(self.structures),
            set(self.invalidated),
            dict(self.contradictions),
        )


@dataclass
class KnowledgeStore:
    knowledge: PersistentKnowledge = field(default_factory=PersistentKnowledge)

    def commit(
        self,
        writes: Sequence[MemoryWrite],
        experience: Experience | None = None,
    ) -> None:
        if (
            experience is not None
            and experience.verification.valid
            and experience.verification.confidence > 0
        ):
            self.knowledge.experiences.append(experience)

        for write in writes:
            if write.confidence <= 0:
                continue
            if write.operation in {"add", "update", "consolidate"}:
                self.knowledge.structures[write.key] = Structure(
                    write.key,
                    write.value,
                    "persisted",
                    provenance="verified",
                )
                self.knowledge.invalidated.discard(write.key)
            elif write.operation in {"invalidate", "delete"}:
                self.knowledge.structures.pop(write.key, None)
                self.knowledge.invalidated.add(write.key)
                self.knowledge.contradictions[write.key] = (
                    self.knowledge.contradictions.get(write.key, 0) + 1
                )

    def record_contradiction(self, key: str) -> int:
        count = self.knowledge.contradictions.get(key, 0) + 1
        self.knowledge.contradictions[key] = count
        return count


@dataclass
class UnifiedPersistentStructuralWorldModel:
    discoverer: StructureDiscoverer
    addressor: Addressor
    predictor: Predictor
    planner: Planner
    executor: Executor
    verifier: Verifier
    writer: WritePolicy
    repairer: RepairPolicy | None = None
    representation_validator: RepresentationValidator | None = None
    store: KnowledgeStore = field(default_factory=KnowledgeStore)
    budget: Budget = field(default_factory=Budget)
    identifiability_gate: Callable[
        [Sequence[Any]], IdentifiabilityReport
    ] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def step(
        self,
        observation: Observation,
        state: PredictiveState,
    ) -> tuple[PredictiveState, Execution, Verification]:
        if self.identifiability_gate is not None:
            report = self.identifiability_gate((observation,))
            if not report.identifiable:
                raise RuntimeError(
                    f"identifiability gate failed: {report.reason}"
                )

        if self.representation_validator is not None:
            report = self.representation_validator.validate((observation,))
            if not report.passed:
                raise RuntimeError(
                    "representation validation failed: "
                    + "; ".join(report.details)
                )

        knowledge = self.store.knowledge.snapshot()
        self.discoverer.discover(observation, knowledge)

        proposals = self.addressor.address(
            state,
            knowledge,
            self.budget,
        )
        if not proposals.candidates:
            raise RuntimeError(
                "proposal miss: no exhaustive hidden-truth fallback permitted"
            )

        plan = self.planner.plan(
            state,
            proposals,
            self.predictor,
            self.budget,
        )

        attempts = 0
        execution = self.executor.execute(plan, state)
        verification = self.verifier.verify(
            state,
            execution,
            observation,
        )

        while (
            not verification.valid
            and self.repairer is not None
            and attempts < self.budget.max_repair_attempts
        ):
            attempts += 1
            repaired = self.repairer.repair(
                state,
                plan,
                verification,
                attempts,
                self.budget,
            )
            if repaired is None:
                break
            plan = repaired
            execution = self.executor.execute(plan, state)
            verification = self.verifier.verify(
                state,
                execution,
                observation,
            )

        writes = tuple(
            self.writer.decide(
                state,
                execution,
                verification,
                self.budget,
            )
        )

        experience = Experience(
            state=state,
            action_key=execution.action_key,
            next_latent=execution.output,
            observation=observation.payload,
            verification=verification,
            step=observation.step,
            modality=observation.modality,
        )
        self.store.commit(writes, experience)

        next_state = PredictiveState(
            latent=execution.output,
            sufficient_for=state.sufficient_for,
            confidence=verification.confidence,
        )
        self.history.append(
            {
                "step": observation.step,
                "modality": observation.modality,
                "proposed": len(proposals.candidates),
                "selected": execution.action_key,
                "verified": verification.valid,
                "repair_attempts": attempts,
                "writes": len(writes),
                "proposal_work": proposals.work_units,
                "execution_work": execution.work_units,
            }
        )
        return next_state, execution, verification


__all__ = [
    "Budget",
    "Observation",
    "PredictiveState",
    "Structure",
    "ProposalSet",
    "Plan",
    "Execution",
    "Verification",
    "MemoryWrite",
    "Experience",
    "IdentifiabilityReport",
    "RepresentationValidationReport",
    "PersistentKnowledge",
    "KnowledgeStore",
    "UnifiedPersistentStructuralWorldModel",
]
