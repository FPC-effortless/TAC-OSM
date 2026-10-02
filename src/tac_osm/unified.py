"""Unified persistent structural world model (UPSWM) reference interfaces.
Dependency-free causal core. Numerical backends are injected so research runs can
use PyTorch/other systems without contaminating TAC-OSM's scientific interface.
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
    def __post_init__(self):
        if min(self.proposal_k, self.route_k, self.planning_depth, self.max_execution_work, self.max_memory_items) < 1:
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

class StructureDiscoverer(Protocol):
    def discover(self, observation: Observation, knowledge: "PersistentKnowledge") -> Sequence[Structure]: ...

class Addressor(Protocol):
    def address(self, state: PredictiveState, knowledge: "PersistentKnowledge", budget: Budget) -> ProposalSet: ...

class Predictor(Protocol):
    def predict(self, state: PredictiveState, structure: Structure, depth: int = 1) -> Sequence[float]: ...

class Planner(Protocol):
    def plan(self, state: PredictiveState, proposals: ProposalSet, predictor: Predictor, budget: Budget) -> Plan: ...

class Executor(Protocol):
    def execute(self, plan: Plan, state: PredictiveState) -> Execution: ...

class Verifier(Protocol):
    def verify(self, state: PredictiveState, execution: Execution, observation: Observation) -> Verification: ...

class WritePolicy(Protocol):
    def decide(self, state: PredictiveState, execution: Execution, verification: Verification, budget: Budget) -> Sequence[MemoryWrite]: ...

class RetractionPolicy(Protocol):
    def decide(self, key: str, contradictions: int, confidence: float) -> MemoryWrite | None: ...

@dataclass
class PersistentKnowledge:
    experiences: list[Experience] = field(default_factory=list)
    structures: dict[str, Structure] = field(default_factory=dict)
    invalidated: set[str] = field(default_factory=set)
    contradictions: dict[str, int] = field(default_factory=dict)
    def snapshot(self) -> "PersistentKnowledge":
        return PersistentKnowledge(list(self.experiences), dict(self.structures), set(self.invalidated), dict(self.contradictions))

@dataclass
class KnowledgeStore:
    knowledge: PersistentKnowledge = field(default_factory=PersistentKnowledge)
    def commit(self, writes: Sequence[MemoryWrite], experience: Experience | None = None) -> None:
        if experience is not None and experience.verification.valid and experience.verification.confidence > 0:
            self.knowledge.experiences.append(experience)
        for w in writes:
            if w.confidence <= 0: continue
            if w.operation in {"add", "update", "consolidate"}:
                self.knowledge.structures[w.key] = Structure(w.key, w.value, "persisted", provenance="verified")
                self.knowledge.invalidated.discard(w.key)
            elif w.operation in {"invalidate", "delete"}:
                self.knowledge.structures.pop(w.key, None)
                self.knowledge.invalidated.add(w.key)

@dataclass
class UnifiedPersistentStructuralWorldModel:
    discoverer: StructureDiscoverer
    addressor: Addressor
    predictor: Predictor
    planner: Planner
    executor: Executor
    verifier: Verifier
    writer: WritePolicy
    store: KnowledgeStore = field(default_factory=KnowledgeStore)
    budget: Budget = field(default_factory=Budget)
    identifiability_gate: Callable[[Sequence[Any]], IdentifiabilityReport] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def step(self, observation: Observation, state: PredictiveState) -> tuple[PredictiveState, Execution, Verification]:
        if self.identifiability_gate is not None:
            report = self.identifiability_gate((observation,))
            if not report.identifiable:
                raise RuntimeError(f"identifiability gate failed: {report.reason}")
        knowledge = self.store.knowledge.snapshot()
        _ = self.discoverer.discover(observation, knowledge)
        proposals = self.addressor.address(state, knowledge, self.budget)
        if not proposals.candidates:
            raise RuntimeError("proposal miss: no candidate fallback permitted")
        plan = self.planner.plan(state, proposals, self.predictor, self.budget)
        if plan.selected < 0 or plan.selected >= len(plan.candidates):
            raise RuntimeError("planner selected an invalid proposal")
        execution = self.executor.execute(plan, state)
        verification = self.verifier.verify(state, execution, observation)
        writes = tuple(self.writer.decide(state, execution, verification, self.budget))
        exp = Experience(state, execution.action_key, execution.output, observation.payload, verification, observation.step, observation.modality)
        self.store.commit(writes, exp)
        next_state = PredictiveState(execution.output, state.sufficient_for, verification.confidence)
        self.history.append({"step": observation.step, "modality": observation.modality, "proposed": len(proposals.candidates), "selected": execution.action_key, "verified": verification.valid, "writes": len(writes)})
        return next_state, execution, verification

__all__ = ["Budget","Observation","PredictiveState","Structure","ProposalSet","Plan","Execution","Verification","MemoryWrite","Experience","IdentifiabilityReport","PersistentKnowledge","KnowledgeStore","UnifiedPersistentStructuralWorldModel"]
