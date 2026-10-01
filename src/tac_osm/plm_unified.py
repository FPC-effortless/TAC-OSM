"""Unified PLM research substrate for TAC-OSM.

This module composes, without claiming to replace, the validated TAC-OSM C5
mechanisms: typed persistent state, selective relevance routing, CASM
operator execution, verified state transitions, continual learning, and
capacity lifecycle. It is dependency-free so the integration repository
remains lightweight and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Callable, Iterable, Mapping, Sequence


class StateKind(str, Enum):
    WORLD = "world"
    TASK = "task"
    EXPERIENCE = "experience"
    COMPUTATIONAL = "computational"
    POLICY = "policy"
    EVIDENCE = "evidence"
    PREDICTION = "prediction"
    HYPOTHESIS = "hypothesis"
    UNVERIFIED = "unverified"


class StateStatus(str, Enum):
    TENTATIVE = "tentative"
    VERIFIED = "verified"
    REJECTED = "rejected"
    RETRACTED = "retracted"


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    namespace: str
    key_bits: tuple[int, ...]
    payload: tuple[int, ...]
    state_kind: StateKind
    status: StateStatus
    created_step: int
    source: str = "synthetic"
    confidence: float = 0.0
    supersedes: str | None = None
    operator_family: str = ""


@dataclass(frozen=True)
class StateProposal:
    record: MemoryRecord
    reason: str


@dataclass(frozen=True)
class Verification:
    passed: bool
    confidence: float
    reason: str
    false_accept: bool = False


@dataclass(frozen=True)
class ActionOutcome:
    success: bool
    value: int | float | None
    feedback: str
    cost: int = 1


@dataclass(frozen=True)
class AddressResult:
    candidate_ids: tuple[str, ...]
    index_lookups: int
    raw_candidates: int
    candidates_scored: int
    entries_touched: int
    full_population: int
    target_admitted: bool
    used_full_scan_fallback: bool


@dataclass(frozen=True)
class CandidateScore:
    record_id: str
    score: float


@dataclass(frozen=True)
class OperatorPlan:
    operator_names: tuple[str, ...]
    confidence: float
    estimated_cost: int


@dataclass(frozen=True)
class ExecutionTrace:
    record_id: str
    operator_names: tuple[str, ...]
    outputs: tuple[int | float, ...]
    halted_at: int


@dataclass(frozen=True)
class StepResult:
    step: int
    record_id: str | None
    plan: OperatorPlan | None
    outcome: ActionOutcome
    verification: Verification
    committed: bool
    repaired: bool
    address: AddressResult
    execution_cost: int
    total_cost: int


@dataclass
class PLMConfig:
    index_levels: tuple[int, ...] = (2, 4, 6)
    max_admitted: int = 16
    max_executions: int = 4
    max_depth: int = 3
    halt_threshold: float = 0.80
    allow_tentative_routing: bool = False
    false_accept_rate: float = 0.0
    fast_lr: float = 0.10
    slow_lr: float = 0.01
    consolidation_threshold: int = 256
    prune_after: int = 32
    seed: int = 0


class TypedPersistentState:
    """Typed persistent state with verifier-gated authority and consolidation."""

    def __init__(self, config: PLMConfig | None = None):
        self.config = config or PLMConfig()
        self._records: dict[str, MemoryRecord] = {}
        self._writes = 0
        self._retractions = 0
        self._supersessions = 0

    def load(self, records: Iterable[MemoryRecord]) -> None:
        self._records = {r.record_id: r for r in records}

    def propose(self, *, record_id: str, namespace: str, key_bits: Sequence[int],
                payload: Sequence[int], state_kind: StateKind, step: int,
                source: str = "experience", confidence: float = 0.0,
                operator_family: str = "") -> StateProposal:
        return StateProposal(MemoryRecord(
            record_id=record_id, namespace=namespace,
            key_bits=tuple(map(int, key_bits)), payload=tuple(map(int, payload)),
            state_kind=state_kind, status=StateStatus.TENTATIVE,
            created_step=step, source=source, confidence=float(confidence),
            operator_family=operator_family,
        ), "proposed")

    def commit(self, proposal: StateProposal, verification: Verification,
               *, step: int) -> bool:
        if not verification.passed:
            return False
        r = proposal.record
        self._records[r.record_id] = MemoryRecord(
            **{**r.__dict__, "status": StateStatus.VERIFIED,
               "confidence": max(r.confidence, verification.confidence)}
        )
        self._writes += 1
        return True

    def supersede(self, old_id: str, proposal: StateProposal,
                  verification: Verification, *, step: int) -> bool:
        if old_id not in self._records or not verification.passed:
            return False
        old = self._records[old_id]
        self._records[old_id] = MemoryRecord(
            **{**old.__dict__, "status": StateStatus.RETRACTED}
        )
        self._retractions += 1
        r = proposal.record
        self._records[r.record_id] = MemoryRecord(
            **{**r.__dict__, "status": StateStatus.VERIFIED,
               "supersedes": old_id,
               "confidence": max(r.confidence, verification.confidence)}
        )
        self._writes += 1
        self._supersessions += 1
        return True

    def retract(self, record_id: str, reason: str) -> bool:
        r = self._records.get(record_id)
        if r is None or r.status is StateStatus.RETRACTED:
            return False
        self._records[record_id] = MemoryRecord(
            **{**r.__dict__, "status": StateStatus.RETRACTED, "source": reason}
        )
        self._retractions += 1
        return True

    def records(self, *, include_retracted: bool = False) -> tuple[MemoryRecord, ...]:
        xs = tuple(self._records.values())
        return xs if include_retracted else tuple(
            r for r in xs if r.status is not StateStatus.RETRACTED
        )

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    @property
    def size(self) -> int:
        return len(self.records())

    @property
    def writes(self) -> int:
        return self._writes

    @property
    def retractions(self) -> int:
        return self._retractions

    @property
    def supersessions(self) -> int:
        return self._supersessions

    def consolidate(self) -> dict[str, int]:
        active = list(self.records())
        over = max(0, len(active) - self.config.consolidation_threshold)
        if not over:
            return {"considered": 0, "retracted": 0}
        eligible = sorted(
            (r for r in active if r.state_kind is StateKind.EXPERIENCE),
            key=lambda r: (r.confidence, r.created_step, r.record_id),
        )
        n = 0
        for r in eligible[:over]:
            n += int(self.retract(r.record_id, "consolidation"))
        return {"considered": len(eligible), "retracted": n}


class HierarchicalStateIndex:
    """Explicit multi-level content index; probe work is fully measured."""

    def __init__(self, levels: Sequence[int] = (2, 4, 6)):
        self.levels = tuple(sorted(set(map(int, levels))))
        if not self.levels or min(self.levels) <= 0:
            raise ValueError("index levels must be positive")
        self._tables: dict[int, dict[tuple[int, ...], list[str]]] = {}

    @staticmethod
    def _prefix(bits: Sequence[int], level: int) -> tuple[int, ...]:
        return tuple(map(int, bits[:level]))

    def build(self, records: Iterable[MemoryRecord]) -> None:
        self._tables = {level: {} for level in self.levels}
        for r in records:
            for level in self.levels:
                key = self._prefix(r.key_bits, level)
                self._tables[level].setdefault(key, []).append(r.record_id)

    def lookup(self, key_bits: Sequence[int], *, max_candidates: int
               ) -> tuple[tuple[str, ...], int, int, bool]:
        selected: list[str] = []
        lookups = 0
        fallback = False
        bits = tuple(map(int, key_bits))
        for level in sorted(self.levels, reverse=True):
            prefix = self._prefix(bits, level)
            probes = [prefix]
            for j in range(min(level, len(prefix))):
                p = list(prefix)
                p[j] ^= 1
                probes.append(tuple(p))
            found = False
            for probe in probes:
                lookups += 1
                bucket = self._tables[level].get(tuple(probe), [])
                if bucket:
                    selected.extend(bucket)
                    found = True
            if found:
                break
        if not selected:
            fallback = True
            lookups += 1
            selected.extend(sum(self._tables[self.levels[0]].values(), []))
        raw = tuple(dict.fromkeys(selected))
        return raw[:max_candidates], lookups, len(raw), fallback


class BinaryCDL:
    """Cheap relevance learner with slow shared and fast local terms."""

    def __init__(self, dim: int, *, config: PLMConfig | None = None):
        self.dim = int(dim)
        self.config = config or PLMConfig()
        self.slow_weights = [0.0] * self.dim
        self.fast_bias: dict[str, float] = {}

    def _features(self, q: Sequence[int], r: MemoryRecord) -> tuple[float, ...]:
        x = [1.0 if int(a) == int(b) else -1.0 for a, b in zip(q, r.key_bits)]
        x += [0.0] * max(0, self.dim - len(x))
        return tuple(x[:self.dim])

    def score(self, q: Sequence[int], r: MemoryRecord) -> float:
        f = self._features(q, r)
        return sum(w * x for w, x in zip(self.slow_weights, f)) + self.fast_bias.get(r.namespace, 0.0)

    def rank(self, q: Sequence[int], records: Sequence[MemoryRecord]) -> tuple[CandidateScore, ...]:
        scored = [(r, self.score(q, r)) for r in records]
        return tuple(CandidateScore(r.record_id, s) for r, s in sorted(
            scored, key=lambda x: (-x[1], x[0].record_id)
        ))

    def update(self, q: Sequence[int], selected: MemoryRecord, reward: float, *,
               correct_record: MemoryRecord | None = None) -> None:
        dslow = self.config.slow_lr * float(reward)
        dfast = self.config.fast_lr * float(reward)
        for i, x in enumerate(self._features(q, selected)):
            self.slow_weights[i] += dslow * x
        self.fast_bias[selected.namespace] = self.fast_bias.get(selected.namespace, 0.0) + dfast
        if reward < 0 and correct_record and correct_record.record_id != selected.record_id:
            for i, x in enumerate(self._features(q, correct_record)):
                self.slow_weights[i] += self.config.slow_lr * x
            self.fast_bias[correct_record.namespace] = (
                self.fast_bias.get(correct_record.namespace, 0.0) + self.config.fast_lr
            )


@dataclass
class OperatorSpec:
    name: str
    arity: int
    fn: Callable[[tuple[int, ...]], int]
    family: str
    cost: int = 1
    fast_weight: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    last_used_step: int = -1

    @property
    def total_uses(self) -> int:
        return self.success_count + self.failure_count


class OperatorPool:
    """CASM operator library with evidence-driven lifecycle hooks."""

    def __init__(self, *, prune_after: int = 32):
        self.operators: dict[str, OperatorSpec] = {}
        self.prune_after = int(prune_after)

    def register(self, op: OperatorSpec) -> None:
        if op.name in self.operators:
            raise ValueError(f"operator exists: {op.name}")
        self.operators[op.name] = op

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.operators))

    def plans(self, family_hint: str = "", *, max_depth: int = 3) -> tuple[OperatorPlan, ...]:
        ops = sorted(
            self.operators.values(),
            key=lambda o: (
                0 if family_hint and (o.family == family_hint or o.name == family_hint) else 1,
                -o.fast_weight, o.cost, o.name,
            ),
        )
        out: list[OperatorPlan] = []
        for op in ops[:max_depth]:
            n = op.total_uses
            empirical = (op.success_count + 1.0) / (n + 2.0)
            conf = 0.5 * empirical + 0.5 / (1.0 + max(0.0, -op.fast_weight))
            out.append(OperatorPlan((op.name,), conf, op.cost))
        return tuple(out)

    def update(self, plan: OperatorPlan, success: bool, step: int) -> None:
        for name in plan.operator_names:
            op = self.operators[name]
            op.last_used_step = step
            if success:
                op.success_count += 1
                op.fast_weight += 0.10
            else:
                op.failure_count += 1
                op.fast_weight -= 0.05

    def maybe_prune(self, step: int) -> tuple[str, ...]:
        doomed = [
            name for name, op in self.operators.items()
            if op.total_uses and op.failure_count > 2 * op.success_count + 2
            and step - op.last_used_step >= self.prune_after
        ]
        for name in doomed:
            del self.operators[name]
        return tuple(sorted(doomed))

    def synthesize_not(self, name: str = "not") -> None:
        if name not in self.operators:
            self.register(OperatorSpec(name, 1, lambda x: 1 - int(x[0]), "not"))

    def synthesize_composite(self, name: str, first: str, second: str) -> None:
        if name in self.operators:
            return
        a, b = self.operators[first], self.operators[second]
        self.register(OperatorSpec(
            name, 2, lambda x: b.fn((a.fn(x),)), name, a.cost + b.cost
        ))


class AdaptiveCASM:
    def __init__(self, pool: OperatorPool, config: PLMConfig | None = None):
        self.pool = pool
        self.config = config or PLMConfig()

    def execute(self, record: MemoryRecord, plan: OperatorPlan) -> ExecutionTrace:
        value = record.payload
        outputs: list[int | float] = []
        names: list[str] = []
        for name in plan.operator_names[:self.config.max_depth]:
            op = self.pool.operators[name]
            if len(value) < op.arity:
                raise ValueError(f"operator {name} needs arity {op.arity}, got {len(value)}")
            value = (op.fn(tuple(value[:op.arity])),)
            outputs.append(value[0])
            names.append(name)
            if plan.confidence >= self.config.halt_threshold:
                break
        return ExecutionTrace(record.record_id, tuple(names), tuple(outputs), len(names))


class ReferenceVerifier:
    def __init__(self, *, false_accept_rate: float = 0.0, seed: int = 0):
        if not 0 <= false_accept_rate <= 1:
            raise ValueError("false_accept_rate must be in [0,1]")
        self.rate = float(false_accept_rate)
        self.rng = random.Random(seed)

    def verify(self, outcome: ActionOutcome) -> Verification:
        if outcome.success:
            return Verification(True, 1.0, "environment_success")
        if self.rng.random() < self.rate:
            return Verification(True, self.rate, "injected_false_accept", True)
        return Verification(False, 0.0, "environment_failure")


class RelationEnvironment:
    FUNCTIONS: Mapping[str, Callable[[tuple[int, ...]], int]] = {
        "xor": lambda x: int(x[0]) ^ int(x[1]),
        "xnor": lambda x: 1 - (int(x[0]) ^ int(x[1])),
        "and": lambda x: int(x[0]) & int(x[1]),
        "or": lambda x: int(x[0]) | int(x[1]),
    }

    @classmethod
    def apply(cls, operation: str, payload: Sequence[int]) -> int:
        return cls.FUNCTIONS[operation](tuple(payload))


@dataclass(frozen=True)
class RelationTask:
    target_id: str
    query_bits: tuple[int, ...]
    operation: str
    expected: int
    namespace: str = "relational"


class UnifiedPLM:
    """Executable state -> address -> compute -> action -> verify -> learn loop."""

    def __init__(self, *, state: TypedPersistentState, router: BinaryCDL,
                 index: HierarchicalStateIndex, operators: OperatorPool,
                 verifier: ReferenceVerifier, config: PLMConfig | None = None):
        self.state, self.router, self.index = state, router, index
        self.operators, self.verifier = operators, verifier
        self.config = config or state.config
        self.casm = AdaptiveCASM(operators, self.config)

    def rebuild_index(self) -> None:
        self.index.build(self.state.records())

    def address(self, query_bits: Sequence[int], target_id: str | None = None) -> AddressResult:
        ids, probes, raw, fallback = self.index.lookup(
            query_bits, max_candidates=self.config.max_admitted
        )
        rows = [
            self.state.get(i) for i in ids
            if self.state.get(i) is not None
            and (self.config.allow_tentative_routing
                 or self.state.get(i).status is StateStatus.VERIFIED)
        ]
        admitted = target_id in {r.record_id for r in rows} if target_id else False
        return AddressResult(
            tuple(r.record_id for r in rows), probes, raw, len(rows),
            raw, self.state.size, admitted, fallback
        )

    def step(self, task: RelationTask, *, step: int) -> StepResult:
        address = self.address(task.query_bits, task.target_id)
        candidates = [self.state.get(i) for i in address.candidate_ids]
        candidates = [r for r in candidates if r is not None]
        if not candidates:
            out = ActionOutcome(False, None, "no_candidates")
            v = self.verifier.verify(out)
            return StepResult(step, None, None, out, v, False, True, address, 0, address.index_lookups)

        by_id = {r.record_id: r for r in candidates}
        correct = self.state.get(task.target_id)
        execution_cost = 0
        attempted: list[str] = []
        repaired = False

        for row in self.router.rank(task.query_bits, candidates):
            record = by_id[row.record_id]
            for plan in self.operators.plans(record.operator_family, max_depth=self.config.max_depth)[:self.config.max_executions]:
                trace = self.casm.execute(record, plan)
                execution_cost += trace.halted_at * plan.estimated_cost
                if trace.operator_names[-1] in RelationEnvironment.FUNCTIONS:
                    value = RelationEnvironment.apply(trace.operator_names[-1], record.payload[:2])
                else:
                    value = None
                success = (
                    record.record_id == task.target_id
                    and trace.operator_names[-1] == task.operation
                    and value == task.expected
                )
                out = ActionOutcome(success, value, "match" if success else "mismatch")
                v = self.verifier.verify(out)
                attempted.append(record.record_id)
                self.operators.update(plan, v.passed, step)

                if v.passed:
                    proposal = self.state.propose(
                        record_id=f"experience:{step}:{record.record_id}:{trace.operator_names[-1]}",
                        namespace=record.namespace, key_bits=record.key_bits,
                        payload=record.payload, state_kind=StateKind.EXPERIENCE,
                        step=step, source="verified_transition",
                        confidence=v.confidence, operator_family=trace.operator_names[-1],
                    )
                    committed = self.state.commit(proposal, v, step=step)
                    self.router.update(task.query_bits, record, 1.0, correct_record=record)
                    return StepResult(
                        step, record.record_id, plan, out, v, committed, repaired,
                        address, execution_cost,
                        address.index_lookups + address.raw_candidates + address.candidates_scored + execution_cost,
                    )

                repaired = True
                self.router.update(task.query_bits, record, -1.0, correct_record=correct)
                if execution_cost >= self.config.max_executions:
                    break
            if execution_cost >= self.config.max_executions:
                break

        out = ActionOutcome(False, None, "repair_budget_exhausted:" + ",".join(attempted))
        v = self.verifier.verify(out)
        return StepResult(
            step, attempted[0] if attempted else None, None, out, v, False,
            repaired, address, execution_cost,
            address.index_lookups + address.candidates_scored + execution_cost,
        )

    def learn_and_consolidate(self) -> dict[str, int]:
        return self.state.consolidate()


def build_operator_pool() -> OperatorPool:
    pool = OperatorPool()
    for name in ("xor", "xnor", "and", "or"):
        pool.register(OperatorSpec(name, 2, RelationEnvironment.FUNCTIONS[name], name))
    return pool


def bitflip(bits: Sequence[int], p: float, rng: random.Random) -> tuple[int, ...]:
    return tuple(1 - int(b) if rng.random() < p else int(b) for b in bits)


def generate_episode(*, rng: random.Random, history: int, records_target: int,
                     key_dim: int = 12, noise: float = 0.10
                     ) -> tuple[RelationTask, tuple[MemoryRecord, ...]]:
    op = rng.choice(tuple(RelationEnvironment.FUNCTIONS))
    target_key = tuple(rng.randrange(2) for _ in range(key_dim))
    payload = (rng.randrange(2), rng.randrange(2))
    target_id = f"target:{history}:{''.join(map(str,target_key))}"
    records = [MemoryRecord(
        target_id, "relational", target_key, payload, StateKind.WORLD,
        StateStatus.VERIFIED, 0, "seed_fact", 1.0, None, op
    )]
    for i in range(records_target - 1):
        key = tuple(rng.randrange(2) for _ in range(key_dim))
        records.append(MemoryRecord(
            f"distractor:{history}:{i}:{''.join(map(str,key))}", "relational", key,
            (rng.randrange(2), rng.randrange(2)),
            StateKind.EXPERIENCE if i % 2 == 0 else StateKind.WORLD,
            StateStatus.VERIFIED, i + 1, "distractor", 0.8, None,
            rng.choice(tuple(RelationEnvironment.FUNCTIONS)),
        ))
    return (
        RelationTask(target_id, bitflip(target_key, noise, rng), op,
                     RelationEnvironment.apply(op, payload)),
        tuple(records),
    )


def new_plm_for_records(records: Sequence[MemoryRecord], *,
                        config: PLMConfig) -> UnifiedPLM:
    state = TypedPersistentState(config)
    state.load(records)
    router = BinaryCDL(len(records[0].key_bits), config=config)
    index = HierarchicalStateIndex(config.index_levels)
    plm = UnifiedPLM(
        state=state, router=router, index=index,
        operators=build_operator_pool(),
        verifier=ReferenceVerifier(
            false_accept_rate=config.false_accept_rate, seed=config.seed
        ),
        config=config,
    )
    plm.rebuild_index()
    return plm


__all__ = [
    "StateKind", "StateStatus", "MemoryRecord", "StateProposal", "Verification",
    "ActionOutcome", "AddressResult", "CandidateScore", "OperatorPlan",
    "ExecutionTrace", "StepResult", "PLMConfig", "TypedPersistentState",
    "HierarchicalStateIndex", "BinaryCDL", "OperatorSpec", "OperatorPool",
    "AdaptiveCASM", "ReferenceVerifier", "RelationEnvironment", "RelationTask",
    "UnifiedPLM", "build_operator_pool", "bitflip", "generate_episode",
    "new_plm_for_records",
]
