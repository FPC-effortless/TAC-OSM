"""Fused CDL -> CASM research loop components.

CDL is represented at runtime by a small query/key student patterned after the
validated CDL Stage-B Q/K student. The expensive language-model teacher stays
offline; this module only contains the cheap student boundary.

CASM is represented by an explicit computation selector that compiles the
selected candidate/reference relation into an exact executable graph. CASM
execution is therefore downstream of CDL admission and remains independent of
the routing decision.

The intended causal loop is:

    S_t -> address -> CDL relevance/admission -> CASM computation selection
        -> exact execution -> O_t -> V_t -> verified learning/write -> S_{t+1}

The implementation deliberately does not claim sublinear end-to-end scaling.
The current student still scores the full candidate set; admission K is measured
as a capability boundary and as a future index interface, not disguised as a
hardware speedup.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

from . import Candidate, PersistentState, Query, RoutingDecision, StateRead, Structure
from .environment import parse_query
from .executor import relevance_program
from .state_addressing import AddressedMemory, StateAddressor
from .execution_feedback import execution_feedback_from_verdict

__all__ = [
    "CDLConfig",
    "CDLDiagnostics",
    "CDLStudentRouter",
    "CasmSelection",
    "CasmComputationSelector",
    "reference_from_read",
]


@dataclass(frozen=True)
class CDLConfig:
    input_dim: int = 8
    latent_dim: int = 16
    learning_rate: float = 0.01
    margin: float = 0.25
    admission_k: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim < 4:
            raise ValueError("input_dim must be >= 4")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative")
        if self.admission_k < 1:
            raise ValueError("admission_k must be positive")


@dataclass(frozen=True)
class CDLDiagnostics:
    candidate_count: int
    candidates_scored: int
    admission_k: int
    admission_count: int
    state_found: bool
    state_inspected_slots: int
    state_pool_size: int
    selected_rank: int
    target_rank: int | None
    target_in_admission: bool | None
    query_encode_ops: int
    candidate_encode_ops: int
    similarity_ops: int

    @property
    def total_score_ops(self) -> int:
        return self.query_encode_ops + self.candidate_encode_ops + self.similarity_ops

    @property
    def admission_fraction(self) -> float:
        if self.candidate_count <= 0:
            return 0.0
        return self.admission_count / self.candidate_count


class CDLStudentRouter:
    """Cheap runtime student approximating CDL-style relevance.

    The parameterization mirrors the validated CDL Stage-B Q/K student:
    separate query and candidate projections followed by a dot product.
    Unlike the external teacher, this model is pure Python and can participate
    in the dependency-free TAC-OSM loop.

    Verifier-derived learning is post-execution. A successful execution makes
    the selected candidate a positive; a verified rejection localizes the
    hidden target after the action and supplies a repair positive. No target or
    outcome is consumed by route().
    """

    def __init__(
        self,
        config: CDLConfig | None = None,
        *,
        addressor: StateAddressor | None = None,
    ) -> None:
        self.config = config or CDLConfig()
        self.addressor = addressor or StateAddressor()
        rng = random.Random(self.config.seed)
        scale = 0.05
        self.wq = [
            [rng.uniform(-scale, scale) for _ in range(self._raw_query_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.wc = [
            [rng.uniform(-scale, scale) for _ in range(self.config.input_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.bq = [0.0] * self.config.latent_dim
        self.bc = [0.0] * self.config.latent_dim
        self._last_memory: AddressedMemory | None = None
        self._last_scores: tuple[float, ...] = ()
        self._last_order: tuple[int, ...] = ()
        self._updates = 0

    @property
    def _raw_query_dim(self) -> int:
        return 3 * self.config.input_dim + 1

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def last_memory(self) -> AddressedMemory | None:
        return self._last_memory

    @property
    def last_order(self) -> tuple[int, ...]:
        return self._last_order

    def _bits(self, query: Query) -> tuple[int, ...]:
        raw = query.text.partition("\t")[0]
        return tuple(int(x) for x in raw.split()) if raw.strip() else ()

    def _pad(self, values: Sequence[float]) -> list[float]:
        out = [0.0] * self.config.input_dim
        for i, value in enumerate(values[: self.config.input_dim]):
            out[i] = float(value)
        return out

    def _query_input(self, query: Query, memory: AddressedMemory) -> list[float]:
        bits = self._bits(query)
        reference = memory.value if memory.found else bits
        ref = self._pad(reference)
        context = self._pad(query.context)
        interaction = [ref[i] * context[i] for i in range(self.config.input_dim)]
        return ref + context + interaction + [memory.present]

    def _candidate_input(self, candidate: Candidate) -> list[float]:
        return self._pad(candidate.descriptor)

    @staticmethod
    def _linear(
        weights: Sequence[Sequence[float]],
        bias: Sequence[float],
        x: Sequence[float],
    ) -> list[float]:
        return [
            sum(w * xv for w, xv in zip(row, x)) + bias[i]
            for i, row in enumerate(weights)
        ]

    @staticmethod
    def _score(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def score(
        self,
        query: Query,
        memory: AddressedMemory,
        candidates: Sequence[Candidate],
    ) -> list[float]:
        zq = self._linear(self.wq, self.bq, self._query_input(query, memory))
        return [
            self._score(zq, self._linear(self.wc, self.bc, self._candidate_input(c)))
            for c in candidates
        ]

    def route(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
    ) -> RoutingDecision:
        if not candidates:
            raise ValueError("cannot route an empty candidate set")
        memory = self.addressor.address(query, state)
        scores = self.score(query, memory, candidates)
        order = tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], i)))
        self._last_memory = memory
        self._last_scores = tuple(scores)
        self._last_order = order
        return RoutingDecision(
            selected=order[0],
            scores=tuple(scores),
            provenance="cdl_student_v1",
        )

    def diagnostics_with_target(
        self,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        target_index: int,
    ) -> CDLDiagnostics:
        if not 0 <= target_index < len(candidates):
            raise IndexError("target_index outside candidate set")
        memory = self.addressor.address(query, state)
        scores = self.score(query, memory, candidates)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        rank = order.index(target_index) + 1
        k = min(self.config.admission_k, len(candidates))
        d = self.config.latent_dim
        raw = self._raw_query_dim
        return CDLDiagnostics(
            candidate_count=len(candidates),
            candidates_scored=len(candidates),
            admission_k=k,
            admission_count=k,
            state_found=memory.found,
            state_inspected_slots=memory.inspected_slots,
            state_pool_size=memory.pool_size,
            selected_rank=1,
            target_rank=rank,
            target_in_admission=target_index in order[:k],
            query_encode_ops=d * raw,
            candidate_encode_ops=len(candidates) * d * self.config.input_dim,
            similarity_ops=len(candidates) * d,
        )

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _pair_update(
        self,
        query_x: Sequence[float],
        candidate_pos: Candidate,
        candidate_neg: Candidate,
    ) -> float:
        zq = self._linear(self.wq, self.bq, query_x)
        zp = self._linear(self.wc, self.bc, self._candidate_input(candidate_pos))
        zn = self._linear(self.wc, self.bc, self._candidate_input(candidate_neg))
        diff = self._score(zq, zp) - self._score(zq, zn)
        gap = self.config.margin - diff
        gate = self._sigmoid(gap)
        lr = self.config.learning_rate

        delta_c = [a - b for a, b in zip(zp, zn)]
        pos_x = self._candidate_input(candidate_pos)
        neg_x = self._candidate_input(candidate_neg)
        for r in range(self.config.latent_dim):
            dq = gate * delta_c[r]
            for j in range(self._raw_query_dim):
                self.wq[r][j] += lr * dq * query_x[j]
            delta = gate * zq[r]
            for j in range(self.config.input_dim):
                self.wc[r][j] += lr * delta * (pos_x[j] - neg_x[j])

        self._updates += 1
        return math.log1p(math.exp(min(60.0, gap)))

    def learn_from_verifier(
        self,
        *,
        query: Query,
        state: PersistentState,
        candidates: Sequence[Candidate],
        selected: int,
        outcome: Any,
        verification: Any,
        scores: Sequence[float] | None = None,
    ) -> tuple[float, Any]:
        feedback = execution_feedback_from_verdict(
            candidates, selected, outcome, verification
        )
        if not feedback.positive_indices:
            return 0.0, feedback
        memory = self._last_memory
        if memory is None:
            memory = self.addressor.address(query, state)
        query_x = self._query_input(query, memory)
        score_values = list(scores) if scores is not None else self.score(query, memory, candidates)
        positive = feedback.positive_indices[0]
        negatives = [i for i in feedback.negative_indices if i != positive]
        negatives.sort(key=lambda i: (-float(score_values[i]), i))
        negatives = negatives[:2]
        losses = [
            self._pair_update(query_x, candidates[positive], candidates[i])
            for i in negatives
        ]
        return (
            sum(losses) / len(losses) if losses else 0.0,
            feedback,
        )

    def set_analytic_relation(self) -> None:
        """Construct a separating witness for the representability gate."""
        if self.config.latent_dim < self.config.input_dim:
            raise ValueError("latent_dim must be >= input_dim")
        for row in self.wq:
            for j in range(len(row)):
                row[j] = 0.0
        for row in self.wc:
            for j in range(len(row)):
                row[j] = 0.0
        for j in range(len(self.bq)):
            self.bq[j] = 0.0
        for j in range(len(self.bc)):
            self.bc[j] = 0.0

        d = self.config.input_dim
        for j in range(d):
            self.wq[j][d + j] = -1.0
            self.wq[j][2 * d + j] = 2.0
            self.wc[j][j] = 2.0
            self.bc[j] = -1.0


@dataclass(frozen=True)
class CasmSelection:
    structure: Structure
    active_nodes: int
    true_edges: int
    candidate_edges: int


class CasmComputationSelector:
    """Explicit CASM computation-selection boundary.

    The selector sees the same reference and marked positions available to the
    executor. It compiles exactly the computation needed for the selected
    candidate and records structural work. It never sees the outcome.
    """

    def __init__(self, *, max_nodes: int = 10) -> None:
        if max_nodes < 4:
            raise ValueError("max_nodes must be >= 4")
        self.max_nodes = max_nodes
        self.last_selection: CasmSelection | None = None

    def select(
        self,
        *,
        reference: Sequence[int],
        candidate: Candidate,
        marks: Sequence[int],
        step_index: int,
    ) -> Structure:
        program = relevance_program(
            reference,
            candidate.descriptor,
            marks,
            max_nodes=self.max_nodes,
        )
        selection = CasmSelection(
            structure=Structure(
                key=candidate.key,
                spec=program,
                provenance=f"casm_selected:{candidate.provenance}:{step_index}",
            ),
            active_nodes=program.active_count,
            true_edges=len(program.true_edges),
            candidate_edges=len(program.candidate_edges),
        )
        self.last_selection = selection
        return selection.structure


def reference_from_read(query: Query, read: StateRead) -> tuple[int, ...]:
    bits, address = parse_query(query)
    if address:
        for key, value in zip(read.keys, read.values):
            if key == address and value:
                return tuple(int(x) for x in value)
    if bits:
        return tuple(int(x) for x in bits)
    for value in read.values:
        if value:
            return tuple(int(x) for x in value)
    return ()
