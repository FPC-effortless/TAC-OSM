"""Teacher-student sparse proposal for continuous persistent-state routing.

The teacher is a full-candidate cosine scorer. The student learns a compact
query-to-prototype distribution with KL distillation, then proposes a bounded
union of prototype buckets. A fixed cosine reranker sees only that shortlist.

No target address, gold output, or evaluation answer is consumed by the
distillation optimizer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from . import Query
from .learned_prototype_state import LearnedPrototypeStateIndex
from .learned_state_index import LearnedSemanticStateIndex


@dataclass(frozen=True)
class DistillationConfig:
    learning_rate: float = 0.02
    epochs: int = 256
    teacher_temperature: float = 0.10
    student_temperature: float = 0.10
    beam_width: int = 2
    max_shortlist: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.teacher_temperature <= 0.0 or self.student_temperature <= 0.0:
            raise ValueError("temperatures must be positive")
        if self.beam_width < 1:
            raise ValueError("beam_width must be positive")
        if self.max_shortlist < 1:
            raise ValueError("max_shortlist must be positive")


@dataclass(frozen=True)
class DistillationDiagnostics:
    epochs: int
    training_queries: int
    optimizer_updates: int
    initial_kl: float
    final_kl: float
    teacher_candidate_score_macs: int
    student_prototype_score_macs: int
    rejected_hard_negatives: int = 0
    accepted_hard_negatives: int = 0


@dataclass(frozen=True)
class TeacherNegativeFilter:
    """Hard-negative screening against the full teacher distribution."""

    accepted_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    positive_teacher_score: float


@dataclass(frozen=True)
class DistilledProposalLookup:
    selected_address: str | None
    candidate_addresses: tuple[str, ...]
    prototype_indices: tuple[int, ...]
    proposal_macs: int
    rerank_macs: int
    total_macs: int
    teacher_target_available: bool = False

    @property
    def shortlist_size(self) -> int:
        return len(self.candidate_addresses)


def _signed_query(query: Query, width: int) -> tuple[float, ...]:
    raw = query.text.partition("\t")[0].strip()
    bits = tuple(int(x) for x in raw.split()) if raw else ()
    if len(bits) != width or any(x not in (0, 1) for x in bits):
        raise ValueError(f"query code must be binary width {width}")
    return tuple(1.0 if x else -1.0 for x in bits)


def _normalize(values: Sequence[float]) -> tuple[list[float], float]:
    norm = math.sqrt(sum(value * value for value in values))
    denom = max(norm, 1e-8)
    return [value / denom for value in values], denom


def _softmax(logits: Sequence[float], temperature: float) -> tuple[float, ...]:
    scaled = [float(value) / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(max(-60.0, min(60.0, value - maximum))) for value in scaled]
    total = sum(exponentials)
    if total <= 0.0:
        raise ValueError("softmax normalization failed")
    return tuple(value / total for value in exponentials)


def _kl(target: Sequence[float], predicted: Sequence[float]) -> float:
    total = 0.0
    for p, q in zip(target, predicted):
        if p <= 0.0:
            continue
        total += p * math.log(max(p, 1e-12) / max(q, 1e-12))
    return total


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class CosineTeacherStudentProposal:
    """Distilled low-work proposal plus full-teacher cosine reranking."""

    def __init__(
        self,
        teacher_index: LearnedSemanticStateIndex,
        prototype_index: LearnedPrototypeStateIndex,
        config: DistillationConfig | None = None,
        teacher_state_addresses: Sequence[str] | None = None,
    ) -> None:
        self.teacher = teacher_index
        self.prototypes = prototype_index
        available = set(self.prototypes.state_embeddings)
        if teacher_state_addresses is None:
            self._teacher_state_addresses = tuple(sorted(available))
        else:
            selected = tuple(sorted(str(address) for address in teacher_state_addresses))
            unknown = set(selected) - available
            if unknown:
                raise ValueError("teacher_state_addresses contains unknown state addresses")
            if not selected:
                raise ValueError("teacher_state_addresses must contain at least one state address")
            self._teacher_state_addresses = selected
        self.config = config or DistillationConfig(
            beam_width=2,
            max_shortlist=prototype_index.config.bucket_capacity * 2,
        )
        if self.config.max_shortlist < self.config.beam_width:
            raise ValueError("max_shortlist must be >= beam_width")
        d = self.teacher.config.latent_dim
        self._wq: list[list[float]] = []
        self._bq: list[float] = []
        self._updates = 0
        self._diagnostics: DistillationDiagnostics | None = None
        rng = random.Random(self.config.seed * 104729 + 19)
        scale = 0.08
        self._wq = [
            [rng.uniform(-scale, scale) for _ in range(self.teacher.config.input_dim)]
            for _ in range(d)
        ]
        self._bq = [0.0] * d
        self._last_kl: tuple[float, ...] = ()
        self._teacher_prototype_assignment = self._build_training_assignment(
            self._teacher_state_addresses
        )

    @property
    def diagnostics(self) -> DistillationDiagnostics | None:
        return self._diagnostics

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def latent_dim(self) -> int:
        return self.teacher.config.latent_dim

    @property
    def input_dim(self) -> int:
        return self.teacher.config.input_dim

    @property
    def query_projection_macs(self) -> int:
        return self.input_dim * self.latent_dim

    @property
    def prototype_count(self) -> int:
        return len(self.prototypes.prototypes)

    def _linear_query(self, x: Sequence[float]) -> list[float]:
        return [
            sum(w * xv for w, xv in zip(row, x)) + self._bq[i]
            for i, row in enumerate(self._wq)
        ]

    def encode_student_query(self, query: Query) -> tuple[float, ...]:
        raw = self._linear_query(_signed_query(query, self.input_dim))
        normalized, _ = _normalize(raw)
        return tuple(normalized)

    def _teacher_state_scores(
        self,
        query: Query,
        addresses: Sequence[str] | None = None,
    ) -> tuple[float, ...]:
        q_raw = self.teacher.encode_query(query)
        q, _ = _normalize(q_raw)
        selected = self._teacher_state_addresses if addresses is None else tuple(addresses)
        return tuple(
            _cosine(q, self.prototypes.state_embeddings[address])
            for address in selected
        )

    def teacher_distribution(
        self,
        query: Query,
        *,
        addresses: Sequence[str] | None = None,
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        selected = self._teacher_state_addresses if addresses is None else tuple(addresses)
        if not selected:
            raise ValueError("teacher distribution requires at least one address")
        scores = self._teacher_state_scores(query, selected)
        return tuple(selected), _softmax(scores, self.config.teacher_temperature)

    def _build_training_assignment(
        self,
        addresses: Sequence[str],
    ) -> dict[str, int]:
        """Assign only training-state addresses to prototypes with capacity."""
        capacity = self.prototypes.config.bucket_capacity
        if len(addresses) > self.prototype_count * capacity:
            raise ValueError("teacher training-state population exceeds prototype capacity")
        pairs = []
        for address in addresses:
            embedding = self.prototypes.state_embeddings[address]
            for prototype_index, center in enumerate(self.prototypes.prototypes):
                pairs.append(
                    (_cosine(embedding, center), address, prototype_index)
                )
        pairs.sort(key=lambda item: (-item[0], item[2], item[1]))
        assigned: dict[str, int] = {}
        counts = [0] * self.prototype_count
        for _, address, prototype_index in pairs:
            if address in assigned or counts[prototype_index] >= capacity:
                continue
            assigned[address] = prototype_index
            counts[prototype_index] += 1
        if len(assigned) != len(addresses):
            raise AssertionError("could not assign all training states within prototype capacity")
        return assigned

    def _teacher_prototype_distribution(
        self,
        addresses: Sequence[str],
        state_distribution: Sequence[float],
    ) -> tuple[float, ...]:
        address_to_proto = self._teacher_prototype_assignment
        mass = [0.0] * self.prototype_count
        for address, probability in zip(addresses, state_distribution):
            try:
                mass[address_to_proto[address]] += probability
            except KeyError as exc:
                raise AssertionError("state address is missing prototype assignment") from exc
        total = sum(mass)
        if total <= 0.0:
            raise AssertionError("teacher prototype distribution is empty")
        return tuple(value / total for value in mass)

    def _student_prototype_distribution(
        self,
        query: Query,
    ) -> tuple[list[float], tuple[float, ...]]:
        q = self.encode_student_query(query)
        scores = [
            _cosine(q, center)
            for center in self.prototypes.prototypes
        ]
        return list(scores), _softmax(scores, self.config.student_temperature)

    def fit(self, training_queries: Sequence[Query]) -> DistillationDiagnostics:
        if not training_queries:
            raise ValueError("at least one training query is required")
        addresses = self._teacher_state_addresses
        train_macs_per_query = len(addresses) * self.latent_dim
        student_macs_per_query = self.prototype_count * self.latent_dim
        first_loss = 0.0
        final_loss = 0.0
        updates = 0
        for epoch in range(self.config.epochs):
            order = list(training_queries)
            random.Random(self.config.seed * 1009 + epoch).shuffle(order)
            for query in order:
                teacher_states, teacher_state_distribution = self.teacher_distribution(query)
                target = self._teacher_prototype_distribution(
                    teacher_states, teacher_state_distribution
                )
                student_scores, predicted = self._student_prototype_distribution(query)
                loss = _kl(target, predicted)
                if updates == 0:
                    first_loss = loss
                final_loss = loss

                x = _signed_query(query, self.input_dim)
                q_raw = self._linear_query(x)
                q_norm, q_denom = _normalize(q_raw)
                grad_q = [0.0] * self.latent_dim
                for prototype_index, (weight, center) in enumerate(
                    zip(
                        (predicted[i] - target[i] for i in range(self.prototype_count)),
                        self.prototypes.prototypes,
                    )
                ):
                    cosine = student_scores[prototype_index]
                    for r in range(self.latent_dim):
                        grad_q[r] += weight * (
                            center[r] - cosine * q_norm[r]
                        ) / q_denom

                lr = self.config.learning_rate / self.config.student_temperature
                for r in range(self.latent_dim):
                    self._bq[r] -= lr * grad_q[r]
                    for j in range(self.input_dim):
                        self._wq[r][j] -= lr * grad_q[r] * x[j]
                updates += 1

        self._updates = updates
        self._diagnostics = DistillationDiagnostics(
            epochs=self.config.epochs,
            training_queries=len(training_queries),
            optimizer_updates=updates,
            initial_kl=first_loss,
            final_kl=final_loss,
            teacher_candidate_score_macs=(
                len(training_queries) * self.config.epochs * train_macs_per_query
            ),
            student_prototype_score_macs=(
                len(training_queries) * self.config.epochs * student_macs_per_query
            ),
        )
        return self._diagnostics

    def filter_hard_negatives(
        self,
        query: Query,
        candidate_addresses: Sequence[str],
        positive_address: str,
        *,
        minimum_teacher_margin: float = 0.02,
        max_negatives: int | None = None,
    ) -> TeacherNegativeFilter:
        if positive_address not in candidate_addresses:
            raise ValueError("positive address must be present in candidate pool")
        addresses = tuple(candidate_addresses)
        scores = dict(zip(
            *self.teacher_distribution(query, addresses=addresses)
        ))
        positive_score = float(scores[positive_address])
        ordered = sorted(
            (
                (float(scores[address]), index, address)
                for index, address in enumerate(addresses)
                if address != positive_address
            ),
            key=lambda item: (-item[0], item[1]),
        )
        accepted: list[int] = []
        rejected: list[int] = []
        for score, index, _ in ordered:
            if (
                positive_score - score >= minimum_teacher_margin
                and (max_negatives is None or len(accepted) < max_negatives)
            ):
                accepted.append(index)
            else:
                rejected.append(index)
        return TeacherNegativeFilter(
            accepted_indices=tuple(accepted),
            rejected_indices=tuple(rejected),
            positive_teacher_score=positive_score,
        )

    def lookup(
        self,
        query: Query,
        *,
        beam_width: int | None = None,
        max_shortlist: int | None = None,
    ) -> DistilledProposalLookup:
        beam = self.config.beam_width if beam_width is None else int(beam_width)
        limit = self.config.max_shortlist if max_shortlist is None else int(max_shortlist)
        if beam < 1 or beam > self.prototype_count:
            raise ValueError("beam_width must be within prototype count")
        if limit < 1:
            raise ValueError("max_shortlist must be positive")

        q = self.encode_student_query(query)
        scores = [
            _cosine(q, center)
            for center in self.prototypes.prototypes
        ]
        selected = tuple(
            sorted(
                range(len(scores)),
                key=lambda i: (-scores[i], i),
            )[:beam]
        )
        addresses = tuple(sorted({
            address
            for prototype_index in selected
            for address in self.prototypes.buckets[prototype_index]
        }))

        # The reranker is allowed to over-fetch and then cap the final state set.
        if len(addresses) > limit:
            q_teacher_raw = self.teacher.encode_query(query)
            q_teacher, _ = _normalize(q_teacher_raw)
            ranked = sorted(
                (
                    (_cosine(q_teacher, self.prototypes.state_embeddings[address]), address)
                    for address in addresses
                ),
                key=lambda item: (-item[0], item[1]),
            )
            addresses = tuple(address for _, address in ranked[:limit])

        q_teacher_raw = self.teacher.encode_query(query)
        q_teacher, _ = _normalize(q_teacher_raw)
        ranked = sorted(
            (
                (_cosine(q_teacher, self.prototypes.state_embeddings[address]), address)
                for address in addresses
            ),
            key=lambda item: (-item[0], item[1]),
        )
        chosen = ranked[0][1] if ranked else None
        teacher_query_macs = self.teacher.query_embedding_macs
        proposal_macs = self.query_projection_macs + self.prototype_count * self.latent_dim
        rerank_macs = teacher_query_macs + len(addresses) * self.latent_dim
        return DistilledProposalLookup(
            selected_address=chosen,
            candidate_addresses=addresses,
            prototype_indices=selected,
            proposal_macs=proposal_macs,
            rerank_macs=rerank_macs,
            total_macs=proposal_macs + rerank_macs,
        )
