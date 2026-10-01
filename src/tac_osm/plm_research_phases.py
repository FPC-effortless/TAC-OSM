"""Orthogonal research phases for the unified PLM substrate.

These experiments continue the integrated PLM program without reopening the
settled C5 representation search. Each phase isolates one remaining interface:

P2: verifier-gated persistent-state stability.
P3: operator/computation addressing as operator population E grows.
P4: slow/fast continual specialization under sequential domain shift.

All outputs are deterministic, CPU-friendly and explicitly separated from
capability claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import random
import statistics

from .plm_unified import (
    ActionOutcome,
    BinaryCDL,
    MemoryRecord,
    PLMConfig,
    StateKind,
    StateStatus,
    StateProposal,
    TypedPersistentState,
    Verification,
    OperatorSpec,
    OperatorPool,
)


# --------------------------------------------------------------------------- #
# P2 — persistent-state stability
# --------------------------------------------------------------------------- #


class StrictVerifier:
    """Exact verifier control: only task-correct transitions are accepted."""

    def verify(self, outcome: ActionOutcome) -> Verification:
        return Verification(
            passed=outcome.success,
            confidence=1.0 if outcome.success else 0.0,
            reason="strict_exact",
            false_accept=False,
        )


class ProbabilisticFalseAcceptVerifier:
    """Injected false-acceptance process for stability stress."""

    def __init__(self, rate: float, seed: int):
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        self.rate = float(rate)
        self.rng = random.Random(seed)

    def verify(self, outcome: ActionOutcome) -> Verification:
        if outcome.success:
            return Verification(True, 1.0, "strict_success", False)
        accepted = self.rng.random() < self.rate
        return Verification(
            accepted,
            self.rate if accepted else 0.0,
            "false_accept" if accepted else "strict_reject",
            accepted,
        )


@dataclass(frozen=True)
class StateStabilityRow:
    seed: int
    arm: str
    false_accept_rate: float
    horizon: int
    attempted: int
    correct_attempts: int
    wrong_attempts: int
    verified_writes: int
    wrong_verified_writes: int
    wrong_write_rate: float
    final_state_size: int
    retractions: int
    supersessions: int


def _proposal(state: TypedPersistentState, *, record_id: str, value: int,
              step: int, confidence: float = 1.0) -> StateProposal:
    return state.propose(
        record_id=record_id,
        namespace="stability",
        key_bits=(1, 0, 1, 0),
        payload=(value,),
        state_kind=StateKind.EXPERIENCE,
        step=step,
        source="p2_stability",
        confidence=confidence,
        operator_family="identity",
    )


def run_state_stability(
    *, seed: int, horizon: int = 256, false_accept_rate: float = 0.0
) -> StateStabilityRow:
    """Inject correct/incorrect transitions and measure verified contamination.

    The world truth is deliberately simple: the only correct payload is 1.
    Each step submits one correct or one incorrect proposal. Since proposals
    are tagged by evaluator-owned truth in this harness, the experiment can
    classify a committed write after the verifier decision without giving the
    classifier information to the verifier.
    """
    state = TypedPersistentState(
        PLMConfig(consolidation_threshold=horizon + 4, seed=seed)
    )
    verifier = ProbabilisticFalseAcceptVerifier(false_accept_rate, seed)
    rng = random.Random(seed + 101)

    attempted = correct = wrong = verified = wrong_verified = 0
    for step in range(horizon):
        is_correct = (step % 2 == 0)
        value = 1 if is_correct else 0
        proposal = _proposal(
            state,
            record_id=f"experience:{step}",
            value=value,
            step=step,
        )
        outcome = ActionOutcome(
            success=is_correct,
            value=value,
            feedback="correct" if is_correct else "wrong",
        )
        verdict = verifier.verify(outcome)
        committed = state.commit(proposal, verdict, step=step)
        attempted += 1
        correct += int(is_correct)
        wrong += int(not is_correct)
        if committed:
            verified += 1
            wrong_verified += int(not is_correct)
    # Exercise explicit lifecycle semantics after the measurement window.
    # Only experience state is eligible for conservative consolidation.
    state.consolidate()
    arm = "strict" if false_accept_rate == 0.0 else "false_accept"
    return StateStabilityRow(
        seed=seed,
        arm=arm,
        false_accept_rate=false_accept_rate,
        horizon=horizon,
        attempted=attempted,
        correct_attempts=correct,
        wrong_attempts=wrong,
        verified_writes=verified,
        wrong_verified_writes=wrong_verified,
        wrong_write_rate=(wrong_verified / verified) if verified else 0.0,
        final_state_size=state.size,
        retractions=state.retractions,
        supersessions=state.supersessions,
    )


def run_no_verifier_control(*, seed: int, horizon: int = 256) -> StateStabilityRow:
    """Control in which every proposal is accepted, including wrong ones."""
    state = TypedPersistentState(
        PLMConfig(consolidation_threshold=horizon + 4, seed=seed)
    )
    attempted = correct = wrong = verified = wrong_verified = 0
    for step in range(horizon):
        is_correct = (step % 2 == 0)
        proposal = _proposal(
            state,
            record_id=f"experience:{step}",
            value=1 if is_correct else 0,
            step=step,
        )
        committed = state.commit(
            proposal,
            Verification(True, 1.0, "no_verifier_control", not is_correct),
            step=step,
        )
        attempted += 1
        correct += int(is_correct)
        wrong += int(not is_correct)
        if committed:
            verified += 1
            wrong_verified += int(not is_correct)
    return StateStabilityRow(
        seed=seed,
        arm="no_verifier",
        false_accept_rate=1.0,
        horizon=horizon,
        attempted=attempted,
        correct_attempts=correct,
        wrong_attempts=wrong,
        verified_writes=verified,
        wrong_verified_writes=wrong_verified,
        wrong_write_rate=wrong_verified / verified,
        final_state_size=state.size,
        retractions=state.retractions,
        supersessions=state.supersessions,
    )


# --------------------------------------------------------------------------- #
# P3 — operator/computation addressing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OperatorRecord:
    name: str
    key_bits: tuple[int, ...]
    cost: int


@dataclass(frozen=True)
class OperatorAddressResult:
    target_name: str
    raw_candidates: int
    admitted_candidates: int
    probes: int
    target_admitted: bool
    target_rank: int


class HierarchicalOperatorIndex:
    """Explicit multi-level operator address index."""

    def __init__(self, levels: Sequence[int] = (4, 8, 12)):
        self.levels = tuple(sorted(set(int(x) for x in levels)))
        if not self.levels:
            raise ValueError("levels must be non-empty")
        self.tables: dict[int, dict[tuple[int, ...], list[str]]] = {}

    @staticmethod
    def _prefix(bits: Sequence[int], level: int) -> tuple[int, ...]:
        return tuple(bits[:level])

    def build(self, operators: Iterable[OperatorRecord]) -> None:
        self.tables = {level: {} for level in self.levels}
        for op in operators:
            for level in self.levels:
                self.tables[level].setdefault(
                    self._prefix(op.key_bits, level), []
                ).append(op.name)

    def lookup(
        self,
        query_bits: Sequence[int],
        *,
        target_name: str,
        max_admitted: int,
    ) -> OperatorAddressResult:
        raw_names: list[str] = []
        probes = 0
        for level in sorted(self.levels, reverse=True):
            prefix = self._prefix(query_bits, level)
            probe_keys = [prefix]
            for i in range(min(level, len(prefix))):
                p = list(prefix)
                p[i] ^= 1
                probe_keys.append(tuple(p))
            found = False
            for key in probe_keys:
                probes += 1
                bucket = self.tables[level].get(key, [])
                if bucket:
                    raw_names.extend(bucket)
                    found = True
            if found:
                break

        raw = tuple(dict.fromkeys(raw_names))
        admitted = raw[:max_admitted]
        target_admitted = target_name in admitted
        target_rank = (
            admitted.index(target_name) + 1
            if target_admitted
            else len(admitted) + 1
        )
        return OperatorAddressResult(
            target_name,
            len(raw),
            len(admitted),
            probes,
            target_admitted,
            target_rank,
        )


def make_operator_population(
    e: int, *, key_dim: int, seed: int
) -> tuple[OperatorRecord, ...]:
    rng = random.Random(seed)
    return tuple(
        OperatorRecord(
            name=f"op:{i}",
            key_bits=tuple(rng.randrange(2) for _ in range(key_dim)),
            cost=1 + i % 4,
        )
        for i in range(e)
    )


@dataclass(frozen=True)
class OperatorScalingRow:
    seed: int
    E: int
    raw_candidates: int
    admitted_candidates: int
    probes: int
    target_admitted: bool
    target_rank: int
    indexed_cost_proxy: int
    full_scan_cost: int
    index_beats_scan: bool


def operator_scaling_row(
    *, seed: int, e: int, key_dim: int = 16,
    max_admitted: int = 8, noise: float = 0.05
) -> OperatorScalingRow:
    operators = make_operator_population(e, key_dim=key_dim, seed=seed)
    rng = random.Random(seed + 17 * e)
    target = operators[rng.randrange(e)]
    query = tuple(
        1 - bit if rng.random() < noise else bit
        for bit in target.key_bits
    )
    index = HierarchicalOperatorIndex((4, 8, 12))
    index.build(operators)
    hit = index.lookup(
        query,
        target_name=target.name,
        max_admitted=max_admitted,
    )
    indexed = hit.probes + hit.raw_candidates + hit.admitted_candidates
    scan = e
    return OperatorScalingRow(
        seed,
        e,
        hit.raw_candidates,
        hit.admitted_candidates,
        hit.probes,
        hit.target_admitted,
        hit.target_rank,
        indexed,
        scan,
        indexed < scan,
    )


# --------------------------------------------------------------------------- #
# P4 — slow/fast continual specialization
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlasticityRow:
    seed: int
    arm: str
    train_domains: tuple[int, ...]
    heldout_domain: int
    retention_after_sequence: tuple[float, ...]
    mean_final_retention: float
    early_adaptation_error: float
    heldout_accuracy: float
    shared_update_l1: float
    specialist_update_l1: float


class ContinualSpecialistModel:
    """Small online model for testing slow shared vs fast local adaptation."""

    def __init__(
        self, *, dim: int, slow_lr: float, fast_lr: float,
        use_specialists: bool
    ):
        self.dim = dim
        self.slow_lr = slow_lr
        self.fast_lr = fast_lr
        self.use_specialists = use_specialists
        self.shared = [0.0] * dim
        self.specialists: dict[int, list[float]] = {}
        self.shared_update_l1 = 0.0
        self.specialist_update_l1 = 0.0

    def _specialist(self, domain: int) -> list[float]:
        if domain not in self.specialists:
            self.specialists[domain] = [0.0] * self.dim
        return self.specialists[domain]

    def score(self, x: Sequence[int], domain: int) -> float:
        delta = self._specialist(domain) if self.use_specialists else ()
        return sum(
            float(x[i])
            * (self.shared[i] + (delta[i] if delta else 0.0))
            for i in range(self.dim)
        )

    def predict(self, x: Sequence[int], domain: int) -> int:
        return int(self.score(x, domain) >= 0.0)

    def update(self, x: Sequence[int], domain: int, y: int) -> None:
        pred = self.predict(x, domain)
        err = int(y) - pred
        if err == 0:
            return
        for i, value in enumerate(x):
            ds = self.slow_lr * err * value
            self.shared[i] += ds
            self.shared_update_l1 += abs(ds)
            if self.use_specialists:
                dl = self.fast_lr * err * value
                delta = self._specialist(domain)
                delta[i] += dl
                self.specialist_update_l1 += abs(dl)


def domain_example(
    rng: random.Random, domain: int, dim: int
) -> tuple[tuple[int, ...], int]:
    x = tuple(rng.randrange(2) for _ in range(dim))
    sign = 1 if ((domain + sum(x)) % 2 == 0) else -1
    threshold = 2 + domain % 3
    y = int(sign * (sum(x) - threshold) >= 0)
    return x, y


def _accuracy(
    model: ContinualSpecialistModel,
    data: Sequence[tuple[tuple[int, ...], int]],
    domain: int,
) -> float:
    return statistics.fmean(
        float(model.predict(x, domain) == y) for x, y in data
    )


def run_plasticity_sequence(
    *, seed: int, train_domains: Sequence[int] = (0, 1, 2),
    heldout_domain: int = 3, dim: int = 8, examples_per_domain: int = 200
) -> tuple[PlasticityRow, ...]:
    arms = (
        ("global_fast", 0.10, 0.00, False),
        ("specialist_fast", 0.00, 0.10, True),
        ("slow_shared_fast_specialist", 0.01, 0.10, True),
    )
    rng = random.Random(seed)
    train = {
        d: [domain_example(rng, d, dim) for _ in range(examples_per_domain)]
        for d in train_domains
    }
    heldout = [
        domain_example(rng, heldout_domain, dim)
        for _ in range(examples_per_domain)
    ]

    rows: list[PlasticityRow] = []
    for arm, slow_lr, fast_lr, specialists in arms:
        model = ContinualSpecialistModel(
            dim=dim,
            slow_lr=slow_lr,
            fast_lr=fast_lr,
            use_specialists=specialists,
        )
        adaptation_errors: list[int] = []
        for domain in train_domains:
            for j, (x, y) in enumerate(train[domain]):
                before = model.predict(x, domain)
                if j < 20:
                    adaptation_errors.append(int(before != y))
                model.update(x, domain, y)

        retention = tuple(
            _accuracy(model, train[d], d) for d in train_domains
        )
        rows.append(
            PlasticityRow(
                seed=seed,
                arm=arm,
                train_domains=tuple(train_domains),
                heldout_domain=heldout_domain,
                retention_after_sequence=retention,
                mean_final_retention=statistics.fmean(retention),
                early_adaptation_error=statistics.fmean(adaptation_errors),
                heldout_accuracy=_accuracy(model, heldout, heldout_domain),
                shared_update_l1=model.shared_update_l1,
                specialist_update_l1=model.specialist_update_l1,
            )
        )
    return tuple(rows)


def aggregate_rows(
    rows: Sequence[object], group_field: str
) -> list[dict[str, object]]:
    groups: dict[object, list[object]] = {}
    for row in rows:
        groups.setdefault(getattr(row, group_field), []).append(row)

    out: list[dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda x: str(x[0])):
        first = group[0]
        if isinstance(first, StateStabilityRow):
            out.append({
                "arm": key,
                "wrong_write_rate_mean": statistics.fmean(
                    r.wrong_write_rate for r in group
                ),
                "verified_writes_mean": statistics.fmean(
                    r.verified_writes for r in group
                ),
                "wrong_verified_writes_mean": statistics.fmean(
                    r.wrong_verified_writes for r in group
                ),
                "final_state_size_mean": statistics.fmean(
                    r.final_state_size for r in group
                ),
            })
        elif isinstance(first, OperatorScalingRow):
            out.append({
                "E": int(key),
                "raw_candidates_mean": statistics.fmean(
                    r.raw_candidates for r in group
                ),
                "admitted_candidates_mean": statistics.fmean(
                    r.admitted_candidates for r in group
                ),
                "target_admission_rate": statistics.fmean(
                    float(r.target_admitted) for r in group
                ),
                "indexed_cost_mean": statistics.fmean(
                    r.indexed_cost_proxy for r in group
                ),
                "full_scan_cost": statistics.fmean(
                    r.full_scan_cost for r in group
                ),
                "index_beats_scan_rate": statistics.fmean(
                    float(r.index_beats_scan) for r in group
                ),
            })
        elif isinstance(first, PlasticityRow):
            out.append({
                "arm": key,
                "mean_final_retention": statistics.fmean(
                    r.mean_final_retention for r in group
                ),
                "early_adaptation_error": statistics.fmean(
                    r.early_adaptation_error for r in group
                ),
                "heldout_accuracy": statistics.fmean(
                    r.heldout_accuracy for r in group
                ),
                "shared_update_l1_mean": statistics.fmean(
                    r.shared_update_l1 for r in group
                ),
                "specialist_update_l1_mean": statistics.fmean(
                    r.specialist_update_l1 for r in group
                ),
            })
    return out
