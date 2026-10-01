"""Full C5 phase: dense CDL shortlisting, calibrated K, OR-LSH admission,
exact CASM execution, verifier selection, and closed-loop experience learning.

This module is intentionally dependency-light. The dense router is a small Q/K
student trained on exhaustive CASM outcomes. The LSH index is inference-time
only; candidate embeddings are precomputed and query-time work touches only
hash tables plus the admitted candidate set.

Research status:
    * all findings are workload-bounded;
    * asymptotic claims require measured exponents;
    * conformal coverage assumes exchangeability of calibration/test trials.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

from . import Candidate, Computation, Outcome, Query, StateUpdate, Structure, VerificationResult
from .executor import ExecutorConfig, StructuralExecutor
from .model import relevance_program
from .structured_verifier import RelationConstraintVerifier
from .temporal import TemporalPersistentState


@dataclass(frozen=True)
class PhaseConfig:
    dim: int = 16
    max_nodes: int = 63
    latent_dim: int = 16
    train_steps: int = 600
    calibration_trials: int = 64
    eval_trials: int = 64
    online_trials: int = 96
    train_min_m: int = 8
    train_max_m: int = 256
    eval_levels: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    learning_rate: float = 0.015
    soft_target_epsilon: float = 0.05
    lsh_tables_cap: int = 128
    alpha: float = 0.10
    alpha_95: float = 0.05
    noisy_query_bits: int = 1

    def __post_init__(self) -> None:
        if self.max_nodes < 4 * self.dim - 1:
            raise ValueError("max_nodes must fit the exact equality circuit")
        if self.train_min_m < 2 or self.train_min_m > self.train_max_m:
            raise ValueError("invalid training population bounds")
        if self.latent_dim < self.dim:
            raise ValueError("latent_dim must be >= dim for the analytic witness")
        if not 0.0 < self.soft_target_epsilon < 1.0:
            raise ValueError("soft_target_epsilon must be in (0,1)")
        if not 0.0 < self.alpha < 1.0 or not 0.0 < self.alpha_95 < 1.0:
            raise ValueError("alpha values must be in (0,1)")


@dataclass(frozen=True)
class Trial:
    candidates: tuple[Candidate, ...]
    target_index: int
    query: Query
    reference: tuple[int, ...]
    address: str


@dataclass(frozen=True)
class CASMOutcome:
    candidate_index: int
    output: float
    verified: bool
    node_count: int
    edge_count: int


@dataclass(frozen=True)
class LSHLookup:
    addresses: tuple[str, ...]
    query_hash_ops: int
    candidate_rerank_count: int
    build_macs_amortized: float
    tables: int
    bits: int
    capped_tables: bool


class CDLDenseOutcomeRouter:
    """Small dense Q/K student trained against exhaustive downstream outcomes."""

    def __init__(
        self,
        *,
        dim: int,
        latent_dim: int,
        seed: int,
        learning_rate: float,
        soft_target_epsilon: float,
        analytic_init: bool = False,
    ) -> None:
        self.dim = dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.soft_target_epsilon = soft_target_epsilon
        self.rng = random.Random(seed)
        raw_q = 3 * dim + 1
        scale = 0.05
        self.wq = [
            [self.rng.uniform(-scale, scale) for _ in range(raw_q)]
            for _ in range(latent_dim)
        ]
        self.wc = [
            [self.rng.uniform(-scale, scale) for _ in range(dim)]
            for _ in range(latent_dim)
        ]
        self.bq = [0.0] * latent_dim
        self.bc = [0.0] * latent_dim
        self.updates = 0
        if analytic_init:
            self.set_analytic_identity()

    @staticmethod
    def _linear(
        weights: Sequence[Sequence[float]],
        bias: Sequence[float],
        x: Sequence[float],
    ) -> tuple[float, ...]:
        return tuple(
            sum(w * xv for w, xv in zip(row, x)) + bias[i]
            for i, row in enumerate(weights)
        )

    def set_analytic_identity(self) -> None:
        """Identity signed-bit witness; used only as an initialization ablation."""
        for row in self.wq:
            for j in range(len(row)):
                row[j] = 0.0
        for row in self.wc:
            for j in range(len(row)):
                row[j] = 0.0
        for j in range(self.latent_dim):
            self.bq[j] = 0.0
            self.bc[j] = 0.0
        for j in range(self.dim):
            self.wq[j][j] = 1.0
            self.wc[j][j] = 1.0

    def _reference(self, query: Query, state: TemporalPersistentState) -> tuple[int, ...]:
        read = state.read(query)
        if read.values:
            return tuple(int(x) for x in read.values[0])
        bits = query.text.partition("\t")[0]
        if not bits.strip():
            return tuple(0 for _ in range(self.dim))
        values = tuple(int(x) for x in bits.split())
        if len(values) != self.dim:
            raise ValueError("query reference width mismatch")
        return values

    def _query_features(
        self, query: Query, state: TemporalPersistentState
    ) -> list[float]:
        ref = self._reference(query, state)
        signed = [2.0 * int(x) - 1.0 for x in ref]
        context = [
            float(query.context[i]) if i < len(query.context) else 0.0
            for i in range(self.dim)
        ]
        interaction = [signed[i] * context[i] for i in range(self.dim)]
        present = 1.0 if state.read(query).values else 0.0
        return signed + context + interaction + [present]

    def _candidate_features(self, candidate: Candidate) -> list[float]:
        if len(candidate.descriptor) != self.dim:
            raise ValueError("candidate descriptor width mismatch")
        return [2.0 * int(x) - 1.0 for x in candidate.descriptor]

    def encode_query(
        self, query: Query, state: TemporalPersistentState
    ) -> tuple[float, ...]:
        return self._linear(self.wq, self.bq, self._query_features(query, state))

    def encode_candidate(self, candidate: Candidate) -> tuple[float, ...]:
        return self._linear(self.wc, self.bc, self._candidate_features(candidate))

    @staticmethod
    def score_embeddings(
        zq: Sequence[float], zc: Sequence[float]
    ) -> float:
        return sum(a * b for a, b in zip(zq, zc))

    def candidate_embeddings(
        self, candidates: Sequence[Candidate]
    ) -> dict[str, tuple[float, ...]]:
        return {c.key: self.encode_candidate(c) for c in candidates}

    def score_all(
        self,
        query: Query,
        state: TemporalPersistentState,
        candidates: Sequence[Candidate],
    ) -> list[float]:
        zq = self.encode_query(query, state)
        return [
            self.score_embeddings(zq, self.encode_candidate(candidate))
            for candidate in candidates
        ]

    def target_rank(
        self,
        query: Query,
        state: TemporalPersistentState,
        candidates: Sequence[Candidate],
        target_index: int,
    ) -> int:
        scores = self.score_all(query, state, candidates)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return order.index(target_index) + 1

    @staticmethod
    def _softmax(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        mx = max(scores)
        exps = [math.exp(max(-60.0, min(60.0, s - mx))) for s in scores]
        total = sum(exps)
        return [x / total for x in exps]

    def learn_from_dense_outcomes(
        self,
        query: Query,
        state: TemporalPersistentState,
        candidates: Sequence[Candidate],
        success_indices: Iterable[int],
    ) -> float:
        positives = {int(i) for i in success_indices}
        if not positives:
            return 0.0
        zq = list(self.encode_query(query, state))
        candidate_features = [self._candidate_features(c) for c in candidates]
        zc = [list(self.encode_candidate(c)) for c in candidates]
        scores = [self.score_embeddings(zq, z) for z in zc]
        probs = self._softmax(scores)

        n = len(candidates)
        target = self.soft_target_epsilon / max(1, n - len(positives))
        target_probs = [target] * n
        pos_prob = (1.0 - self.soft_target_epsilon) / len(positives)
        for i in positives:
            target_probs[i] = pos_prob

        # Gradient ascent on log-likelihood / KL-compatible cross entropy:
        # dL/dscore_i = target_i - model_i.
        grad_q = [0.0] * self.latent_dim
        grad_c = [[0.0] * self.latent_dim for _ in candidates]
        for i, (t, p) in enumerate(zip(target_probs, probs)):
            g = t - p
            for r in range(self.latent_dim):
                grad_q[r] += g * zc[i][r]
                grad_c[i][r] += g * zq[r]

        lr = self.learning_rate
        qx = self._query_features(query, state)
        for r in range(self.latent_dim):
            for j in range(len(qx)):
                self.wq[r][j] += lr * grad_q[r] * qx[j]
            self.bq[r] += lr * grad_q[r]

        for i in range(n):
            cx = candidate_features[i]
            for r in range(self.latent_dim):
                for j in range(self.dim):
                    self.wc[r][j] += lr * grad_c[i][r] * cx[j]
                self.bc[r] += lr * grad_c[i][r]
        self.updates += 1
        return -sum(
            t * math.log(max(p, 1e-12))
            for t, p in zip(target_probs, probs)
        )

    def learn_from_verified(
        self,
        query: Query,
        state: TemporalPersistentState,
        candidates: Sequence[Candidate],
        positive_index: int,
        negative_index: int | None,
    ) -> None:
        """Pairwise post-verification update for the closed loop."""
        if negative_index is None:
            return
        zq = list(self.encode_query(query, state))
        zp = list(self.encode_candidate(candidates[positive_index]))
        zn = list(self.encode_candidate(candidates[negative_index]))
        diff = self.score_embeddings(zq, zp) - self.score_embeddings(zq, zn)
        gate = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, diff))))
        dq = [gate * (zp[r] - zn[r]) for r in range(self.latent_dim)]
        qx = self._query_features(query, state)
        lr = self.learning_rate
        for r in range(self.latent_dim):
            for j in range(len(qx)):
                self.wq[r][j] += lr * dq[r] * qx[j]
            self.bq[r] += lr * dq[r]
            for j in range(self.dim):
                self.wc[r][j] += lr * gate * zq[r] * self._candidate_features(candidates[positive_index])[j]
                self.wc[r][j] -= lr * gate * zq[r] * self._candidate_features(candidates[negative_index])[j]
        self.updates += 1


class ORLSHIndex:
    """Classic OR-over-tables random-hyperplane index over frozen candidate vectors."""

    def __init__(
        self,
        *,
        dim: int,
        bits: int,
        tables: int,
        seed: int,
        tables_cap: int,
    ) -> None:
        if bits < 1 or tables < 1:
            raise ValueError("bits and tables must be positive")
        self.dim = dim
        self.bits = bits
        self.requested_tables = tables
        self.tables = min(tables, tables_cap)
        self.tables_cap = tables_cap
        self.rng = random.Random(seed)
        self.planes = [
            [
                [self.rng.gauss(0.0, 1.0) for _ in range(dim)]
                for _ in range(bits)
            ]
            for _ in range(self.tables)
        ]
        self.buckets: list[dict[int, tuple[str, ...]]] = []
        self.embeddings: dict[str, tuple[float, ...]] = {}
        self.build_macs = 0
        self._built = False

    @staticmethod
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _hash(self, vector: Sequence[float], table: int) -> int:
        code = 0
        for bit, plane in enumerate(self.planes[table]):
            code <<= 1
            if self._dot(vector, plane) >= 0.0:
                code |= 1
        return code

    def build(
        self, embeddings: dict[str, tuple[float, ...]]
    ) -> None:
        self.embeddings = dict(embeddings)
        maps: list[dict[int, list[str]]] = [
            {} for _ in range(self.tables)
        ]
        for address, vector in embeddings.items():
            for t in range(self.tables):
                code = self._hash(vector, t)
                maps[t].setdefault(code, []).append(address)
        self.buckets = [
            {code: tuple(sorted(addresses)) for code, addresses in table.items()}
            for table in maps
        ]
        self.build_macs = (
            len(embeddings) * self.tables * self.bits * self.dim
        )
        self._built = True

    def lookup(
        self,
        query_vector: Sequence[float],
        candidate_limit: int,
        *,
        score_fn,
    ) -> LSHLookup:
        if not self._built:
            raise RuntimeError("LSH index has not been built")
        addresses: set[str] = set()
        for t in range(self.tables):
            code = self._hash(query_vector, t)
            addresses.update(self.buckets[t].get(code, ()))
        scored = sorted(
            ((score_fn(address), address) for address in addresses),
            key=lambda item: (-item[0], item[1]),
        )
        selected = tuple(address for _, address in scored[:candidate_limit])
        return LSHLookup(
            addresses=selected,
            query_hash_ops=self.tables * self.bits * self.dim,
            candidate_rerank_count=len(addresses),
            build_macs_amortized=(
                self.build_macs / max(1, len(self.embeddings))
            ),
            tables=self.tables,
            bits=self.bits,
            capped_tables=self.requested_tables > self.tables,
        )


def empirical_quantile(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(int(v) for v in values)
    idx = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[idx]


def conformal_k(ranks: Sequence[int], alpha: float, max_m: int) -> int:
    """Split-conformal-style rank threshold using the finite sample quantile."""
    k = empirical_quantile(ranks, 1.0 - alpha)
    return max(1, min(max_m, k))


def sample_log_uniform_int(rng: random.Random, low: int, high: int) -> int:
    if low < 1 or high < low:
        raise ValueError("invalid log-uniform range")
    return max(low, min(high, int(round(math.exp(
        rng.uniform(math.log(low), math.log(high))
    )))))


def fresh_unique_code(rng: random.Random, dim: int) -> tuple[int, ...]:
    return tuple(rng.randrange(2) for _ in range(dim))


def build_population(
    rng: random.Random,
    *,
    seed_tag: str,
    m: int,
    target: tuple[int, ...],
) -> tuple[Candidate, ...]:
    if m < 2 or m > 1 << len(target):
        raise ValueError("population cannot exceed the unique code space")
    values: set[tuple[int, ...]] = {target}
    while len(values) < m:
        values.add(fresh_unique_code(rng, len(target)))
    distractors = list(values - {target})
    rng.shuffle(distractors)
    descriptors = [target] + distractors
    rng.shuffle(descriptors)
    return tuple(
        Candidate(
            key=f"{seed_tag}-c-{i:05d}",
            descriptor=tuple(desc),
            action=i,
            provenance="c5_full_phase_population",
        )
        for i, desc in enumerate(descriptors)
    )


def make_persistent_trial(
    rng: random.Random,
    *,
    seed_tag: str,
    step: int,
    m: int,
    dim: int,
    state: TemporalPersistentState,
) -> Trial:
    target = fresh_unique_code(rng, dim)
    candidates = build_population(rng, seed_tag=seed_tag, m=m, target=target)
    target_index = next(
        i for i, c in enumerate(candidates) if c.descriptor == target
    )
    address = f"world:state:{seed_tag}:{step:06d}"
    # Fresh code is introduced at each step and becomes visible only after a
    # causal delay. This is the persistence intervention boundary.
    state.stage_world_write(
        StateUpdate(key=address, value=target, step=state.current_step),
        delay=1,
    )
    state.advance_to(state.current_step + 1)
    query_bits = list(target)
    for _ in range(0):
        pass
    # The address is the public query. A noisy copy is retained as an optional
    # diagnostic field in provenance but is not used by the main state-aware
    # router. The target remains invisible except through state.
    query = Query(
        text="\t" + address,
        context=tuple(1 for _ in range(dim)),
        step=state.current_step,
        provenance="c5_full_phase_persistent",
    )
    return Trial(
        candidates=candidates,
        target_index=target_index,
        query=query,
        reference=target,
        address=address,
    )


class CASMVerifier:
    """Exact CASM relation execution + independent post-execution verifier."""

    def __init__(self, *, dim: int, max_nodes: int) -> None:
        self.dim = dim
        self.executor = StructuralExecutor(
            ExecutorConfig(
                type="learned",
                dim=dim,
                max_nodes=max_nodes,
                temperature=1.0,
            )
        )
        self.verifier = RelationConstraintVerifier()
        self.relation = "equality"
        self.context = tuple(1 for _ in range(dim))

    def execute_and_verify(
        self,
        *,
        candidate: Candidate,
        reference: Sequence[int],
    ) -> CASMOutcome:
        program = relevance_program(
            reference,
            candidate.descriptor,
            self.context,
            max_nodes=self.executor.max_nodes,
        )
        structure = Structure(
            key=candidate.key,
            spec=program,
            provenance="c5_full_phase_casm_exact",
        )
        execution = self.executor.execute(structure, ())
        computation = Computation(
            structure=structure,
            action=candidate.action,
            trace=(execution.output,) + tuple(execution.node_values),
        )
        success = bool(execution.output >= 0.5)
        outcome = Outcome(
            success=success,
            value=float(execution.output),
            feedback="relation_satisfied" if success else "relation_unsatisfied",
            detail=None,
        )
        verdict = self.verifier.verify_task(
            computation,
            outcome,
            candidate=candidate,
            reference=reference,
            context=self.context,
            relation=self.relation,
        )
        return CASMOutcome(
            candidate_index=candidate.action,
            output=float(execution.output),
            verified=verdict.valid,
            node_count=len(execution.node_values),
            edge_count=len(getattr(program, "true_edges", getattr(program, "true_edge_set", ()))),
        )


def run_exhaustive(
    casm: CASMVerifier,
    trial: Trial,
) -> tuple[list[CASMOutcome], list[int]]:
    outcomes = [
        casm.execute_and_verify(candidate=c, reference=trial.reference)
        for c in trial.candidates
    ]
    successes = [
        i for i, outcome in enumerate(outcomes)
        if outcome.verified and outcome.output >= 0.5
    ]
    return outcomes, successes


def run_sparse(
    *,
    router: CDLDenseOutcomeRouter,
    index: ORLSHIndex,
    state: TemporalPersistentState,
    trial: Trial,
    k: int,
    casm: CASMVerifier,
) -> dict[str, object]:
    zq = router.encode_query(trial.query, state)
    embeddings = index.embeddings
    lookup = index.lookup(
        zq,
        candidate_limit=k,
        score_fn=lambda address: router.score_embeddings(
            zq, embeddings[address]
        ),
    )
    admitted_indices = [
        next(i for i, c in enumerate(trial.candidates) if c.key == address)
        for address in lookup.addresses
    ]
    executions = [
        casm.execute_and_verify(
            candidate=trial.candidates[i],
            reference=trial.reference,
        )
        for i in admitted_indices
    ]
    selected_success = next(
        (x.candidate_index for x in executions if x.verified and x.output >= 0.5),
        None,
    )
    target_admitted = trial.target_index in admitted_indices
    success = selected_success == trial.target_index
    return {
        "target_admitted": target_admitted,
        "success": success,
        "admitted_count": len(admitted_indices),
        "rerank_count": lookup.candidate_rerank_count,
        "query_hash_ops": lookup.query_hash_ops,
        "build_macs_amortized": lookup.build_macs_amortized,
        "tables": lookup.tables,
        "bits": lookup.bits,
        "tables_capped": lookup.capped_tables,
        "executed_candidates": len(admitted_indices),
        "executed_nodes": sum(x.node_count for x in executions),
        "executed_edges": sum(x.edge_count for x in executions),
    }


def estimate_p1_p2(
    *,
    index: ORLSHIndex,
    router: CDLDenseOutcomeRouter,
    trials: Sequence[Trial],
    state_factory,
    samples_per_trial: int = 4,
) -> tuple[float, float]:
    agreements_target: list[float] = []
    agreements_decoy: list[float] = []
    for trial in trials:
        state = state_factory(trial)
        zq = router.encode_query(trial.query, state)
        target = trial.candidates[trial.target_index]
        zt = router.encode_candidate(target)
        for _ in range(samples_per_trial):
            seed_material = f"{trial.address}:{_}".encode("utf-8")
            stable_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            decoy_index = random.Random(stable_seed).randrange(len(trial.candidates))
            if decoy_index == trial.target_index:
                decoy_index = (decoy_index + 1) % len(trial.candidates)
            zd = router.encode_candidate(trial.candidates[decoy_index])
            agreements_target.extend(
                1.0 if (
                    (1 if index._dot(zq, index.planes[t][b]) >= 0.0 else 0)
                    ==
                    (1 if index._dot(zt, index.planes[t][b]) >= 0.0 else 0)
                ) else 0.0
                for t in range(index.tables)
                for b in range(index.bits)
            )
            agreements_decoy.extend(
                1.0 if (
                    (1 if index._dot(zq, index.planes[t][b]) >= 0.0 else 0)
                    ==
                    (1 if index._dot(zd, index.planes[t][b]) >= 0.0 else 0)
                ) else 0.0
                for t in range(index.tables)
                for b in range(index.bits)
            )
    return statistics.fmean(agreements_target), statistics.fmean(agreements_decoy)


def oracle_success(
    *,
    trials: Sequence[Trial],
    state_factory,
    reset: bool,
) -> float:
    successes = 0
    for trial in trials:
        state = state_factory(trial)
        if reset:
            state.clear()
        read = state.read(trial.query)
        if not read.values:
            selected = 0
        else:
            reference = tuple(read.values[0])
            selected = next(
                (
                    i for i, c in enumerate(trial.candidates)
                    if c.descriptor == reference
                ),
                0,
            )
        successes += int(selected == trial.target_index)
    return successes / max(1, len(trials))
