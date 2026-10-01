"""TACOSM-C5-NOISY-FULL-PHASE-001.

The main arm removes the clean-target leak from C5-FULL-PHASE-001.

Task:
  * 64 fixed binary semantic classes, minimum Hamming distance >= 3;
  * one-bit query corruption;
  * candidate populations repeat every class equally, so valid-action count
    grows with M while the relevant class fraction remains 1/64;
  * CDL sees only the noisy public query and candidate descriptors;
  * CASM executes against the noisy public query, never the hidden class;
  * environment outcome is the hidden action success signal, exposed only after
    execution;
  * verification checks execution/outcome consistency without target access;
  * verified successes can update CDL after the action;
  * a persistent-clean control is registered separately.

This is a workload-specific research instrument, not a claim about language
semantics or general memory.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

from . import Candidate, Computation, Outcome, Query, StateUpdate, Structure
from .executor import ExecutorConfig, StructuralExecutor
from .model import relevance_program
from .structured_verifier import VerificationEvidence, SemanticVerifier, BoundedExecutableRepair
from .temporal import TemporalPersistentState

DIM = 16
N_CLASSES = 64
CLASS_COPIES = 16
M_LEVELS = (64, 128, 256, 512, 1024)
NOISE_BITS = 1
CONTEXT = tuple(1 for _ in range(DIM))
WORK_PER_EXECUTION = 1


@dataclass(frozen=True)
class PhaseConfig:
    dim: int = DIM
    classes: int = N_CLASSES
    train_steps: int = 800
    calibration_trials: int = 64
    eval_trials: int = 64
    online_trials: int = 96
    eval_levels: tuple[int, ...] = M_LEVELS
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    learning_rate: float = 0.012
    soft_target_epsilon: float = 0.05
    alpha: float = 0.10
    alpha_95: float = 0.05
    lsh_tables_cap: int = 128
    repair_budget: int = 4

    def __post_init__(self) -> None:
        if self.dim != DIM:
            raise ValueError("this registered task fixes dim=16")
        if self.classes != N_CLASSES:
            raise ValueError("this registered task fixes 64 semantic classes")
        if any(m < self.classes or m % self.classes for m in self.eval_levels):
            raise ValueError("M levels must be divisible by 64")
        if self.repair_budget < 1:
            raise ValueError("repair_budget must be >= 1")


@dataclass(frozen=True)
class NoisyClassTrial:
    candidates: tuple[Candidate, ...]
    valid_indices: frozenset[int]
    target_class: tuple[int, ...]
    noisy_query: Query
    step: int


@dataclass(frozen=True)
class ExecutionRecord:
    index: int
    casm_output: float
    environment_success: bool
    verified: bool


@dataclass(frozen=True)
class SparseRun:
    success: bool
    first_attempt_success: bool
    admitted_valid_count: int
    admitted_count: int
    executed_count: int
    verified_count: int
    repair_attempts: int
    rerank_count: int
    routing_ops: int
    target_admitted: bool


def build_error_correcting_codebook(
    *, classes: int = N_CLASSES, dim: int = DIM, min_distance: int = 3
) -> tuple[tuple[int, ...], ...]:
    """Deterministically construct a fixed 64-word binary code."""
    codes: list[tuple[int, ...]] = []
    for value in range(1 << dim):
        bits = tuple((value >> (dim - 1 - j)) & 1 for j in range(dim))
        if all(
            sum(a != b for a, b in zip(bits, other)) >= min_distance
            for other in codes
        ):
            codes.append(bits)
            if len(codes) == classes:
                break
    if len(codes) != classes:
        raise RuntimeError("could not construct registered codebook")
    return tuple(codes)


CODEBOOK = build_error_correcting_codebook()


def corrupt_one_bit(code: Sequence[int], *, seed: int) -> tuple[int, ...]:
    values = list(code)
    flip = seed % len(values)
    values[flip] ^= 1
    return tuple(values)


def build_population(seed: int, m: int) -> tuple[Candidate, ...]:
    if m not in M_LEVELS:
        raise ValueError(f"unregistered M={m}")
    copies = m // N_CLASSES
    rng = random.Random(seed * 1_000_003 + m * 97)
    candidates: list[Candidate] = []
    for class_id, descriptor in enumerate(CODEBOOK):
        for copy_id in range(copies):
            candidates.append(
                Candidate(
                    key=f"noisy-s{seed}-m{m}-c{class_id:02d}-v{copy_id:02d}",
                    descriptor=descriptor,
                    action=len(candidates),
                    provenance="c5_noisy_repeated_class",
                )
            )
    rng.shuffle(candidates)
    return tuple(
        Candidate(
            key=c.key,
            descriptor=c.descriptor,
            action=i,
            provenance=c.provenance,
        )
        for i, c in enumerate(candidates)
    )


def make_trial(
    *,
    seed: int,
    step: int,
    candidates: Sequence[Candidate],
) -> NoisyClassTrial:
    rng = random.Random(seed * 1_000_003 + step * 7919 + 17)
    target_class = CODEBOOK[rng.randrange(N_CLASSES)]
    valid_indices = frozenset(
        i for i, candidate in enumerate(candidates)
        if candidate.descriptor == target_class
    )
    noisy = corrupt_one_bit(target_class, seed=seed + step * 13 + 5)
    query = Query(
        text=" ".join(str(int(x)) for x in noisy),
        context=CONTEXT,
        step=step,
        provenance="c5_noisy_public_query",
    )
    return NoisyClassTrial(
        candidates=tuple(candidates),
        valid_indices=valid_indices,
        target_class=target_class,
        noisy_query=query,
        step=step,
    )


class NoisyEnvironment:
    """Hidden action outcome. The target is exposed only to this environment."""

    def __init__(self, trial: NoisyClassTrial) -> None:
        self.valid_indices = trial.valid_indices

    def act(self, index: int, *, casm_output: float) -> Outcome:
        return Outcome(
            success=index in self.valid_indices,
            value=float(casm_output),
            feedback="environment_success" if index in self.valid_indices else "environment_failure",
            detail=None,
        )


class NoisyCASMExecutor:
    """CASM execution over public noisy query + candidate structure.

    The CASM circuit computes exact equality against the observed noisy query.
    It therefore never receives the hidden target. The environment's action
    outcome is a separate signal, which allows verification to distinguish
    computation validity from task success.
    """

    def __init__(self, *, dim: int) -> None:
        self.dim = dim
        self.executor = StructuralExecutor(
            ExecutorConfig(type="learned", dim=dim, max_nodes=4 * dim - 1, temperature=1.0)
        )

    def execute(self, trial: NoisyClassTrial, candidate: Candidate) -> tuple[Computation, float]:
        observed = tuple(int(x) for x in trial.noisy_query.text.split())
        program = relevance_program(
            observed,
            candidate.descriptor,
            CONTEXT,
            max_nodes=self.executor.max_nodes,
        )
        structure = Structure(
            key=candidate.key,
            spec=program,
            provenance="c5_noisy_public_casm",
        )
        execution = self.executor.execute(structure, ())
        computation = Computation(
            structure=structure,
            action=candidate.action,
            trace=(execution.output,) + tuple(execution.node_values),
        )
        return computation, float(execution.output)


class NoisyOutcomeVerifier(SemanticVerifier):
    """Verify only public execution semantics and the post-action outcome."""

    def verify_after_action(
        self,
        computation: Computation,
        outcome: Outcome,
        *,
        candidate: Candidate,
        query: Query,
    ) -> VerificationEvidence:
        base = self.verify(computation, outcome)
        if not base.valid:
            return base
        observed = tuple(int(x) for x in query.text.split())
        if candidate.descriptor == observed:
            expected = 1.0
        else:
            expected = 0.0
        actual = float(computation.trace[0]) if computation.trace else 0.0
        if actual != expected:
            return VerificationEvidence(
                valid=False,
                failed_constraint="public_casm_semantics",
                counterexample=(
                    f"candidate={candidate.key} expected_observed_match={int(expected)} "
                    f"computed={actual:.6f}"
                ),
                repair_target=f"candidate:{candidate.key}",
                confidence=1.0,
                evidence=("noisy_public_query", "candidate_descriptor", "casm_trace"),
            )
        if not outcome.success:
            return VerificationEvidence(
                valid=False,
                failed_constraint="environment_action_success",
                counterexample=f"candidate={candidate.key} environment rejected action",
                repair_target=f"candidate:{candidate.key}",
                confidence=1.0,
                evidence=("environment_outcome", "casm_trace"),
            )
        return VerificationEvidence(
            valid=True,
            confidence=1.0,
            evidence=("environment_outcome", "casm_trace", "noisy_public_query"),
        )


class CDLDenseNoisyRouter:
    """Outcome-trained CDL student whose query input is always the noisy code."""

    def __init__(
        self,
        *,
        seed: int,
        learning_rate: float,
        soft_target_epsilon: float,
    ) -> None:
        from .c5_full_phase import CDLDenseOutcomeRouter
        self.inner = CDLDenseOutcomeRouter(
            dim=DIM,
            latent_dim=16,
            seed=seed,
            learning_rate=learning_rate,
            soft_target_epsilon=soft_target_epsilon,
            analytic_init=False,
        )

    @property
    def updates(self) -> int:
        return self.inner.updates

    def score(self, trial: NoisyClassTrial) -> list[float]:
        empty = TemporalPersistentState()
        return self.inner.score_all(
            trial.noisy_query, empty, trial.candidates
        )

    def rank(self, trial: NoisyClassTrial) -> list[int]:
        scores = self.score(trial)
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    def best_valid_rank(self, trial: NoisyClassTrial) -> int:
        ranking = self.rank(trial)
        return min(ranking.index(i) + 1 for i in trial.valid_indices)

    def top1_valid(self, trial: NoisyClassTrial) -> bool:
        return self.rank(trial)[0] in trial.valid_indices

    def train_exhaustive(
        self,
        trial: NoisyClassTrial,
        successes: Sequence[int],
    ) -> None:
        empty = TemporalPersistentState()
        self.inner.learn_from_dense_outcomes(
            trial.noisy_query,
            empty,
            trial.candidates,
            successes,
        )

    def learn_verified(
        self,
        trial: NoisyClassTrial,
        positive_index: int,
        negative_index: int | None,
    ) -> None:
        empty = TemporalPersistentState()
        self.inner.learn_from_verified(
            trial.noisy_query,
            empty,
            trial.candidates,
            positive_index,
            negative_index,
        )


class HammingBaseline:
    """Reference nearest-code router using only the public noisy query."""

    @staticmethod
    def distance(a: Sequence[int], b: Sequence[int]) -> int:
        return sum(int(x != y) for x, y in zip(a, b))

    def rank(self, trial: NoisyClassTrial) -> list[int]:
        observed = tuple(int(x) for x in trial.noisy_query.text.split())
        return sorted(
            range(len(trial.candidates)),
            key=lambda i: (self.distance(observed, trial.candidates[i].descriptor), i),
        )

    def best_valid_rank(self, trial: NoisyClassTrial) -> int:
        ranking = self.rank(trial)
        return min(ranking.index(i) + 1 for i in trial.valid_indices)


@dataclass(frozen=True)
class LSHLookup:
    addresses: tuple[str, ...]
    hash_ops: int
    rerank_count: int
    requested_tables: int
    actual_tables: int
    bits: int
    capped: bool


class ORLSHIndex:
    """OR-over-tables random-hyperplane index over frozen CDL candidate vectors."""

    def __init__(
        self,
        *,
        latent_dim: int,
        bits: int,
        tables: int,
        cap: int,
        seed: int,
    ) -> None:
        self.latent_dim = latent_dim
        self.bits = max(1, bits)
        self.requested_tables = max(1, tables)
        self.actual_tables = min(self.requested_tables, cap)
        self.cap = cap
        self.rng = random.Random(seed)
        self.planes = [
            [
                [self.rng.gauss(0.0, 1.0) for _ in range(latent_dim)]
                for _ in range(self.bits)
            ]
            for _ in range(self.actual_tables)
        ]
        self.embeddings: dict[str, tuple[float, ...]] = {}
        self.buckets: list[dict[int, tuple[str, ...]]] = []
        self.build_macs = 0
        self.built = False

    @staticmethod
    def dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def hash(self, vector: Sequence[float], table: int) -> int:
        code = 0
        for plane in self.planes[table]:
            code = (code << 1) | int(self.dot(vector, plane) >= 0.0)
        return code

    def build(self, embeddings: dict[str, tuple[float, ...]]) -> None:
        self.embeddings = dict(embeddings)
        tables: list[dict[int, list[str]]] = [{} for _ in range(self.actual_tables)]
        for key, vector in embeddings.items():
            for t in range(self.actual_tables):
                code = self.hash(vector, t)
                tables[t].setdefault(code, []).append(key)
        self.buckets = [
            {code: tuple(sorted(keys)) for code, keys in table.items()}
            for table in tables
        ]
        self.build_macs = (
            len(embeddings) * self.actual_tables * self.bits * self.latent_dim
        )
        self.built = True

    def lookup(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        score,
    ) -> LSHLookup:
        if not self.built:
            raise RuntimeError("index is not built")
        addresses: set[str] = set()
        for t in range(self.actual_tables):
            code = self.hash(query_vector, t)
            addresses.update(self.buckets[t].get(code, ()))
        ordered = sorted(
            ((score(key), key) for key in addresses),
            key=lambda x: (-x[0], x[1]),
        )
        return LSHLookup(
            addresses=tuple(key for _, key in ordered[:k]),
            hash_ops=self.actual_tables * self.bits * self.latent_dim,
            rerank_count=len(addresses),
            requested_tables=self.requested_tables,
            actual_tables=self.actual_tables,
            bits=self.bits,
            capped=self.requested_tables > self.actual_tables,
        )


def empirical_quantile(values: Sequence[int], probability: float) -> int:
    ordered = sorted(int(x) for x in values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def conformal_k(ranks: Sequence[int], alpha: float, max_k: int) -> int:
    return max(1, min(max_k, empirical_quantile(ranks, 1.0 - alpha)))


def estimate_scaling(xs: Sequence[int], ys: Sequence[float]) -> float:
    pairs = [(math.log(x), math.log(max(y, 1e-9))) for x, y in zip(xs, ys)]
    if len(pairs) < 2:
        return 0.0
    mx = statistics.fmean(x for x, _ in pairs)
    my = statistics.fmean(y for _, y in pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    return (
        sum((x - mx) * (y - my) for x, y in pairs) / den
        if den else 0.0
    )


def estimate_p1_p2(
    router: CDLDenseNoisyRouter,
    index: ORLSHIndex,
    trial: NoisyClassTrial,
    *,
    samples: int = 8,
) -> tuple[float, float]:
    qz = router.inner.encode_query(trial.noisy_query, TemporalPersistentState())
    target_indices = tuple(trial.valid_indices)
    target = trial.candidates[target_indices[0]]
    p1_values: list[int] = []
    p2_values: list[int] = []
    for sample in range(samples):
        valid_seed = int.from_bytes(
            hashlib.sha256(f"{trial.step}:{sample}".encode()).digest()[:8],
            "big",
        )
        rng = random.Random(valid_seed)
        decoy_index = rng.randrange(len(trial.candidates))
        while decoy_index in trial.valid_indices:
            decoy_index = rng.randrange(len(trial.candidates))
        decoy = trial.candidates[decoy_index]
        tz = router.inner.encode_candidate(target)
        dz = router.inner.encode_candidate(decoy)
        for t in range(index.actual_tables):
            for plane in index.planes[t]:
                p1_values.append(
                    int(
                        (index.dot(qz, plane) >= 0.0)
                        == (index.dot(tz, plane) >= 0.0)
                    )
                )
                p2_values.append(
                    int(
                        (index.dot(qz, plane) >= 0.0)
                        == (index.dot(dz, plane) >= 0.0)
                    )
                )
    return statistics.fmean(p1_values), statistics.fmean(p2_values)


def derive_tables(p1: float, p2: float, m: int) -> tuple[float, int]:
    if p1 <= p2 or p2 <= 0.0 or p2 >= 1.0:
        rho = 1.0
    else:
        rho = max(0.0, min(1.0, math.log(p1) / math.log(p2)))
    requested = max(1, math.ceil(m ** rho))
    return rho, requested


def execute_one(
    trial: NoisyClassTrial,
    index: int,
    executor: NoisyCASMExecutor,
    environment: NoisyEnvironment,
    verifier: NoisyOutcomeVerifier,
) -> ExecutionRecord:
    computation, casm_output = executor.execute(trial, trial.candidates[index])
    outcome = environment.act(index, casm_output=casm_output)
    evidence = verifier.verify_after_action(
        computation,
        outcome,
        candidate=trial.candidates[index],
        query=trial.noisy_query,
    )
    return ExecutionRecord(
        index=index,
        casm_output=casm_output,
        environment_success=outcome.success,
        verified=evidence.valid,
    )


def evaluate_sparse(
    *,
    trial: NoisyClassTrial,
    router: CDLDenseNoisyRouter,
    lsh: ORLSHIndex,
    k: int,
    executor: NoisyCASMExecutor,
    environment: NoisyEnvironment,
    verifier: NoisyOutcomeVerifier,
    repair_budget: int,
) -> SparseRun:
    empty = TemporalPersistentState()
    qz = router.inner.encode_query(trial.noisy_query, empty)
    lookup = lsh.lookup(
        qz,
        k=min(k, len(trial.candidates)),
        score=lambda key: router.inner.score_embeddings(
            qz, lsh.embeddings[key]
        ),
    )
    position = {c.key: i for i, c in enumerate(trial.candidates)}
    admitted = [position[key] for key in lookup.addresses]
    first = False
    verified_count = 0
    executed = 0
    final_success = False
    attempts = 0
    for index in admitted[:repair_budget]:
        attempts += 1
        rec = execute_one(trial, index, executor, environment, verifier)
        executed += 1
        verified_count += int(rec.verified)
        if attempts == 1:
            first = rec.verified
        if rec.verified:
            final_success = True
            break
    return SparseRun(
        success=final_success,
        first_attempt_success=first,
        admitted_valid_count=sum(i in trial.valid_indices for i in admitted),
        admitted_count=len(admitted),
        executed_count=executed,
        verified_count=verified_count,
        repair_attempts=max(0, attempts - 1),
        rerank_count=lookup.rerank_count,
        routing_ops=lookup.hash_ops + lookup.rerank_count * router.inner.config.latent_dim,
        target_admitted=any(i in trial.valid_indices for i in admitted),
    )


def hamming_best_valid_rank(trial: NoisyClassTrial) -> int:
    return HammingBaseline().best_valid_rank(trial)


def persistent_clean_control(
    *,
    seed: int,
    m: int,
    trials: int,
) -> tuple[float, float]:
    successes = 0
    resets = 0
    candidates = build_population(seed, m)
    for step in range(trials):
        trial = make_trial(seed=seed + 5000, step=step, candidates=candidates)
        state = TemporalPersistentState()
        address = f"state:{seed}:{step}"
        state.stage_world_write(
            StateUpdate(key=address, value=trial.target_class, step=0),
            delay=1,
        )
        state.advance_to(1)
        read = state.read(Query(text="\t" + address, step=1))
        if read.values:
            clean = tuple(read.values[0])
            chosen = next(
                i for i, c in enumerate(candidates) if c.descriptor == clean
            )
            successes += int(chosen in trial.valid_indices)
        state.clear()
        read = state.read(Query(text="\t" + address, step=1))
        if read.values:
            clean = tuple(read.values[0])
            chosen = next(
                i for i, c in enumerate(candidates) if c.descriptor == clean
            )
            resets += int(chosen in trial.valid_indices)
    return successes / trials, resets / trials
