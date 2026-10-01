"""C5 persistent relational full loop.

This phase removes the exact-Hamming shortcut and puts temporal persistence in
the main routing path.

Task:
  * each episode writes two random operand vectors plus an operation id to
    persistent state;
  * the target candidate is the result of the public relation
    R(left, right, op), but the target itself is never stored in state;
  * the query is an opaque state address plus a one-bit-corrupted operation
    hint in Query.context;
  * the router must read the persisted operands and operation to rank the
    derived target among random candidate descriptors;
  * CASM/verifier derive the expected descriptor from persistent state only;
  * reset control clears state before routing.

The workload is deliberately relational rather than semantic language data.
It tests persistence plus learned compositional relevance, and keeps the
C5 sparse funnel intact.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, Computation, Outcome, Query, StateUpdate, Structure
from .executor import ExecutorConfig, StructuralExecutor
from .model import relevance_program
from .structured_verifier import SemanticVerifier, VerificationEvidence
from .temporal import TemporalPersistentState

DIM = 16
N_OPS = 4
OPS = ("xor", "xnor", "and", "or")
M_LEVELS = (64, 128, 256, 512, 1024)
TRAIN_M_LEVELS = (64, 128, 256)
TRAIN_STEPS = 800
CALIBRATION_TRIALS = 64
EVAL_TRIALS = 64
ONLINE_TRIALS = 64
STATE_DELAY = 3
LATENT_DIM = 16
LSH_CAP = 128
REPAIR_BUDGET = 4


@dataclass(frozen=True)
class PersistentRelationConfig:
    dim: int = DIM
    ops: tuple[str, ...] = OPS
    train_steps: int = TRAIN_STEPS
    calibration_trials: int = CALIBRATION_TRIALS
    eval_trials: int = EVAL_TRIALS
    online_trials: int = ONLINE_TRIALS
    eval_levels: tuple[int, ...] = M_LEVELS
    train_m_levels: tuple[int, ...] = TRAIN_M_LEVELS
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    learning_rate: float = 0.012
    soft_target_epsilon: float = 0.05
    alpha: float = 0.10
    repair_budget: int = REPAIR_BUDGET
    lsh_cap: int = LSH_CAP

    def __post_init__(self) -> None:
        if self.dim != DIM or self.ops != OPS:
            raise ValueError("registered relational task is fixed")
        if self.repair_budget < 1:
            raise ValueError("repair_budget must be >= 1")
        if any(m < N_OPS or m & (m - 1) for m in self.eval_levels):
            raise ValueError("M levels must be powers of two")


@dataclass(frozen=True)
class PersistentRelationTrial:
    candidates: tuple[Candidate, ...]
    target_descriptor: tuple[int, ...]
    target_index: int
    address: str
    query: Query
    step: int


@dataclass(frozen=True)
class ExecutionRecord:
    index: int
    output: float
    environment_success: bool
    verified: bool


def apply_relation(
    left: Sequence[int], right: Sequence[int], op: int
) -> tuple[int, ...]:
    if len(left) != DIM or len(right) != DIM:
        raise ValueError("operand width mismatch")
    if not 0 <= op < N_OPS:
        raise ValueError("invalid op")
    out: list[int] = []
    for a, b in zip(left, right):
        if op == 0:
            v = a ^ b
        elif op == 1:
            v = 1 - (a ^ b)
        elif op == 2:
            v = a & b
        else:
            v = a | b
        out.append(int(v))
    return tuple(out)


def encode_state_value(
    left: Sequence[int], right: Sequence[int], op: int
) -> tuple[int, ...]:
    return tuple(int(x) for x in left) + tuple(int(x) for x in right) + (int(op),)


def decode_state_value(value: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if len(value) != 2 * DIM + 1:
        raise ValueError("persistent relation state width mismatch")
    left = tuple(int(x) for x in value[:DIM])
    right = tuple(int(x) for x in value[DIM:2 * DIM])
    op = int(value[-1])
    if not 0 <= op < N_OPS:
        raise ValueError("invalid persisted op")
    return left, right, op


def corrupted_op_context(op: int, *, seed: int) -> tuple[int, ...]:
    values = [0] * N_OPS
    values[op] = 1
    flip = seed % N_OPS
    values[flip] ^= 1
    return tuple(values)


def prepare_state(
    *, seed: int, step: int, address: str, left: tuple[int, ...],
    right: tuple[int, ...], op: int
) -> tuple[TemporalPersistentState, Query]:
    state = TemporalPersistentState()
    state.stage_world_write(
        StateUpdate(
            key=address,
            value=encode_state_value(left, right, op),
            step=0,
        ),
        delay=STATE_DELAY,
    )
    rng = random.Random(seed * 1009 + step * 9176 + 13)
    # Interleave unrelated writes across the temporal boundaries. These values
    # are never addressed by the query and provide persistence interference.
    for boundary in range(1, STATE_DELAY + 1):
        decoy_key = f"decoy:{seed}:{step}:{boundary}"
        decoy = tuple(rng.randrange(2) for _ in range(2 * DIM + 1))
        state.stage_world_write(
            StateUpdate(key=decoy_key, value=decoy, step=boundary - 1),
            delay=1,
        )
        state.advance_to(boundary)
    query = Query(
        text="	" + address,
        context=corrupted_op_context(op, seed=seed + step * 17 + 3),
        step=STATE_DELAY,
        provenance="c5_persistent_relational_query",
    )
    return state, query


def build_population(seed: int, m: int, target: tuple[int, ...]) -> tuple[Candidate, ...]:
    if m < 1 or m > (1 << DIM):
        raise ValueError("invalid population size")
    rng = random.Random(seed * 1_000_003 + m * 97)
    values = {target}
    while len(values) < m:
        values.add(tuple(rng.randrange(2) for _ in range(DIM)))
    descriptors = list(values)
    rng.shuffle(descriptors)
    return tuple(
        Candidate(
            key=f"prl-s{seed}-m{m}-c{i:04d}",
            descriptor=tuple(desc),
            action=i,
            provenance="c5_persistent_relational",
        )
        for i, desc in enumerate(descriptors)
    )


def make_trial(*, seed: int, step: int, m: int) -> PersistentRelationTrial:
    rng = random.Random(seed * 1_000_003 + step * 7919 + m)
    left = tuple(rng.randrange(2) for _ in range(DIM))
    right = tuple(rng.randrange(2) for _ in range(DIM))
    op = rng.randrange(N_OPS)
    target = apply_relation(left, right, op)
    candidates = build_population(seed + 31, m, target)
    target_index = next(i for i, c in enumerate(candidates) if c.descriptor == target)
    address = f"world:relation:{seed}:{step:06d}"
    query = Query(
        text="	" + address,
        context=corrupted_op_context(op, seed=seed + step * 17 + 3),
        step=STATE_DELAY,
        provenance="c5_persistent_relational_query",
    )
    return PersistentRelationTrial(
        candidates=candidates,
        target_descriptor=target,
        target_index=target_index,
        address=address,
        query=query,
        step=step,
    )


class PersistentRelationEnvironment:
    def __init__(self, trial: PersistentRelationTrial) -> None:
        self.target_index = trial.target_index

    def act(self, index: int, *, casm_output: float) -> Outcome:
        ok = index == self.target_index
        return Outcome(
            success=ok,
            value=float(casm_output),
            feedback="relation_action_success" if ok else "relation_action_failure",
            detail=None,
        )


class CDLPersistentRelationRouter:
    """Dual linear Q/K learner over persisted operands and a noisy op hint."""

    def __init__(
        self, *, seed: int, learning_rate: float, soft_target_epsilon: float
    ) -> None:
        self.dim = DIM
        self.latent_dim = LATENT_DIM
        self.learning_rate = learning_rate
        self.soft_target_epsilon = soft_target_epsilon
        self.rng = random.Random(seed)
        # Four operation blocks, each containing per-bit a, b, a*b features.
        self.query_dim = 4 * 3 * DIM + N_OPS + 1
        scale = 0.04
        self.wq = [
            [self.rng.uniform(-scale, scale) for _ in range(self.query_dim)]
            for _ in range(self.latent_dim)
        ]
        self.bq = [0.0] * self.latent_dim
        self.wc = [
            [self.rng.uniform(-scale, scale) for _ in range(DIM)]
            for _ in range(self.latent_dim)
        ]
        self.bc = [0.0] * self.latent_dim
        self.updates = 0

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

    def _query_features(
        self, query: Query, state: TemporalPersistentState
    ) -> list[float]:
        read = state.read(query)
        if not read.values:
            left = right = tuple(0 for _ in range(DIM))
            state_op = 0
            present = 0.0
        else:
            left, right, state_op = decode_state_value(read.values[0])
            present = 1.0
        a = [2.0 * x - 1.0 for x in left]
        b = [2.0 * x - 1.0 for x in right]
        p = [x * y for x, y in zip(a, b)]
        blocks: list[float] = []
        for op in range(N_OPS):
            gate = 1.0 if op == state_op else 0.0
            blocks.extend(gate * a[i] for i in range(DIM))
            blocks.extend(gate * b[i] for i in range(DIM))
            blocks.extend(gate * p[i] for i in range(DIM))
        ctx = list(query.context[:N_OPS])
        ctx.extend([0.0] * (N_OPS - len(ctx)))
        return blocks + ctx + [present]

    def _candidate_features(self, candidate: Candidate) -> list[float]:
        if len(candidate.descriptor) != DIM:
            raise ValueError("candidate descriptor width mismatch")
        return [2.0 * int(x) - 1.0 for x in candidate.descriptor]

    def encode_query(
        self, query: Query, state: TemporalPersistentState
    ) -> tuple[float, ...]:
        return self._linear(self.wq, self.bq, self._query_features(query, state))

    def encode_candidate(self, candidate: Candidate) -> tuple[float, ...]:
        return self._linear(self.wc, self.bc, self._candidate_features(candidate))

    @staticmethod
    def score_embeddings(zq: Sequence[float], zc: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(zq, zc))

    def candidate_embeddings(
        self, candidates: Sequence[Candidate]
    ) -> dict[str, tuple[float, ...]]:
        return {c.key: self.encode_candidate(c) for c in candidates}

    def score_all(
        self, query: Query, state: TemporalPersistentState, candidates: Sequence[Candidate]
    ) -> list[float]:
        zq = self.encode_query(query, state)
        return [self.score_embeddings(zq, self.encode_candidate(c)) for c in candidates]

    def rank(
        self, query: Query, state: TemporalPersistentState,
        candidates: Sequence[Candidate]
    ) -> list[int]:
        scores = self.score_all(query, state, candidates)
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    @staticmethod
    def _softmax(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        mx = max(scores)
        exps = [math.exp(max(-60.0, min(60.0, s - mx))) for s in scores]
        total = sum(exps)
        return [x / total for x in exps]

    def train_exhaustive(
        self,
        query: Query,
        state: TemporalPersistentState,
        candidates: Sequence[Candidate],
        positive_index: int,
    ) -> None:
        zq = list(self.encode_query(query, state))
        candidate_features = [self._candidate_features(c) for c in candidates]
        zc = [list(self.encode_candidate(c)) for c in candidates]
        scores = [self.score_embeddings(zq, z) for z in zc]
        probs = self._softmax(scores)
        n = len(candidates)
        target = self.soft_target_epsilon / max(1, n - 1)
        target_probs = [target] * n
        target_probs[positive_index] = 1.0 - self.soft_target_epsilon

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

        for i, cx in enumerate(candidate_features):
            for r in range(self.latent_dim):
                g = grad_c[i][r]
                for j in range(DIM):
                    self.wc[r][j] += lr * g * cx[j]
                self.bc[r] += lr * g
        self.updates += 1

    def learn_verified(
        self,
        query: Query,
        state: TemporalPersistentState,
        candidates: Sequence[Candidate],
        positive_index: int,
        negative_index: int | None,
    ) -> None:
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
        pf = self._candidate_features(candidates[positive_index])
        nf = self._candidate_features(candidates[negative_index])
        for r in range(self.latent_dim):
            for j in range(len(qx)):
                self.wq[r][j] += lr * dq[r] * qx[j]
            self.bq[r] += lr * dq[r]
            for j in range(DIM):
                self.wc[r][j] += lr * gate * zq[r] * (pf[j] - nf[j])
        self.updates += 1


class PersistentRelationCASMVerifier:
    """CASM computes R(state_left,state_right,state_op) and verifies the action."""

    def __init__(self) -> None:
        self.executor = StructuralExecutor(
            ExecutorConfig(type="learned", dim=DIM, max_nodes=4 * DIM - 1, temperature=1.0)
        )
        self.verifier = SemanticVerifier()

    def derive_reference(
        self, state: TemporalPersistentState, query: Query
    ) -> tuple[int, ...] | None:
        read = state.read(query)
        if not read.values:
            return None
        left, right, op = decode_state_value(read.values[0])
        return apply_relation(left, right, op)

    def execute_and_verify(
        self,
        trial: PersistentRelationTrial,
        state: TemporalPersistentState,
        index: int,
    ) -> ExecutionRecord:
        reference = self.derive_reference(state, trial.query)
        if reference is None:
            return ExecutionRecord(index, 0.0, False, False)
        candidate = trial.candidates[index]
        program = relevance_program(
            reference,
            candidate.descriptor,
            tuple(1 for _ in range(DIM)),
            max_nodes=self.executor.max_nodes,
        )
        structure = Structure(
            key=candidate.key,
            spec=program,
            provenance="c5_persistent_relational_casm",
        )
        execution = self.executor.execute(structure, ())
        computation = Computation(
            structure=structure,
            action=candidate.action,
            trace=(execution.output,) + tuple(execution.node_values),
        )
        outcome = PersistentRelationEnvironment(trial).act(
            index, casm_output=float(execution.output)
        )
        base = self.verifier.verify(computation, outcome)
        expected = bool(tuple(candidate.descriptor) == reference)
        actual = bool(execution.output >= 0.5)
        valid = base.valid and actual == expected and outcome.success and expected
        return ExecutionRecord(
            index=index,
            output=float(execution.output),
            environment_success=outcome.success,
            verified=valid,
        )


def state_factory(trial: PersistentRelationTrial) -> TemporalPersistentState:
    rng = random.Random(trial.step * 7919 + sum(trial.candidates[trial.target_index].descriptor))
    # Reconstruct the episode deterministically from the trial seed embedded in
    # its address. The address itself is sufficient to identify the stored item.
    parts = trial.address.split(":")
    seed = int(parts[2]) if len(parts) > 3 else 0
    left_seed = seed * 1_000_003 + trial.step * 7919 + 1
    local = random.Random(left_seed)
    left = tuple(local.randrange(2) for _ in range(DIM))
    right = tuple(local.randrange(2) for _ in range(DIM))
    op = local.randrange(N_OPS)
    state, _ = prepare_state(
        seed=seed, step=trial.step, address=trial.address,
        left=left, right=right, op=op
    )
    return state


def make_episode(
    *, seed: int, step: int, m: int
) -> tuple[PersistentRelationTrial, TemporalPersistentState]:
    rng = random.Random(seed * 1_000_003 + step * 7919 + m)
    left = tuple(rng.randrange(2) for _ in range(DIM))
    right = tuple(rng.randrange(2) for _ in range(DIM))
    op = rng.randrange(N_OPS)
    target = apply_relation(left, right, op)
    candidates = build_population(seed + 31, m, target)
    target_index = next(i for i, c in enumerate(candidates) if c.descriptor == target)
    address = f"world:relation:{seed}:{step:06d}"
    state, query = prepare_state(
        seed=seed, step=step, address=address,
        left=left, right=right, op=op
    )
    trial = PersistentRelationTrial(
        candidates=candidates,
        target_descriptor=target,
        target_index=target_index,
        address=address,
        query=query,
        step=step,
    )
    return trial, state


class HammingOperandBaseline:
    @staticmethod
    def distance(a: Sequence[int], b: Sequence[int]) -> int:
        return sum(int(x != y) for x, y in zip(a, b))

    def rank(self, trial: PersistentRelationTrial, state: TemporalPersistentState) -> int:
        read = state.read(trial.query)
        if not read.values:
            return len(trial.candidates) // 2
        left, right, _ = decode_state_value(read.values[0])
        scores = [
            min(
                self.distance(c.descriptor, left),
                self.distance(c.descriptor, right),
            )
            for c in trial.candidates
        ]
        return sorted(range(len(scores)), key=lambda i: (scores[i], i)).index(trial.target_index) + 1


def persistent_oracle_rank(trial: PersistentRelationTrial) -> int:
    return 1


def hamming_reset_rank(trial: PersistentRelationTrial) -> int:
    return len(trial.candidates) // 2


def fit_power(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(math.log(float(x)), math.log(max(float(y), 1e-9))) for x, y in zip(xs, ys)]
    if len(pairs) < 2:
        return 0.0
    mx = statistics.fmean(x for x, _ in pairs)
    my = statistics.fmean(y for _, y in pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    return sum((x - mx) * (y - my) for x, y in pairs) / den if den else 0.0


def empirical_quantile(values: Sequence[int], probability: float) -> int:
    ordered = sorted(int(v) for v in values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))]


def derive_tables(p1: float, p2: float, m: int) -> tuple[float, int]:
    if p1 <= p2 or not 0.0 < p2 < 1.0:
        rho = 1.0
    else:
        rho = max(0.0, min(1.0, math.log(p1) / math.log(p2)))
    return rho, max(1, math.ceil(m ** rho))


def estimate_lsh_geometry(
    router: CDLPersistentRelationRouter,
    index_planes,
    trials: Sequence[PersistentRelationTrial],
    states: Sequence[TemporalPersistentState],
) -> tuple[float, float]:
    values1: list[int] = []
    values2: list[int] = []
    for trial, state in zip(trials, states):
        qz = router.encode_query(trial.query, state)
        target = trial.candidates[trial.target_index]
        target_z = router.encode_candidate(target)
        decoy_index = next(i for i in range(len(trial.candidates)) if i != trial.target_index)
        decoy_z = router.encode_candidate(trial.candidates[decoy_index])
        for planes in index_planes:
            for plane in planes:
                qh = int(sum(a * b for a, b in zip(qz, plane)) >= 0)
                th = int(sum(a * b for a, b in zip(target_z, plane)) >= 0)
                dh = int(sum(a * b for a, b in zip(decoy_z, plane)) >= 0)
                values1.append(int(qh == th))
                values2.append(int(qh == dh))
    return statistics.fmean(values1), statistics.fmean(values2)


def run_sparse(
    router: CDLPersistentRelationRouter,
    trial: PersistentRelationTrial,
    state: TemporalPersistentState,
    *,
    k: int,
    tables: int,
    seed: int,
) -> dict[str, object]:
    from .c5_noisy_full_phase import ORLSHIndex
    index = ORLSHIndex(
        latent_dim=LATENT_DIM,
        bits=max(1, math.ceil(math.log2(len(trial.candidates)))),
        tables=tables,
        cap=LSH_CAP,
        seed=seed,
    )
    index.build(router.candidate_embeddings(trial.candidates))
    qz = router.encode_query(trial.query, state)
    lookup = index.lookup(
        qz,
        k=min(k, len(trial.candidates)),
        score=lambda key: router.score_embeddings(qz, index.embeddings[key]),
    )
    positions = {c.key: i for i, c in enumerate(trial.candidates)}
    admitted = [positions[key] for key in lookup.addresses]
    verifier = PersistentRelationCASMVerifier()
    first = False
    final = False
    repairs = 0
    for j, idx in enumerate(admitted[:REPAIR_BUDGET]):
        record = verifier.execute_and_verify(trial, state, idx)
        if j == 0:
            first = record.verified
        if record.verified:
            final = True
            repairs = j
            break
    return {
        "target_admitted": trial.target_index in admitted,
        "success": final,
        "first_attempt_success": first,
        "admitted_count": len(admitted),
        "rerank_count": lookup.rerank_count,
        "executed_count": min(len(admitted), REPAIR_BUDGET),
        "repair_attempts": repairs,
        "routing_ops": lookup.hash_ops + lookup.rerank_count * LATENT_DIM,
        "tables": tables,
    }
