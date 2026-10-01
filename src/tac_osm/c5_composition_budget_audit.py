"""C5 composition-representation and execution-budget audit."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from typing import Callable, Sequence

from . import Candidate, Query
from .c5_noisy_full_phase import ORLSHIndex
from .c5_persistent_relational_loop import (
    DIM,
    LATENT_DIM,
    N_OPS,
    OPS,
    PersistentRelationCASMVerifier,
    PersistentRelationTrial,
    apply_relation,
    empirical_quantile,
    make_episode,
    decode_state_value,
    CDLPersistentRelationRouter,
)
from .temporal import TemporalPersistentState

M_LEVELS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
REP_LEVELS = (1024,)
SEEDS = (0, 1, 2, 3, 4)
TRAIN_STEPS = 1200
BASELINE_STEPS = 800
DISTILL_STEPS = 400
CALIBRATION_TRIALS = 64
HELDOUT_TRIALS = 64
DISTILL_TABLES = 8
DISTILL_K = 16
LSH_MAX_TABLES = 128
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 91013
BUDGETS = (1, 2, 4, 8, 16, 32, 64)
LATENT_HALF = LATENT_DIM // 2

SPLIT_OFFSETS = {
    "calibration": 300_000,
    "heldout": 500_000,
    "distill": 200_000,
    "online": 600_000,
}


def split_step(split: str, index: int) -> int:
    return SPLIT_OFFSETS[split] + index


def episode_key(seed: int, m: int, split: str, index: int) -> str:
    return f"{split}|s={seed}|m={m}|i={index}|step={split_step(split, index)}"


def split_manifest(seed: int, m: int, split: str, count: int) -> tuple[str, ...]:
    return tuple(episode_key(seed, m, split, i) for i in range(count))


def verify_split_disjointness(seed: int, m: int) -> dict[str, object]:
    cal = set(split_manifest(seed, m, "calibration", CALIBRATION_TRIALS))
    ho = set(split_manifest(seed, m, "heldout", HELDOUT_TRIALS))
    return {
        "calibration_count": len(cal),
        "heldout_count": len(ho),
        "intersection_count": len(cal & ho),
        "calibration_digest": hashlib.sha256("\n".join(sorted(cal)).encode()).hexdigest(),
        "heldout_digest": hashlib.sha256("\n".join(sorted(ho)).encode()).hexdigest(),
    }


def trials_for_split(
    seed: int, m: int, split: str, count: int
) -> list[tuple[PersistentRelationTrial, TemporalPersistentState]]:
    return [
        make_episode(seed=seed, step=split_step(split, i) + m * 17, m=m)
        for i in range(count)
    ]


class ProductKeyRelationRouter:
    """Two 8-bit product-key subspaces with a linear relational query tower."""

    latent_dim = LATENT_DIM

    def __init__(self, *, seed: int, learning_rate: float = 0.012, soft_target_epsilon: float = 0.05) -> None:
        self.rng = random.Random(seed)
        self.learning_rate = learning_rate
        self.soft_target_epsilon = soft_target_epsilon
        self.query_dim = 4 * 3 * DIM + N_OPS + 1
        scale = 0.04
        self.wq = [
            [self.rng.uniform(-scale, scale) for _ in range(self.query_dim)]
            for _ in range(LATENT_DIM)
        ]
        self.bq = [0.0] * LATENT_DIM
        self.keys0 = [
            [self.rng.uniform(-scale, scale) for _ in range(LATENT_HALF)]
            for _ in range(1 << (DIM // 2))
        ]
        self.keys1 = [
            [self.rng.uniform(-scale, scale) for _ in range(LATENT_HALF)]
            for _ in range(1 << (DIM // 2))
        ]
        self.updates = 0

    @staticmethod
    def _linear(weights, bias, x):
        return tuple(
            sum(w * xv for w, xv in zip(row, x)) + bias[i]
            for i, row in enumerate(weights)
        )

    def _query_features(self, query: Query, state: TemporalPersistentState) -> list[float]:
        read = state.read(query)
        if not read.values:
            left = right = (0,) * DIM
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

    def _code(self, descriptor: Sequence[int], half: int) -> int:
        width = DIM // 2
        bits = descriptor[half * width : (half + 1) * width]
        code = 0
        for bit in bits:
            code = (code << 1) | int(bit)
        return code

    def encode_query(self, query, state):
        return self._linear(self.wq, self.bq, self._query_features(query, state))

    def encode_candidate(self, candidate):
        return tuple(self.keys0[self._code(candidate.descriptor, 0)] + self.keys1[self._code(candidate.descriptor, 1)])

    def candidate_embeddings(self, candidates):
        return {c.key: self.encode_candidate(c) for c in candidates}

    @staticmethod
    def score_embeddings(zq, zc):
        return sum(a * b for a, b in zip(zq, zc))

    def rank(self, query, state, candidates):
        qz = self.encode_query(query, state)
        scores = [self.score_embeddings(qz, self.encode_candidate(c)) for c in candidates]
        return sorted(range(len(candidates)), key=lambda i: (-scores[i], i))

    @staticmethod
    def _softmax(scores):
        mx = max(scores)
        exps = [math.exp(max(-60.0, min(60.0, s - mx))) for s in scores]
        total = sum(exps)
        return [x / total for x in exps]

    def _update_query(self, grad_q, qx):
        lr = self.learning_rate
        for r in range(LATENT_DIM):
            for j in range(len(qx)):
                self.wq[r][j] += lr * grad_q[r] * qx[j]
            self.bq[r] += lr * grad_q[r]

    def train_exhaustive(self, query, state, candidates, positive_index):
        qx = self._query_features(query, state)
        zq = list(self.encode_query(query, state))
        zc = [list(self.encode_candidate(c)) for c in candidates]
        probs = self._softmax([self.score_embeddings(zq, z) for z in zc])
        n = len(candidates)
        target = self.soft_target_epsilon / max(1, n - 1)
        targets = [target] * n
        targets[positive_index] = 1.0 - self.soft_target_epsilon
        grad_q = [0.0] * LATENT_DIM
        grad_k0 = {}
        grad_k1 = {}
        for i, (t, p) in enumerate(zip(targets, probs)):
            g = t - p
            for r in range(LATENT_DIM):
                grad_q[r] += g * zc[i][r]
            c = candidates[i]
            c0 = self._code(c.descriptor, 0)
            c1 = self._code(c.descriptor, 1)
            g0 = grad_k0.setdefault(c0, [0.0] * LATENT_HALF)
            g1 = grad_k1.setdefault(c1, [0.0] * LATENT_HALF)
            for j in range(LATENT_HALF):
                g0[j] += g * zq[j]
                g1[j] += g * zq[LATENT_HALF + j]
        self._update_query(grad_q, qx)
        lr = self.learning_rate
        for code, grad in grad_k0.items():
            for j in range(LATENT_HALF):
                self.keys0[code][j] += lr * grad[j]
        for code, grad in grad_k1.items():
            for j in range(LATENT_HALF):
                self.keys1[code][j] += lr * grad[j]
        self.updates += 1

    def learn_verified(self, query, state, candidates, positive_index, negative_index):
        if negative_index is None:
            return
        qx = self._query_features(query, state)
        zq = list(self.encode_query(query, state))
        zp = list(self.encode_candidate(candidates[positive_index]))
        zn = list(self.encode_candidate(candidates[negative_index]))
        diff = self.score_embeddings(zq, zp) - self.score_embeddings(zq, zn)
        gate = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, diff))))
        self._update_query([gate * (zp[r] - zn[r]) for r in range(LATENT_DIM)], qx)
        cp, cn = self._code(candidates[positive_index].descriptor, 0), self._code(candidates[negative_index].descriptor, 0)
        dp, dn = self._code(candidates[positive_index].descriptor, 1), self._code(candidates[negative_index].descriptor, 1)
        lr = self.learning_rate
        for j in range(LATENT_HALF):
            self.keys0[cp][j] += lr * gate * zq[j]
            self.keys1[dp][j] += lr * gate * zq[LATENT_HALF + j]
            self.keys0[cn][j] -= lr * gate * zq[j]
            self.keys1[dn][j] -= lr * gate * zq[LATENT_HALF + j]
        self.updates += 1


class AnalyticRelationalRouter(CDLPersistentRelationRouter):
    """Baseline learner with an explicit persisted-operation constant channel.

    The baseline has a,b,a*b features gated by the persisted operation but no
    operation-specific constant. AND/OR therefore need a constant term. This
    arm adds four persisted-op gate features so the exact Boolean signed
    relations are representable without consulting the corrupted public hint.
    """

    def __init__(self, *, seed: int, learning_rate: float = 0.012, soft_target_epsilon: float = 0.05) -> None:
        super().__init__(
            seed=seed,
            learning_rate=learning_rate,
            soft_target_epsilon=soft_target_epsilon,
        )
        self.query_dim = 4 * 3 * DIM + N_OPS + N_OPS + 1
        scale = 0.04
        self.wq = [
            [self.rng.uniform(-scale, scale) for _ in range(self.query_dim)]
            for _ in range(LATENT_DIM)
        ]
        self.bq = [0.0] * LATENT_DIM

    def _query_features(self, query: Query, state: TemporalPersistentState) -> list[float]:
        read = state.read(query)
        if not read.values:
            left = right = (0,) * DIM
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
        op_constants = [1.0 if op == state_op else 0.0 for op in range(N_OPS)]
        ctx = list(query.context[:N_OPS])
        ctx.extend([0.0] * (N_OPS - len(ctx)))
        return blocks + op_constants + ctx + [present]


class BitwiseLateInteractionRouter:
    """Independent per-bit query/candidate channels with summed interaction."""

    latent_dim = LATENT_DIM

    def __init__(self, *, seed: int, learning_rate: float = 0.012, soft_target_epsilon: float = 0.05) -> None:
        self.rng = random.Random(seed)
        self.learning_rate = learning_rate
        self.soft_target_epsilon = soft_target_epsilon
        self.w = [
            [self.rng.uniform(-0.04, 0.04) for _ in range(3 + N_OPS + 1)]
            for _ in range(DIM)
        ]
        self.updates = 0

    def _features(self, query, state):
        read = state.read(query)
        if not read.values:
            left = right = (0,) * DIM
            present = 0.0
        else:
            left, right, op = decode_state_value(read.values[0])
            present = 1.0
        ctx = list(query.context[:N_OPS])
        ctx.extend([0.0] * (N_OPS - len(ctx)))
        out = []
        for i in range(DIM):
            a = 2.0 * left[i] - 1.0
            b = 2.0 * right[i] - 1.0
            out.append([a, b, a * b] + ctx + [present])
        return out

    def encode_query(self, query, state):
        return tuple(sum(w * x for w, x in zip(self.w[i], feat)) for i, feat in enumerate(self._features(query, state)))

    def encode_candidate(self, candidate):
        return tuple(2.0 * int(x) - 1.0 for x in candidate.descriptor)

    def candidate_embeddings(self, candidates):
        return {c.key: self.encode_candidate(c) for c in candidates}

    @staticmethod
    def score_embeddings(zq, zc):
        return sum(a * b for a, b in zip(zq, zc))

    def rank(self, query, state, candidates):
        qz = self.encode_query(query, state)
        scores = [self.score_embeddings(qz, self.encode_candidate(c)) for c in candidates]
        return sorted(range(len(candidates)), key=lambda i: (-scores[i], i))

    @staticmethod
    def _softmax(scores):
        mx = max(scores)
        exps = [math.exp(max(-60.0, min(60.0, s - mx))) for s in scores]
        total = sum(exps)
        return [x / total for x in exps]

    def train_exhaustive(self, query, state, candidates, positive_index):
        feats = self._features(query, state)
        q = list(self.encode_query(query, state))
        zc = [list(self.encode_candidate(c)) for c in candidates]
        probs = self._softmax([self.score_embeddings(q, z) for z in zc])
        n = len(candidates)
        target = self.soft_target_epsilon / max(1, n - 1)
        targets = [target] * n
        targets[positive_index] = 1.0 - self.soft_target_epsilon
        dq = [0.0] * DIM
        for z, t, p in zip(zc, targets, probs):
            g = t - p
            for i in range(DIM):
                dq[i] += g * z[i]
        lr = self.learning_rate
        for i in range(DIM):
            for j in range(len(feats[i])):
                self.w[i][j] += lr * dq[i] * feats[i][j]
        self.updates += 1

    def learn_verified(self, query, state, candidates, positive_index, negative_index):
        if negative_index is None:
            return
        feats = self._features(query, state)
        q = list(self.encode_query(query, state))
        p = self.encode_candidate(candidates[positive_index])
        n = self.encode_candidate(candidates[negative_index])
        diff = self.score_embeddings(q, p) - self.score_embeddings(q, n)
        gate = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, diff))))
        lr = self.learning_rate
        for i in range(DIM):
            delta = gate * (p[i] - n[i])
            for j in range(len(feats[i])):
                self.w[i][j] += lr * delta * feats[i][j]
        self.updates += 1


def analytic_relational_initialize(router) -> None:
    """Initialize an exact signed-bit relational solution.

    XOR  = -a*b
    XNOR = +a*b
    AND  = (a+b+a*b-1)/2
    OR   = (a+b-a*b+1)/2

    The explicit persisted-operation constant channel supplies the +/-1/2
    offset required by AND/OR without using the corrupted public hint.
    """
    if not isinstance(router, AnalyticRelationalRouter):
        raise TypeError("analytic initialization requires AnalyticRelationalRouter")
    router.wq = [[0.0] * router.query_dim for _ in range(LATENT_DIM)]
    router.bq = [0.0] * LATENT_DIM
    router.wc = [[0.0] * DIM for _ in range(LATENT_DIM)]
    router.bc = [0.0] * LATENT_DIM
    const_base = 4 * 3 * DIM
    for i in range(DIM):
        router.wc[i][i] = 1.0
        for op in range(N_OPS):
            base = op * 3 * DIM
            ai = base + i
            bi = base + DIM + i
            pi = base + 2 * DIM + i
            gate_i = const_base + op
            if op == 0:
                router.wq[i][pi] = -1.0
            elif op == 1:
                router.wq[i][pi] = 1.0
            elif op == 2:
                router.wq[i][ai] = 0.5
                router.wq[i][bi] = 0.5
                router.wq[i][pi] = 0.5
                router.wq[i][gate_i] = -0.5
            else:
                router.wq[i][ai] = 0.5
                router.wq[i][bi] = 0.5
                router.wq[i][pi] = -0.5
                router.wq[i][gate_i] = 0.5

def train_exhaustive(router, seed: int, steps: int) -> None:
    levels = (64, 128, 256)
    for step in range(steps):
        m = levels[step % len(levels)]
        trial, state = make_episode(seed=seed, step=step, m=m)
        router.train_exhaustive(trial.query, state, trial.candidates, trial.target_index)


def hard_negative_distill(router, seed: int, steps: int) -> int:
    verifier = PersistentRelationCASMVerifier()
    harvested = 0
    for step in range(steps):
        m = (64, 128, 256, 512)[step % 4]
        trial, state = make_episode(seed=seed, step=SPLIT_OFFSETS["distill"] + 1000 + step, m=m)
        index = ORLSHIndex(
            latent_dim=LATENT_DIM,
            bits=max(1, math.ceil(math.log2(m))),
            tables=DISTILL_TABLES,
            cap=DISTILL_TABLES,
            seed=seed * 100_003 + step,
        )
        index.build(router.candidate_embeddings(trial.candidates))
        qz = router.encode_query(trial.query, state)
        lookup = index.lookup(
            qz,
            k=min(DISTILL_K, m),
            score=lambda key: router.score_embeddings(qz, index.embeddings[key]),
        )
        positions = {c.key: i for i, c in enumerate(trial.candidates)}
        admitted = [positions[k] for k in lookup.addresses]
        negative = None
        for idx in admitted[:4]:
            rec = verifier.execute_and_verify(trial, state, idx)
            if not rec.verified:
                negative = idx
                break
        if negative is None and admitted:
            negative = admitted[0]
        if negative is not None and negative != trial.target_index:
            router.learn_verified(trial.query, state, trial.candidates, trial.target_index, negative)
            harvested += 1
    return harvested


def build_router(arm: str, seed: int):
    if arm == "analytic_init":
        router = AnalyticRelationalRouter(
            seed=seed, learning_rate=0.012, soft_target_epsilon=0.05
        )
        analytic_relational_initialize(router)
        train_exhaustive(router, seed, TRAIN_STEPS)
        return router

    if arm in {"control", "outcome_distill"}:
        router = CDLPersistentRelationRouter(
            seed=seed, learning_rate=0.012, soft_target_epsilon=0.05
        )
        if arm == "outcome_distill":
            train_exhaustive(router, seed, BASELINE_STEPS)
            hard_negative_distill(router, seed, DISTILL_STEPS)
        else:
            train_exhaustive(router, seed, TRAIN_STEPS)
        return router

    if arm in {"product_key", "product_key_distill"}:
        router = ProductKeyRelationRouter(seed=seed)
        train_exhaustive(router, seed, BASELINE_STEPS)
        if arm == "product_key_distill":
            hard_negative_distill(router, seed, DISTILL_STEPS)
        else:
            train_exhaustive(router, seed, DISTILL_STEPS)
        return router

    if arm in {"late_interaction", "late_interaction_distill"}:
        router = BitwiseLateInteractionRouter(seed=seed)
        train_exhaustive(router, seed, BASELINE_STEPS)
        if arm == "late_interaction_distill":
            hard_negative_distill(router, seed, DISTILL_STEPS)
        else:
            train_exhaustive(router, seed, DISTILL_STEPS)
        return router

    raise ValueError(arm)

def dense_rank(router, trial, state):
    qz = router.encode_query(trial.query, state)
    scores = [router.score_embeddings(qz, router.encode_candidate(c)) for c in trial.candidates]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order.index(trial.target_index) + 1, scores


def exact_relation_hash_lookup(trial: PersistentRelationTrial, state: TemporalPersistentState) -> tuple[int, int]:
    read = state.read(trial.query)
    if not read.values:
        return -1, 0
    left, right, op = decode_state_value(read.values[0])
    by_descriptor = {c.descriptor: i for i, c in enumerate(trial.candidates)}
    matches = []
    for candidate_op in range(N_OPS):
        matches.append(by_descriptor.get(apply_relation(left, right, candidate_op)))
    selected = matches[op]
    return (-1 if selected is None else 1), N_OPS


def operand_hamming_rank(trial, state) -> int:
    read = state.read(trial.query)
    if not read.values:
        return len(trial.candidates) // 2
    left, right, _ = decode_state_value(read.values[0])
    def dist(desc, operand):
        return sum(int(a != b) for a, b in zip(desc, operand))
    scores = [min(dist(c.descriptor, left), dist(c.descriptor, right)) for c in trial.candidates]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return order.index(trial.target_index) + 1


def rank_op(trial, state) -> int:
    read = state.read(trial.query)
    return decode_state_value(read.values[0])[2] if read.values else 0


def bootstrap_ci(values: Sequence[float], statistic: Callable[[Sequence[float]], float], seed: int) -> tuple[float, float]:
    if len(values) < 2:
        x = float(statistic(values))
        return x, x
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(float(statistic(sample)))
    estimates.sort()
    return estimates[max(0, int(0.025 * len(estimates)) - 1)], estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]


def rank_summary(values: Sequence[int], seed: int) -> dict[str, object]:
    vals = [float(x) for x in values]
    p90_stat = lambda xs: empirical_quantile([int(x) for x in xs], 0.90)
    mean_ci = bootstrap_ci(vals, statistics.fmean, seed)
    p90_ci = bootstrap_ci(vals, p90_stat, seed + 1)
    return {
        "n": len(values),
        "mean_rank": statistics.fmean(values),
        "P90_rank": empirical_quantile(values, 0.90),
        "top1": sum(r == 1 for r in values) / len(values),
        "mean_rank_ci95": [mean_ci[0], mean_ci[1]],
        "P90_rank_ci95": [p90_ci[0], p90_ci[1]],
    }


def empirical_l90(router, seed: int, m: int, calibration) -> tuple[int, list[dict[str, float]]]:
    bits = max(1, math.ceil(math.log2(m)))
    counts = [0] * LSH_MAX_TABLES
    total = len(calibration)
    for trial_idx, (trial, state) in enumerate(calibration):
        index = ORLSHIndex(
            latent_dim=LATENT_DIM,
            bits=bits,
            tables=LSH_MAX_TABLES,
            cap=LSH_MAX_TABLES,
            seed=seed * 1_000_003 + m * 97 + trial_idx,
        )
        qz = router.encode_query(trial.query, state)
        tz = router.encode_candidate(trial.candidates[trial.target_index])
        found = False
        for t in range(LSH_MAX_TABLES):
            found = found or (index.hash(qz, t) == index.hash(tz, t))
            if found:
                counts[t] += 1
    curve = []
    l90 = LSH_MAX_TABLES
    for i, c in enumerate(counts, 1):
        recall = c / max(1, total)
        curve.append({"L": i, "admission_recall": recall})
        if recall >= 0.90 and l90 == LSH_MAX_TABLES:
            l90 = i
    return l90, curve


def sparse_run(router, trial, state, *, tables: int, k: int, budget: int, seed: int):
    index = ORLSHIndex(
        latent_dim=LATENT_DIM,
        bits=max(1, math.ceil(math.log2(len(trial.candidates)))),
        tables=tables,
        cap=tables,
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
    success = False
    executed = 0
    for idx in admitted[:budget]:
        executed += 1
        if verifier.execute_and_verify(trial, state, idx).verified:
            success = True
            break
    return {
        "success": success,
        "target_admitted": trial.target_index in admitted,
        "executed": executed,
        "admitted_rank": admitted.index(trial.target_index) + 1 if trial.target_index in admitted else None,
        "rerank_count": lookup.rerank_count,
    }


def adaptive_budget(scores: Sequence[float], order: Sequence[int], cap: int = 16) -> int:
    if len(order) < 4:
        return min(cap, len(order))
    ordered = [scores[i] for i in order]
    std = statistics.pstdev(ordered)
    if std <= 1e-12:
        return min(cap, len(order))
    margin = (ordered[0] - ordered[3]) / std
    if margin >= 1.0:
        return min(1, cap, len(order))
    if margin >= 0.5:
        return min(4, cap, len(order))
    if margin >= 0.25:
        return min(8, cap, len(order))
    return min(16, cap, len(order))


def sparse_adaptive_run(router, trial, state, *, tables: int, k: int, seed: int):
    index = ORLSHIndex(
        latent_dim=LATENT_DIM,
        bits=max(1, math.ceil(math.log2(len(trial.candidates)))),
        tables=tables,
        cap=tables,
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
    scores = [router.score_embeddings(qz, router.encode_candidate(c)) for c in trial.candidates]
    budget = adaptive_budget(scores, admitted, 16)
    verifier = PersistentRelationCASMVerifier()
    executed = 0
    success = False
    for idx in admitted[:budget]:
        executed += 1
        if verifier.execute_and_verify(trial, state, idx).verified:
            success = True
            break
    return {"success": success, "budget": budget, "executed": executed, "target_admitted": trial.target_index in admitted}


def fit_power(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(math.log(float(x)), math.log(max(float(y), 1e-9))) for x, y in zip(xs, ys) if y > 0]
    if len(pairs) < 2:
        return 0.0
    mx = statistics.fmean(x for x, _ in pairs)
    my = statistics.fmean(y for _, y in pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    return sum((x - mx) * (y - my) for x, y in pairs) / den if den else 0.0


def evaluate_baseline(seed: int, router) -> dict[str, object]:
    dense = []
    sparse = []
    per_op = {}
    split_checks = {}
    mean_exec = []
    for m in M_LEVELS:
        calibration = trials_for_split(seed, m, "calibration", CALIBRATION_TRIALS)
        heldout = trials_for_split(seed, m, "heldout", HELDOUT_TRIALS)
        split_checks[str(m)] = verify_split_disjointness(seed, m)
        cal_ranks = [dense_rank(router, *x)[0] for x in calibration]
        k90 = empirical_quantile(cal_ranks, 0.90)
        l90, lcurve = empirical_l90(router, seed, m, calibration)
        eval_ranks = []
        op_rows = {op: [] for op in range(N_OPS)}
        for trial, state in heldout:
            rank, _ = dense_rank(router, trial, state)
            eval_ranks.append(rank)
            op_rows[rank_op(trial, state)].append(rank)
        summary = rank_summary(eval_ranks, BOOTSTRAP_SEED + seed * 100 + m)
        summary.update({
            "M": m,
            "K90_calibration": k90,
            "L90_calibration": l90,
            "mean_expected_dense_executions": statistics.fmean(eval_ranks),
        })
        dense.append(summary)
        mean_exec.append(statistics.fmean(eval_ranks))
        per_op[str(m)] = {
            OPS[op]: {
                "n": len(v),
                "top1": sum(x == 1 for x in v) / len(v) if v else 0.0,
                "mean_rank": statistics.fmean(v) if v else None,
            }
            for op, v in op_rows.items()
        }
        budgets = []
        for budget in BUDGETS:
            rows = [
                sparse_run(router, trial, state, tables=l90, k=k90, budget=budget, seed=seed * 700_003 + m * 31)
                for trial, state in heldout
            ]
            admitted_ranks = [r["admitted_rank"] for r in rows if r["admitted_rank"] is not None]
            budgets.append({
                "budget": budget,
                "success": statistics.fmean(r["success"] for r in rows),
                "mean_executed": statistics.fmean(r["executed"] for r in rows),
                "admission": statistics.fmean(r["target_admitted"] for r in rows),
                "mean_admitted_rank": statistics.fmean(admitted_ranks) if admitted_ranks else None,
            })
        adaptive = [
            sparse_adaptive_run(router, trial, state, tables=l90, k=k90, seed=seed * 900_001 + m)
            for trial, state in heldout
        ]
        sparse.append({
            "M": m,
            "K90": k90,
            "L90": l90,
            "L_curve": lcurve,
            "budgets": budgets,
            "adaptive": {
                "success": statistics.fmean(r["success"] for r in adaptive),
                "mean_executed": statistics.fmean(r["executed"] for r in adaptive),
                "mean_budget": statistics.fmean(r["budget"] for r in adaptive),
                "admission": statistics.fmean(r["target_admitted"] for r in adaptive),
            },
        })
    return {
        "dense": dense,
        "sparse": sparse,
        "per_op": per_op,
        "split_checks": split_checks,
        "dense_expected_execution_fit_gamma": fit_power(M_LEVELS, mean_exec),
        "exact_reference": {
            "capability_top1": 1.0,
            "query_relation_computations": N_OPS,
            "query_hash_lookups": N_OPS,
            "runtime_in_M": "O(1) after descriptor-to-index hash tables are built",
        },
    }


def stable_arm_seed(arm: str) -> int:
    return int.from_bytes(hashlib.sha256(arm.encode()).digest()[:4], "big")


def evaluate_representation_arm(seed: int, arm: str, router) -> dict[str, object]:
    out = {}
    for m in REP_LEVELS:
        calibration = trials_for_split(seed, m, "calibration", CALIBRATION_TRIALS)
        heldout = trials_for_split(seed, m, "heldout", HELDOUT_TRIALS)
        cal_ranks = [dense_rank(router, *x)[0] for x in calibration]
        k90 = empirical_quantile(cal_ranks, 0.90)
        l90, _ = empirical_l90(router, seed, m, calibration)
        ranks = [dense_rank(router, trial, state)[0] for trial, state in heldout]
        summary = rank_summary(ranks, BOOTSTRAP_SEED + seed * 100 + stable_arm_seed(arm) % 1000)
        budgets = []
        for budget in BUDGETS:
            rows = [
                sparse_run(router, trial, state, tables=l90, k=k90, budget=budget, seed=seed * 700_003 + m * 31)
                for trial, state in heldout
            ]
            budgets.append({
                "budget": budget,
                "success": statistics.fmean(r["success"] for r in rows),
                "mean_executed": statistics.fmean(r["executed"] for r in rows),
                "admission": statistics.fmean(r["target_admitted"] for r in rows),
            })
        adaptive = [
            sparse_adaptive_run(router, trial, state, tables=l90, k=k90, seed=seed * 900_001 + m)
            for trial, state in heldout
        ]
        out[str(m)] = {
            **summary,
            "K90": k90,
            "L90": l90,
            "budgets": budgets,
            "adaptive": {
                "success": statistics.fmean(r["success"] for r in adaptive),
                "mean_executed": statistics.fmean(r["executed"] for r in adaptive),
                "mean_budget": statistics.fmean(r["budget"] for r in adaptive),
            },
        }
    return out


def main_measure() -> dict[str, object]:
    arms = (
        "control",
        "analytic_init",
        "product_key",
        "late_interaction",
        "outcome_distill",
        "product_key_distill",
        "late_interaction_distill",
    )
    result = {
        "protocol": {
            "id": "TACOSM-C5-COMPOSITION-BUDGET-AUDIT-001",
            "status": "measured",
            "M_levels": list(M_LEVELS),
            "representation_M": list(REP_LEVELS),
            "seeds": list(SEEDS),
            "training_updates": TRAIN_STEPS,
            "calibration_trials": CALIBRATION_TRIALS,
            "heldout_trials": HELDOUT_TRIALS,
            "budgets": list(BUDGETS),
        },
        "diagnostics": {
            "baseline_query_tower_hidden_nonlinearity": False,
            "baseline_explicit_pairwise_feature": True,
            "baseline_relation_basis": ["a", "b", "a*b"],
            "xor_xnor_have_explicit_pairwise_basis": True,
            "selection_rule": "all registered arms reported; no held-out arm selection",
        },
        "architectures": {
            "control": "existing CDLPersistentRelationRouter with 1200 exhaustive updates",
            "analytic_init": "exact Boolean relation initialization then 1200 exhaustive updates",
            "product_key": "two 8-bit product-key subspaces with 8-D subkeys",
            "late_interaction": "independent per-bit interaction channels",
            "outcome_distill": "800 exhaustive + 400 CASM-failure hard-negative updates",
            "product_key_distill": "product-key, 800 exhaustive + 400 CASM-failure hard-negative updates",
            "late_interaction_distill": "per-bit, 800 exhaustive + 400 CASM-failure hard-negative updates",
        },
        "arms": {},
    }

    for seed in SEEDS:
        result["arms"].setdefault("control", {})[str(seed)] = evaluate_baseline(seed, build_router("control", seed))

    for arm in arms:
        if arm == "control":
            continue
        result["arms"].setdefault(arm, {})
        for seed in SEEDS:
            result["arms"][arm][str(seed)] = evaluate_representation_arm(seed, arm, build_router(arm, seed))

    reset_checks = {}
    for arm in arms:
        reset_checks[arm] = {}
        for seed in SEEDS:
            router = build_router(arm, seed)
            heldout = trials_for_split(seed, 1024, "heldout", HELDOUT_TRIALS)
            persistent = []
            reset = []
            per_op = {name: [] for name in OPS}
            for trial, state in heldout:
                p_rank, _ = dense_rank(router, trial, state)
                persistent.append(p_rank)
                read = state.read(trial.query)
                state.clear()
                qz = router.encode_query(trial.query, state)
                scores = [router.score_embeddings(qz, router.encode_candidate(c)) for c in trial.candidates]
                order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                reset.append(order.index(trial.target_index) + 1)
                if read.values:
                    per_op[OPS[decode_state_value(read.values[0])[2]]].append(p_rank)
            reset_checks[arm][str(seed)] = {
                "persistent_top1": sum(x == 1 for x in persistent) / len(persistent),
                "reset_top1": sum(x == 1 for x in reset) / len(reset),
                "persistent_P90": empirical_quantile(persistent, 0.90),
                "reset_P90": empirical_quantile(reset, 0.90),
                "per_op_top1": {k: sum(x == 1 for x in v) / len(v) if v else None for k, v in per_op.items()},
            }
    result["persistent_reset"] = reset_checks

    pooled_l90 = []
    pooled_exec = []
    pooled_p90 = []
    pooled_k90 = []
    for idx, m in enumerate(M_LEVELS):
        rows = [result["arms"]["control"][str(seed)]["dense"][idx] for seed in SEEDS]
        pooled_l90.append(statistics.fmean(r["L90_calibration"] for r in rows))
        pooled_exec.append(statistics.fmean(r["mean_expected_dense_executions"] for r in rows))
        pooled_p90.append(statistics.fmean(r["P90_rank"] for r in rows))
        pooled_k90.append(statistics.fmean(r["K90_calibration"] for r in rows))
    result["control_scaling"] = {
        "pooled_L90": [{"M": m, "L90": v} for m, v in zip(M_LEVELS, pooled_l90)],
        "L90_fit_gamma": fit_power(M_LEVELS, pooled_l90),
        "pooled_mean_expected_dense_executions": [{"M": m, "mean_executions": v} for m, v in zip(M_LEVELS, pooled_exec)],
        "mean_execution_fit_gamma": fit_power(M_LEVELS, pooled_exec),
        "pooled_P90": [{"M": m, "P90": v} for m, v in zip(M_LEVELS, pooled_p90)],
        "pooled_K90": [{"M": m, "K90": v} for m, v in zip(M_LEVELS, pooled_k90)],
        "L90_definition": "smallest integer L in [1,128] with >=0.90 calibration target admission",
    }

    bootstrap = {}
    for arm in arms:
        bootstrap[arm] = {}
        for seed in SEEDS:
            row = result["arms"][arm][str(seed)]["dense"][4] if arm == "control" else result["arms"][arm][str(seed)]["1024"]
            bootstrap[arm][str(seed)] = {
                "mean_rank": row["mean_rank"],
                "mean_rank_ci95": row["mean_rank_ci95"],
                "P90_rank": row["P90_rank"],
                "P90_rank_ci95": row["P90_rank_ci95"],
                "top1": row["top1"],
            }
    result["bootstrap_M1024"] = bootstrap
    result["scope"] = {
        "semantic_language_claim": False,
        "complexity_theorem": False,
        "causal_learning_claim": False,
        "exact_reference_is_task_semantic_upper_bound": True,
        "heldout_selection": False,
        "calibration_heldout_disjoint_verified": all(
            result["arms"]["control"][str(s)]["split_checks"][str(m)]["intersection_count"] == 0
            for s in SEEDS for m in M_LEVELS
        ),
    }
    return result
