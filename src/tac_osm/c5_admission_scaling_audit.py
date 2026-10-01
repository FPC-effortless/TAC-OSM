"""C5 admission/scaling audit.

This phase follows TACOSM-C5-NOISY-FULL-PHASE-001 without changing its
representation learner. It answers two audit questions before any new
representation intervention:

1. Does the dense-CDL rank curve continue toward linear growth at
   M=2048, 4096, and 8192?
2. Can the OR-LSH admission layer recover the dense K90 coverage when the
   number of tables is swept explicitly with no table-cap intervention?

The main workload remains the registered synthetic binary task. Therefore
the module is an audit of routing/admission mechanics, not a semantic
benchmark.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence

from . import Candidate
from .c5_noisy_full_phase import (
    CDLDenseNoisyRouter,
    CODEBOOK,
    DIM,
    NoisyClassTrial,
    build_population,
    make_trial,
    estimate_p1_p2,
    ORLSHIndex,
)
from .executor import StructuralExecutor, ExecutorConfig
from .c5_noisy_full_phase import NoisyCASMExecutor


DENSE_M_LEVELS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
TRAIN_M_LEVELS = (64, 128, 256)
LSH_SWEEP_M = 1024
LSH_TABLE_SWEEP = (1, 2, 4, 8, 10, 12, 16, 24, 32, 48, 64)
LSH_TABLE_CAP = 128
SEEDS = (0, 1, 2, 3, 4)
TRAIN_STEPS = 800
CALIBRATION_TRIALS = 64
HELDOUT_TRIALS = 64
P1_SAMPLES = 4
LATENT_DIM = 16
ALPHA = 0.10


@dataclass(frozen=True)
class AuditConfig:
    dense_m_levels: tuple[int, ...] = DENSE_M_LEVELS
    train_m_levels: tuple[int, ...] = TRAIN_M_LEVELS
    sweep_m: int = LSH_SWEEP_M
    lsh_tables: tuple[int, ...] = LSH_TABLE_SWEEP
    lsh_cap: int = LSH_TABLE_CAP
    seeds: tuple[int, ...] = SEEDS
    train_steps: int = TRAIN_STEPS
    calibration_trials: int = CALIBRATION_TRIALS
    heldout_trials: int = HELDOUT_TRIALS
    p1_samples: int = P1_SAMPLES

    def __post_init__(self) -> None:
        for m in self.dense_m_levels:
            if m < len(CODEBOOK) or m % len(CODEBOOK):
                raise ValueError(f"M={m} must be divisible by {len(CODEBOOK)}")
        if any(l < 1 for l in self.lsh_tables):
            raise ValueError("all LSH table counts must be >= 1")
        if max(self.lsh_tables) >= self.lsh_cap:
            raise ValueError("LSH sweep must remain strictly below the cap")
        if self.sweep_m not in self.dense_m_levels:
            raise ValueError("sweep_m must be present in dense_m_levels")


def build_population_extended(seed: int, m: int) -> tuple[Candidate, ...]:
    """Build the same equal-repeat population for M beyond C5's 1024 cap."""
    if m < len(CODEBOOK) or m % len(CODEBOOK):
        raise ValueError("M must be divisible by the semantic class count")
    copies = m // len(CODEBOOK)
    rng = random.Random(seed * 1_000_003 + m * 97)
    candidates: list[Candidate] = []
    for class_id, descriptor in enumerate(CODEBOOK):
        for copy_id in range(copies):
            candidates.append(
                Candidate(
                    key=f"audit-s{seed}-m{m}-c{class_id:02d}-v{copy_id:03d}",
                    descriptor=descriptor,
                    action=len(candidates),
                    provenance="c5_audit_repeated_class",
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


def audit_trial(seed: int, step: int, m: int) -> NoisyClassTrial:
    return make_trial(
        seed=seed,
        step=step,
        candidates=build_population_extended(seed, m),
    )


def train_router(seed: int, config: AuditConfig) -> CDLDenseNoisyRouter:
    """Mirror the leakage-corrected C5 training distribution exactly."""
    router = CDLDenseNoisyRouter(
        seed=seed,
        learning_rate=0.012,
        soft_target_epsilon=0.05,
    )
    executor = NoisyCASMExecutor(dim=DIM)
    for step in range(config.train_steps):
        m = config.train_m_levels[step % len(config.train_m_levels)]
        trial = audit_trial(seed + step * 11, step, m)
        successes = []
        valid = trial.valid_indices
        for i in range(len(trial.candidates)):
            _, value = executor.execute(trial, trial.candidates[i])
            # Keep the same hidden-outcome construction used in C5.
            if i in valid:
                successes.append(i)
        router.train_exhaustive(trial, successes)
    return router


def empirical_quantile(values: Sequence[int], probability: float) -> int:
    ordered = sorted(int(v) for v in values)
    idx = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[idx]


def fit_power(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [
        (math.log(float(x)), math.log(max(float(y), 1e-9)))
        for x, y in zip(xs, ys)
    ]
    if len(pairs) < 2:
        return 0.0
    mx = statistics.fmean(x for x, _ in pairs)
    my = statistics.fmean(y for _, y in pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    return sum((x - mx) * (y - my) for x, y in pairs) / den if den else 0.0


def _class_ids(candidates: Sequence[Candidate]) -> list[int]:
    lookup = {descriptor: i for i, descriptor in enumerate(CODEBOOK)}
    return [lookup[c.descriptor] for c in candidates]


def exact_best_valid_rank_from_class_scores(
    candidates: Sequence[Candidate],
    target_class: tuple[int, ...],
    class_scores: Sequence[float],
) -> int:
    """Exact candidate-list rank without scoring duplicate class copies.

    The candidate population repeats each descriptor. All copies of a class
    therefore share the same dense score. We count all strictly better class
    copies and then reproduce the candidate-index tie break exactly.
    """
    ids = _class_ids(candidates)
    target_id = {descriptor: i for i, descriptor in enumerate(CODEBOOK)}[target_class]
    target_score = class_scores[target_id]
    copies = len(candidates) // len(CODEBOOK)
    better_classes = sum(score > target_score for i, score in enumerate(class_scores) if i != target_id)
    better = better_classes * copies
    target_first = next(i for i, class_id in enumerate(ids) if class_id == target_id)
    tied_before = sum(
        1
        for i in range(target_first)
        if class_scores[ids[i]] == target_score
    )
    return better + tied_before + 1


def dense_rank(router: CDLDenseNoisyRouter, trial: NoisyClassTrial) -> int:
    empty = __import__(
        "tac_osm.temporal", fromlist=["TemporalPersistentState"]
    ).TemporalPersistentState()
    qz = router.inner.encode_query(trial.noisy_query, empty)
    reps = tuple(
        Candidate(
            key=f"audit-class-{i}",
            descriptor=descriptor,
            action=i,
            provenance="c5_audit_class_rep",
        )
        for i, descriptor in enumerate(CODEBOOK)
    )
    embeddings = router.inner.candidate_embeddings(reps)
    class_scores = [
        router.inner.score_embeddings(qz, embeddings[c.key])
        for c in reps
    ]
    return exact_best_valid_rank_from_class_scores(
        trial.candidates, trial.target_class, class_scores
    )


def dense_scaling_for_seed(
    router: CDLDenseNoisyRouter,
    seed: int,
    config: AuditConfig,
) -> dict[str, object]:
    per_m = []
    for m in config.dense_m_levels:
        ranks = [
            dense_rank(
                router,
                audit_trial(seed + 100, step=9000 + i, m=m),
            )
            for i in range(config.heldout_trials)
        ]
        p90 = empirical_quantile(ranks, 0.90)
        per_m.append(
            {
                "M": m,
                "mean_best_valid_rank": statistics.fmean(ranks),
                "P90_best_valid_rank": p90,
                "Top1_valid": sum(r == 1 for r in ranks) / len(ranks),
            }
        )
    return {"seed": seed, "by_m": per_m}


def theoretical_l90(p1: float, bits: int) -> int:
    """Smallest integer L with iid collision approximation >= 90%."""
    collision = p1 ** bits
    if collision <= 0.0:
        return 1
    return max(1, math.ceil(math.log(0.10) / math.log(1.0 - collision)))


def prefix_lookup(
    index: ORLSHIndex,
    query_vector: Sequence[float],
    *,
    tables: int,
    k: int,
    score,
) -> tuple[tuple[str, ...], int, int]:
    """Lookup through a prefix of a single max-table LSH family."""
    if tables < 1 or tables > index.actual_tables:
        raise ValueError("invalid table prefix")
    addresses: set[str] = set()
    for t in range(tables):
        code = index.hash(query_vector, t)
        addresses.update(index.buckets[t].get(code, ()))
    ordered = sorted(
        ((score(key), key) for key in addresses),
        key=lambda item: (-item[0], item[1]),
    )
    return (
        tuple(key for _, key in ordered[:k]),
        len(addresses),
        tables * index.bits * index.latent_dim,
    )


def lsh_sweep_for_seed(
    router: CDLDenseNoisyRouter,
    seed: int,
    config: AuditConfig,
) -> dict[str, object]:
    m = config.sweep_m
    calibration = [
        audit_trial(seed + 7, 7000 + i, m) for i in range(config.calibration_trials)
    ]
    heldout = [
        audit_trial(seed + 107, 9000 + i, m) for i in range(config.heldout_trials)
    ]
    ranks = [dense_rank(router, trial) for trial in calibration]
    k90 = empirical_quantile(ranks, 0.90)

    bits = max(1, math.ceil(math.log2(m)))
    probe = ORLSHIndex(
        latent_dim=LATENT_DIM,
        bits=bits,
        tables=1,
        cap=config.lsh_cap,
        seed=seed * 1009 + m,
    )
    probe.build(
        router.inner.candidate_embeddings(
            build_population_extended(seed, m)
        )
    )
    p1s: list[float] = []
    p2s: list[float] = []
    for trial in calibration[: min(32, len(calibration))]:
        p1, p2 = estimate_p1_p2(
            router,
            probe,
            trial,
            samples=config.p1_samples,
        )
        p1s.append(p1)
        p2s.append(p2)
    p1 = statistics.fmean(p1s)
    p2 = statistics.fmean(p2s)
    l90_theory = theoretical_l90(p1, bits)

    index = ORLSHIndex(
        latent_dim=LATENT_DIM,
        bits=bits,
        tables=max(config.lsh_tables),
        cap=config.lsh_cap,
        seed=seed * 9176 + m,
    )
    index.build(
        router.inner.candidate_embeddings(heldout[0].candidates)
    )
    position = {c.key: i for i, c in enumerate(heldout[0].candidates)}
    # Build positions independently because every held-out trial has the same
    # candidate population by construction.
    results = []
    for tables in config.lsh_tables:
        admitted: list[bool] = []
        rerank_counts: list[int] = []
        routing_ops: list[int] = []
        for trial in heldout:
            empty = __import__(
                "tac_osm.temporal", fromlist=["TemporalPersistentState"]
            ).TemporalPersistentState()
            qz = router.inner.encode_query(trial.noisy_query, empty)
            addresses, rerank_count, hash_ops = prefix_lookup(
                index,
                qz,
                tables=tables,
                k=min(k90, m),
                score=lambda key, qz=qz: router.inner.score_embeddings(
                    qz, index.embeddings[key]
                ),
            )
            admitted_indices = {
                position[key]
                for key in addresses
            }
            admitted.append(bool(admitted_indices.intersection(trial.valid_indices)))
            rerank_counts.append(rerank_count)
            routing_ops.append(hash_ops + rerank_count * LATENT_DIM)
        results.append(
            {
                "tables": tables,
                "admission_recall": statistics.fmean(admitted),
                "rerank_fraction": statistics.fmean(rerank_counts) / m,
                "mean_rerank_count": statistics.fmean(rerank_counts),
                "mean_routing_ops": statistics.fmean(routing_ops),
                "routing_fraction": statistics.fmean(routing_ops) / m,
                "cap_bound": tables > index.actual_tables,
                "is_theoretical_L90": tables == l90_theory,
            }
        )
    return {
        "seed": seed,
        "M": m,
        "bits": bits,
        "K90": k90,
        "p1": p1,
        "p2": p2,
        "theoretical_L90": l90_theory,
        "index_actual_tables": index.actual_tables,
        "results": results,
    }


def aggregate_dense(per_seed: Sequence[dict[str, object]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for i, m in enumerate(DENSE_M_LEVELS):
        rows = [seed["by_m"][i] for seed in per_seed]
        p90 = statistics.fmean(float(r["P90_best_valid_rank"]) for r in rows)
        mean_rank = statistics.fmean(float(r["mean_best_valid_rank"]) for r in rows)
        top1 = statistics.fmean(float(r["Top1_valid"]) for r in rows)
        out.append(
            {
                "M": float(m),
                "mean_best_valid_rank": mean_rank,
                "P90_best_valid_rank": p90,
                "Top1_valid": top1,
            }
        )
    return out


def local_slopes(dense: Sequence[dict[str, float]]) -> list[dict[str, float]]:
    out = []
    for prev, cur in zip(dense, dense[1:]):
        prev_rank = float(prev["P90_best_valid_rank"])
        cur_rank = float(cur["P90_best_valid_rank"])
        slope = (
            math.log(cur_rank / prev_rank) / math.log(float(cur["M"]) / float(prev["M"]))
            if prev_rank > 0 and cur_rank > 0
            else 0.0
        )
        out.append(
            {
                "M_from": float(prev["M"]),
                "M_to": float(cur["M"]),
                "local_P90_exponent": slope,
            }
        )
    return out
