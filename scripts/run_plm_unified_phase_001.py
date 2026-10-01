#!/usr/bin/env python3
"""TACOSM-PLM-UNIFIED-PHASE-001 integration benchmark.

The benchmark continues C5 from PR #61 and does not re-select a C5
representation winner. Existing C5 evidence is recorded as fixed input.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.plm_unified import BinaryCDL, PLMConfig, StateKind, build_operator_pool
from tac_osm.plm_unified import generate_episode, new_plm_for_records

EXPERIMENT_ID = "TACOSM-PLM-UNIFIED-PHASE-001"
H_LEVELS = (64, 256, 1024, 4096, 16384)
SEEDS = (0, 1, 2, 3, 4)
TRAIN_EPISODES = 600
KEY_DIM = 12
QUERY_NOISE = 0.10
BOUNDED_STATE_CAPACITY = 256


def train_router(seed: int) -> BinaryCDL:
    cfg = PLMConfig(slow_lr=0.01, fast_lr=0.10, seed=seed)
    router = BinaryCDL(KEY_DIM, config=cfg)
    rng = random.Random(seed)
    for _ in range(TRAIN_EPISODES):
        task, records = generate_episode(
            rng=rng, history=32, records_target=33,
            key_dim=KEY_DIM, noise=QUERY_NOISE,
        )
        ranked = router.rank(task.query_bits, records)
        selected = next(r for r in records if r.record_id == ranked[0].record_id)
        target = next(r for r in records if r.record_id == task.target_id)
        router.update(
            task.query_bits, selected,
            1.0 if selected.record_id == target.record_id else -1.0,
            correct_record=target,
        )
    return router


def rank_target(router, task, records):
    ranked = router.rank(task.query_bits, records)
    ids = [x.record_id for x in ranked]
    return ids.index(task.target_id) + 1, ids[0] == task.target_id


def evaluate_seed(seed: int, router: BinaryCDL):
    rows = []
    for h in H_LEVELS:
        task, records = generate_episode(
            rng=random.Random(seed * 100000 + h),
            history=h, records_target=h + 1,
            key_dim=KEY_DIM, noise=QUERY_NOISE,
        )
        dense_rank, dense_top1 = rank_target(router, task, records)

        cfg = PLMConfig(
            index_levels=(2,4,6),
            max_admitted=16,
            max_executions=4,
            fast_lr=0.0,
            slow_lr=0.0,
            seed=seed,
        )
        plm = new_plm_for_records(records, config=cfg)
        plm.router.slow_weights = list(router.slow_weights)
        plm.router.fast_bias = dict(router.fast_bias)

        address = plm.address(task.query_bits, target_id=task.target_id)
        local = [plm.state.get(i) for i in address.candidate_ids]
        local = [r for r in local if r is not None]
        local_rank, local_top1 = rank_target(plm.router, task, local) if local else (h+2, False)
        step = plm.step(task, step=h)

        bounded = records[-BOUNDED_STATE_CAPACITY:]
        rows.append({
            "H": h,
            "M": h + 1,
            "dense_target_rank": dense_rank,
            "dense_top1": dense_top1,
            "address_raw_candidates": address.raw_candidates,
            "address_candidate_count": address.candidates_scored,
            "address_lookups": address.index_lookups,
            "address_full_scan_fallback": address.used_full_scan_fallback,
            "address_target_admitted": address.target_admitted,
            "local_target_rank": local_rank,
            "local_top1": local_top1,
            "unified_success": step.outcome.success and step.verification.passed,
            "unified_committed": step.committed,
            "repaired": step.repaired,
            "execution_cost": step.execution_cost,
            "total_cost_proxy": step.total_cost,
            "state_population": plm.state.size,
            "experience_writes": sum(1 for r in plm.state.records() if r.state_kind is StateKind.EXPERIENCE),
            "bounded_state_recall": any(r.record_id == task.target_id for r in bounded),
        })
    return rows


def summarize(rows):
    out = []
    for h in H_LEVELS:
        g = [r for r in rows if r["H"] == h]
        out.append({
            "H": h, "M": h + 1,
            "dense_mean_rank": statistics.fmean(r["dense_target_rank"] for r in g),
            "dense_top1_rate": statistics.fmean(float(r["dense_top1"]) for r in g),
            "address_mean_raw_candidates": statistics.fmean(r["address_raw_candidates"] for r in g),
            "address_mean_candidates": statistics.fmean(r["address_candidate_count"] for r in g),
            "address_mean_lookups": statistics.fmean(r["address_lookups"] for r in g),
            "address_full_scan_fallback_rate": statistics.fmean(float(r["address_full_scan_fallback"]) for r in g),
            "address_recall": statistics.fmean(float(r["address_target_admitted"]) for r in g),
            "local_top1_rate": statistics.fmean(float(r["local_top1"]) for r in g),
            "unified_success_rate": statistics.fmean(float(r["unified_success"]) for r in g),
            "unified_commit_rate": statistics.fmean(float(r["unified_committed"]) for r in g),
            "repair_rate": statistics.fmean(float(r["repaired"]) for r in g),
            "mean_execution_cost": statistics.fmean(r["execution_cost"] for r in g),
            "mean_total_cost_proxy": statistics.fmean(r["total_cost_proxy"] for r in g),
            "bounded_state_recall": statistics.fmean(float(r["bounded_state_recall"]) for r in g),
        })
    return out


def lifecycle_probe():
    pool = build_operator_pool()
    initial = len(pool.operators)
    pool.synthesize_not()
    pool.synthesize_composite("xor_then_not", "xor", "not")
    return {
        "initial_operator_count": initial,
        "expanded_operator_count": len(pool.operators),
        "created": ["not", "xor_then_not"],
        "interpretation": "mechanism probe only; no capability promotion",
    }


def false_acceptance_probe(seed: int):
    router = train_router(seed)
    wrong_commits = 0
    for h in H_LEVELS:
        task, records = generate_episode(
            rng=random.Random(seed * 100000 + h + 7),
            history=h, records_target=h + 1,
            key_dim=KEY_DIM, noise=QUERY_NOISE,
        )
        cfg = PLMConfig(
            max_admitted=16, max_executions=1,
            false_accept_rate=1.0, seed=seed,
        )
        plm = new_plm_for_records(records, config=cfg)
        plm.router.slow_weights = list(router.slow_weights)
        plm.router.fast_bias = dict(router.fast_bias)
        result = plm.step(task, step=h)
        wrong_commits += int(result.committed and not result.outcome.success)
    return {
        "seed": seed,
        "wrong_verified_commits": wrong_commits,
        "note": "deterministic false-acceptance stability stress test; not real-world verifier error modeling",
    }


def main():
    routers = {s: train_router(s) for s in SEEDS}
    per_seed = []
    for s in SEEDS:
        per_seed.extend(evaluate_seed(s, routers[s]))

    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "source_base": {
            "parent_branch": "research/c5-composition-budget-audit-001",
            "parent_sha": "e3c59d1c630d810205c7e38b82ecb60f4a8410af",
            "settled_inputs": [
                "C5 dense-rank scaling and LSH admission/work",
                "C5 persistent relational loop",
                "representability gate",
                "verified-write separation",
                "CDL teacher/cheap-router evidence boundary",
                "CASM structural execution boundary",
            ],
            "not_reopened": [
                "C5 representation-arm winner selection",
                "C5 LSH operating-point selection",
                "generic Transformer/SSM implementation benchmark",
            ],
        },
        "protocol": {
            "H": list(H_LEVELS),
            "M": "H+1",
            "seeds": list(SEEDS),
            "train_episodes": TRAIN_EPISODES,
            "key_dim": KEY_DIM,
            "query_noise": QUERY_NOISE,
            "bounded_state_capacity": BOUNDED_STATE_CAPACITY,
            "max_admitted": 16,
            "max_executions": 4,
            "slow_lr": 0.01,
            "fast_lr": 0.10,
        },
        "cost_accounting": {
            "address_proxy": "index_lookups + candidates_scored",
            "raw_address_population": "raw candidates before admission cap",
            "execution_cost": "operator steps actually attempted before verification",
            "total_cost_proxy": "address proxy + execution cost",
            "future_requirement": "wall-clock + memory + I/O on compute-backed benchmark",
        },
        "summary": summarize(per_seed),
        "per_seed": per_seed,
        "lifecycle_probe": lifecycle_probe(),
        "false_acceptance_probe": [false_acceptance_probe(s) for s in SEEDS],
        "decision_rules": [
            "No asymptotic claim from finite-range measurements.",
            "No semantic-language claim from synthetic relations.",
            "Addressing, admission, execution and verification are separate costs.",
            "The bounded-state arm is an explicit capacity analogue, not a named SSM implementation.",
            "No C5 representation arm is promoted by this phase.",
        ],
    }
    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
