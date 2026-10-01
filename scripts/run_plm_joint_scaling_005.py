#!/usr/bin/env python3
"""TACOSM-PLM-JOINT-SCALING-005.

End-to-end mechanistic fusion of persistent-state addressing and
computational/operator addressing. This phase deliberately avoids reopening
the C5 representation search.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.plm_unified import (
    HierarchicalStateIndex,
    MemoryRecord,
    RelationEnvironment,
    StateKind,
    StateStatus,
)
from tac_osm.plm_research_phases import (
    HierarchicalOperatorIndex,
    OperatorRecord,
    make_operator_population,
)

EXPERIMENT_ID = "TACOSM-PLM-JOINT-SCALING-005"
SEEDS = tuple(range(5))
M_LEVELS = (64, 256, 1024, 4096, 16384)
E_LEVELS = (16, 64, 256, 1024, 4096, 16384)
STATE_CAP = 16
OP_CAP = 8
STATE_KEY_DIM = 12
OP_KEY_DIM = 16
QUERY_NOISE = 0.10
FAMILIES = ("xor", "and", "or", "xnor")


@dataclass(frozen=True)
class JointRow:
    seed: int
    M: int
    E: int
    state_raw_candidates: int
    state_admitted: int
    state_probes: int
    state_target_admitted: bool
    operator_raw_candidates: int
    operator_admitted: int
    operator_probes: int
    operator_target_admitted: bool
    state_rank: int
    operator_rank: int
    joint_success: bool
    execution_cost: int
    address_cost: int
    total_cost_proxy: int
    full_scan_proxy: int
    cost_ratio_to_scan: float


def noisy(bits: tuple[int, ...], p: float, rng: random.Random) -> tuple[int, ...]:
    return tuple(1 - b if rng.random() < p else b for b in bits)


def agreement_rank(
    query: tuple[int, ...],
    rows: list[MemoryRecord],
    target_id: str,
) -> tuple[int, bool]:
    if not rows:
        return 1, False
    scored = []
    for row in rows:
        score = sum(int(a == b) for a, b in zip(query, row.key_bits))
        scored.append((score, row.record_id))
    scored.sort(key=lambda x: (-x[0], x[1]))
    ids = [x[1] for x in scored]
    if target_id not in ids:
        return len(ids) + 1, False
    rank = ids.index(target_id) + 1
    return rank, rank == 1


def make_state_population(m: int, *, seed: int) -> tuple[MemoryRecord, ...]:
    rng = random.Random(seed)
    target_key = tuple(rng.randrange(2) for _ in range(STATE_KEY_DIM))
    target_payload = (rng.randrange(2), rng.randrange(2))
    target_op = rng.choice(tuple(RelationEnvironment.FUNCTIONS))
    target = MemoryRecord(
        record_id="target",
        namespace="relational",
        key_bits=target_key,
        payload=target_payload,
        state_kind=StateKind.WORLD,
        status=StateStatus.VERIFIED,
        created_step=0,
        source="joint_target",
        confidence=1.0,
        operator_family=target_op,
    )
    rows = [target]
    for i in range(m - 1):
        rows.append(
            MemoryRecord(
                record_id=f"distractor:{i}",
                namespace="relational",
                key_bits=tuple(
                    rng.randrange(2) for _ in range(STATE_KEY_DIM)
                ),
                payload=(rng.randrange(2), rng.randrange(2)),
                state_kind=StateKind.WORLD,
                status=StateStatus.VERIFIED,
                created_step=i + 1,
                source="joint_distractor",
                confidence=0.8,
                operator_family=rng.choice(
                    tuple(RelationEnvironment.FUNCTIONS)
                ),
            )
        )
    return tuple(rows)


def target_operator(
    operators: tuple[OperatorRecord, ...],
    family: str,
    rng: random.Random,
) -> OperatorRecord:
    family_ops = [
        op for i, op in enumerate(operators)
        if FAMILIES[i % len(FAMILIES)] == family
    ]
    if not family_ops:
        raise RuntimeError(f"operator population E does not contain family {family}")
    return family_ops[rng.randrange(len(family_ops))]


def evaluate(seed: int, m: int, e: int) -> JointRow:
    state_rows = make_state_population(
        m, seed=seed * 1000003 + m + e
    )
    target = state_rows[0]

    op_rng = random.Random(seed * 9001 + 17 * e + m)
    operators = make_operator_population(
        e, key_dim=OP_KEY_DIM, seed=seed * 101 + e
    )
    target_op = target_operator(operators, target.operator_family, op_rng)

    state_query = noisy(target.key_bits, QUERY_NOISE, op_rng)
    operator_query = noisy(target_op.key_bits, QUERY_NOISE, op_rng)

    state_index = HierarchicalStateIndex((2, 4, 6))
    state_index.build(state_rows)
    state_ids, state_probes, state_raw, _fallback = state_index.lookup(
        state_query, max_candidates=STATE_CAP
    )
    state_set = set(state_ids)
    admitted_state = [
        r for r in state_rows if r.record_id in state_set
    ]
    state_rank, _state_top1 = agreement_rank(
        state_query, admitted_state, target.record_id
    )

    operator_index = HierarchicalOperatorIndex((4, 8, 12))
    operator_index.build(operators)
    op_hit = operator_index.lookup(
        operator_query,
        target_name=target_op.name,
        max_admitted=OP_CAP,
    )
    admitted_ops = [
        r for r in operators if r.name in set(op_hit.selected)
    ]

    op_rows = [
        MemoryRecord(
            record_id=r.name,
            namespace="operator",
            key_bits=r.key_bits,
            payload=(0, 0),
            state_kind=StateKind.COMPUTATIONAL,
            status=StateStatus.VERIFIED,
            created_step=0,
            source="joint_operator",
            confidence=1.0,
            operator_family=FAMILIES[i % len(FAMILIES)],
        )
        for i, r in enumerate(operators)
        if r.name in set(op_hit.selected)
    ]
    operator_rank, _op_top1 = agreement_rank(
        operator_query, op_rows, target_op.name
    )

    joint_success = (
        state_rank == 1
        and operator_rank == 1
        and target.record_id in state_set
        and op_hit.target_admitted
    )

    state_work = state_probes + state_raw + len(admitted_state)
    operator_work = (
        op_hit.probes + op_hit.raw_candidates + op_hit.admitted_candidates
    )
    address_cost = state_work + operator_work
    execution_cost = 1
    total = address_cost + execution_cost
    full_scan = m + e

    return JointRow(
        seed=seed,
        M=m,
        E=e,
        state_raw_candidates=state_raw,
        state_admitted=len(admitted_state),
        state_probes=state_probes,
        state_target_admitted=target.record_id in state_set,
        operator_raw_candidates=op_hit.raw_candidates,
        operator_admitted=op_hit.admitted_candidates,
        operator_probes=op_hit.probes,
        operator_target_admitted=op_hit.target_admitted,
        state_rank=state_rank,
        operator_rank=operator_rank,
        joint_success=joint_success,
        execution_cost=execution_cost,
        address_cost=address_cost,
        total_cost_proxy=total,
        full_scan_proxy=full_scan,
        cost_ratio_to_scan=total / full_scan,
    )


def aggregate(rows: list[JointRow]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[JointRow]] = {}
    for row in rows:
        groups.setdefault((row.M, row.E), []).append(row)
    out = []
    for (m, e), group in sorted(groups.items()):
        out.append({
            "M": m,
            "E": e,
            "state_raw_mean": sum(r.state_raw_candidates for r in group) / len(group),
            "operator_raw_mean": sum(r.operator_raw_candidates for r in group) / len(group),
            "state_admission_rate": sum(float(r.state_target_admitted) for r in group) / len(group),
            "operator_admission_rate": sum(float(r.operator_target_admitted) for r in group) / len(group),
            "joint_success_rate": sum(float(r.joint_success) for r in group) / len(group),
            "address_cost_mean": sum(r.address_cost for r in group) / len(group),
            "total_cost_mean": sum(r.total_cost_proxy for r in group) / len(group),
            "full_scan_cost": sum(r.full_scan_proxy for r in group) / len(group),
            "mean_cost_ratio": sum(r.cost_ratio_to_scan for r in group) / len(group),
        })
    return out


def main() -> None:
    rows = [
        evaluate(seed, m, e)
        for seed in SEEDS
        for m in M_LEVELS
        for e in E_LEVELS
    ]
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "lineage": {
            "stacked_from": "research/plm-unified-phase-001",
            "consumes": [
                "P1 state address accounting",
                "P3 operator address accounting",
                "P2 verifier-gated state discipline",
            ],
            "not_reopened": [
                "C5 representation-arm selection",
                "C5 LSH operating-point selection",
            ],
        },
        "protocol": {
            "seeds": SEEDS,
            "M": M_LEVELS,
            "E": E_LEVELS,
            "state_cap": STATE_CAP,
            "operator_cap": OP_CAP,
            "state_key_dim": STATE_KEY_DIM,
            "operator_key_dim": OP_KEY_DIM,
            "query_noise": QUERY_NOISE,
        },
        "metrics": [
            "state target admission",
            "operator target admission",
            "state target rank",
            "operator target rank",
            "joint success",
            "state raw/admitted/probe work",
            "operator raw/admitted/probe work",
            "total cost proxy",
            "full-scan proxy",
        ],
        "cost_accounting": {
            "state_address": "state_probes + state_raw_candidates + state_admitted",
            "operator_address": "operator_probes + operator_raw_candidates + operator_admitted",
            "execution": "one exact reference operation",
            "full_scan_proxy": "M + E",
        },
        "rows": [r.__dict__ for r in rows],
        "summary": aggregate(rows),
        "decision_rules": [
            "Joint success requires target rank 1 in both admitted sets.",
            "Low cost with failed target admission is not a successful routing result.",
            "Raw address work remains charged even when K is capped.",
            "The exact agreement reranker is a mechanistic control; it is not a learned CDL capability result.",
            "Finite M/E grids are descriptive, not asymptotic proofs.",
        ],
    }
    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
