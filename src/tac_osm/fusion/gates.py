"""Pre-run integrity and representability gates for the fused model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .. import Candidate, Query, StateRead
from ..environment import build_relational_task, build_lookup_task, build_replay_task
from ..state import PersistentStore, StateConfig


@dataclass(frozen=True)
class GateResult:
    passed: bool
    name: str
    detail: dict[str, object]


FORBIDDEN_ROUTER_FIELDS = frozenset({
    "target_action",
    "gold_index",
    "outcome",
    "reward",
    "success",
    "answer",
})


def router_observation_gate() -> GateResult:
    """Static schema gate: forbidden target-bearing fields are not in Query."""
    query_fields = set(Query.__dataclass_fields__)
    candidate_fields = set(Candidate.__dataclass_fields__)
    forbidden_present = sorted(
        (query_fields | candidate_fields) & FORBIDDEN_ROUTER_FIELDS
    )
    return GateResult(
        passed=not forbidden_present,
        name="router_observation_schema",
        detail={
            "query_fields": sorted(query_fields),
            "candidate_fields": sorted(candidate_fields),
            "forbidden_present": forbidden_present,
        },
    )


def task_uniqueness_gate(
    *,
    seeds: Sequence[int] = (11, 17, 23),
) -> GateResult:
    """Verify each emitted task has exactly one relation satisfier."""
    failures: list[dict[str, object]] = []
    for seed in seeds:
        store = PersistentStore(StateConfig(seed=seed, n_slots=64))
        tasks = [
            build_relational_task(seed, dim=8, n_candidates=8),
            build_lookup_task(seed + 101, store, dim=8, n_candidates=8),
            build_replay_task(seed + 202, store, dim=8, n_candidates=8),
        ]
        for task in tasks:
            winners = [
                i for i, candidate in enumerate(task.candidates)
                if task.target_action == i
            ]
            if len(winners) != 1:
                failures.append({
                    "seed": seed,
                    "family": task.family,
                    "winners": winners,
                })
    return GateResult(
        passed=not failures,
        name="task_uniqueness",
        detail={"failures": failures},
    )


def same_present_task_signature(seed: int = 31) -> dict[str, object]:
    """Return a target-free signature for same-present/different-past tests."""
    store = PersistentStore(StateConfig(seed=seed, n_slots=64))
    task = build_relational_task(seed, dim=8, n_candidates=8)
    query = task.public()
    read = store.read(query)
    return {
        "query": query,
        "candidate_descriptors": tuple(c.descriptor for c in task.candidates),
        "state_keys": read.keys,
        "state_values": read.values,
    }


def all_gates() -> tuple[GateResult, ...]:
    return (router_observation_gate(), task_uniqueness_gate())
