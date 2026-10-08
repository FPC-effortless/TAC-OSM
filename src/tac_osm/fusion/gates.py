"""Pre-run integrity and representability gates for the fused model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .. import Candidate, Query
from ..environment import (
    build_lookup_task,
    build_relational_task,
    build_replay_task,
    parse_query,
    satisfies_relation,
)
from ..state import PersistentStore, StateConfig


@dataclass(frozen=True)
class GateResult:
    passed: bool
    name: str
    detail: dict[str, object]


FORBIDDEN_ROUTER_FIELDS = frozenset({
    "target_action", "gold_index", "outcome", "reward", "success", "answer",
})


def router_observation_gate() -> GateResult:
    """Static schema gate: target-bearing fields are absent from router inputs."""
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


def task_uniqueness_gate(*, seeds: Sequence[int] = (11, 17, 23)) -> GateResult:
    """Verify the intended public relation has exactly one satisfier."""
    failures: list[dict[str, object]] = []
    for seed in seeds:
        store = PersistentStore(StateConfig(seed=seed, n_slots=64))
        tasks = (
            build_relational_task(seed, dim=8, n_candidates=8),
            build_lookup_task(seed + 101, store, dim=8, n_candidates=8),
            build_replay_task(seed + 202, store, dim=8, n_candidates=8),
        )
        for task in tasks:
            bits, address = parse_query(task.public())
            if task.family == "relational":
                reference = bits
            else:
                reference = tuple(getattr(task.detail, "written_bits", ()))
            if not reference:
                failures.append({
                    "seed": seed,
                    "family": task.family,
                    "error": "missing_reference",
                })
                continue
            winners = [
                i for i, candidate in enumerate(task.candidates)
                if satisfies_relation(bits or reference, task.query.context, candidate.descriptor)
                and (
                    task.family != "replay"
                    or bool(address)
                )
            ]
            if len(winners) != 1 or winners[0] != task.target_action:
                failures.append({
                    "seed": seed,
                    "family": task.family,
                    "winners": winners,
                    "target_action_hidden_diagnostic": task.target_action,
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
    read = store.read(task.public())
    return {
        "query": task.public(),
        "candidate_descriptors": tuple(c.descriptor for c in task.candidates),
        "state_keys": read.keys,
        "state_values": read.values,
    }


def all_gates() -> tuple[GateResult, ...]:
    return (router_observation_gate(), task_uniqueness_gate())
