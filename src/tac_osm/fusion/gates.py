"""Pre-run integrity and representability gates for the fused model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .. import Candidate, Query, StateUpdate
from ..environment import (
    build_lookup_task,
    build_relational_task,
    build_replay_task,
    parse_query,
    satisfies_relation,
)
from ..representability import representable
from ..state import PersistentStore, StateConfig
from .interfaces import MemoryContext, RegimeContext
from .operators import candidate_features


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
                # Stateful families must be validated against the value actually
                # addressable from the public query, not only against Task.detail.
                read = store.read(task.public())
                if not address:
                    failures.append({
                        "seed": seed,
                        "family": task.family,
                        "error": "missing_address",
                    })
                    continue
                if not read.keys or read.keys[0] != address or not read.values:
                    failures.append({
                        "seed": seed,
                        "family": task.family,
                        "error": "address_not_resolved",
                        "address": address,
                        "resolved_keys": list(read.keys),
                    })
                    continue
                reference = tuple(read.values[0])
                expected = tuple(getattr(task.detail, "written_bits", ()))
                if not expected:
                    failures.append({
                        "seed": seed,
                        "family": task.family,
                        "error": "missing_reference",
                    })
                    continue
                if reference != expected:
                    failures.append({
                        "seed": seed,
                        "family": task.family,
                        "error": "state_reference_mismatch",
                    })
                    continue

            if not reference:
                failures.append({
                    "seed": seed,
                    "family": task.family,
                    "error": "missing_reference",
                })
                continue

            # The relation is evaluated against the target-defining reference.
            # For replay this is the value retrieved by the public address; the
            # public query remains an input to the generator but is not the gold
            # vector for the relation.
            winners = [
                i for i, candidate in enumerate(task.candidates)
                if satisfies_relation(reference, task.query.context, candidate.descriptor)
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



def operator_representability_gate(*, n_episodes: int = 12) -> GateResult:
    """Run a shared-weight gate against the fused operator feature map."""
    if n_episodes < 3:
        raise ValueError("n_episodes must be >= 3")

    dim = 8
    feature_dim = 1 + 7 * dim + 7
    weights = [0.0] * feature_dim
    for j in range(dim):
        weights[1 + dim + j] = 1.0
        weights[1 + 2 * dim + j] = 16.0

    neutral_memory = MemoryContext((), (), (), 0.0, 0.0)
    neutral_regime = RegimeContext((0.0, 0.0, 0.0, 0.0), 0.5, 0.0, 0.0, 0.0)
    episodes = []
    reads = {}

    for family_offset, family in enumerate(("relational", "state_lookup", "replay")):
        for i in range(n_episodes):
            if family == "relational":
                store = PersistentStore(StateConfig(seed=20000 + i, n_slots=64))
                task = build_relational_task(
                    10000 + family_offset * 1000 + i, dim=dim, n_candidates=8
                )
            else:
                store = PersistentStore(
                    StateConfig(seed=20000 + family_offset * 100 + i, n_slots=64)
                )
                builder = build_lookup_task if family == "state_lookup" else build_replay_task
                task = builder(
                    11000 + family_offset * 1000 + i, store,
                    dim=dim, n_candidates=8
                )
            episodes.append(task)
            reads[id(task)] = store.read(task.public())

    def feature_fn(episode):
        read = reads[id(episode)]
        rows = [
            candidate_features(
                episode.public(), read, neutral_memory, neutral_regime, candidate
            )
            for candidate in episode.candidates
        ]
        return [
            tuple(row) + (0.0,) * max(0, feature_dim - len(row))
            if len(row) < feature_dim else tuple(row[:feature_dim])
            for row in rows
        ]

    result = representable(
        score_fn=lambda w, row: sum(a * b for a, b in zip(w, row)),
        feature_fn=feature_fn,
        gold_fn=lambda episode: episode.target_action,
        episodes=episodes,
        min_margin=1e-6,
        shared_weights=weights,
    )
    return GateResult(
        passed=bool(result["pass"]),
        name="operator_representability",
        detail=result,
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
    return (router_observation_gate(), task_uniqueness_gate(), operator_representability_gate())
