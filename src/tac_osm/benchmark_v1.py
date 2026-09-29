"""Leakage-resistant generalized task generation for the hardened path.

The v0 generator starts from a known gold descriptor and manufactures every
distractor to violate the gold relation. That is useful for mechanism
isolation, but it is not a general relevance benchmark.

v1 samples the candidate population first and then enforces the requested
validity regime. The generator supports unique, multiple-valid, and no-valid
sets plus several relations. Hidden acceptable actions remain environment
state; the Query contains only public information and, for persistence tasks,
the address.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Sequence

from . import Candidate, Query

RelationName = Literal["equality", "xor_parity", "majority", "any_match"]
ValidityMode = Literal["unique", "multiple", "none"]


@dataclass(frozen=True)
class GeneratedTask:
    query: Query
    candidates: tuple[Candidate, ...]
    acceptable_actions: frozenset[int]
    relation: RelationName
    validity: ValidityMode
    reference_bits: tuple[int, ...]

    def public(self) -> Query:
        return Query(
            text=self.query.text,
            context=self.query.context,
            step=self.query.step,
            provenance=self.query.provenance,
        )


def relation_holds(
    relation: RelationName,
    reference: Sequence[int],
    descriptor: Sequence[int],
    marks: Sequence[int],
) -> bool:
    values = [
        int(reference[j] == descriptor[j])
        for j in range(min(len(reference), len(descriptor), len(marks)))
        if marks[j]
    ]
    if not values:
        return False
    if relation == "equality":
        return all(values)
    if relation == "xor_parity":
        return sum(values) % 2 == 1
    if relation == "majority":
        return sum(values) > len(values) / 2.0
    if relation == "any_match":
        return any(values)
    raise ValueError(f"unknown relation {relation!r}")


def _random_descriptor(rng: random.Random, dim: int) -> tuple[int, ...]:
    return tuple(rng.randrange(2) for _ in range(dim))


def _valid_indices(
    candidates: Sequence[Candidate],
    relation: RelationName,
    reference: Sequence[int],
    marks: Sequence[int],
) -> set[int]:
    return {
        i for i, c in enumerate(candidates)
        if relation_holds(relation, reference, c.descriptor, marks)
    }


def _force_mode(
    rng: random.Random,
    candidates: list[Candidate],
    *,
    relation: RelationName,
    reference: Sequence[int],
    marks: Sequence[int],
    validity: ValidityMode,
) -> set[int]:
    def holds(desc: Sequence[int]) -> bool:
        return relation_holds(relation, reference, desc, marks)

    if validity == "unique":
        target = rng.randrange(len(candidates))
        good = list(candidates[target].descriptor)
        # Build a satisfying descriptor for the equality relation and related
        # predicates. Search is bounded by the binary descriptor space; the
        # dimensions used by the benchmark are intentionally small.
        for _ in range(2 ** min(12, len(good)) + 1):
            good = list(_random_descriptor(rng, len(good)))
            if holds(good):
                break
        else:
            raise RuntimeError("could not construct a unique-valid candidate")
        candidates[target] = Candidate(
            key=candidates[target].key,
            descriptor=tuple(good),
            action=target,
            provenance="sampled_valid",
        )
        for i, c in enumerate(candidates):
            if i == target:
                continue
            if holds(c.descriptor):
                bad = list(c.descriptor)
                for j in range(len(bad)):
                    if marks[j]:
                        bad[j] ^= 1
                        if not holds(bad):
                            candidates[i] = Candidate(
                                key=c.key,
                                descriptor=tuple(bad),
                                action=i,
                                provenance=c.provenance,
                            )
                            break
        return _valid_indices(candidates, relation, reference, marks)

    if validity == "multiple":
        target_a, target_b = rng.sample(range(len(candidates)), 2)
        seed_desc = None
        for _ in range(2 ** min(12, len(reference)) + 1):
            candidate = _random_descriptor(rng, len(reference))
            if holds(candidate):
                seed_desc = candidate
                break
        if seed_desc is None:
            raise RuntimeError("could not construct multiple-valid candidates")
        for idx in (target_a, target_b):
            candidates[idx] = Candidate(
                key=candidates[idx].key,
                descriptor=tuple(seed_desc),
                action=idx,
                provenance="sampled_valid",
            )
        return _valid_indices(candidates, relation, reference, marks)

    if validity == "none":
        for i, c in enumerate(candidates):
            desc = list(c.descriptor)
            for _ in range(len(desc) * 2 + 2):
                if not holds(desc):
                    break
                j = rng.choice([j for j, mark in enumerate(marks) if mark])
                desc[j] ^= 1
            if holds(desc):
                raise RuntimeError("could not construct a no-valid candidate")
            candidates[i] = Candidate(
                key=c.key, descriptor=tuple(desc), action=i, provenance=c.provenance
            )
        return set()

    raise ValueError(f"unknown validity {validity!r}")


def generate_task(
    seed: int,
    *,
    dim: int = 8,
    n_candidates: int = 64,
    relation: RelationName = "equality",
    validity: ValidityMode = "unique",
    state_address: str = "",
    reference_bits: Sequence[int] | None = None,
    step: int = 0,
) -> GeneratedTask:
    if dim < 2:
        raise ValueError("dim must be >= 2")
    if n_candidates < 2:
        raise ValueError("n_candidates must be >= 2")
    rng = random.Random(seed)
    reference = (
        tuple(int(x) for x in reference_bits)
        if reference_bits is not None
        else tuple(rng.randrange(2) for _ in range(dim))
    )
    if len(reference) != dim:
        raise ValueError("reference_bits length must equal dim")
    n_marked = max(2, dim // 4)
    marks = tuple(1 if j < n_marked else 0 for j in range(dim))
    candidates = [
        Candidate(
            key=f"c{i}",
            descriptor=_random_descriptor(rng, dim),
            action=i,
            provenance="sampled",
        )
        for i in range(n_candidates)
    ]
    acceptable = _force_mode(
        rng,
        candidates,
        relation=relation,
        reference=reference,
        marks=marks,
        validity=validity,
    )
    bits = "" if state_address else " ".join(str(x) for x in reference)
    query = Query(
        text=f"{bits}\t{state_address}",
        context=marks,
        step=step,
        provenance="generalized_v1",
    )
    return GeneratedTask(
        query=query,
        candidates=tuple(candidates),
        acceptable_actions=frozenset(acceptable),
        relation=relation,
        validity=validity,
        reference_bits=tuple(reference),
    )
