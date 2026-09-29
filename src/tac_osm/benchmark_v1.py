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


def _find_descriptor_with_truth(
    rng: random.Random,
    dim: int,
    relation: RelationName,
    reference: Sequence[int],
    marks: Sequence[int],
    desired: bool,
) -> tuple[int, ...]:
    """Find a descriptor with the requested relation truth value."""
    mark_positions = [j for j, mark in enumerate(marks) if mark]
    for _ in range(512):
        desc = _random_descriptor(rng, dim)
        if relation_holds(relation, reference, desc, marks) is desired:
            return desc
    if len(mark_positions) <= 16:
        for mask in range(1 << len(mark_positions)):
            desc = _random_descriptor(rng, dim)
            for bit_index, position in enumerate(mark_positions):
                desc[position] = (mask >> bit_index) & 1
            if relation_holds(relation, reference, desc, marks) is desired:
                return tuple(desc)
    raise RuntimeError(
        f"could not construct descriptor with relation={relation!r}, desired={desired}"
    )


def _force_mode(
    rng: random.Random,
    candidates: list[Candidate],
    *,
    relation: RelationName,
    reference: Sequence[int],
    marks: Sequence[int],
    validity: ValidityMode,
) -> set[int]:
    if validity == "unique":
        target = rng.randrange(len(candidates))
        good = _find_descriptor_with_truth(
            rng, len(reference), relation, reference, marks, True
        )
        candidates[target] = Candidate(
            key=candidates[target].key,
            descriptor=good,
            action=target,
            provenance="sampled_valid",
        )
        for i, c in enumerate(candidates):
            if i == target:
                continue
            bad = _find_descriptor_with_truth(
                rng, len(reference), relation, reference, marks, False
            )
            candidates[i] = Candidate(
                key=c.key,
                descriptor=bad,
                action=i,
                provenance=c.provenance,
            )
        return {target}

    if validity == "multiple":
        target_a, target_b = rng.sample(range(len(candidates)), 2)
        good = _find_descriptor_with_truth(
            rng, len(reference), relation, reference, marks, True
        )
        for idx in (target_a, target_b):
            candidates[idx] = Candidate(
                key=candidates[idx].key,
                descriptor=good,
                action=idx,
                provenance="sampled_valid",
            )
        for i, c in enumerate(candidates):
            if i in (target_a, target_b):
                continue
            bad = _find_descriptor_with_truth(
                rng, len(reference), relation, reference, marks, False
            )
            candidates[i] = Candidate(
                key=c.key,
                descriptor=bad,
                action=i,
                provenance=c.provenance,
            )
        return {target_a, target_b}

    if validity == "none":
        for i, c in enumerate(candidates):
            bad = _find_descriptor_with_truth(
                rng, len(reference), relation, reference, marks, False
            )
            candidates[i] = Candidate(
                key=c.key,
                descriptor=bad,
                action=i,
                provenance=c.provenance,
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


def repeated_equality_population(
    seed: int,
    *,
    dim: int = 8,
    n_candidates: int = 64,
    marked_positions: Sequence[int] | None = None,
) -> tuple[Candidate, ...]:
    """Build a static candidate universe with balanced marked-bit buckets.

    This is the C5 cost control. The candidate universe is built once and then
    reused across many queries. For the default two marked positions, every
    2-bit signature is repeated evenly, creating multiple-valid tasks while
    keeping the relevant signature independent of H.
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")
    marks = tuple(marked_positions) if marked_positions is not None else (0, 1)
    if not marks or any(j < 0 or j >= dim for j in marks):
        raise ValueError("marked_positions must contain valid dimension indices")
    n_signatures = 1 << len(marks)
    if n_candidates % n_signatures:
        raise ValueError(
            f"n_candidates={n_candidates} must be divisible by {n_signatures} "
            "to keep equality buckets balanced"
        )
    rng = random.Random(seed)
    candidates: list[Candidate] = []
    per_bucket = n_candidates // n_signatures
    for signature_id in range(n_signatures):
        bits = tuple((signature_id >> bit) & 1 for bit in range(len(marks)))
        for _ in range(per_bucket):
            desc = [rng.randrange(2) for _ in range(dim)]
            for bit, position in zip(bits, marks):
                desc[position] = bit
            index = len(candidates)
            candidates.append(
                Candidate(
                    key=f"c{index}",
                    descriptor=tuple(desc),
                    action=index,
                    provenance="static_population_v1",
                )
            )
    rng.shuffle(candidates)
    # Actions are the candidate's fixed global ids, not their post-shuffle
    # positions. This keeps action identity independent of candidate ordering.
    return tuple(
        Candidate(
            key=c.key,
            descriptor=c.descriptor,
            action=i,
            provenance=c.provenance,
        )
        for i, c in enumerate(candidates)
    )


def task_from_population(
    *,
    candidates: Sequence[Candidate],
    reference_bits: Sequence[int],
    step: int,
    relation: RelationName = "equality",
    state_address: str = "",
    validity: ValidityMode = "multiple",
) -> GeneratedTask:
    """Make a query against an existing candidate universe without mutating it."""
    if not candidates:
        raise ValueError("candidate population must be non-empty")
    reference = tuple(int(x) for x in reference_bits)
    dim = len(reference)
    marks = tuple(1 if j < 2 else 0 for j in range(dim))
    acceptable = {
        i
        for i, candidate in enumerate(candidates)
        if relation_holds(relation, reference, candidate.descriptor, marks)
    }
    if validity == "unique" and len(acceptable) != 1:
        raise ValueError(
            "population does not realize validity=unique: "
            f"found {len(acceptable)} acceptable candidates"
        )
    if validity == "multiple" and len(acceptable) < 2:
        raise ValueError(
            "population does not realize validity=multiple: "
            f"found {len(acceptable)} acceptable candidates"
        )
    if validity == "none" and acceptable:
        raise ValueError(
            "population does not realize validity=none: "
            f"found {len(acceptable)} acceptable candidates"
        )
    bits = "" if state_address else " ".join(str(x) for x in reference)
    return GeneratedTask(
        query=Query(
            text=f"{bits}\\t{state_address}",
            context=marks,
            step=step,
            provenance="static_population_v1",
        ),
        candidates=tuple(candidates),
        acceptable_actions=frozenset(acceptable),
        relation=relation,
        validity=validity,
        reference_bits=reference,
    )
