"""Direct C5 execution-subset control.

The workload is deliberately different from SELECTIVE-001: the correct output
depends on aggregating contributions from every relevant candidate program.
The exhaustive arm executes all H candidate programs; the selective arm retains
the relevant bucket R and executes only that bucket.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, Query
from .addressing import ContentAddressIndex
from .benchmark_v1 import relation_holds

MARKS = (1, 1, 1, 1, 1, 1, 0, 0)
WORK_UNITS_PER_CANDIDATE = 9


@dataclass(frozen=True)
class SelectiveExecutionTask:
    query: Query
    candidates: tuple[Candidate, ...]
    expected_output: int
    relevant_indices: frozenset[int]
    step: int


class SelectiveProgramExecutor:
    """Fixed-work candidate program used only for C5 execution accounting."""

    def execute_candidate(self, candidate: Candidate, query: Query) -> tuple[int, int]:
        bits = tuple(int(x) for x in query.text.partition("\t")[0].split())
        equalities = sum(
            int(candidate.descriptor[j] == bits[j])
            for j, mark in enumerate(query.context)
            if mark and j < len(bits) and j < len(candidate.descriptor)
        )
        payload = int(candidate.descriptor[6]) + 2 * int(candidate.descriptor[7])
        matches = int(equalities == sum(query.context))
        return matches * (1 + payload + sum(candidate.descriptor)), WORK_UNITS_PER_CANDIDATE

    def execute_population(
        self, candidates: Sequence[Candidate], query: Query
    ) -> tuple[int, int, int]:
        output = 0
        work = 0
        invocations = 0
        for candidate in candidates:
            value, units = self.execute_candidate(candidate, query)
            output += value
            work += units
            invocations += 1
        return output, work, invocations


def build_population(seed: int, h: int) -> tuple[Candidate, ...]:
    if h not in (64, 128, 256):
        raise ValueError("H must be 64, 128, or 256")
    rng = random.Random(seed)
    per_signature = h // 64
    candidates: list[Candidate] = []
    for signature_id in range(64):
        prefix = tuple((signature_id >> j) & 1 for j in range(6))
        for _ in range(per_signature):
            payload = (rng.randrange(2), rng.randrange(2))
            descriptor = prefix + payload
            i = len(candidates)
            candidates.append(
                Candidate(
                    key=f"exec-{seed}-{i}",
                    descriptor=descriptor,
                    action=i,
                    provenance="c5_exec_static_population",
                )
            )
    rng.shuffle(candidates)
    return tuple(
        Candidate(
            key=c.key, descriptor=c.descriptor, action=i, provenance=c.provenance
        )
        for i, c in enumerate(candidates)
    )


def build_task(seed: int, h: int, step: int) -> SelectiveExecutionTask:
    candidates = build_population(seed, h)
    rng = random.Random(seed * 1009 + step * 9176 + 31)
    signature_id = rng.randrange(64)
    reference = tuple((signature_id >> j) & 1 for j in range(6)) + (0, 0)
    query = Query(
        text=" ".join(str(x) for x in reference) + "\t",
        context=MARKS,
        step=step,
        provenance="c5_exec_subset",
    )
    relevant = frozenset(
        i
        for i, candidate in enumerate(candidates)
        if relation_holds("equality", reference, candidate.descriptor, MARKS)
    )
    if len(relevant) != h // 64:
        raise AssertionError("population must contain exactly H/64 relevant programs")
    executor = SelectiveProgramExecutor()
    expected, work, invocations = executor.execute_population(
        tuple(candidates[i] for i in relevant), query
    )
    assert work == invocations * WORK_UNITS_PER_CANDIDATE
    return SelectiveExecutionTask(
        query=query,
        candidates=candidates,
        expected_output=expected,
        relevant_indices=relevant,
        step=step,
    )


def retrieve(task: SelectiveExecutionTask, k: int) -> tuple[int, ...]:
    index = ContentAddressIndex.build(task.candidates, context=task.query.context)
    hit = index.lookup(task.query, reference=tuple(int(x) for x in task.query.text.partition("\t")[0].split()), k=k, relation="equality")
    return hit.candidate_indices