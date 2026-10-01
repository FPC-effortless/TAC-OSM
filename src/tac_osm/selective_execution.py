"""Direct C5 execution-subset control.

The workload differs from SELECTIVE-001: the correct output depends on
aggregating contributions from every relevant candidate program. The
exhaustive arm executes all H candidates; the selective arm retains the
relevant bucket R and executes only that bucket.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, Query
from .addressing import ContentAddressIndex
from .benchmark_v1 import relation_holds

MATCH_MARKS = (1, 1, 1, 1, 1, 1, 0, 0)
WORK_UNITS_PER_CANDIDATE = 9
QUERY_PREFIXES = tuple(range(16))


@dataclass(frozen=True)
class SelectiveExecutionTask:
    query: Query
    candidates: tuple[Candidate, ...]
    expected_output: int
    relevant_indices: frozenset[int]
    step: int


class SelectiveProgramExecutor:
    """Fixed-work candidate program used for C5 execution accounting."""

    def execute_candidate(
        self, candidate: Candidate, query: Query
    ) -> tuple[int, int]:
        bits = tuple(int(x) for x in query.text.partition("\t")[0].split())
        equalities = sum(
            int(candidate.descriptor[j] == bits[j])
            for j, mark in enumerate(query.context)
            if mark and j < len(bits) and j < len(candidate.descriptor)
        )
        payload = int(candidate.descriptor[6]) + 2 * int(candidate.descriptor[7])
        matches = int(equalities == sum(query.context))
        contribution = matches * (1 + payload + sum(candidate.descriptor))
        return contribution, WORK_UNITS_PER_CANDIDATE

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
        raise ValueError('H must be 64, 128, or 256')
    if h % 4:
        raise ValueError('H must be divisible by 4')
    rng = random.Random(seed)
    # Prefixes 0..15 are queryable relevance classes. Each has four payload
    # variants, so every query has exactly R=4 relevant candidates. Larger H
    # adds complete distractor prefix classes without changing R.
    candidates: list[Candidate] = []
    for prefix in range(h // 4):
        prefix_bits = tuple((prefix >> j) & 1 for j in range(6))
        for payload in range(4):
            descriptor = prefix_bits + (payload & 1, (payload >> 1) & 1)
            index = len(candidates)
            candidates.append(
                Candidate(
                    key=f"exec-{seed}-{index}",
                    descriptor=descriptor,
                    action=index,
                    provenance="c5_exec_static_population",
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


def build_task(seed: int, h: int, step: int) -> SelectiveExecutionTask:
    candidates = build_population(seed, h)
    rng = random.Random(seed * 1009 + step * 9176 + 31)
    prefix = rng.choice(QUERY_PREFIXES)
    reference = tuple((prefix >> j) & 1 for j in range(6)) + (0, 0)
    query = Query(
        text=" ".join(str(x) for x in reference) + "\t",
        context=MATCH_MARKS,
        step=step,
        provenance="c5_exec_subset",
    )
    relevant = frozenset(
        i
        for i, candidate in enumerate(candidates)
        if relation_holds("equality", reference, candidate.descriptor, MATCH_MARKS)
    )
    if len(relevant) != 4:
        raise AssertionError('fixed-R population must contain exactly four relevant programs')
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


def build_index(task: SelectiveExecutionTask) -> ContentAddressIndex:
    return ContentAddressIndex.build(
        task.candidates,
        context=task.query.context,
    )


def retrieve(task: SelectiveExecutionTask, index: ContentAddressIndex, k: int) -> tuple[int, ...]:
    reference = tuple(
        int(x) for x in task.query.text.partition("\t")[0].split()
    )
    hit = index.lookup(
        task.query,
        reference=reference,
        k=k,
        relation="equality",
    )
    return hit.candidate_indices