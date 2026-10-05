"""Budget-capped terminal utility for finite evidence partitions."""
from __future__ import annotations
from collections import Counter
from typing import Hashable, Sequence

def budget_capped_utility(candidate_evidence: Sequence[Hashable], budget: int) -> float:
    if not candidate_evidence:
        raise ValueError("candidate_evidence must not be empty")
    if budget <= 0:
        raise ValueError("budget must be positive")
    counts = Counter(candidate_evidence)
    return sum(min(budget, n) for n in counts.values()) / len(candidate_evidence)
