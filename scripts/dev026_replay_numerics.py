"""Proposed DEV-026: narrow float32 tolerances; exact scientific decisions/costs."""
from __future__ import annotations
import math

# Float32 inference quantities only; never apply to actions or realized costs.
FLOAT32_FIELDS = frozenset(('logits', 'decision_probabilities', 'estimated_gains', 'hypothetical_costs'))
REL_TOL = 1e-5
ABS_TOL = 1e-6


def compare_diagonal(actual, archived, inference_float=False):
    """Raise on any decision/cost change, nonfinite number or excess float error."""
    assert type(actual) is type(archived), 'type changed'
    if isinstance(actual, dict):
        assert actual.keys() == archived.keys(), 'fields changed'
        for key in actual:
            compare_diagonal(actual[key], archived[key], inference_float or key in FLOAT32_FIELDS)
    elif isinstance(actual, list):
        assert len(actual) == len(archived), 'length changed'
        for a, b in zip(actual, archived):
            compare_diagonal(a, b, inference_float)
    elif isinstance(actual, float):
        assert math.isfinite(actual) and math.isfinite(archived), 'nonfinite evidence'
        assert (math.isclose(actual, archived, rel_tol=REL_TOL, abs_tol=ABS_TOL)
                if inference_float else actual == archived), 'numeric evidence changed'
    else:
        assert actual == archived, 'discrete evidence changed'
