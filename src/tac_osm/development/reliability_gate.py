"""DEV-049 finite empirical reliability gates; no distribution-free safety claim."""
from __future__ import annotations

import math

THRESHOLDS = (-1.0,) + tuple(i / 40 for i in range(41))


def services_for(probabilities, threshold):
    if threshold not in THRESHOLDS:
        raise ValueError("unregistered threshold")
    values = list(probabilities)
    if not values or any(len(p) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1
                                           for v in p) for p in values):
        raise ValueError("finite two-action probabilities required")
    # Explicit endpoint implements the known safe action, including exact 0/1 predictions.
    return tuple(i for i, p in enumerate(values)
                 if threshold == -1 or p[0] - p[1] > threshold)


def outcome_totals(probabilities, outcomes, threshold):
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("nonempty matching world rows required")
    failures = services = failed_worlds = 0
    for p, y in zip(probabilities, outcomes):
        chosen = set(services_for(p, threshold))
        if len(y) != len(p) or any(len(v) != 2 or any(b not in (0, 1) for b in v) for v in y):
            raise ValueError("binary two-action simulator outcomes required")
        count = sum(v[int(i in chosen)] for i, v in enumerate(y))
        failures += count
        failed_worlds += count > 0
        services += len(chosen)
    return dict(failures=failures, failed_worlds=failed_worlds,
                services=services, total_cost=4 * failures + services)


def select_threshold(probabilities, outcomes, *, zero_failures):
    candidates = [(t, outcome_totals(probabilities, outcomes, t)) for t in THRESHOLDS]
    eligible = [(t, r) for t, r in candidates if not zero_failures or r["failures"] == 0]
    if not eligible:
        raise ValueError("no empirically feasible policy; service is not guaranteed safe")
    threshold, _ = min(eligible, key=lambda item: (item[1]["total_cost"], item[0]))
    return threshold, [{"threshold": t, **r} for t, r in candidates]


def zero_event_upper_bound(independent_blocks, alpha=.05):
    """One-sided bound for zero events under independent Bernoulli blocks.

    With unequal stratum risks this bounds their average via the product bound.
    It is not valid after selecting a policy on the same observations.
    """
    if independent_blocks < 1 or not 0 < alpha < 1:
        raise ValueError("positive block count and alpha in (0,1) required")
    return -math.expm1(math.log(alpha) / independent_blocks)
