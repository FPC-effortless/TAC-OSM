"""The materiality verdicts, defined once and used by every measurement script.

## Why this module exists

``delta > 0`` is an improvement; that part is obvious. What is not obvious,
and what used to be written out separately in ``measure_surrogate.py`` and
``measure_learn.py``, is the threshold: a difference is material only if it
exceeds the baseline's own seed spread, floored at 0.02.

Two implementations of that rule is exactly the drift class the contract
module exists to prevent — one script's threshold drifting from the other's
would let the same measurement be reported as material in one experiment and
within noise in another, with nothing in the output to show it. It is now
defined once, and the 0.02 floor is defined once with it.

## Layer 0 — infrastructure, not evidence

No verdict is a scientific conclusion. A verdict is the pre-registered
decision rule applied to a measurement; the rule's *consequences* were
committed before the run and live in the contract. See
``docs/EVIDENCE_REGISTER.md``: a verdict answers "did this row fire?", not "is
the hypothesis true?".
"""

from __future__ import annotations

from typing import Sequence

__all__ = [
    "MATERIAL_IMPROVEMENT",
    "MATERIAL_HARM",
    "WITHIN_SEED_NOISE",
    "spread_floor",
    "materiality_threshold",
    "verdict",
    "delta_detail",
]

#: The three verdicts a pre-registered decision rule may return. A fourth value
#: appearing anywhere would be a verdict the rule never committed to, which is
#: why they are constants and not ad-hoc strings in each script.
MATERIAL_IMPROVEMENT = "MATERIAL improvement"
MATERIAL_HARM = "MATERIAL harm"
WITHIN_SEED_NOISE = "within seed noise"

#: The floor on a materiality threshold. A seed spread below 2% is sampling
#: error at 100 evaluation steps, not the experiment's noise floor; using it
#: as the threshold would let a 1.5% difference be called material. The floor
#: is a pre-registered constant, not a per-run decision, so it is named here
#: and does not appear as a bare literal in a measurement script.
SPREAD_FLOOR = 0.02


def spread_floor() -> float:
    """The floor on any materiality threshold, as a value not a literal.

    Scripts print this in their gate headers, and printing a name rather than
    ``0.02`` keeps a reader's eye on the fact that the number is a
    pre-registered choice.
    """
    return SPREAD_FLOOR


def materiality_threshold(spread: float) -> float:
    """The threshold a delta must exceed to count as material.

    ``max(spread, SPREAD_FLOOR)``. The spread is the baseline's own seed
    spread, measured before any arm is compared — the pre-registered standard,
    chosen because it is available before the comparison and does not depend
    on the arm being judged.
    """
    return max(spread, SPREAD_FLOOR)


def verdict(delta: float, spread: float) -> str:
    """The pre-registered materiality verdict for one delta.

    Called from exactly one place per delta per script, and consumed by both
    the terminal report and the machine-readable record — two computations of
    a verdict is how the report and the record come to disagree about what the
    decision rule decided.
    """
    threshold = materiality_threshold(spread)
    if delta > threshold:
        return MATERIAL_IMPROVEMENT
    if delta < -threshold:
        return MATERIAL_HARM
    return WITHIN_SEED_NOISE


def seed_spread(values: Sequence[float]) -> float:
    """Max minus min across seeds — the materiality threshold's input.

    Shared because it was duplicated verbatim in every measurement script, and
    because a run that computes it differently computes a different threshold
    from the same numbers.
    """
    return max(values) - min(values) if values else 0.0


def delta_detail(
    *,
    arm: str,
    metric: str,
    baseline: float,
    observed: float,
    baseline_spread: float,
    context: dict[str, object] | None = None,
) -> dict[str, object]:
    """One decision-rule comparison, as fields.

    ``context`` carries whatever else the experiment's rule reads — F3 tags
    each row with its candidate count, and a different experiment may tag one
    with a different quantity. The shared part is the verdict and the
    threshold that produced it, which are what must not drift between scripts.
    """
    delta = observed - baseline
    out: dict[str, object] = {
        "arm": arm,
        "metric": metric,
        "baseline": baseline,
        "observed": observed,
        "delta": delta,
        "baseline_spread": baseline_spread,
        "threshold": materiality_threshold(baseline_spread),
        "verdict": verdict(delta, baseline_spread),
    }
    if context:
        out.update(context)
    return out
