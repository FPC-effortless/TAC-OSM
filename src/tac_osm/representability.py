"""Mechanism representability gate — PNDS-URP v0.4 §34.

**Admissibility condition.** No learned-arm measurement may be interpreted
until the mechanism under test has been shown to *represent* the relation it
is being asked to learn.

This module exists because of a specific, documented failure. At
``cdl-attention-experiment`` commit ``dd8f63c``,
``PersistentStateRouter._features`` never read ``state.key``: all 70 key
blocks were computed for every candidate, making the feature map a function
of ``(candidate, target)`` alone. Under the exact-``T`` constraint the
analytically ideal weights then scored **exactly 0** for every candidate — a
constant, sitting in its own null space. The intended relation was
identically unrepresentable, so no training signal could have beaten chance.

Meanwhile ``true_key``, a *control arm*, read the relation directly and
never passed through the learned feature map. It reported 1.0000 for four
consecutive commits while the basis was broken.

    oracle success  =/=>  learned-map representability

A perfect oracle conceals a broken basis indefinitely. So this gate is a
**compulsory pre-training unit test**, not a retrospective audit.
"""

from __future__ import annotations

from typing import Callable, Sequence


def representable(
    score_fn: Callable[..., Sequence[float]],
    feature_fn: Callable[..., Sequence[Sequence[float]]],
    gold_fn: Callable[..., int],
    episodes: Sequence,
    *,
    min_margin: float = 1e-6,
    min_episodes: int = 1,
) -> dict:
    """Check that the *actual* learned feature map can express the relation.

    Pass analytically ideal weights through the **real** feature basis and
    require a positive margin separating the gold candidate from the best
    distractor, over many episodes.

    Parameters
    ----------
    score_fn:
        ``(weights, features) -> scores``. Must be linear in ``features``
        given ``weights``; the point of the gate is that the ideal weights
        *exist*, so the relation lives in the hypothesis class.
    feature_fn:
        ``(episode) -> feature rows``, one row per candidate, produced by the
        actual learned feature map under test. This is the object that
        failed at ``dd8f63c``.
    gold_fn:
        ``(episode) -> gold index``. Used only to locate the target; it is
        never an input to the mechanism.

    Returns
    -------
    dict with ``pass``, ``n_episodes``, ``min_margin``, and ``detail``.
    A ``False`` result means the run measures model-class expressiveness, not
    learnability from outcomes, and any learned-arm result is uninterpretable.
    """
    if len(episodes) < min_episodes:
        return {
            "pass": False,
            "n_episodes": len(episodes),
            "min_margin": None,
            "detail": f"insufficient episodes: {len(episodes)} < {min_episodes}",
        }

    worst = None
    for episode in episodes:
        rows = list(feature_fn(episode))
        if not rows:
            return {
                "pass": False,
                "n_episodes": len(episodes),
                "min_margin": None,
                "detail": "feature_fn returned no rows",
            }
        gold = gold_fn(episode)
        if not 0 <= gold < len(rows):
            return {
                "pass": False,
                "n_episodes": len(episodes),
                "min_margin": None,
                "detail": f"gold index {gold} out of range for {len(rows)} rows",
            }

        # Choose weights that maximise the gold row's score, then measure the
        # margin against the best alternative. If the ideal weights cannot
        # separate, no learned weights can.
        margins = []
        for other in range(len(rows)):
            if other == gold:
                continue
            diff = [g - o for g, o in zip(rows[gold], rows[other])]
            weight = [1.0 if d >= 0 else -1.0 for d in diff]
            margin = sum(w * d for w, d in zip(weight, diff))
            margins.append(margin)

        best_margin = min(margins) if margins else 0.0
        if worst is None or best_margin < worst:
            worst = best_margin

    passed = worst is not None and worst >= min_margin
    return {
        "pass": bool(passed),
        "n_episodes": len(episodes),
        "min_margin": worst,
        "detail": (
            "relation is representable through the actual feature map"
            if passed
            else "analytically ideal weights do not separate gold from the "
            "best distractor: the relation is in the null space of the "
            "feature basis (the dd8f63c failure)"
        ),
    }


__all__ = ["representable"]
