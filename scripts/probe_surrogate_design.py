#!/usr/bin/env python
"""Pre-implementation probe for TACOSM-SURROGATE-001 (F3).

Three questions, answered from the task distribution alone:

  1. Is the baseline reward `float(outcome.success)` a gold-indexed scalar?
  2. Is the analytic-margin surrogate actually dense where the outcome reward
     is sparse?  (P(reward != 0) vs P(surrogate != 0))
  3. How often does the surrogate's sign agree with the outcome indicator?
     i.e. is `sgn(s_gold - max_{j!=gold} s_j)` == `1[argmax == gold]`?

Writes a plain-text report. No training loop, no model construction beyond
task builders, so it is cheap and read-only w.r.t. the published results.
"""
from __future__ import annotations

import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from tac_osm.environment import (  # noqa: E402
    WorldConfig,
    WorldEnvironment,
    build_lookup_task,
    build_relational_task,
    build_replay_task,
)
from tac_osm.router import (  # noqa: E402
    LearnedRelationalRouter,
    RouterConfig,
    analytic_weights,
    basis_size,
    features,
)
from tac_osm.state import PersistentStore, StateConfig  # noqa: E402

DIM = 8
MAX_SLOTS = 4


def _episode_score_row(task, store, w):
    """Score every candidate of one task under weight vector w."""
    router = LearnedRelationalRouter(
        RouterConfig(dim=DIM, max_state_slots=MAX_SLOTS)
    )
    router.w = list(w)
    return router.score(task.query, store, task.candidates)


def _probe(h: int, n_episodes: int, seed0: int = 500000) -> dict:
    store = PersistentStore(StateConfig(seed=seed0, n_slots=512))
    analytic = analytic_weights(DIM, MAX_SLOTS)
    zero = [0.0] * basis_size(DIM, MAX_SLOTS)

    # Track for both the analytic vector and the zero vector.
    acc = {"analytic": _acc(), "zero": _acc()}

    families = {
        "relational": lambda i: build_relational_task(
            seed0 + i, dim=DIM, n_candidates=h),
        "state_lookup": lambda i: build_lookup_task(
            seed0 + 1000 + i, store, dim=DIM, n_candidates=h),
        "replay": lambda i: build_replay_task(
            seed0 + 2000 + i, store, dim=DIM, n_candidates=h),
    }
    for name, build in families.items():
        store.reset()
        for i in range(n_episodes):
            task = build(i)
            # Write the world facts the family's relation depends on, mirroring
            # what the loop would have persisted.
            address = task.query.text.partition("\t")[2]
            if address:
                detail = task.detail
                written = getattr(detail, "written_bits", None)
                if written:
                    store.write(_dummy_update(address, tuple(written), task))

            gold = task.target_action
            for tag, w in (("analytic", analytic), ("zero", zero)):
                raw = _episode_score_row(task, store, w)
                _update_acc(acc[tag], raw, gold, h)
    return acc


def _dummy_update(key, value, task):
    from tac_osm import StateUpdate
    return StateUpdate(key=key, value=value, task_key=key,
                       success_score=1.0, step=task.query.step)


def _acc():
    return {
        "n": 0,
        "reward_nonzero": 0,       # outcome indicator is 1
        "surrogate_nonzero": 0,    # margin != 0
        "agree": 0,                # sgn(margin) == indicator
        "disagree": 0,
        "margins": [],
        "indicator": [],
    }


def _update_acc(a, raw, gold, h):
    others = [s for i, s in enumerate(raw) if i != gold]
    margin = raw[gold] - max(others)
    # The outcome reward the loop would have issued for the *greedy* action.
    # `route` samples, so `success` depends on the draw; the relevant
    # comparison for the leakage question is against the indicator that the
    # greedy action is gold, which is what a dense surrogate would approximate.
    greedy = max(range(len(raw)), key=lambda i: raw[i])
    indicator = 1.0 if greedy == gold else 0.0

    a["n"] += 1
    a["reward_nonzero"] += int(indicator > 0)
    a["surrogate_nonzero"] += int(margin != 0.0)
    a["margins"].append(margin)
    a["indicator"].append(indicator)
    if (margin > 0) == (indicator > 0):
        a["agree"] += 1
    else:
        a["disagree"] += 1


def _centring_probe(h: int, n_episodes: int, seed0: int = 700000) -> list[str]:
    """Does the pre-registered centring rule preserve the surrogate's sign?

    The registered rule is

        centred_reward = (s_gold - max_{j != gold} s_j) - mean(scores)

    The stated purpose of the centring term is, quoting the pre-registration,
    that "a margin that is positive on every step would otherwise be a
    constant positive reward and the update would degrade to plain gradient
    ascent on the selected features". So the rule is designed to *remove* a
    constant positive offset.

    This measures the quantity the rule actually produces, for two weight
    vectors:

      * ``analytic`` -- the successful case the arm is meant to move toward;
      * ``zero`` -- the untrained start point, where every run begins.

    For each it reports the sign agreement between ``centred_reward`` and the
    margin, because the update is ``lr * centred * (1 - p_sel) * features``
    applied to the *selected* candidate only: a centred reward whose sign is
    opposite the margin's pushes the selected action the wrong way relative to
    the quantity the arm claims to encode.
    """
    store = PersistentStore(StateConfig(seed=seed0, n_slots=512))
    vectors = {
        "analytic": analytic_weights(DIM, MAX_SLOTS),
        "zero": [0.0] * basis_size(DIM, MAX_SLOTS),
    }
    out: list[str] = []
    families = {
        "relational": lambda i: build_relational_task(
            seed0 + i, dim=DIM, n_candidates=h),
        "state_lookup": lambda i: build_lookup_task(
            seed0 + 1000 + i, store, dim=DIM, n_candidates=h),
        "replay": lambda i: build_replay_task(
            seed0 + 2000 + i, store, dim=DIM, n_candidates=h),
    }
    for tag, w in vectors.items():
        agree = 0
        n = 0
        centred_vals: list[float] = []
        margin_vals: list[float] = []
        mean_vals: list[float] = []
        for build in families.values():
            store.reset()
            for i in range(n_episodes):
                task = build(i)
                address = task.query.text.partition("\t")[2]
                if address:
                    written = getattr(task.detail, "written_bits", None)
                    if written:
                        store.write(_dummy_update(address, tuple(written), task))
                raw = _episode_score_row(task, store, w)
                gold = task.target_action
                others = [s for i, s in enumerate(raw) if i != gold]
                margin = raw[gold] - max(others)
                mean = statistics.fmean(raw)
                centred = margin - mean
                centred_vals.append(centred)
                margin_vals.append(margin)
                mean_vals.append(mean)
                n += 1
                if (centred > 0) == (margin > 0):
                    agree += 1
        out.append(
            f"  centring probe [{tag}]:\n"
            f"    sgn(margin - mean) == sgn(margin) on "
            f"{agree}/{n} = {agree / n:.4f} of steps\n"
            f"    margin : mean={statistics.fmean(margin_vals):+.4f} "
            f"min={min(margin_vals):+.4f} max={max(margin_vals):+.4f}\n"
            f"    mean(scores): mean={statistics.fmean(mean_vals):+.4f} "
            f"min={min(mean_vals):+.4f} max={max(mean_vals):+.4f}\n"
            f"    centred: mean={statistics.fmean(centred_vals):+.4f} "
            f"min={min(centred_vals):+.4f} max={max(centred_vals):+.4f}"
        )
    return out


def _report(h: int, acc: dict) -> str:
    lines = [f"H = {h}", "-" * 60]
    for tag in ("analytic", "zero"):
        a = acc[tag]
        n = a["n"]
        if not n:
            continue
        p_rew = a["reward_nonzero"] / n
        p_sur = a["surrogate_nonzero"] / n
        p_agree = a["agree"] / n
        m = a["margins"]
        lines.append(f"  [{tag}]  n={n}")
        lines.append(f"    P(outcome reward != 0)      = {p_rew:.4f}")
        lines.append(f"    P(surrogate margin != 0)    = {p_sur:.4f}")
        lines.append(f"    P(sgn(margin) == indicator) = {p_agree:.4f}")
        lines.append(
            f"    margin: mean={statistics.fmean(m):+.4f} "
            f"min={min(m):+.4f} max={max(m):+.4f}"
        )
        lines.append("")
    lines.extend(_centring_probe(h, n_episodes=60))
    return "\n".join(lines)


def main() -> None:
    out = ["F3 leakage / density probe", "=" * 60, ""]
    for h in (8, 64, 256):
        acc = _probe(h, n_episodes=60)
        out.append(_report(h, acc))
        out.append("")

    text = "\n".join(out)
    print(text)


if __name__ == "__main__":
    main()
