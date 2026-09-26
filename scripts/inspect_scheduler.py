#!/usr/bin/env python
"""Log the family schedule and cumulative exposure N_f(t) for a run.

Answers a question that must be settled before any family-level curve is
interpreted: is that curve a learning curve, or an exposure curve?

The schedule is strict round-robin over ``WorldConfig.families``
(``families[_task_index % len(families)]``), so in a mixed run each family
receives exactly ``ceil(t / |F|)`` tasks — ``N_f(t)`` is therefore a
consequence of the schedule, not something to be assumed.

But the *task stream* is also a function of the derived seed

    seed = cfg.seed * 100_003 + task_index * 7919

and the three family builders share candidate generation. ``relational`` and
``state_lookup`` both call ``_build_candidates`` with the same seed, marks,
dim, n_candidates and noise, so under a single-family config — where the task
index advances by 1 per step for both — they emit the **same candidate set and
the same gold index**. The two families then score identically not because
they are equally learnable but because the tasks are identical.

The detector is therefore **across-family**, not within-run: a single-family
run contains one family by construction and cannot exhibit a cross-family
collision no matter how degenerate its stream is.

Usage:
    python scripts/inspect_scheduler.py --n-steps 500 --seed 0
    python scripts/inspect_scheduler.py --mode cross-family
"""
from __future__ import annotations

import argparse
from collections import Counter

from tac_osm.ablation import AblationConfig, RouterSwitch
from tac_osm.builder import build_model

FAMILIES = ("relational", "state_lookup", "replay")


def _drain(n_steps: int, fams: tuple[str, ...], seed: int) -> list[dict]:
    """Run the scheduler for n_steps and record what it emitted."""
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.families = fams

    log: list[dict] = []
    for _ in range(n_steps):
        state = bm.model.state
        task = bm.model.environment.next_task(state)
        log.append({
            "task_index": bm.model.environment._task_index,
            "family": task.family,
            "n_candidates": len(task.candidates),
            "gold_action": task.target_action,
            "candidate_key": tuple(c.descriptor for c in task.candidates),
        })
        # ``transition`` clears _current and advances _step, so the next
        # next_task sees a scheduler in the same state the loop would leave it.
        bm.model.environment.transition(
            state, task.target_action, task.public()
        )
    return log


def report_schedule(n_steps: int, seed: int, fams: tuple[str, ...]) -> None:
    log = _drain(n_steps, fams, seed)
    counts = Counter(r["family"] for r in log)
    print(f"steps={n_steps} seed={seed} families={fams}")
    print(f"cumulative exposure N_f(t): {dict(counts)}")
    print(f"expected (round-robin):    "
          f"{dict(Counter({f: n_steps // len(fams) for f in fams}))}")
    print()
    print("  t   idx  family           n_cand  gold")
    for t, r in enumerate(log[:15], start=1):
        print(f"{t:4d} {r['task_index']:5d}  {r['family']:<15} "
              f"{r['n_candidates']:6d} {r['gold_action']:5d}")
    if len(log) > 15:
        print(f"  ... ({len(log) - 15} more)")


def report_cross_family(seed: int, n_steps: int) -> None:
    """The comparison the family table depends on.

    Each family is driven from the same task index, so if two of them emit the
    same candidate set the corresponding rows of the table measure one task
    stream under two labels.
    """
    streams: dict[str, list[dict]] = {}
    for fam in FAMILIES:
        streams[fam] = _drain(n_steps, (fam,), seed)

    print(f"cross-family stream identity (seed={seed}, {n_steps} steps each)")
    print()
    print(f"{'pair':<40} {'identical tasks':>16}")
    for i, a in enumerate(FAMILIES):
        for b in FAMILIES[i + 1:]:
            same = sum(1 for x, y in zip(streams[a], streams[b])
                       if x["candidate_key"] == y["candidate_key"])
            print(f"{a} / {b:<20} {same:>10}/{n_steps:>4}")

    print()
    n_exposure = min(len(v) for v in streams.values())
    print(f"N_f(t) under a single-family config: {n_exposure} for the one family")
    print("A mixed run of the same length gives "
          f"{n_exposure // len(FAMILIES)} per family — the family table is "
          "therefore an exposure table unless streams are separated.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", default="relational,state_lookup,replay")
    ap.add_argument("--mode", default="schedule",
                    choices=["schedule", "cross-family"],
                    help="'schedule' logs N_f(t); 'cross-family' tests stream identity")
    args = ap.parse_args()

    if args.mode == "cross-family":
        report_cross_family(args.seed, args.n_steps)
    else:
        fams = tuple(f for f in args.families.split(",") if f)
        report_schedule(args.n_steps, args.seed, fams)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    main()
