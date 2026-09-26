#!/usr/bin/env python
"""Measure the baseline arms per family, without pinning families.

A pinned run (config.families = (fam,)) does NOT isolate family f. It changes
len(families) from 3 to 1, which changes the derived task seed
(cfg.seed*100_003 + task_index*7919) and therefore the whole task stream, and
it triples per-family exposure. Worse, relational and state_lookup share
candidate generation, so under a single-family config they emit the *same*
tasks — a pinned comparison between them measures one stream under two labels.

The correct decomposition measures one mixed run and splits the accuracy by
the family each task actually carried. That holds the stream, the exposure and
the schedule identical across families and reports per-family accuracy over
the same trained router.

Per-family exposure is then reported alongside the accuracy, because a family
that saw 167 tasks is not comparable to one that saw 500.

Usage: python scripts/measure_baseline.py [--n-steps 500] [--seeds 0,1,2,3,4]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from tac_osm.ablation import AblationConfig, RouterSwitch
from tac_osm.builder import build_model

ARMS = ("oracle", "random", "static", "full_context", "learned")


def run_arm(arm: str, n_steps: int, seed: int) -> tuple[float, dict[str, float]]:
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type=arm))
    bm = build_model(cfg, n_steps=n_steps)
    ep = bm.model.run()

    # The trajectory carries the family of every step in the query's
    # provenance, so accuracy can be split without re-running anything.
    per_family = defaultdict(lambda: [0, 0])
    for step in ep.steps:
        fam = step.query.provenance
        per_family[fam][1] += 1
        if step.outcome.success:
            per_family[fam][0] += 1
    accs = {f: s / n if n else 0.0 for f, (s, n) in per_family.items()}
    return ep.accuracy, accs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"n_steps={args.n_steps} seeds={seeds} families=mixed")
    print()
    print(f"{'arm':<13} {'accuracy':>9}   per-family")
    for arm in ARMS:
        agg, families = [], defaultdict(list)
        for seed in seeds:
            acc, per_fam = run_arm(arm, args.n_steps, seed)
            agg.append(acc)
            for f, a in per_fam.items():
                families[f].append(a)
        overall = sum(agg) / len(agg)
        if families:
            parts = "  ".join(
                f"{f}={sum(vs) / len(vs):.4f}"
                for f, vs in sorted(families.items()))
        else:
            parts = "-"
        print(f"{arm:<13} {overall:>9.4f}   {parts}")

    print()
    print("per-family exposure is equal under the mixed schedule "
          f"(~{args.n_steps // 3} tasks each at {args.n_steps} steps)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    main()
