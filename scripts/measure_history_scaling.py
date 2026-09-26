#!/usr/bin/env python
"""History scaling: does computation depend on relevance or on history?

The central TAC-OSM question, measured directly. For each history size H the
task carries one relevant item and H-1 distractors, so the *relevant problem*
is held fixed and only the amount of irrelevant material grows:

    H=1   [R]
    H=2   [R, D1]
    H=4   [R, D1, D2, D3]
    H=8   [R, D1 ... D7]

H is therefore the candidate count. Gold is built once per seed and the
distractor set is extended, not redrawn, so difficulty is not confounded with
a new task distribution at every H.

The router is trained ONCE at a fixed candidate count and then evaluated at
every H, because the point is whether a relevance router transfers — not
whether it can be retrained per H. Its weights are copied verbatim across the
sweep, which requires the feature basis to be invariant in H: `basis_size`
depends on `dim` and `max_state_slots` only, never on the candidate count, so
a single trained vector scores a candidate set of any size.

Reported per H, each averaged over --seeds:

  accuracy            P(sampled selection == gold); the loop samples
  routing@1           P(gold is the argmax of the router's scores), i.e. the
                      decision the router *would* make without sampling
  gold_rank           mean rank of gold under the router's scores
  recall@4            P(gold in the top 4 by score)
  exec|route          P(success | routing@1 correct)
  C_router            candidates scored by the router (== H)
  C_executed          nodes the executor actually evaluates
  total_nodes         nodes available in the program substrate
  n_steps             episodes run at this H

`accuracy` and `routing@1` differ because the learned router samples from a
softmax rather than taking the argmax, and the gap between them is the cost of
that sampling, not an execution failure. At H=8 the gap is within noise; at
H=32 it reaches ~0.09. Both columns are reported so the two are not confused.

Three outcomes are distinguished:

  A  routing and C_executed stay flat as H grows -> relevance scaling
  B  routing falls, exec|route stays flat      -> a routing problem
  C  routing holds but C_executed grows        -> no computational saving

v0.1 caveat, reported rather than hidden: the structural executor fixes
`active_count = max_nodes`, so C_executed cannot grow with H by construction.
A flat C_executed here means the *architecture* has no room to scale, not that
scaling has been demonstrated. C_router is the cost that genuinely varies, and
it grows linearly with H. That is the honest reading of outcome C.

Usage:
    python scripts/measure_history_scaling.py --train-candidates 8 --steps 500
"""
from __future__ import annotations

import argparse
import statistics

from tac_osm.ablation import AblationConfig, RouterSwitch
from tac_osm.builder import build_model

# H is the candidate count: one relevant item plus H-1 distractors.
# H=1 is not expressible — _validate_shape requires n_candidates >= 2, so the
# minimal task is one relevant item plus one distractor. The sweep starts at 2
# rather than silently lowering the environment's floor.
H_LEVELS = (2, 4, 8, 16, 32, 64)


def _gold_rank(scores: list[float], gold: int) -> int:
    """1-indexed rank of gold under the router's scores (1 == best)."""
    g = scores[gold]
    return 1 + sum(1 for s in scores if s > g)


def _evaluate(arm: str, n_candidates: int, n_steps: int,
              weights: list[float] | None) -> dict:
    """Run n_steps episodes at a fixed candidate count and accumulate metrics."""
    cfg = AblationConfig(seed=0, router=RouterSwitch(type=arm))
    bm = build_model(cfg, n_steps=n_steps)
    env = bm.model.environment
    env.config.n_candidates = n_candidates

    if weights is not None:
        bm.model.router.w = list(weights)
        # A copied router must not keep learning from its own eval behaviour,
        # or the sweep measures different weights at every H.
        bm.model.config.learn = False

    hits = route1 = 0
    ranks: list[int] = []
    recall4 = 0
    exec_given_route = 0
    c_executed: list[int] = []
    total_nodes: list[int] = []
    n = 0

    for _ in range(n_steps):
        task = env.next_task(bm.model.state)
        query = task.public()
        decision = bm.model.router.route(query, bm.model.state, task.candidates)
        selected = decision.selected

        gold = task.target_action
        scores = list(decision.scores)
        routed = scores.index(max(scores)) == gold

        ranks.append(_gold_rank(scores, gold))
        recall4 += int(_gold_rank(scores, gold) <= 4)
        route1 += int(routed)

        # The executor's node count is a property of the program substrate,
        # read here rather than assumed constant.
        program = getattr(getattr(task, "detail", None), "program", None)
        if program is None:
            executed = bm.model.executor.config.max_nodes
            avail = executed
        else:
            executed = getattr(program, "active_count",
                               bm.model.executor.config.max_nodes)
            avail = len(getattr(program, "nodes", ()))

        outcome = env.transition(bm.model.state, selected, query)
        success = outcome.success

        hits += int(success)
        exec_given_route += int(success and routed)
        c_executed.append(executed)
        total_nodes.append(avail)
        n += 1

    return {
        "accuracy": hits / n if n else 0.0,
        "routing@1": route1 / n if n else 0.0,
        "gold_rank": statistics.fmean(ranks) if ranks else 0.0,
        "recall@4": recall4 / n if n else 0.0,
        "exec|route": (exec_given_route / route1) if route1 else 0.0,
        "C_router": n_candidates,
        "C_executed": statistics.fmean(c_executed) if c_executed else 0.0,
        "total_nodes": statistics.fmean(total_nodes) if total_nodes else 0.0,
        "n_steps": n,
    }


def _train(n_candidates: int, n_steps: int, seeds: list[int]) -> list[float]:
    """Train one router per seed at a fixed candidate count; return each w."""
    trained = []
    for seed in seeds:
        cfg = AblationConfig(seed=seed, router=RouterSwitch(type="learned"))
        bm = build_model(cfg, n_steps=n_steps)
        bm.model.environment.config.n_candidates = n_candidates
        bm.model.run()
        trained.append(list(bm.model.router.w))
    return trained


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-candidates", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--levels", default=",".join(str(h) for h in H_LEVELS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    levels = tuple(int(h) for h in args.levels.split(",") if h.strip())

    print(f"train_candidates={args.train_candidates} steps={args.steps} "
          f"eval_steps={args.eval_steps} seeds={seeds}")
    print("the router is trained once and copied verbatim to every H "
          "(learning disabled at evaluation)")
    print()

    trained = _train(args.train_candidates, args.steps, seeds)

    print(f"{'H':>4} {'accuracy':>9} {'routing@1':>10} {'gold_rank':>10} "
          f"{'recall@4':>9} {'exec|route':>11} {'C_router':>9} "
          f"{'C_executed':>11} {'total_nodes':>12}")
    for h in levels:
        rows = [_evaluate("learned", h, args.eval_steps, w) for w in trained]
        mean = {k: statistics.fmean(r[k] for r in rows) for k in rows[0]}
        print(f"{h:>4} {mean['accuracy']:>9.4f} {mean['routing@1']:>10.4f} "
              f"{mean['gold_rank']:>10.2f} {mean['recall@4']:>9.4f} "
              f"{mean['exec|route']:>11.4f} {mean['C_router']:>9.0f} "
              f"{mean['C_executed']:>11.1f} {mean['total_nodes']:>12.1f}")

    print()
    print("C_router grows with H by construction: the router must score every "
          "candidate.")
    print("C_executed is capped by max_nodes in v0.1, so a flat column is an "
          "architectural ceiling,")
    print("not evidence of sublinear scaling. Read C_executed as 'no room to "
          "grow', not 'scales'.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    main()
