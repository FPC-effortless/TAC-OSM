#!/usr/bin/env python
"""F0 — the retrieval ceiling of the existing router.

Pre-registered in ``docs/TACOSM-RETRIEVAL-001.md``. The question this script
answers comes before the index: *does the relevance signal survive in the
existing router's score distribution, at large H?*

The router scores all H candidates. Instead of sampling or taking an argmax,
retain the top-K by score and ask whether the gold candidate is among them:

    H -> score all H -> keep top-K -> is gold in K?

If recall@K saturates well — the curve rises steeply with K — then the scoring
representation is not the immediate problem, and the bottleneck is efficient
*search*: an index that narrows H to K before scoring is worth building (F1).

If recall@K saturates badly — even K=16 does not recover the gold at large H —
then the problem is upstream of search, in the state, query, candidate or
scoring representation, and an index built around an inadequate representation
would return the wrong candidates faster. That distinction is the point of
running this before F1.

## What is held fixed

Following TACOSM-HS-001, the *relevant problem* is fixed and only the
irrelevant material grows: each task carries exactly one relevant item and
H-1 distractors. Gold is built once per seed and the distractor set is
extended rather than redrawn, so difficulty is not confounded with a new task
distribution at every H.

The router is trained ONCE at a fixed candidate count and then evaluated at
every H. Its weights are loaded with
:meth:`~tac_osm.router.LearnedRelationalRouter.load_weights`, and the
model-state integrity gate runs before any measurement is taken, so an
evaluation cannot silently score with an untrained vector — the error that
made TACOSM-HS-001 report a ``routing@1`` of ~1/40 from a router that had no
weights at all.

## Reported

For each (H, K):

  recall@K      P(gold in the top K by score)
  acc|retrieved P(task success | gold in top K), from executing the loop's
                decision — separates a retrieval failure from an execution one
  gold_rank     mean rank of gold under the score
  margin        mean score gap between gold and the K-th-best candidate
  C_scored      candidates scored, == H (the exhaustive cost)
  C_inspected   candidates retained, == K (the retrieval budget)

Also reported per K: `acc|retrieved`, `P(task success | gold ∈ top-K)`, from
executing the loop's own decision. A low value there means the loss is not
retrieval at all and the top-K curve is measuring something other than
task-relevant relevance.

Outcomes:

  A  recall@K saturates well  -> the representation is adequate; F1 (search)
  B  recall@K saturates badly -> the problem is upstream of search; the
     router itself must change, not the search around it

Usage:
    python scripts/measure_retrieval_ceiling.py --train-candidates 8 --steps 500
"""
from __future__ import annotations

import argparse
import statistics

from tac_osm.ablation import AblationConfig, RouterSwitch
from tac_osm.builder import build_model
from tac_osm.integrity import assert_trained, snapshot_router

# H is the candidate count: one relevant item plus H-1 distractors.
# H=1 is not expressible (_validate_shape requires n_candidates >= 2), so the
# sweep starts at 2, as in TACOSM-HS-001.
H_LEVELS = (8, 32, 64, 128, 256)
K_LEVELS = (1, 2, 4, 8, 16)


def _gold_rank(scores: list[float], gold: int) -> int:
    """1-indexed rank of gold under the router's scores (1 == best)."""
    g = scores[gold]
    return 1 + sum(1 for s in scores if s > g)


def _top_k(scores: list[float], k: int) -> set[int]:
    """Indices of the k highest scores, ties broken by position for stability."""
    return set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])


def _budgets(h: int, k_levels: tuple[int, ...]) -> tuple[int, ...]:
    """The K levels that are well-defined at this H.

    A retrieval budget larger than the candidate population is not a budget —
    it recalls everything by construction and its ``margin`` would index past
    the end of the score list. The interesting part of the curve is always
    ``K < H`` anyway, so levels at or above H are dropped rather than clamped:
    a clamped column would silently report a different K from the one a reader
    asked for.
    """
    return tuple(k for k in k_levels if k < h)


def _measure_at(arm: str, h: int, n_steps: int, weights: list[float],
                k_levels: tuple[int, ...]) -> dict:
    """Evaluate one (H, arm, weights) cell, returning per-K retrieval metrics."""
    cfg = AblationConfig(seed=0, router=RouterSwitch(type=arm))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = h
    if weights is not None:
        bm.model.router.load_weights(weights)
        bm.model.config.learn = False
    # The gate runs unconditionally: a learned arm that has not loaded a
    # trained state must not be able to measure anything.
    assert_trained(snapshot_router(bm.model.router), context=f"retrieval H={h}")

    model = bm.model
    env = model.environment
    ks = _budgets(h, k_levels)
    if not ks:
        raise ValueError(
            f"no retrieval budget is well-defined at H={h}: every requested K "
            f"is >= H, so there is no subset to retrieve"
        )

    n = 0
    ranks: list[int] = []
    in_top: dict[int, int] = {k: 0 for k in ks}
    exec_ok_when_retrieved: dict[int, int] = {k: 0 for k in ks}
    margins: dict[int, list[float]] = {k: [] for k in ks}

    for _ in range(n_steps):
        task = env.next_task(model.state)
        query = task.public()
        decision = model.router.route(query, model.state, task.candidates)
        scores = list(decision.scores)
        gold = task.target_action

        ranks.append(_gold_rank(scores, gold))

        # Execute the loop's own decision once. The decision is fixed by the
        # router and independent of K, and ``transition`` consumes the task it
        # was issued, so it must run exactly once per step: calling it inside
        # the K loop advances the stream once per budget and the second call
        # raises "no task has been issued for this step". The outcome is
        # reused across budgets, so the per-K columns are computed over one
        # shared trajectory.
        outcome = env.transition(model.state, decision.selected, query)

        for k in ks:
            retrieved = gold in _top_k(scores, k)
            if retrieved:
                in_top[k] += 1
                # P(success | gold in top K): what accuracy would look like if
                # retrieval were the only bottleneck at this budget. A low
                # value here means the loss is not retrieval at all.
                exec_ok_when_retrieved[k] += int(outcome.success)
            kth = sorted(scores, reverse=True)[k - 1]
            margins[k].append(scores[gold] - kth)
        n += 1

    out: dict[str, float] = {
        "gold_rank": statistics.fmean(ranks) if ranks else 0.0,
        "C_scored": float(h),
        "n_steps": float(n),
    }
    for k in ks:
        hit = in_top[k]
        out[f"recall@{k}"] = hit / n if n else 0.0
        out[f"acc|retr@{k}"] = (exec_ok_when_retrieved[k] / hit) if hit else 0.0
        out[f"margin@{k}"] = statistics.fmean(margins[k]) if margins[k] else 0.0
        # The fraction of steps lost *solely* to retrieval at this budget:
        # gold was reachable in principle (it was scored at all) but did not
        # make the cut. C_inspected is the retrieval budget.
        out[f"C_inspected@{k}"] = float(k)
    return out


def _train(n_candidates: int, n_steps: int, seeds: list[int]) -> list[list[float]]:
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
    ap.add_argument("--k-levels", default=",".join(str(k) for k in K_LEVELS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    levels = tuple(int(h) for h in args.levels.split(",") if h.strip())
    k_levels = tuple(int(k) for k in args.k_levels.split(",") if k.strip())

    print(f"train_candidates={args.train_candidates} steps={args.steps} "
          f"eval_steps={args.eval_steps} seeds={seeds}")
    print("the router is trained once; weights are loaded into each "
          "evaluation router (learning disabled)")
    print("the model-state integrity gate runs before every measurement")
    print()

    trained = _train(args.train_candidates, args.steps, seeds)

    # The budget set is per-H: K >= H is not a retrieval budget, so a row at a
    # small H legitimately has fewer columns than a row at a large one. The
    # header prints the widest set and each row prints only its own.
    widest = max((_budgets(h, k_levels) for h in levels), key=len)
    header_k = "  ".join(f"rec@{k:<2}" for k in widest)
    print(f"{'H':>5} {'gold_rank':>10}  {header_k}   {'C_scored':>9}")
    print(f"{'':>5} {'(mean)':>10}  "
          f"{''.join(' ' * len(f'rec@{k:<2}') for k in widest)}   "
          f"{'(exhaustive)':>9}")
    print("-" * 60)

    for h in levels:
        rows = [_measure_at("learned", h, args.eval_steps, w, k_levels)
                for w in trained]
        mean = {k2: statistics.fmean(r[k2] for r in rows) for k2 in rows[0]}
        ks = _budgets(h, k_levels)
        cells = "  ".join(f"{mean[f'recall@{k}']:>5.3f} " for k in ks)
        print(f"{h:>5} {mean['gold_rank']:>10.2f}  {cells}   {mean['C_scored']:>9.0f}")

    print()
    print("The recall curve is the F0 question. If it saturates well at large "
          "H the")
    print("scoring representation is adequate and the bottleneck is search "
          "(F1).")
    print("If it saturates badly the problem is upstream — state, query, "
          "candidate")
    print("or scoring representation — and an index would retrieve the wrong "
          "candidates faster.")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "src")
    main()
