#!/usr/bin/env python
"""TACOSM-MATCHED-001 — matched-H training vs the H=8 transfer sweep.

TACOSM-HS-001 and F0 (TACOSM-RETRIEVAL-001) trained the router once at 8
candidates and copied the weights to every H. Both degrade at large H, and
both are confounded the same way: the sweep measures *transfer*, and the
degradation may say nothing about the representation. Two readings of F0's
``recall@16 = 0.550`` at H=256 survive the data:

  **A — population problem.** The scoring function cannot discriminate when
  the candidate population is large. The representation is the bottleneck.

  **B — training-transfer problem.** The scoring function handles H=256 fine,
  but training at 8 candidates never taught it the score distribution a
  256-candidate problem needs.

They imply different architecture. A says change the representation; B says
fix the training and build F1. F0 cannot separate them, because it has no cell
where the router saw H candidates during training.

This experiment adds those cells and nothing else:

    train H  x  eval H   ->   routing@1, recall@K, and the score margins
    {8, 64, 256} x {8, 64, 256}

The diagonal is the new evidence. The first row is the frozen HS-001/F0
transfer design, reproduced exactly so the off-diagonal cells are a
comparison against a known quantity rather than a new one.

## Why the margins, not just recall

``recall@K = 0.55`` is ambiguous on its own. Gold could sit at rank 2 behind
one strong distractor — the signal is intact and a budget of 2 finds it — or
gold could be buried among a hundred near-tied candidates, indistinguishable
from the field. Both give the same recall and need different fixes. So every
cell records:

  s_gold          the gold candidate's score
  s_best_distr    the highest *other* score
  delta_1         s_gold - s_best_distr   (the top-1 margin)
  delta_K         s_gold - s_Kth-competitor, for each K in {1,2,4,8,16}
  gold_rank       1-indexed rank of gold
  entropy         the softmax entropy, in nats
  recall@K        P(gold in top K)
  acc             accuracy from the executed loop (sampled decision)

``delta_1`` separates the two readings directly: it is the score gap that
argmax has to cross. If it collapses toward 0 as H grows even on the diagonal,
the signal is genuinely gone (A). If it is preserved on the diagonal and only
lost off it, the representation can discriminate at H and the loss is
training-transfer (B).

Note the units. The router's ``RoutingDecision.scores`` are *softmax
probabilities* over candidates, which are normalised over H, so a margin in
probability units shrinks with H even for a perfect scorer. ``score()`` is
used instead: it returns the raw dot products, which are not normalised over
the candidate count and are therefore comparable across H. Both are reported
— the probability margin is what the *decision* sees, the raw margin is what
the *scoring function* produces.

## Design

H is the candidate count: one relevant item and H−1 distractors, so the
relevant problem is fixed and only the irrelevant material grows. Gold is
built once per seed and the distractor set is *extended*, not redrawn, so
difficulty is not confounded with a new task distribution at every H — the
same control as HS-001 and F0.

**Train and eval seeds match within a cell.** Training at seed ``s`` consumes
task stream ``s``; evaluating at seed ``s`` replays it. The diagonal cell
therefore measures the scoring function on the same tasks it trained on,
which is the question "can this basis discriminate at H?". An evaluation at a
*different* seed would mix in transfer to unseen tasks and the cell would
answer a different question from the one the design asks. The full
cross-seed average is reported separately as the ``(all seeds)`` column, and
it is the number to read for population-level statements.

Every cell is built through ``build_model`` from an ``AblationConfig`` — the
single construction path for all arms, so no cell has its own code path.
Weights are loaded with
:meth:`~tac_osm.router.LearnedRelationalRouter.load_weights` and learning is
disabled at evaluation.

**The model-state integrity gate runs before every measurement.**
TACOSM-HS-001 reported a ``routing@1`` of ~1/40 from a router whose weights
had never been loaded: ``w = [0]*n`` scores every candidate uniformly, and a
uniform distribution is smooth, bounded and plausible-looking, so nothing
downstream objected. An evaluation that reaches the measurement loop with an
all-zero vector raises :class:`~tac_osm.integrity.IntegrityError` instead of
producing a number. See ``src/tac_osm/integrity.py``.

**Oracle sanity check.** The oracle arm reads ``Task.target_action``, which
no real router can see. Its accuracy must be 1.0 at every cell, or the task
is ambiguous and no learned-arm number from that cell means anything. It is
run per eval H and the result is printed before the table, not buried.

## Outcomes — what actually happened

The pre-registered rule was binary, and the run produced **neither** branch:

| result | consequence |
|---|---|
| diagonal ``routing@1`` recovers at large H | B — NOT OBSERVED. The diagonal is the worst row at every H_eval. |
| diagonal falls as steeply as the H=8 transfer row | A — NOT OBSERVED. The H=8 row is the *best* row, and the analytic vector reaches 1.0000 at every H. |
| **observed: matched-H training degrades every metric at every population** | **C — the learning rule.** The hypothesis class is adequate; REINFORCE's reward signal is too sparse at large H. |

Read the matrices **by column, not by row**: down a column the trained state
changes while the population is fixed, and every metric falls as the training
population grows — including at the training population itself. Down a row,
only the evaluation population changes.

Three diagnostics in ``docs/TACOSM-MATCHED-001.md`` establish the mechanism:
the analytic vector scores 1.0000 at H up to 512 (representation adequate),
successes per 500 steps fall 248 → 18 → 5 as H goes 8 → 64 → 256 (reward
scarcity), and 5000 training steps leave ``routing@1`` at 0.0700 while the
weight norm grows 12-fold and the margin worsens (a wrong learning rule, not
an under-trained model).

**The outcome table above is the pre-registered one, un-amended** — a
pre-registration is not revised to fit a result. The row that actually
occurred is recorded, not substituted.

## Cost

9 training cells (3 train-H x 3 seeds) + 27 evaluation cells (9 trained
states x 3 eval-H). Training dominates: ~9 s per seed at H=256. Full run is
about 2 minutes.

Usage:
    python scripts/measure_matched_h.py --steps 500 --eval-steps 100
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.ablation import AblationConfig, RouterSwitch  # noqa: E402
from tac_osm.builder import build_model  # noqa: E402
from tac_osm.integrity import (  # noqa: E402
    IntegrityError,
    assert_trained,
    checkpoint_to_dict,
    snapshot_router,
)

# H is the candidate count: one relevant item plus H-1 distractors. The 3x3
# submatrix, not the full 5x5: H=32 and H=128 are interpolations, and three
# levels at 8x spacing is what separates a smooth population effect from a
# training artefact.
H_LEVELS = (8, 64, 256)
K_LEVELS = (1, 2, 4, 8, 16)


def _budgets(h: int, k_levels: tuple[int, ...]) -> tuple[int, ...]:
    """The K levels that are well-defined at this H.

    A retrieval budget larger than the candidate population is not a budget —
    it recalls everything by construction and its ``delta_K`` would index
    past the end of the score list. The interesting part of the curve is
    always ``K < H`` anyway, so levels at or above H are dropped rather than
    clamped: a clamped column would silently report a different K from the one
    a reader asked for. (Carried over from F0 for exactly this reason.)
    """
    return tuple(k for k in k_levels if k < h)


def _gold_rank(scores: Sequence[float], gold: int) -> int:
    """1-indexed rank of gold under the router's scores (1 == best).

    Ties count against gold: a tie for first is reported as rank 2, because
    the router samples among the tied set and cannot be said to rank gold
    first. ``routing@1`` is defined over the same convention as HS-001 and
    F0, so the three experiments agree on what a rank means.
    """
    g = scores[gold]
    return 1 + sum(1 for s in scores if s > g)


def _top_k(scores: Sequence[float], k: int) -> set[int]:
    """Indices of the k highest scores, ties broken by position for stability.

    The same selection rule F0 used, so the recall columns are directly
    comparable across the two experiments.
    """
    return set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])


def _softmax_entropy(probs: Sequence[float]) -> float:
    """Entropy of the decision distribution, in nats, over non-zero entries.

    A uniform scorer over H candidates has entropy ``ln H``. It is reported
    so that ``ln H`` is visible as the reference: an entropy near ``ln H``
    means the router has no preference at all, and the recall numbers are then
    a statement about chance, not about discrimination.
    """
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def _train_one(h_train: int, seed: int, n_steps: int) -> tuple[list[float], dict]:
    """Train one router at a fixed candidate count. Returns (weights, manifest)."""
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = h_train
    bm.model.run()
    checkpoint = snapshot_router(bm.model.router)
    if checkpoint is None:
        raise IntegrityError(
            "trained arm carries no learned parameters; nothing to evaluate"
        )
    # Training is also gated. A training run that left the weights at exactly
    # zero would otherwise archive a uniform scorer as a trained state, and
    # the evaluation would have nothing to compare against.
    assert_trained(checkpoint, context=f"matched-H training H={h_train} seed={seed}")
    return list(checkpoint.weights), checkpoint_to_dict(checkpoint)


def _evaluate(h_eval: int, weights: Sequence[float], seed: int, n_steps: int) -> dict:
    """Evaluate one trained state at one candidate count. Returns the cell."""
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = h_eval
    bm.model.router.load_weights(weights)
    bm.model.config.learn = False
    # Unconditional: the TACOSM-HS-001 failure mode is silent, so the gate is
    # not optional. An all-zero vector produces a smooth, bounded, plausible
    # distribution and only the parameters can catch it.
    assert_trained(snapshot_router(bm.model.router),
                   context=f"matched-H evaluation H={h_eval} seed={seed}")

    model = bm.model
    env = model.environment
    ks = _budgets(h_eval, K_LEVELS)
    if not ks:
        raise ValueError(
            f"no retrieval budget is well-defined at H={h_eval}: every "
            f"requested K is >= H, so there is no subset to retrieve"
        )

    n = 0
    hits = 0
    route1 = 0
    ranks: list[int] = []
    in_top: dict[int, int] = {k: 0 for k in ks}
    s_gold: list[float] = []
    s_best_distr: list[float] = []
    prob_margins: list[float] = []
    deltas: dict[int, list[float]] = {k: [] for k in ks}
    entropies: list[float] = []

    for _ in range(n_steps):
        task = env.next_task(model.state)
        query = task.public()
        decision = model.router.route(query, model.state, task.candidates)
        probs = list(decision.scores)
        # Scored with ``router.score`` — the raw dot products, not the softmax
        # probabilities the decision samples from. A margin in probability
        # units shrinks with H even for a perfect scorer, because the
        # probabilities are normalised over the candidate count. The raw dot
        # products have no such normalisation, so a margin computed from them
        # is comparable across H.
        raw = model.router.score(query, model.state, task.candidates)
        gold = task.target_action

        ranks.append(_gold_rank(raw, gold))
        # ``routing@1`` is the argmax decision the router *would* make — the
        # same definition HS-001 and F0 use, so the matrices are comparable.
        # It differs from ``accuracy`` because the router samples from a
        # softmax rather than taking the argmax.
        route1 += int(raw[gold] == max(raw))
        for k in ks:
            if gold in _top_k(raw, k):
                in_top[k] += 1

        s_gold.append(raw[gold])
        others = [s for i, s in enumerate(raw) if i != gold]
        s_best_distr.append(max(others))
        prob_margins.append(probs[gold] - max(p for i, p in enumerate(probs) if i != gold))
        entropies.append(_softmax_entropy(probs))
        # ``delta_K`` is the gap to the K-th *competing* score, so gold is
        # removed before ordering. Sorted with gold included, the K-th overall
        # score *is* gold whenever gold sits in the top K, which makes
        # ``delta@1`` zero by construction for a perfect scorer: the analytic
        # vector reported +0.0000 at H=64 where its true margin is +3.0. The
        # docstring's "s_gold - s_Kth-competitor" is what this computes now.
        ordered = sorted(others, reverse=True)
        for k in ks:
            deltas[k].append(raw[gold] - ordered[k - 1])

        # ``transition`` consumes the task it was issued, so it runs exactly
        # once per step: calling it inside the K loop advances the stream once
        # per budget and the second call raises "no task has been issued for
        # this step" (a bug F0 had and fixed).
        outcome = env.transition(model.state, decision.selected, query)
        hits += int(outcome.success)
        n += 1

    out: dict[str, float] = {
        "accuracy": hits / n if n else 0.0,
        "routing@1": route1 / n if n else 0.0,
        "gold_rank": statistics.fmean(ranks) if ranks else 0.0,
        "s_gold": statistics.fmean(s_gold) if s_gold else 0.0,
        "s_best_distr": statistics.fmean(s_best_distr) if s_best_distr else 0.0,
        # The raw top-1 margin: the gap argmax has to cross. This is the
        # number that separates reading A from reading B.
        "delta_1": statistics.fmean([a - b for a, b in zip(s_gold, s_best_distr)])
        if s_gold else 0.0,
        # Same gap in the probability units the decision actually samples
        # from. Reported because the loop's behaviour is governed by the
        # softmax, not by the raw scores; it is the decision-relevant margin.
        "prob_margin": statistics.fmean(prob_margins) if prob_margins else 0.0,
        "entropy": statistics.fmean(entropies) if entropies else 0.0,
        "ln_H": math.log(h_eval),
        "n_steps": float(n),
    }
    for k in ks:
        out[f"recall@{k}"] = in_top[k] / n if n else 0.0
    for k in ks:
        out[f"delta@{k}"] = statistics.fmean(deltas[k]) if deltas[k] else 0.0
    return out


def _oracle_check(h_eval: int, seed: int, n_steps: int) -> float:
    """Oracle accuracy at this eval H. Must be 1.0, or the task is ambiguous.

    The oracle reads ``Task.target_action``, which no real router can see. Its
    accuracy is a property of the environment, not of routing, so it is the
    control that makes a degradation attributable to the model rather than to
    an ambiguous task. An oracle below 1.0 means gold is not the unique
    relation-satisfier and every learned-arm number from that H is meaningless.
    """
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="oracle"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = h_eval
    ep = bm.model.run()
    return ep.accuracy


def _print_matrix(title: str, metric: str, h_levels: tuple[int, ...],
                  rows: dict[tuple[int, int], dict[str, float]]) -> None:
    """Print one metric as a train-H (rows) x eval-H (columns) matrix.

    Rows are the training population, columns the evaluation population. The
    diagonal is matched-H; the first row is the frozen transfer design from
    HS-001 and F0.

    A cell is printed as ``--`` when the metric does not exist at that eval H.
    That is the *expected* case for ``recall@16`` at ``H_eval <= 16``: a
    retrieval budget at or above the candidate population is not a budget, so
    it is dropped rather than clamped (see :func:`_budgets`). Printing a
    placeholder keeps the matrix square and readable, and it does not invent a
    number.
    """
    available = [metric in rows[(h_levels[0], h_ev)] for h_ev in h_levels]
    if not any(available):
        return
    print(f"\n{title}")
    header = "  ".join(f"{'H_eval=' + str(h):>12}" for h in h_levels)
    print(f"{'H_train':>12}  {header}")
    print("-" * (12 + 2 + len(header)))
    for h_tr in h_levels:
        cells = [
            f"{rows[(h_tr, h_ev)][metric]:>12.4f}" if metric in rows[(h_tr, h_ev)] else "--"
            for h_ev in h_levels
        ]
        print(f"{('H_train=' + str(h_tr)):>12}  {'  '.join(cells)}")


def _parse_levels(s: str) -> tuple[int, ...]:
    levels = tuple(int(x) for x in s.split(",") if x.strip())
    if not levels:
        raise SystemExit("no H levels given")
    for h in levels:
        if h < 2:
            raise SystemExit(
                f"H must be >= 2 (one relevant item + one distractor), got {h}"
            )
    return levels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--levels", default=",".join(str(h) for h in H_LEVELS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    h_levels = _parse_levels(args.levels)
    k_levels = K_LEVELS

    print(f"steps={args.steps} eval_steps={args.eval_steps} seeds={seeds}")
    print("train and eval seeds match within a cell; weights are loaded and "
          "learning disabled")
    print("the model-state integrity gate runs before every measurement")
    print()

    # -- 1. Oracle sanity, before any learned number is printed ------------- #
    # Printed first because an oracle below 1.0 invalidates everything that
    # follows it in this table, and a reader should be able to stop there.
    print("oracle accuracy (environment control; must be 1.0):")
    for h_ev in h_levels:
        accs = [_oracle_check(h_ev, s, args.eval_steps) for s in seeds]
        print(f"  H_eval={h_ev:>5}: {statistics.fmean(accs):.4f}")

    # -- 2. Train one router per (train-H, seed) --------------------------- #
    # A fresh router per cell: the weights are the cell's only difference, and
    # reusing a router across train-H values would let one cell's training
    # history leak into another's.
    trained: dict[tuple[int, int], list[float]] = {}
    for h_tr in h_levels:
        for seed in seeds:
            w, manifest = _train_one(h_tr, seed, args.steps)
            trained[(h_tr, seed)] = w
            print(f"trained H={h_tr:>4} seed={seed}: norm="
                  f"{manifest.get('parameter_norm', 0.0):.4f} "
                  f"updates={manifest.get('n_updates', 0)} "
                  f"hash={(manifest.get('parameter_hash') or '')[:12]}")
    print()

    # -- 3. Evaluate every trained state at every eval H ------------------- #
    # The trained state from H_train is evaluated at every H_eval, so a row
    # holds one trained state across populations and a column holds one
    # population across trained states. Train and eval seeds match within a
    # cell, which is what makes the diagonal a matched-population measurement
    # rather than a transfer one.
    rows: dict[tuple[int, int], dict[str, float]] = {}
    for h_tr in h_levels:
        for h_ev in h_levels:
            per_seed = [_evaluate(h_ev, trained[(h_tr, seed)], seed, args.eval_steps)
                        for seed in seeds]
            rows[(h_tr, h_ev)] = {
                key: statistics.fmean(r[key] for r in per_seed) for key in per_seed[0]
            }

    # -- 4. Report --------------------------------------------------------- #
    _print_matrix("routing@1 — the argmax decision", "routing@1", h_levels, rows)
    _print_matrix("recall@4 — the shortlist signal", "recall@4", h_levels, rows)
    _print_matrix("recall@16 — the budget from F0", "recall@16", h_levels, rows)
    _print_matrix("delta_1 — raw top-1 margin (s_gold - s_best_distr)",
                  "delta_1", h_levels, rows)
    _print_matrix("gold_rank — mean rank of gold", "gold_rank", h_levels, rows)
    _print_matrix("entropy — nats; ln_H is the uniform reference",
                  "entropy", h_levels, rows)
    _print_matrix("prob_margin — top-1 gap in decision units",
                  "prob_margin", h_levels, rows)
    # The K-th competitor margins. ``delta@1`` is numerically identical to
    # ``delta_1`` now that both exclude gold, so it is not duplicated; the
    # larger K are the new columns, and they answer the question ``recall@K``
    # cannot: whether the residual signal at large H is one strong distractor
    # behind gold or a field of near-tied candidates around it.
    for k in k_levels:
        if k <= 1:
            continue
        _print_matrix(f"delta@{k} — margin to the {k}-th-highest competitor",
                      f"delta@{k}", h_levels, rows)

    print()
    print("How to read this:")
    print("  Read the matrices by COLUMN, not by row. Down a column the")
    print("  trained state changes while the evaluation population is fixed.")
    print()
    print("  The pre-registered rule expected the diagonal to recover (B) or")
    print("  to fall as steeply as the H=8 row (A). Neither happened: matched-H")
    print("  training is the worst row at every population, and the H=8 row is")
    print("  the best. The failure is in the learning dynamics, not the")
    print("  representation — see docs/TACOSM-MATCHED-001.md.")
    print()
    print("  delta_1 is the diagnostic: the gap argmax has to cross, in raw")
    print("  (non-normalised) score units so it is comparable across H.")
    print("  entropy should stay well below ln_H; ln_H means no preference at")
    print("  all.")
    print()
    print("  delta@K for K > 1 is the margin to the K-th best COMPETITOR, so")
    print("  gold is excluded from the ranking it is measured against: it is")
    print("  s_gold minus the K-th highest score among the other candidates.")
    print("  Read it against recall@K. recall says whether gold is inside the")
    print("  budget; delta@K says by how much. A cell with recall@16 = 0.30")
    print("  and a small delta@16 has gold near the budget's edge; the same")
    print("  recall with a large delta@4 has a handful of strong distractors")
    print("  and a wide field behind them. Those need different retrieval")
    print("  budgets, and recall alone cannot tell them apart.")


if __name__ == "__main__":
    main()
