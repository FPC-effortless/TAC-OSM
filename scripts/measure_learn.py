#!/usr/bin/env python
"""TACOSM-LEARN-001 (F2) — does named exploration training fix large-H routing?

Pre-registered in ``docs/TACOSM-LEARN-001.md``. Run that document, not this
script, for the design; the script is the instrument.

TACOSM-MATCHED-001 located the large-H failure in the learning dynamics, not
the representation: the analytic vector reaches ``routing@1 = 1.0000`` at
H up to 512, while REINFORCE sees 5 successes in 500 steps at H=256 (1%,
against a 0.39% chance rate) and 5000 steps leave ``routing@1`` at 0.0700
while the weight norm grows 12-fold. F2 tests whether two *named* training-
time interventions move the number:

    baseline         no schedule; REINFORCE as MATCHED-001 trained it
    epsilon_greedy   eps_0 = 0.30, annealed linearly to 0 over T=500
    temperature      tau 2.0 -> 0.5, annealed linearly over T=500

Both act on **training-time selection only**. Neither touches the feature
basis, the environment, the candidate generator, the update rule, or the
evaluator.

## The endpoint is the deterministic ranking, not the sampled action

This is the load-bearing distinction in the experiment. ``epsilon``-greedy
changes which candidate is *sampled during training*; the scientific question
is whether that changes the learned parameters enough to change the
*deterministic ranking at evaluation*. The comparison is

    training intervention  ->  learned parameters  ->  fixed evaluator

not the reward accumulated under the exploratory policy. Reporting the latter
would measure how well the schedule explores, which is a claim about the
schedule rather than about routing.

So the evaluation cell never consults the exploration schedule: the router is
built fresh, only the trained weights are loaded onto it, and ``route`` then
samples at the configured temperature of 0.5 with ``epsilon = 0`` — the
MATCHED-001 protocol, which is what the baseline arm has to reproduce. The
training statistics (explored fraction, successes per step, the schedule's
actual epsilon/tau values) are reported **separately**, so a reader can see
the exploration ran without the endpoint being contaminated by it.

## The baseline reproduction gate

The baseline arm is the MATCHED-001 protocol. If its column does not reproduce
the published numbers the run is **invalid**: the frozen baseline is the
reference, and a run that cannot reach it has changed something it did not
declare. No arm is reported on a failed gate. The tolerance is the pre-
registered one — the seed spread, computed *before* any arm is compared.

## Design, inherited from MATCHED-001

H is the candidate count: one relevant item and H-1 distractors. Train and
eval seeds match within a cell, so the matched-H cell measures the scorer on
the tasks it trained on. Every cell is built through ``build_model`` from an
``AblationConfig`` — the single construction path — and the weights are loaded
with ``load_weights`` for evaluation. The model-state integrity gate runs
before every training and evaluation cell, and the oracle control is printed
first, per eval H.

## Outcomes

Recorded in ``docs/TACOSM-LEARN-001.md`` after the run. The pre-registered
decision rule is fixed there and is not amended by the result.

## Cost

3 arms x 3 train-H x 5 seeds = 45 training runs, plus 3 x 3 x 3 x 5 = 135
evaluation cells. Roughly 5-8 minutes on this device.

Usage:
    python scripts/measure_learn.py --steps 500 --eval-steps 100
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.ablation import AblationConfig, RouterSwitch  # noqa: E402
from tac_osm.builder import build_model  # noqa: E402
from tac_osm.contract import load_contract  # noqa: E402
from tac_osm.integrity import (  # noqa: E402
    IntegrityError,
    assert_trained,
    checkpoint_to_dict,
    snapshot_router,
)
from tac_osm.measurement import results as _results  # noqa: E402
from tac_osm.measurement import verdicts as _verdicts  # noqa: E402
from tac_osm.measurement.results import (  # noqa: E402
    Design,
    Gate,
    GateCell,
    MeasurementRecord,
    Provenance,
)
from tac_osm.router import (  # noqa: E402
    EPS_0,
    SCHEDULE_LENGTH,
    TAU_0,
    TAU_END,
    LearnedRelationalRouter,
    arm_exploration,
)

#: The experiment this script implements. Used for the contract lookup, the
#: record's provenance and the smoke report, so the three cannot disagree
#: about which pre-registration a run claims — a disagreement there is the one
#: that silently invalidates every other check.
EXPERIMENT_ID = "TACOSM-LEARN-001"

# The registered levels, unchanged from MATCHED-001. H=32 and H=128 are
# interpolations; three levels at 8x spacing is what separates a smooth
# population effect from a training artefact.
H_LEVELS = (8, 64, 256)
K_LEVELS = (1, 2, 4, 8, 16)

# The registered schedule length, written as a literal so the machine-readable
# contract in ``contracts/TACOSM-LEARN-001.json`` can be compared with this
# script without executing it. The schedules anneal over this length, so a run
# at a different length is a different intervention.
REGISTERED_STEPS = 500
assert REGISTERED_STEPS == SCHEDULE_LENGTH, (
    "the registered schedule length disagrees with the router's "
    "SCHEDULE_LENGTH; the annealing schedules run over SCHEDULE_LENGTH, so a "
    "contract that pins a different value describes a different intervention"
)

ARMS = ("baseline", "epsilon_greedy", "temperature")

# The MATCHED-001 baseline column, published in docs/TACOSM-MATCHED-001.md.
# The reproduction gate compares against these. They are the *trained-state*
# diagonal, which is what the baseline arm here trains and evaluates, so the
# H_train and H_eval match in every cell.
MATCHED_001_BASELINE = {
    (8, 8): {"routing@1": 0.6240, "recall@4": 0.9800, "delta_1": 0.6317},
    (64, 64): {"routing@1": 0.1320, "recall@4": 0.3200, "delta_1": -1.1857},
    (256, 256): {"routing@1": 0.0740, "recall@4": 0.1340, "delta_1": -0.4383},
}


def _budgets(h: int, k_levels: tuple[int, ...]) -> tuple[int, ...]:
    """The K levels that are well-defined at this H.

    A retrieval budget at or above the candidate population recalls everything
    by construction, and indexing past the end of the score list would raise.
    The interesting curve is always ``K < H``, so levels at or above H are
    dropped rather than clamped: a clamped column would silently report a
    different K from the one a reader asked for. Carried over from F0.
    """
    return tuple(k for k in k_levels if k < h)


def _gold_rank(scores: Sequence[float], gold: int) -> int:
    """1-indexed rank of gold under the router's raw scores (1 == best).

    Ties count against gold: a tie for first is rank 2, because a scorer that
    cannot break a tie cannot be said to rank gold first. The same convention
    as HS-001, F0 and MATCHED-001, so the four experiments agree on "rank".
    """
    g = scores[gold]
    return 1 + sum(1 for s in scores if s > g)


def _top_k(scores: Sequence[float], k: int) -> set[int]:
    """Indices of the k highest scores, ties broken by position for stability."""
    return set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])


def _softmax_entropy(probs: Sequence[float]) -> float:
    """Entropy of the decision distribution, in nats.

    A uniform scorer over H candidates has entropy ``ln H``, reported so the
    reference is visible: an entropy near ``ln H`` means no preference at all,
    and the recall numbers are then a statement about chance.
    """
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def _train_one(arm: str, h_train: int, seed: int, n_steps: int
               ) -> tuple[list[float], dict, dict]:
    """Train one arm at one candidate count.

    Returns ``(weights, checkpoint_manifest, training_stats)``. The training
    stats record what the exploration schedule actually did — the fraction of
    steps taken exploratorily, the successes, and the schedule's epsilon and
    tau at the first and last steps — because that is how a reader verifies
    the intervention ran as registered without having to trust the docstring.
    """
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = h_train

    router = bm.model.router
    # The arm is named, not parameterised: ``arm_exploration`` looks up the
    # pre-registered schedule and raises on an unknown name. There is no path
    # to a schedule with caller-chosen values, which is what keeps the run
    # honest about which intervention it tested.
    if isinstance(router, LearnedRelationalRouter):
        router.exploration = arm_exploration(arm)
    else:  # pragma: no cover - defensive; the config says learned
        raise IntegrityError(
            f"arm {arm!r} requires a learned router, got {type(router).__name__}"
        )

    successes = 0
    episode = bm.model.run()

    # ``route`` stamps ``"learned:explore"`` on the decision's provenance on
    # exactly the steps where the uniform draw replaced the argmax. Counting
    # from the recorded trajectory is what makes the number auditable from the
    # episode alone, without trusting a router attribute that is live only
    # during the step it describes: after ``run`` returns it holds the last
    # step's value, so the loop would report 0 or n instead of the true count.
    explored = sum(
        1 for s in episode.steps if s.decision.provenance == "learned:explore"
    )
    for step in episode.steps:
        if step.outcome.success:
            successes += 1

    checkpoint = snapshot_router(router)
    if checkpoint is None:
        raise IntegrityError(
            "trained arm carries no learned parameters; nothing to evaluate"
        )
    # Training is gated too: a run that left the weights at exactly zero would
    # archive a uniform scorer as a trained state. Same rule as MATCHED-001.
    assert_trained(checkpoint, context=f"F2 training arm={arm} H={h_train} seed={seed}")

    sched = router.exploration
    stats = {
        "arm": arm,
        "h_train": h_train,
        "seed": seed,
        "n_steps": n_steps,
        "explored_steps": explored,
        "explored_frac": explored / n_steps if n_steps else 0.0,
        "successes": successes,
        "success_rate": successes / n_steps if n_steps else 0.0,
        "chance": 1.0 / h_train,
        # The schedule at its endpoints, proving it annealed as registered.
        "epsilon_first": sched.epsilon(0) if sched else 0.0,
        "epsilon_last": sched.epsilon(n_steps) if sched else 0.0,
        "tau_first": sched.tau(0) if sched else TAU_END,
        "tau_last": sched.tau(n_steps) if sched else TAU_END,
        "updates": router.updates,
    }
    return list(checkpoint.weights), checkpoint_to_dict(checkpoint), stats


def _evaluate(h_eval: int, weights: Sequence[float], seed: int, n_steps: int) -> dict:
    """Evaluate one trained state at one candidate count, at the fixed evaluator.

    This is MATCHED-001's evaluation cell, deliberately not rewritten: the
    baseline arm must reproduce that experiment's published numbers, and
    rewriting the evaluator would be a silent change to the protocol.

    The exploration schedule is **not** consulted here. The router is built
    fresh by ``build_model`` and only ``load_weights`` sets its parameters, so
    ``router.exploration`` is ``None`` and ``route`` falls back to the
    configured temperature (0.5 at evaluation). That is how ``epsilon = 0``
    holds: the arm's schedule was on the training-time router, not this one.

    The ranking endpoints are computed from ``router.score`` — the raw dot
    products, not the softmax probabilities the decision samples from — and
    ``routing@1`` is MATCHED-001's ``raw[gold] == max(raw)``, ties included.
    ``LearnedRelationalRouter.evaluate`` is also called each step, to expose
    the strict tie-broken reading as the ``argmax@1`` audit column; it is not
    the basis of any endpoint in the decision rule.
    """
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = h_eval
    router = bm.model.router
    if not isinstance(router, LearnedRelationalRouter):  # pragma: no cover
        raise IntegrityError(
            f"evaluation requires a learned router, got {type(router).__name__}"
        )
    router.load_weights(weights)
    bm.model.config.learn = False
    # Unconditional. The HS-001 failure mode is silent, so the gate is not
    # optional: an all-zero vector yields a smooth, bounded, plausible
    # distribution and only the parameters can catch it.
    assert_trained(snapshot_router(router),
                   context=f"F2 evaluation H={h_eval} seed={seed}")

    env = bm.model.environment
    state = bm.model.state
    ks = _budgets(h_eval, K_LEVELS)
    if not ks:
        raise ValueError(
            f"no retrieval budget is well-defined at H={h_eval}: every "
            f"requested K is >= H, so there is no subset to retrieve"
        )

    n = 0
    route1 = 0
    exec_successes = 0
    argmax_hits = 0
    ranks: list[int] = []
    in_top: dict[int, int] = {k: 0 for k in ks}
    s_gold: list[float] = []
    s_best_distr: list[float] = []
    prob_margins: list[float] = []
    deltas: dict[int, list[float]] = {k: [] for k in ks}
    entropies: list[float] = []

    for _ in range(n_steps):
        task = env.next_task(state)
        query = task.public()
        gold = task.target_action

        # The step's decision comes from ``route``, not ``evaluate`` — this is
        # MATCHED-001's cell unchanged, and it is the cell the baseline arm
        # must reproduce. ``route`` samples at the configured temperature
        # (0.5 at evaluation, with ``exploration`` unset on this fresh router),
        # so the *action that drives the loop* is the sampled one and the
        # trajectory the later steps see is the same one MATCHED-001 took.
        decision = router.route(query, state, task.candidates)
        probs = list(decision.scores)
        # ``score`` is the raw dot products — not the softmax probabilities
        # the decision samples from. A margin in probability units shrinks
        # with H even for a perfect scorer, because the probabilities are
        # normalised over the candidate count; the raw products have no such
        # normalisation, so margins from them are comparable across H.
        raw = router.score(query, state, task.candidates)

        # ``routing@1`` on MATCHED-001's convention: gold's raw score is the
        # maximum, ties included. ``LearnedRelationalRouter.evaluate`` breaks
        # ties toward the last tied index instead, so counting its argmax
        # would score a tied first place as a miss and fail the gate for a
        # reason that is an artefact of the tie rule, not a property of the
        # arm. ``argmax_hits`` below carries the strict reading as an audit.
        ranks.append(_gold_rank(raw, gold))
        route1 += int(raw[gold] == max(raw))
        # The endpoint the pre-registration names, held beside the primary
        # one: the deterministic argmax. Under a trained scorer with a unique
        # top score this equals ``routing@1``; where it differs, the
        # difference is ties, and that is worth a reader being able to see.
        amax, _ = router.evaluate(query, state, task.candidates)
        argmax_hits += int(amax == gold)
        for k in ks:
            if gold in _top_k(raw, k):
                in_top[k] += 1

        s_gold.append(raw[gold])
        others = [s for i, s in enumerate(raw) if i != gold]
        s_best_distr.append(max(others))
        prob_margins.append(probs[gold] - max(p for i, p in enumerate(probs) if i != gold))
        entropies.append(_softmax_entropy(probs))
        # ``delta_K`` is the gap to the K-th *competing* score, so gold is
        # removed before ordering — the same fix as measure_matched_h.py.
        # Sorted with gold included, ``delta@1`` is zero by construction
        # whenever gold is the argmax, which is not a margin. This key is not
        # part of the reproduction gate (that compares ``delta_1`` only), so
        # the gate's published numbers are untouched by this change.
        ordered = sorted(others, reverse=True)
        for k in ks:
            deltas[k].append(raw[gold] - ordered[k - 1])

        # ``transition`` consumes the issued task, so it runs exactly once per
        # step. Calling it inside the K loop would advance the stream once per
        # budget and raise on the second call (the F0 bug). Its success is the
        # pre-registered ``accuracy`` endpoint: the system consequence, from
        # the executed loop rather than from the ranking alone. The action is
        # the sampled one, as in MATCHED-001.
        outcome = env.transition(state, decision.selected, query)
        exec_successes += int(outcome.success)
        n += 1

    out: dict[str, float] = {
        "routing@1": route1 / n if n else 0.0,
        "gold_rank": statistics.fmean(ranks) if ranks else 0.0,
        "s_gold": statistics.fmean(s_gold) if s_gold else 0.0,
        "s_best_distr": statistics.fmean(s_best_distr) if s_best_distr else 0.0,
        # The raw top-1 margin — the gap argmax has to cross, in units that
        # are comparable across H because they are not normalised over the
        # candidate count.
        "delta_1": statistics.fmean([a - b for a, b in zip(s_gold, s_best_distr)])
        if s_gold else 0.0,
        "prob_margin": statistics.fmean(prob_margins) if prob_margins else 0.0,
        "entropy": statistics.fmean(entropies) if entropies else 0.0,
        "ln_H": math.log(h_eval),
        "c_router": float(h_eval),
        # The pre-registered ``accuracy`` endpoint: the system consequence,
        # from the executed loop. The action is the one ``route`` sampled, so
        # this differs from ``routing@1`` wherever the sampler did not draw
        # the argmax — the loop's behaviour is governed by the softmax, not by
        # the raw scores, and the endpoint exists to make that visible.
        "accuracy": exec_successes / n if n else 0.0,
        # ``argmax@1`` is *not* a pre-registered endpoint. It is the strict
        # tie-broken reading of the primary one, reported because it shows
        # where a tie rule could be doing the work: it equals ``routing@1``
        # unless gold shares the maximum with a distractor. Held beside the
        # primary endpoint as an audit, never used in the decision rule.
        "argmax@1": argmax_hits / n if n else 0.0,
        "n_steps": float(n),
    }
    for k in ks:
        out[f"recall@{k}"] = in_top[k] / n if n else 0.0
    for k in ks:
        out[f"delta@{k}"] = statistics.fmean(deltas[k]) if deltas[k] else 0.0
    return out


def _oracle_check(h_eval: int, seed: int, n_steps: int) -> float:
    """Oracle accuracy at this eval H. Must be 1.0, or the task is ambiguous.

    The oracle reads ``Task.target_action``, which no real router can see, so
    its accuracy is a property of the environment. An oracle below 1.0 means
    gold is not the unique relation-satisfier and every learned-arm number
    from that H is meaningless. Printed first, before any arm table.
    """
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="oracle"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = h_eval
    ep = bm.model.run()
    return ep.accuracy


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


def _check_reproduction(rows: dict[tuple[str, int, int], dict[str, float]],
                        per_seed: dict[tuple[str, int, int], list[float]],
                        seeds: list[int],
                        h_levels: tuple[int, ...]) -> bool:
    """Gate: the baseline arm must reproduce the published MATCHED-001 numbers.

    On failure the run is invalid and no arm is reported — the frozen baseline
    is the reference, and a run that cannot reach it has changed something it
    did not declare. The tolerance is the pre-registered one: the baseline's
    own seed spread, so the gate does not demand a reproduction tighter than
    the experiment's own noise floor.

    Only the levels actually in the run are checked. A level absent from the
    run has no baseline cell to compare, and a subset run is a smoke test
    whose gate is informative rather than binding — ``main`` prints that note
    when ``--steps`` differs from the registered length.
    """
    cells = _gate_cells(rows, per_seed, h_levels)
    print()
    print("=" * 72)
    print("GATE: baseline reproduction of TACOSM-MATCHED-001")
    print("=" * 72)
    print(f"tolerance = the baseline arm's own seed spread, over {len(seeds)} seeds")
    skipped = sorted(k for k in MATCHED_001_BASELINE if k[0] not in h_levels)
    if skipped:
        print(f"not checked (absent from this run): {[k[0] for k in skipped]}")
    print()
    for cell in cells:
        mark = "PASS" if cell.passed else "FAIL"
        print(f"  H={int(cell.cell.split('x')[0]):>3} {cell.metric:<12} "
              f"published={cell.published:+.4f} observed={cell.observed:+.4f} "
              f"diff={cell.diff:+.4f} tol={cell.tol:.4f} {mark}")
    print()
    ok = all(c.passed for c in cells)
    if ok:
        print("GATE PASSED — the frozen baseline reproduces. Reporting all arms.")
    else:
        print("GATE FAILED — the baseline arm does not reproduce TACOSM-MATCHED-001")
        print("within the pre-registered tolerance. The run is invalid: no arm is")
        print("reported, because a comparison against a moved baseline is a")
        print("comparison against nothing. See docs/TACOSM-LEARN-001.md.")
    return ok


def _gate_cells(rows: dict[tuple[str, int, int], dict[str, float]],
                per_seed: dict[tuple[str, int, int], list[float]],
                h_levels: tuple[int, ...]) -> tuple[GateCell, ...]:
    """The reproduction gate's cells, computed once for the report and the record.

    The tolerance is the seed spread floored at the pre-registered 0.02 (see
    :func:`tac_osm.measurement.verdicts.materiality_threshold`). A gate with a
    zero spread would otherwise demand a bit-exact reproduction, which is
    stronger than the protocol promises.
    """
    out: list[GateCell] = []
    for (h_tr, h_ev), expected in sorted(MATCHED_001_BASELINE.items()):
        if h_tr not in h_levels:
            continue
        key = ("baseline", h_tr, h_ev)
        if key not in rows or key not in per_seed:
            continue
        for metric, want in expected.items():
            got = rows[key][metric]
            tol = _verdicts.materiality_threshold(
                _verdicts.seed_spread(per_seed[key]))
            out.append(GateCell(
                cell=f"{h_tr}x{h_ev}",
                metric=metric,
                published=want,
                observed=got,
                diff=got - want,
                tol=tol,
                passed=abs(got - want) <= tol,
            ))
    return tuple(out)


def _print_training_table(stats: dict[tuple[str, int], list[dict]],
                          h_levels: tuple[int, ...]) -> None:
    """Training-time behaviour, reported separately from the endpoint.

    These are *not* scientific endpoints. They show the intervention ran as
    registered — the explored fraction and the schedule's endpoint values —
    and the success count the scarcity hypothesis is about. The endpoint table
    below is what answers the research question; this one is its audit trail.
    """
    print()
    print("training-time behaviour (audit trail, NOT the scientific endpoint)")
    print("-" * 72)
    print(f"{'arm':<15} {'H':>4} {'explored':>9} {'successes':>10} "
          f"{'rate':>7} {'chance':>7} {'eps_f':>7} {'eps_l':>7} "
          f"{'tau_f':>7} {'tau_l':>7}")
    for arm in ARMS:
        for h in h_levels:
            cells = stats[(arm, h)]
            explored = statistics.fmean(c["explored_frac"] for c in cells)
            succ = statistics.fmean(c["successes"] for c in cells)
            rate = statistics.fmean(c["success_rate"] for c in cells)
            chance = statistics.fmean(c["chance"] for c in cells)
            print(f"{arm:<15} {h:>4} {explored:>9.3f} {succ:>10.1f} "
                  f"{rate:>7.4f} {chance:>7.4f} "
                  f"{cells[0]['epsilon_first']:>7.3f} {cells[-1]['epsilon_last']:>7.3f} "
                  f"{cells[0]['tau_first']:>7.3f} {cells[-1]['tau_last']:>7.3f}")


def _print_endpoint_table(title: str, metric: str,
                          rows: dict[tuple[str, int, int], dict[str, float]],
                          h_levels: tuple[int, ...]) -> None:
    """One endpoint as an arm x train-H matrix, evaluated at the matched H.

    Each cell is ``train-H == eval-H``: F2 evaluates the arm where it trained,
    which is the cell MATCHED-001 showed to be the worst row and the one the
    scarcity hypothesis is about. Cross-H evaluation is MATCHED-001's question,
    not this one.
    """
    values = {key: cell.get(metric) for key, cell in rows.items()}
    if all(v is None for v in values.values()):
        return
    print()
    print(title)
    header = "  ".join(f"{'H=' + str(h):>13}" for h in h_levels)
    print(f"{'arm':<16}{header}")
    print("-" * (16 + len(header)))
    for arm in ARMS:
        cells = [
            f"{rows[(arm, h, h)][metric]:>13.4f}"
            if rows[(arm, h, h)].get(metric) is not None else "--"
            for h in h_levels
        ]
        print(f"{arm:<16}{'  '.join(cells)}")


def _print_deltas(rows: dict[tuple[str, int, int], dict[str, float]],
                  per_seed: dict[tuple[str, int, int], list[float]],
                  metric: str, h: int) -> None:
    """The exploration arms minus the baseline, with the materiality verdict.

    ``delta > 0`` is an improvement. A difference counts as material only if it
    exceeds the baseline's seed spread at that H — the pre-registered standard,
    chosen because it is measurable before any arm is compared.

    The spread is taken from the table matching the metric being reported.
    ``per_seed`` carries ``routing@1`` only; without the ``recall@16`` table a
    secondary-endpoint delta would have been judged against the *primary*'s
    spread, and the verdict on the paired endpoint would have been computed
    from a number the decision rule never states.
    """
    for line in _delta_lines(rows, per_seed, metric, h):
        print(line)


def _delta_lines(rows: dict[tuple[str, int, int], dict[str, float]],
                 per_seed: dict[tuple[str, int, int], list[float]],
                 per_seed_recall: dict[tuple[str, int, int], list[float]],
                 metric: str, h: int) -> list[str]:
    """``_print_deltas`` as lines, so the JSON record and the report agree.

    The verdicts are computed once, in :func:`_delta_details`, rather than once
    for the terminal and once for the JSON; two computations of a verdict is
    how the record and the report come to disagree about what the rule decided.
    """
    details = _delta_details(rows, per_seed, per_seed_recall, metric, h)
    if not details:
        return []
    spread = details[0]["baseline_spread"]
    base = details[0]["baseline"]
    return [
        "",
        f"{metric} at H={h}: exploration arm minus baseline",
        f"  (baseline = {base:+.4f}; materiality threshold = the baseline's "
        f"seed spread, {spread:.4f})",
        *[f"  {d['arm']:<16} {d['observed']:+.4f}  "
          f"delta={d['delta']:+.4f}  {d['verdict']}" for d in details],
    ]


def _delta_details(rows: dict[tuple[str, int, int], dict[str, float]],
                   per_seed: dict[tuple[str, int, int], list[float]],
                   per_seed_recall: dict[tuple[str, int, int], list[float]],
                   metric: str, h: int) -> list[dict]:
    """One decision-rule comparison, as fields.

    The verdict comes from :func:`tac_osm.measurement.verdicts.verdict`, which
    is the one definition of the rule the repository has. It was duplicated
    here and in ``measure_surrogate.py`` before the measurement package
    existed, each with its own copy of the 0.02 floor; a drift between the two
    would have let the same delta read as material in one experiment and as
    noise in the other, with nothing in either output to show it.

    The spread is per metric: ``routing@1`` from ``per_seed``,
    ``recall@16`` from ``per_seed_recall``. A secondary endpoint judged
    against the primary's spread is not the threshold the decision rule names.
    """
    key = ("baseline", h, h)
    if key not in rows:
        return []
    table = per_seed_recall if metric == "recall@16" else per_seed
    if key not in table:
        return []
    base = rows[key][metric]
    spread = _verdicts.seed_spread(table[key])
    return [
        _verdicts.delta_detail(
            arm=arm, metric=metric, baseline=base,
            observed=rows[(arm, h, h)][metric],
            baseline_spread=spread, context={"h": h},
        )
        for arm in ARMS if arm != "baseline"
    ]


def _build_provenance(contract_source: str) -> Provenance:
    """What produced the record, so it is self-describing.

    A reader with the JSON should be able to answer "what executable
    specification produced this number?" without reconstructing a CLI
    invocation from shell history. The three identifiers are the
    pre-registration id, a content hash of that contract file, and the commit
    the script ran from.
    """
    return Provenance(
        experiment_id=EXPERIMENT_ID,
        contract_source=contract_source,
        contract_sha256=_results.contract_fingerprint(
            _results.contract_path_for(__file__, EXPERIMENT_ID)),
        git_commit=_results.git_commit(),
        script=Path(__file__).name,
        python=_results._python_version(),
        recorded_at=_results.now(),
    )


def _results_path() -> Path:
    """Where the machine-readable summary is written.

    ``results/`` is gitignored, deliberately: the committed part of a result
    is the script and the gates, and the JSON is regenerable from the
    deterministic seeds.
    """
    return _results.results_dir_for(__file__) / "learn_001.json"


def _build_record(args: argparse.Namespace,
                  seeds: list[int],
                  h_levels: tuple[int, ...],
                  rows: dict[tuple[str, int, int], dict[str, float]],
                  per_seed: dict[tuple[str, int, int], list[float]],
                  per_seed_recall: dict[tuple[str, int, int], list[float]],
                  train_stats: dict[tuple[str, int], list[dict]],
                  gate_ok: bool,
                  h_levels_for_gate: tuple[int, ...],
                  *,
                  contract_source: str,
                  deviations: tuple[dict[str, Any], ...] = (),
                  ) -> MeasurementRecord:
    """The run as one record.

    The envelope — provenance, design, gate, endpoints, decision rule, audit,
    per-seed values — is shared with the other measurement scripts through
    :mod:`tac_osm.measurement.results`, because it is what a reader needs from
    *every* result. What goes inside ``endpoints``, ``audit`` and ``per_seed``
    is F2's own: the audit here is the *training-behaviour* trail — the
    explored fraction and the schedule's epsilon/tau endpoints, which prove the
    intervention ran as registered — where F3's is the reward density. Same
    envelope, different evidence, because the two experiments have to prove
    different things.

    Nothing here is a *claim*. Every field is a number the script measured or
    a constant it was given; the one field that would be a scientific
    judgement — ``status`` — is absent on purpose, and is set on the contract
    after the run.
    """
    contract = load_contract(EXPERIMENT_ID)
    primary = contract.primary_endpoint()
    decision: list[dict] = []
    for h in h_levels:
        decision.extend(_delta_details(rows, per_seed, per_seed_recall, primary, h))
    if 256 in h_levels:
        decision.extend(
            _delta_details(rows, per_seed, per_seed_recall, "recall@16", 256))

    # The training-behaviour audit: the explored fraction, the schedule's
    # endpoints and the success count the scarcity hypothesis is about. These
    # are *not* scientific endpoints — the endpoint table answers the research
    # question, and this is what shows the registered schedule actually ran.
    audit: list[dict] = []
    for arm in ARMS:
        for h in h_levels:
            cells = train_stats[(arm, h)]
            audit.append({
                "arm": arm,
                "h": h,
                "explored_frac": statistics.fmean(c["explored_frac"] for c in cells),
                "successes_per_schedule": statistics.fmean(
                    c["successes"] for c in cells),
                "epsilon_first": cells[0]["epsilon_first"],
                "epsilon_last": cells[-1]["epsilon_last"],
                "tau_first": cells[0]["tau_first"],
                "tau_last": cells[-1]["tau_last"],
            })

    # Per-seed values, because the spread is the materiality threshold and a
    # mean alone does not show whether the spread and the delta are the same
    # order of magnitude.
    per_seed_out: list[dict] = []
    for arm in ARMS:
        for h in h_levels:
            vals = per_seed[(arm, h, h)]
            rec = per_seed_recall[(arm, h, h)]
            cell: dict = {
                "arm": arm,
                "h": h,
                "routing@1": {
                    "seeds": {str(s): v for s, v in zip(seeds, vals)},
                    "mean": statistics.fmean(vals),
                    "spread": _verdicts.seed_spread(vals),
                },
            }
            if all(v == v for v in rec):  # NaN-free, i.e. 16 < H
                cell["recall@16"] = {
                    "seeds": {str(s): v for s, v in zip(seeds, rec)},
                    "mean": statistics.fmean(rec),
                    "spread": _verdicts.seed_spread(rec),
                }
            per_seed_out.append(cell)

    smoke = bool(getattr(args, "smoke", False))
    return MeasurementRecord(
        provenance=_build_provenance(contract_source),
        design=Design(
            steps=args.steps,
            eval_steps=args.eval_steps,
            seeds=tuple(seeds),
            h_levels=tuple(h_levels),
            k_levels=tuple(K_LEVELS),
            arms=tuple(ARMS),
            smoke=smoke,
            contract_checked=not smoke,
            deviations=deviations,
        ),
        gate=Gate(
            name="baseline reproduction of TACOSM-MATCHED-001",
            tolerance=f"the baseline arm's own seed spread, floored at "
                      f"{_verdicts.SPREAD_FLOOR:g}",
            passed=gate_ok,
            cells=_gate_cells(rows, per_seed, h_levels_for_gate),
        ),
        endpoints={
            f"{arm}@H={h}": {
                k: v for k, v in rows[(arm, h, h)].items() if k != "n_steps"
            }
            for arm in ARMS for h in h_levels
        },
        decision_rule=tuple(decision),
        audit={"training": audit},
        per_seed={"cells": per_seed_out},
    )


def _emit_results(record: MeasurementRecord, out_path: Path) -> None:
    """Write the machine-readable summary, after the gate and the report.

    The contract made the *design* machine-readable; this makes the *outcome*
    machine-readable, which is the other half of the same check. A result that
    exists only as terminal output exists only as long as the terminal's
    scrollback does.

    The record is written whether the gate passed or failed. A failed gate is
    a fact about the run, not a reason to delete the evidence: the record is
    what a later reader would use to see that the run was invalid and why.
    """
    _results.write_record(record, out_path)
    print()
    print(f"machine-readable summary written to {out_path}")
    print("  (status is left to the reader: the instrument measures, it does")
    print("   not arbitrate the decision rule)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=REGISTERED_STEPS)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--levels", default=",".join(str(h) for h in H_LEVELS))
    ap.add_argument(
        "--smoke", action="store_true",
        help="declare this run as a smoke test and skip the contract check",
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    h_levels = _parse_levels(args.levels)

    # -- 0b. The machine-readable contract ------------------------------- #
    # The pre-registration lives in ``contracts/TACOSM-LEARN-001.json`` as
    # well as in ``docs/``, and this check is what makes the run unable to
    # drift from it silently. ``require_steps`` matters most for F2: the
    # epsilon and tau schedules anneal over the registered length, so a
    # shorter run ends at a *higher* epsilon and a *higher* temperature than
    # the registered arm, and has tested a different exploration regime. A
    # run that did that used to print a NOTE and then report numbers that
    # read as though they came from the registered design.
    #
    # ``--smoke`` is the declared way to run something that is not the
    # registered design. It weakens no check; it states out loud that this
    # run is not a result, and prints its own deviation, so the output cannot
    # be mistaken for a measurement.
    contract_source = f"contracts/{EXPERIMENT_ID}.json"
    contract = load_contract(EXPERIMENT_ID)
    deviations: tuple[dict[str, Any], ...] = ()
    if args.smoke:
        deviations = tuple(_results.report_smoke(
            contract, EXPERIMENT_ID, steps=args.steps, eval_steps=args.eval_steps,
            h_levels=h_levels, seeds=seeds, arms=ARMS,
        ))
    else:
        contract.require_steps(args.steps)
        contract.require_eval_steps(args.eval_steps)
        contract.require_levels(h_levels)
        contract.require_seeds(seeds)
        contract.require_arms(ARMS)

    if args.steps != SCHEDULE_LENGTH:
        # Retained for the ``--smoke`` path and for any reader who finds the
        # contract output unclear: the same statement in the script's own
        # voice, naming the consequence for the gate rather than for the
        # contract.
        print(f"NOTE: --steps={args.steps} but the pre-registered schedule "
              f"length is {SCHEDULE_LENGTH}. The schedules anneal over "
              f"{SCHEDULE_LENGTH} steps, so a shorter run explores more than "
              f"the registered arm at its end. The reproduction gate is "
              f"therefore informative, not binding, at this length.")

    # -- 0. The frozen protocol, printed so a reader can verify it ---------- #
    print(f"experiment: TACOSM-LEARN-001 (F2)")
    print(f"steps={args.steps} eval_steps={args.eval_steps} seeds={seeds}")
    print(f"arms={list(ARMS)}")
    print(f"evaluation: deterministic argmax over router.score "
          f"(epsilon=0, temperature={TAU_END}); exploration does not reach it")
    print(f"registered constants: eps_0={EPS_0} tau_0={TAU_0} "
          f"tau_end={TAU_END} schedule_length={SCHEDULE_LENGTH}")
    print("the model-state integrity gate runs before every measurement")
    print()

    # -- 1. Oracle sanity, before any learned number is printed ------------- #
    print("oracle accuracy (environment control; must be 1.0):")
    for h_ev in h_levels:
        accs = [_oracle_check(h_ev, s, args.eval_steps) for s in seeds]
        print(f"  H_eval={h_ev:>5}: {statistics.fmean(accs):.4f}")

    # -- 2. Train one router per (arm, train-H, seed) ---------------------- #
    # A fresh router per cell, built through the same construction path as
    # every other arm: the only difference between two cells is the named
    # exploration schedule. This is what closes the warm-start hazard the
    # pre-registration's Layer 1 section names — no arm continues from
    # another arm's weights.
    train_stats: dict[tuple[str, int], list[dict]] = {
        (arm, h): [] for arm in ARMS for h in h_levels
    }
    weights: dict[tuple[str, int, int], list[float]] = {}
    for arm in ARMS:
        for h_tr in h_levels:
            for seed in seeds:
                w, manifest, stats = _train_one(arm, h_tr, seed, args.steps)
                weights[(arm, h_tr, seed)] = w
                train_stats[(arm, h_tr)].append(stats)
                print(f"trained arm={arm:<14} H={h_tr:>4} seed={seed}: "
                      f"norm={manifest.get('parameter_norm', 0.0):.4f} "
                      f"updates={manifest.get('n_updates', 0)} "
                      f"explored={stats['explored_frac']:.3f} "
                      f"successes={stats['successes']}")
    _print_training_table(train_stats, h_levels)

    # -- 3. Evaluate every trained state at its matched H ------------------- #
    rows: dict[tuple[str, int, int], dict[str, float]] = {}
    per_seed: dict[tuple[str, int, int], list[float]] = {}
    per_seed_recall: dict[tuple[str, int, int], list[float]] = {}
    for arm in ARMS:
        for h in h_levels:
            cells = [_evaluate(h, weights[(arm, h, seed)], seed, args.eval_steps)
                     for seed in seeds]
            rows[(arm, h, h)] = {
                key: statistics.fmean(c[key] for c in cells) for key in cells[0]
            }
            # Kept per metric so the spread test and the deltas use the same
            # underlying numbers the means were computed from. The
            # ``recall@16`` table is kept separately: the secondary endpoint's
            # verdict is judged against its own spread, and a single table
            # holding only ``routing@1`` would have the paired endpoint read
            # against the primary's noise floor.
            per_seed[(arm, h, h)] = [c["routing@1"] for c in cells]
            per_seed_recall[(arm, h, h)] = [c.get("recall@16", float("nan"))
                                            for c in cells]

    # -- 4. The reproduction gate, before any arm is reported -------------- #
    gate_ok = _check_reproduction(rows, per_seed, seeds, h_levels)
    if not gate_ok:
        print()
        print("Stopping. No exploration arm is reported.")
        _emit_results(_build_record(args, seeds, h_levels, rows, per_seed,
                                    per_seed_recall, train_stats,
                                    gate_ok, h_levels,
                                    contract_source=contract_source,
                                    deviations=deviations),
                      _results_path())
        return

    # -- 5. The endpoints --------------------------------------------------- #
    print()
    print("=" * 72)
    print("ENDPOINTS — deterministic ranking at evaluation (the scientific result)")
    print("=" * 72)
    _print_endpoint_table(
        "routing@1 — argmax accuracy (the primary endpoint)",
        "routing@1", rows, h_levels)
    _print_endpoint_table("recall@4 — the shortlist signal", "recall@4", rows, h_levels)
    _print_endpoint_table("recall@8 — the retrieval budget", "recall@8", rows, h_levels)
    _print_endpoint_table("recall@16 — the budget from F0", "recall@16", rows, h_levels)
    _print_endpoint_table("delta_1 — raw top-1 margin (s_gold - s_best_distr)",
                          "delta_1", rows, h_levels)
    _print_endpoint_table("gold_rank — mean rank of gold", "gold_rank", rows, h_levels)
    _print_endpoint_table("entropy — nats; ln_H is the uniform reference",
                          "entropy", rows, h_levels)
    _print_endpoint_table("accuracy — the system consequence, from the executed loop",
                          "accuracy", rows, h_levels)
    _print_endpoint_table(
        "argmax@1 — AUDIT, not a registered endpoint: routing@1 under the strict tie rule",
        "argmax@1", rows, h_levels)
    _print_endpoint_table("prob_margin — top-1 gap in decision units",
                          "prob_margin", rows, h_levels)
    _print_endpoint_table("C_router — candidates scored (constant by construction)",
                          "c_router", rows, h_levels)

    # -- 6. The decision-rule comparison, at the primary endpoint ---------- #
    print()
    print("=" * 72)
    print("DECISION RULE — delta vs baseline at the primary endpoint")
    print("=" * 72)
    for h in h_levels:
        _print_deltas(rows, per_seed, per_seed_recall, "routing@1", h)
    # ``recall@16`` at the largest registered H is the paired secondary
    # endpoint: ``routing@1`` and ``recall@16`` together are the strong
    # outcome per the decision rule. It is only well-defined where 16 < H.
    if 256 in h_levels:
        print()
        _print_deltas(rows, per_seed, per_seed_recall, "recall@16", 256)

    # -- 7. Per-seed values, so a mean cannot hide the shape --------------- #
    # Audit item 5: per-seed values for the endpoints the decision rule reads,
    # not only the means. The spread test runs on these, and a 5-seed mean that
    # hides a 4-seed collapse and a 1-seed success is not a result.
    print()
    print("=" * 72)
    print("PER-SEED VALUES — the endpoints the decision rule reads")
    print("=" * 72)
    for metric, table in (("routing@1", per_seed), ("recall@16", per_seed_recall)):
        for h in h_levels:
            if 16 not in _budgets(h, K_LEVELS) and metric == "recall@16":
                continue
            print()
            print(f"{metric} at H={h}, per seed")
            print(f"{'arm':<16}" + "".join(f"{'seed' + str(s):>10}" for s in seeds)
                  + f"{'mean':>10}{'spread':>9}")
            print("-" * (16 + 10 * len(seeds) + 19))
            for arm in ARMS:
                vals = table[(arm, h, h)]
                mean = statistics.fmean(vals)
                spread = _verdicts.seed_spread(vals)
                cells = "".join(f"{v:>10.4f}" for v in vals)
                print(f"{arm:<16}{cells}{mean:>10.4f}{spread:>9.4f}")

    print()
    print("How to read this:")
    print("  The endpoint tables are the ranking at evaluation, computed from")
    print("  router.score with the exploration schedule unset on the eval")
    print("  router — epsilon = 0, temperature = 0.5, no sampling in the")
    print("  ranking. The training-behaviour table above is an audit trail")
    print("  proving the schedules ran as registered; it is not a scientific")
    print("  endpoint.")
    print()
    print("  A MATERIAL improvement in routing@1 at H=256 is the outcome the")
    print("  scarcity hypothesis predicts: more successes per step during")
    print("  training -> a scorer with a usable large-H ranking. 'Within seed")
    print("  noise' is not a representation verdict — see the pre-registration's")
    print("  'What a negative result does not license' section, and C10.")
    print()
    print("  Interpretation and the consequence that fires are recorded in")
    print("  docs/TACOSM-LEARN-001.md under 'Outcomes'.")

    # -- 8. The machine-readable summary ---------------------------------- #
    _emit_results(_build_record(args, seeds, h_levels, rows, per_seed,
                                per_seed_recall, train_stats, gate_ok, h_levels,
                                contract_source=contract_source,
                                deviations=deviations),
                  _results_path())


if __name__ == "__main__":
    main()
