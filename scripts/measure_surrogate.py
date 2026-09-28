#!/usr/bin/env python
"""TACOSM-SURROGATE-001 (F3) — does a dense surrogate reward move large-H routing?

Pre-registered in ``docs/TACOSM-SURROGATE-001.md``. Run that document, not
this script, for the design; the script is the instrument.

TACOSM-LEARN-001 (F2) closed two exploration interventions and neither moved
the endpoint, while ``epsilon_greedy`` explored 15.2% of steps and saw *more*
successes than the baseline (3.8 vs 2.4 per 500 at H=256, against a 0.0039
chance rate). Selection is not the binding constraint. What both arms left
alone is the reward itself: binary, terminal, non-zero on 0.0056 of H=256
steps. F3 changes one thing — the *density* of the training signal — and
nothing else:

    baseline              float(outcome.success); the MATCHED-001 protocol
    analytic_margin       s_gold - max_{j != gold} s_j, centred by a
                          running mean of itself
    analytic_margin_clipped   as above, clipped to [0, 1] before centring

All three act on the scalar passed as ``reward`` to
``LearnedRelationalRouter.update``. Neither the feature basis, the update
rule, the exploration (``epsilon = 0`` throughout, as baseline), the executed
action, the evaluator, nor the task stream differs between arms.

## The leakage boundary, and why the surrogate is not "gold-free"

The pre-registration separates two questions. The *interface* question — does
``features()`` or the router see the gold index — is settled by ``leakage.py``
and is unchanged: the surrogate is computed outside the router and reaches
``update`` through the one scalar argument it already accepted. The
*supervision* question is different, and the first draft of the
pre-registration got it wrong: the environment sets ``target_action =
gold_index`` and scores ``success = action == target_action``, so
``float(outcome.success)`` is itself ``1[a_t = gold]``. **The baseline reward
is already a gold-anchored scalar.** Both rewards are the same logical kind;
the intervention is the *density* of a signal with the anchor held fixed.
What a positive result licenses and does not license is stated in the
pre-registration, before the run.

## The hook, and the loop change it required

The reward is computed by the *loop*, not the router, so the arm cannot be a
router config field the way F2's exploration was. ``ModelConfig`` gained one
field, ``reward_fn``, consulted once per step from ``TacOsmModel.step``;
``None`` is the baseline and is bit-for-bit identical to the pre-F3
hardcoded ``float(outcome.success)`` call. The three arms are three named
hooks reached by name through ``arm_reward`` — a lookup, not a constructor,
so no caller can reach the training path with a hook of its own. The hook
cannot widen the router's inputs, because the router still receives a bare
``float``.

## The endpoint is the deterministic ranking, not the sampled action

As in F2: the intervention is training-time, the scientific question is
whether the learned *parameters* changed enough to change the deterministic
ranking at evaluation. The eval router is rebuilt fresh, only
``load_weights`` is applied, ``learn = False``, and ``routing@1`` comes from
``router.score`` exactly as MATCHED-001 computed it. There is no path from
the surrogate to the reported endpoint.

## The baseline reproduction gate

The ``baseline`` arm is MATCHED-001's protocol. If its column does not
reproduce the published numbers the run is **invalid** and no arm is
reported. The tolerance is the pre-registered one: the baseline's own seed
spread, computed before any arm is compared.

## Two findings measured before any arm ran

Both are recorded in the pre-registration, and both changed the design:

* **The centring rule defeated itself.** The first draft centred by
  subtracting ``mean(scores)``, which is *negative* under the analytic vector
  (−0.99 at H=8, −1.65 at H=256) because distractors score strongly negative
  on the marked positions. Subtracting a negative adds, so the centred reward
  was positive on 180/180 steps — exactly the constant-positive-reward
  failure the rule was written to prevent. Replaced with a running mean of
  the surrogate itself.
* **The clip's binding constraint is the floor, not the ceiling.** The
  pre-registration's saturation caveat is written about the ceiling, where a
  margin above 1.0 collapses to a constant unit reward. Measured from a zero
  start — where every run begins — the pre-clip margin's maximum at H=256 is
  +0.088, so the ceiling is never reached; instead the clip *floors* a
  mostly-negative margin, on 89.6% of steps at H=256 and 60.2% at H=8. The
  clip arm's signal is therefore bounded not at the top but at the bottom,
  and the run reports both saturation directions rather than only the one
  the caveat anticipated.

## Outcomes

Recorded in ``docs/TACOSM-SURROGATE-001.md`` after the run. The
pre-registered decision rule is fixed there and is not amended by the
result.

## Cost

3 arms x 3 train-H x 5 seeds = 45 training runs, plus 3 x 3 x 3 x 5 = 135
evaluation cells — the same order as F2. The surrogate adds one margin
computation per training step, already dominated by ``score``'s dot products.

Usage:
    python scripts/measure_surrogate.py --steps 500 --eval-steps 100
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
from tac_osm.model import baseline_reward  # noqa: E402
from tac_osm.rewards import AnalyticMarginReward, arm_reward, baseline_arm_names  # noqa: E402
from tac_osm.router import (  # noqa: E402
    SCHEDULE_LENGTH,
    TAU_END,
    LearnedRelationalRouter,
)

#: The experiment this script implements. Used for the contract lookup, the
#: record's provenance and the smoke report, so the three cannot disagree
#: about which pre-registration a run claims — a disagreement there is the one
#: that silently invalidates every other check.
EXPERIMENT_ID = "TACOSM-SURROGATE-001"

# The registered levels, unchanged from MATCHED-001: three at 8x spacing,
# which is what separates a smooth population effect from a training
# artefact.
H_LEVELS = (8, 64, 256)
K_LEVELS = (1, 2, 4, 8, 16)

# The registered schedule length. This is ``SCHEDULE_LENGTH`` from the router
# — the two are the same value, and the assert below keeps them from drifting
# apart — but it is written here as a literal so the machine-readable contract
# in ``contracts/TACOSM-SURROGATE-001.json`` can be checked against it without
# executing this script. A contract that cannot be compared with the script
# statically is a contract that cannot detect drift, which is the whole point
# of having one.
REGISTERED_STEPS = 500
assert REGISTERED_STEPS == SCHEDULE_LENGTH, (
    "the registered schedule length disagrees with the router's "
    "SCHEDULE_LENGTH; the contract and the annealing schedule must describe "
    "one intervention, and a different length is a different one"
)

# The registered arms, written out for the same reason: a static reader sees
# the literal, a runtime reader sees the package's own registry, and the
# assert keeps the two from drifting.
ARMS = ("baseline", "analytic_margin", "analytic_margin_clipped")
assert tuple(ARMS) == baseline_arm_names(), (
    "the registered arm tuple drifted from rewards.baseline_arm_names(); the "
    "contract in contracts/TACOSM-SURROGATE-001.json and the package's own "
    "registry must name the same arms"
)

# The MATCHED-001 baseline column, published in docs/TACOSM-MATCHED-001.md.
# The reproduction gate compares against these: the *trained-state* diagonal,
# so H_train and H_eval match in every cell.
MATCHED_001_BASELINE = {
    (8, 8): {"routing@1": 0.6240, "recall@4": 0.9800, "delta_1": 0.6317},
    (64, 64): {"routing@1": 0.1320, "recall@4": 0.3200, "delta_1": -1.1857},
    (256, 256): {"routing@1": 0.0740, "recall@4": 0.1340, "delta_1": -0.4383},
}


def _budgets(h: int, k_levels: tuple[int, ...]) -> tuple[int, ...]:
    """The K levels that are well-defined at this H.

    A retrieval budget at or above the candidate population recalls
    everything by construction, and indexing past the end of the score list
    would raise. The interesting curve is always ``K < H``, so levels at or
    above H are dropped rather than clamped: a clamped column would silently
    report a different K from the one a reader asked for. Carried over from
    F0.
    """
    return tuple(k for k in k_levels if k < h)


def _gold_rank(scores: Sequence[float], gold: int) -> int:
    """1-indexed rank of gold under the router's raw scores (1 == best).

    Ties count against gold: a tie for first is rank 2, because a scorer that
    cannot break a tie cannot be said to rank gold first. The same convention
    as HS-001, F0, MATCHED-001 and LEARN-001, so the five experiments agree
    on "rank".
    """
    g = scores[gold]
    return 1 + sum(1 for s in scores if s > g)


def _top_k(scores: Sequence[float], k: int) -> set[int]:
    """Indices of the k highest scores, ties broken by position for stability."""
    return set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])


def _softmax_entropy(probs: Sequence[float]) -> float:
    """Entropy of the decision distribution, in nats.

    A uniform scorer over H candidates has entropy ``ln H``, reported so the
    reference is visible: an entropy near ``ln H`` means no preference at
    all, and the recall numbers are then a statement about chance.
    """
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def _train_one(arm: str, h_train: int, seed: int, n_steps: int
               ) -> tuple[list[float], dict, dict]:
    """Train one arm at one candidate count.

    Returns ``(weights, checkpoint_manifest, training_stats)``. The training
    stats record the reward the arm actually consumed — the density audit the
    pre-registration requires — because that is how a reader verifies the
    intervention ran as registered without having to trust the docstring.
    """
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = h_train

    router = bm.model.router
    if not isinstance(router, LearnedRelationalRouter):  # pragma: no cover
        raise IntegrityError(
            f"arm {arm!r} requires a learned router, got {type(router).__name__}"
        )
    # The arm is named, not parameterised: ``arm_reward`` looks up the
    # pre-registered hook and raises on an unknown name. There is no path to
    # a hook with caller-chosen clip bounds, which is what keeps the run
    # honest about which intervention it tested. The fresh instance is
    # deliberate: the hook carries a running mean, a property of one
    # trajectory, so sharing it between cells would silently change the
    # centring without changing any visible config.
    #
    # The baseline arm is set to ``baseline_reward`` rather than left at
    # ``None`` deliberately. ``None`` and ``baseline_reward`` are provably the
    # same update — that is what the pre-F3 reproduction gate asserts — but
    # naming the function here means every arm goes through the same code
    # path, so two arms differ only in *which named hook* is installed and
    # never in *whether a hook is consulted at all*.
    hook = arm_reward(arm)
    bm.model.config.reward_fn = hook if hook is not None else baseline_reward

    episode = bm.model.run()
    successes = sum(1 for s in episode.steps if s.outcome.success)

    checkpoint = snapshot_router(router)
    if checkpoint is None:
        raise IntegrityError(
            "trained arm carries no learned parameters; nothing to evaluate"
        )
    # Training is gated too: a run that left the weights at exactly zero
    # would archive a uniform scorer as a trained state. Same rule as
    # MATCHED-001 and LEARN-001.
    assert_trained(checkpoint,
                   context=f"F3 training arm={arm} H={h_train} seed={seed}")

    # The baseline arm has no hook, so its reward density is computed from
    # the recorded trajectory: the outcome indicator *is* the reward, which
    # is the point of the density comparison. The surrogate arms' equivalent
    # comes from the hook; the two are normalised to one schema so the audit
    # table reads them identically.
    if hook is None:
        reward_stats: dict[str, float] = {
            "reward_nonzero_frac": successes / n_steps if n_steps else 0.0,
            "reward_mean": successes / n_steps if n_steps else 0.0,
            "reward_min": 0.0,
            "reward_max": 1.0,
            "sat_hi_frac": 0.0,
            "sat_lo_frac": 0.0,
        }
    else:
        reward_stats = hook.stats()

    stats = {
        "arm": arm,
        "h_train": h_train,
        "seed": seed,
        "n_steps": n_steps,
        "successes": successes,
        "success_rate": successes / n_steps if n_steps else 0.0,
        "chance": 1.0 / h_train,
        "updates": router.updates,
        **reward_stats,
    }
    return list(checkpoint.weights), checkpoint_to_dict(checkpoint), stats


def _evaluate(h_eval: int, weights: Sequence[float], seed: int, n_steps: int) -> dict:
    """Evaluate one trained state at one candidate count, at the fixed evaluator.

    This is MATCHED-001's evaluation cell, deliberately not rewritten: the
    baseline arm must reproduce that experiment's published numbers, and
    rewriting the evaluator would be a silent change to the protocol.

    The reward hook is **not** consulted here. The router is built fresh by
    ``build_model`` and only ``load_weights`` sets its parameters, so
    ``ModelConfig.reward_fn`` is ``None`` and ``learn`` is ``False``. That is
    how the surrogate is kept out of the endpoint: there is no path from it
    to the reported ``routing@1``.

    The ranking endpoints are computed from ``router.score`` — the raw dot
    products, not the softmax probabilities the decision samples from — and
    ``routing@1`` is MATCHED-001's ``raw[gold] == max(raw)``, ties included.
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
                   context=f"F3 evaluation H={h_eval} seed={seed}")

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

        # The step's decision comes from ``route`` — MATCHED-001's cell
        # unchanged, and the cell the baseline arm must reproduce. ``route``
        # samples at the configured temperature (0.5 at evaluation, with no
        # exploration on this fresh router), so the *action that drives the
        # loop* is the sampled one.
        decision = router.route(query, state, task.candidates)
        probs = list(decision.scores)
        # ``score`` is the raw dot products — not the softmax probabilities
        # the decision samples from. A margin in probability units shrinks
        # with H even for a perfect scorer, because the probabilities are
        # normalised over the candidate count; the raw products have no such
        # normalisation, so margins from them are comparable across H.
        raw = router.score(query, state, task.candidates)

        # ``routing@1`` on MATCHED-001's convention: gold's raw score is the
        # maximum, ties included. ``evaluate`` breaks ties toward the last
        # tied index instead, so counting its argmax would score a tied
        # first place as a miss. ``argmax_hits`` below carries the strict
        # reading as an audit.
        ranks.append(_gold_rank(raw, gold))
        route1 += int(raw[gold] == max(raw))
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
        ordered = sorted(others, reverse=True)
        for k in ks:
            deltas[k].append(raw[gold] - ordered[k - 1])

        # ``transition`` consumes the issued task, so it runs exactly once
        # per step. Its success is the pre-registered ``accuracy`` endpoint:
        # the system consequence, from the executed loop rather than from the
        # ranking alone. The action is the sampled one, as in MATCHED-001.
        outcome = env.transition(state, decision.selected, query)
        exec_successes += int(outcome.success)
        n += 1

    out: dict[str, float] = {
        "routing@1": route1 / n if n else 0.0,
        "gold_rank": statistics.fmean(ranks) if ranks else 0.0,
        "s_gold": statistics.fmean(s_gold) if s_gold else 0.0,
        "s_best_distr": statistics.fmean(s_best_distr) if s_best_distr else 0.0,
        "delta_1": statistics.fmean([a - b for a, b in zip(s_gold, s_best_distr)])
        if s_gold else 0.0,
        "prob_margin": statistics.fmean(prob_margins) if prob_margins else 0.0,
        "entropy": statistics.fmean(entropies) if entropies else 0.0,
        "ln_H": math.log(h_eval),
        "c_router": float(h_eval),
        "accuracy": exec_successes / n if n else 0.0,
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
        print("comparison against nothing. See docs/TACOSM-SURROGATE-001.md.")
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


def _print_reward_audit(train_stats: dict[tuple[str, int], list[dict]],
                        h_levels: tuple[int, ...]) -> None:
    """The density audit — the reward each arm actually consumed, per H.

    This is the audit trail the pre-registration's leakage boundary rests on.
    The claim is not that the surrogate is gold-free (it is not, and neither
    is the baseline) but that it is *dense*: non-zero on far more steps than
    the outcome indicator at the H where the failure lives. A surrogate that
    was zero on 99% of steps would be the terminal reward with extra steps,
    and would not test H_sparse.

    For the clip arm, both saturation directions are reported. The
    pre-registration's caveat is written about the *ceiling* — a margin above
    1.0 collapsing to a constant unit reward — but from a zero start the
    margin at H=256 never reaches 1.0 and the clip is *flooring* a mostly
    negative margin on ~90% of steps. Reporting only the ceiling would let
    the arm appear untested-by-saturation while the clip was silently
    zeroing most of its signal.
    """
    print()
    print("reward audit — the signal each arm's update actually consumed")
    print("-" * 72)
    print(f"{'arm':<22} {'H':>4} {'nonzero':>8} {'mean':>8} "
          f"{'min':>8} {'max':>8} {'sat_hi':>7} {'sat_lo':>7} {'succ':>6}")
    for arm in ARMS:
        for h in h_levels:
            cells = train_stats[(arm, h)]
            nz = statistics.fmean(c["reward_nonzero_frac"] for c in cells)
            mean = statistics.fmean(c["reward_mean"] for c in cells)
            rmin = min(c["reward_min"] for c in cells)
            rmax = max(c["reward_max"] for c in cells)
            hi = statistics.fmean(c["sat_hi_frac"] for c in cells)
            lo = statistics.fmean(c["sat_lo_frac"] for c in cells)
            succ = statistics.fmean(c["successes"] for c in cells)
            print(f"{arm:<22} {h:>4} {nz:>8.4f} {mean:>8.4f} "
                  f"{rmin:>8.4f} {rmax:>8.4f} {hi:>7.4f} {lo:>7.4f} {succ:>6.1f}")


def _print_endpoint_table(title: str, metric: str,
                          rows: dict[tuple[str, int, int], dict[str, float]],
                          h_levels: tuple[int, ...]) -> None:
    """One endpoint as an arm x train-H matrix, evaluated at the matched H.

    Each cell is ``train-H == eval-H``: F3 evaluates the arm where it
    trained, which is the cell MATCHED-001 showed to be the worst row and the
    one the scarcity hypothesis is about.
    """
    values = {key: cell.get(metric) for key, cell in rows.items()}
    if all(v is None for v in values.values()):
        return
    print()
    print(title)
    header = "  ".join(f"{'H=' + str(h):>13}" for h in h_levels)
    print(f"{'arm':<22}{header}")
    print("-" * (22 + len(header)))
    for arm in ARMS:
        cells = [
            f"{rows[(arm, h, h)][metric]:>13.4f}"
            if rows[(arm, h, h)].get(metric) is not None else "--"
            for h in h_levels
        ]
        print(f"{arm:<22}{'  '.join(cells)}")


def _print_deltas(rows: dict[tuple[str, int, int], dict[str, float]],
                  per_seed: dict[tuple[str, int, int], list[float]],
                  metric: str, h: int) -> None:
    """The surrogate arms minus the baseline, with the materiality verdict.

    ``delta > 0`` is an improvement. A difference counts as material only if
    it exceeds the baseline's seed spread at that H — the pre-registered
    standard, chosen because it is measurable before any arm is compared.
    """
    for line in _delta_lines(rows, per_seed, metric, h):
        print(line)


def _delta_lines(rows: dict[tuple[str, int, int], dict[str, float]],
                 per_seed: dict[tuple[str, int, int], list[float]],
                 metric: str, h: int) -> list[str]:
    """``_print_deltas`` as lines, so the JSON record and the report agree.

    The decision-rule verdicts are computed once, in
    :func:`_delta_details`, rather than once for the terminal and once for
    the JSON; two computations of a verdict is how the record and the report
    come to disagree about what the rule decided. This function renders those
    verdicts for the terminal, and it renders them only.
    """
    details = _delta_details(rows, per_seed, metric, h)
    if not details:
        return []
    spread = details[0]["baseline_spread"]
    base = details[0]["baseline"]
    lines = [
        "",
        f"{metric} at H={h}: surrogate arm minus baseline",
        f"  (baseline = {base:+.4f}; materiality threshold = the baseline's "
        f"seed spread, {spread:.4f})",
    ]
    for d in details:
        lines.append(f"  {d['arm']:<22} {d['observed']:+.4f}  "
                     f"delta={d['delta']:+.4f}  {d['verdict']}")
    return lines


def _delta_details(rows: dict[tuple[str, int, int], dict[str, float]],
                   per_seed: dict[tuple[str, int, int], list[float]],
                   metric: str, h: int) -> list[dict]:
    """One decision-rule comparison, as fields.

    The verdict comes from :func:`tac_osm.measurement.verdicts.verdict`, which
    is the one definition of the rule the repository has. It was duplicated
    here and in ``measure_learn.py`` before the measurement package existed,
    each with its own copy of the 0.02 floor; a drift between the two would
    have let the same delta read as material in one experiment and as noise in
    the other, with nothing in either output to show it.
    """
    key = ("baseline", h, h)
    if key not in rows or key not in per_seed:
        return []
    base = rows[key][metric]
    spread = _verdicts.seed_spread(per_seed[key])
    out: list[dict] = []
    for arm in ARMS:
        if arm == "baseline":
            continue
        got = rows[(arm, h, h)][metric]
        out.append(_verdicts.delta_detail(
            arm=arm, metric=metric, baseline=base, observed=got,
            baseline_spread=spread, context={"h": h},
        ))
    return out


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


def _emit_results(record: MeasurementRecord, out_path: Path) -> None:
    """Write the machine-readable summary, after the gate and the report.

    The contract made the *design* machine-readable; this makes the *outcome*
    machine-readable, which is the other half of the same check. A result that
    exists only as terminal output exists only as long as the terminal's
    scrollback does; a reader with the JSON can compare an outcome against the
    contract's decision rule without re-running anything.

    The record is written whether the gate passed or failed. A failed gate is
    a fact about the run, not a reason to delete the evidence: the record is
    what a later reader would use to see that the run was invalid and why.
    """
    _results.write_record(record, out_path)
    print()
    print(f"machine-readable summary written to {out_path}")
    print("  (status is left to the reader: the instrument measures, it does")
    print("   not arbitrate the decision rule)")


def _results_path() -> Path:
    """Where the machine-readable summary is written.

    ``results/`` is gitignored, deliberately: the committed part of a result
    is the script and the gates, and the JSON is regenerable from the
    deterministic seeds. The summary is written there so it is available to
    whoever runs the script without becoming an artifact that has to be kept
    consistent by hand.
    """
    return _results.results_dir_for(__file__) / "surrogate_001.json"


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
    *every* result. What goes inside ``endpoints``, ``reward_audit`` and
    ``per_seed`` is F3's own: this experiment keys its endpoints by arm and
    level, MATCHED-001 keys its by ``h_train x h_eval``, and neither is forced
    into the other's shape.

    Nothing here is a *claim*. Every field is a number the script measured or
    a constant it was given; the one field that would be a scientific
    judgement — ``status`` — is absent on purpose, and is set on the contract
    after the run.
    """
    contract = load_contract(EXPERIMENT_ID)
    primary = contract.primary_endpoint()
    decision: list[dict] = []
    for h in h_levels:
        decision.extend(_delta_details(rows, per_seed, primary, h))
    if 256 in h_levels:
        decision.extend(_delta_details(rows, per_seed, "recall@16", 256))

    # The reward audit is the leakage boundary's evidence, so it is recorded
    # per arm and H rather than only as a printed table.
    audit: list[dict] = []
    for arm in ARMS:
        for h in h_levels:
            cells = train_stats[(arm, h)]
            audit.append({
                "arm": arm,
                "h": h,
                "reward_nonzero_frac": statistics.fmean(
                    c["reward_nonzero_frac"] for c in cells),
                "reward_mean": statistics.fmean(c["reward_mean"] for c in cells),
                "reward_min": min(c["reward_min"] for c in cells),
                "reward_max": max(c["reward_max"] for c in cells),
                "sat_hi_frac": statistics.fmean(c["sat_hi_frac"] for c in cells),
                "sat_lo_frac": statistics.fmean(c["sat_lo_frac"] for c in cells),
                "successes_per_schedule": statistics.fmean(
                    c["successes"] for c in cells),
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
                k: v for k, v in rows[(arm, h, h)].items()
                if k != "n_steps"
            }
            for arm in ARMS for h in h_levels
        },
        decision_rule=tuple(decision),
        audit={"reward": audit},
        per_seed={"cells": per_seed_out},
    )


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
    # The pre-registration is not only a document: ``contracts/`` holds it as
    # JSON, and this is the check that makes the run unable to drift from it
    # silently. ``require_steps`` is the one that matters most here — the
    # annealing schedule is defined over the registered length, so a shorter
    # run explores *more* than registered and is not the intervention the
    # decision rule was written about. Before this check existed, a run with
    # ``--steps 20`` printed a NOTE and then reported numbers that read as
    # though they came from the registered design.
    #
    # ``--smoke`` is the declared way to run something that is not the
    # registered design. It does not weaken any check; it says out loud that
    # this run is not a result, which is the difference between a smoke test
    # and an unreported deviation. A smoke run still prints its deviation in
    # full, below, so its output cannot be mistaken for a measurement.
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
        print(f"NOTE: --steps={args.steps} but the pre-registered length is "
              f"{SCHEDULE_LENGTH}. The reproduction gate is therefore "
              f"informative, not binding, at this length.")

    # -- 0. The frozen protocol, printed so a reader can verify it ---------- #
    print(f"experiment: TACOSM-SURROGATE-001 (F3)")
    print(f"steps={args.steps} eval_steps={args.eval_steps} seeds={seeds}")
    print(f"arms={list(ARMS)}")
    print(f"intervention: the scalar passed as `reward` to update, and nothing else")
    print(f"exploration: none, in any arm (epsilon=0, temperature={TAU_END})")
    print(f"evaluation: router.score raw dot products; the surrogate does not "
          f"reach it (learn=False, reward_fn=None on the eval router)")
    print("the model-state integrity gate runs before every measurement")
    print()

    # -- 1. Oracle sanity, before any learned number is printed ------------- #
    print("oracle accuracy (environment control; must be 1.0):")
    for h_ev in h_levels:
        accs = [_oracle_check(h_ev, s, args.eval_steps) for s in seeds]
        print(f"  H_eval={h_ev:>5}: {statistics.fmean(accs):.4f}")

    # -- 2. Train one router per (arm, train-H, seed) ---------------------- #
    # A fresh router and a fresh hook per cell, built through the same
    # construction path as every other arm: the only difference between two
    # cells is the named reward. This is what closes the warm-start hazard —
    # no arm continues from another arm's weights, and no arm carries another
    # arm's running-mean baseline.
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
                print(f"trained arm={arm:<20} H={h_tr:>4} seed={seed}: "
                      f"norm={manifest.get('parameter_norm', 0.0):.4f} "
                      f"updates={manifest.get('n_updates', 0)} "
                      f"successes={stats['successes']}")
    _print_reward_audit(train_stats, h_levels)

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
            # underlying numbers the means were computed from. Both are also
            # printed per seed below — audit item 5, which a 5-seed mean that
            # hides a 4-seed collapse and a 1-seed success would fail.
            per_seed[(arm, h, h)] = [c["routing@1"] for c in cells]
            per_seed_recall[(arm, h, h)] = [c.get("recall@16", float("nan"))
                                            for c in cells]

    # -- 4. The reproduction gate, before any arm is reported -------------- #
    gate_ok = _check_reproduction(rows, per_seed, seeds, h_levels)
    if not gate_ok:
        print()
        print("Stopping. No surrogate arm is reported.")
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
    _print_endpoint_table("delta@2 — gap to the 2nd competitor",
                          "delta@2", rows, h_levels)
    _print_endpoint_table("delta@4 — gap to the 4th competitor",
                          "delta@4", rows, h_levels)
    _print_endpoint_table("delta@8 — gap to the 8th competitor",
                          "delta@8", rows, h_levels)
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
        _print_deltas(rows, per_seed, "routing@1", h)
    # ``recall@16`` at the largest registered H is the paired secondary
    # endpoint: ``routing@1`` and ``recall@16`` together are the strong
    # outcome per the decision rule. It is only well-defined where 16 < H.
    if 256 in h_levels:
        print()
        _print_deltas(rows, per_seed, "recall@16", 256)

    # -- 7. Per-seed values, so a mean cannot hide the shape --------------- #
    # Audit item 5: per-seed values for every metric, not only the mean. The
    # spread test runs on these, and a 5-seed mean that hides a 4-seed
    # collapse and a 1-seed success is not a result. Both endpoints the
    # decision rule reads are printed, because ``analytic_margin`` at H=256
    # is within seed noise on the primary endpoint while its seeds span
    # 0.0200-0.1400 — a mean alone would not show that the spread and the
    # delta are the same order of magnitude.
    print()
    print("=" * 72)
    print("PER-SEED VALUES — the endpoints the decision rule reads")
    print("=" * 72)
    for metric, table in (("routing@1", per_seed), ("recall@16", per_seed_recall)):
        for h in h_levels:
            ks = _budgets(h, K_LEVELS)
            if 16 not in ks and metric == "recall@16":
                continue
            print()
            print(f"{metric} at H={h}, per seed")
            print(f"{'arm':<22}" + "".join(f"{'seed' + str(s):>10}" for s in seeds)
                  + f"{'mean':>10}{'spread':>9}")
            print("-" * (22 + 10 * len(seeds) + 19))
            for arm in ARMS:
                vals = table[(arm, h, h)]
                mean = statistics.fmean(vals)
                spread = _verdicts.seed_spread(vals)
                cells = "".join(f"{v:>10.4f}" for v in vals)
                print(f"{arm:<22}{cells}{mean:>10.4f}{spread:>9.4f}")

    print()
    print("How to read this:")
    print("  The endpoint tables are the ranking at evaluation, computed from")
    print("  router.score with learning disabled and no reward hook on the")
    print("  eval router. The reward audit above is the density the arms'")
    print("  updates actually consumed; it is the audit trail the leakage")
    print("  boundary rests on, not a scientific endpoint.")
    print()
    print("  The surrogate is anchored to the gold index, exactly as")
    print("  float(outcome.success) is: the intervention is the DENSITY of a")
    print("  signal of the same logical kind, with the anchor held fixed. See")
    print("  the pre-registration's 'The leakage boundary' for what a")
    print("  positive and a negative result each license.")
    print()
    print("  Interpretation and the consequence that fires are recorded in")
    print("  docs/TACOSM-SURROGATE-001.md under 'Status after the run'.")

    # -- 8. The machine-readable summary ---------------------------------- #
    _emit_results(_build_record(args, seeds, h_levels, rows, per_seed,
                                per_seed_recall, train_stats, True, h_levels,
                                contract_source=contract_source,
                                deviations=deviations),
                  _results_path())


if __name__ == "__main__":
    main()
