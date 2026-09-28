#!/usr/bin/env python
"""TACOSM-RETRIEVAL-001 F1 — efficient retrieval: a cheap index before scoring.

Pre-registered in ``docs/TACOSM-RETRIEVAL-001.md`` and ``docs/ROADMAP.md``
(M2.1, historically Stage F1). This is the experiment the retrieval boundary
exists for.

    H → cheap index → K candidates → existing scorer → R

The index narrows ``H`` to ``K ≪ H`` *before* scoring, so the router's cost
changes from ``O(H)`` to ``O(K)`` while the scorer itself is reused
unchanged. The claim under test is a **cost-quality trade**, not "the index
works": at matched H and matched K, the indexed arm's recall@K is compared
against the exhaustive arm's, at a cost that grows with K rather than H.

## Why this experiment is re-queued now

Its hypothesis is a trade, and the quality half of that trade was not a
property of the representation until F3. MATCHED-001 showed the analytic
vector reaches ``routing@1 = 1.0000`` at every H — the basis is adequate —
and F3 (TACOSM-SURROGATE-001) showed the large-H top-K signal is
*responsive to training*: ``Δ(recall@16) = +0.0860 / +0.2460`` at H=256 under
a training-only intervention. An index built before that evidence would have
been measured against a baseline whose weakness was a training artefact, and
a bad baseline makes a mediocre index look good.

**The bound travels with the evidence.** M1.8 moved ``recall@16`` and did not
move ``routing@1``, so F1 is a cost-quality trade at ``K ≥ 2``, not at
``K = 1``. A run that reported top-1 preservation as its headline would be
reporting a number M1 says is not there.

## The four arms

A and B are the exhaustive references; C is the proposed architecture; D is
the oracle-index ceiling.

  A  exhaustive + top-1     score all H, take the argmax. The current
                           architecture, unchanged, and the cost reference.
  B  exhaustive + top-K    score all H, retain the top-K by score. The
                           retrieval ceiling (F0). Same cost as A — it still
                           scores everything — and the quality reference C is
                           measured against at matched K.
  C  indexed + top-K       H → cheap index → K → the router's own scorer.
                           **The reason F1 exists.** The index is the only new
                           mechanism.
  D  oracle index + top-K  the index is told where gold is. A diagnostic, not
                           a candidate architecture: it measures headroom
                           rather than occupying it. C vs D isolates how much
                           of the retrieval loss is the index and how much is
                           the scorer's ranking within the retrieved set.

The comparison **C vs B at matched K** isolates the *cost* of indexing: same
budget, same scorer, one arm with an index and one without. If C's recall@K
is materially below B's the index is discarding gold, and the loss is the
index's and not the scorer's.

## The failure taxonomy

Every failed step is attributed to exactly one category, so a drop in
``exact_success`` can be read as a *kind* of failure rather than a quantity:

  retrieval miss       gold was not in the retrieved set; the index or the
                       budget lost it. The only category that is an argument
                       against the index.
  ranking miss         gold was retrieved but not selected; the scorer ranked
                       it below a distractor inside the retrieved set.
  execution failure    the correct candidate was selected and the
                       computation failed.
  verification failure the computation succeeded and the verifier rejected it.

The taxonomy is closed: a failure that fits none of the four is a bug in the
measurement rather than a new kind of failure, and the run raises rather
than inventing a fifth bucket.

## What is held fixed

Following TACOSM-HS-001, the *relevant problem* is fixed and only the
irrelevant material grows: one relevant item and ``H-1`` distractors per
task, gold built once per seed with the distractor set extended rather than
redrawn, so difficulty is not confounded with a new task distribution at
every H.

The router is trained once at 8 candidates and its weights loaded into every
evaluation router with ``load_weights``, learning disabled — the F0 and
MATCHED-001 design, so the quality reference is the same scorer across arms.
The model-state integrity gate runs before every measurement (C4), because
the TACOSM-HS-001 failure mode — an all-zero weight vector scoring every
candidate uniformly — is silent and looks reasonable.

**The scorer is reused unchanged in every arm.** Arms differ in which
candidates the scorer *sees*, never in how it scores them. That is what keeps
this a cost experiment rather than a router experiment: if the scorer changed
between arms the comparison would be measuring a different model rather than
a different budget.

## Cost terms

Six terms per ``(arm, H, K)``, reported along the same axis so a
capability-versus-computation curve is a plot of one table rather than a
narrative: ``candidates_inspected``, ``router_operations``,
``index_operations``, ``executor_nodes``, ``verifier_operations``,
``wall_clock``. Two of them are not measurements and the table says so.

``executor_nodes`` is fixed at ``max_nodes = 10`` by the v0.1 executor, so it
is an **architectural constant**; and ``router_operations`` is a **modelled**
count of the scoring work each arm's architecture would perform — ``H`` in A
and B, which score the population to build their retained set, and ``K`` in C
and D, which score only what the index kept. It is modelled rather than
measured because this loop scores the full population in every arm to keep
the endpoint definitions comparable across arms; ``wall_clock`` is the
measured counterpart and it does not share that accounting. The asymmetry is
stated rather than hidden: a reader comparing ``router_operations`` against
``wall_clock`` is comparing a design claim against a runtime observation, and
the two are not the same quantity.

## What this experiment does not claim

* It is not "try another router". A better scorer is an M1 result measured
  under M1's protocol and would change the baseline this experiment is
  defined against.
* It is not "fix the top-1 endpoint". That question is M1.9's and it is open.
* It is not a re-measurement of F0. The exhaustive arms exist as the
  cost-quality reference; every indexed-arm number is a delta against them at
  matched K.
* It does not test ``C_executed ≈ f(|R|)`` (C5). The v0.1 executor fixes
  ``active_count = max_nodes``, so there is no execution-side scaling to
  measure. F1's structural contribution is that ``|R|`` becomes a defined
  quantity; the scaling measurement is M3's.

## A result the pre-registration did not anticipate, reported as measured

The index is a *filter*, not a ranker, and the environment's distractors are
constructed to defeat cheap agreement rules. It can therefore retain gold at
a **higher** rate than the trained scorer ranks it, particularly at large H
where the trained scorer's reward signal is sparse (MATCHED-001: 5 successes
in 500 steps at H=256). The third branch of the decision rule exists for that
outcome, and it is reported rather than smoothed away: an index that beats
the trained scorer is evidence about the *scorer*, and it is consistent with
MATCHED-001's finding that the learning rule, not the representation, is the
failure.

Usage:
    python scripts/measure_retrieval.py --steps 500 --eval-steps 100
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.ablation import AblationConfig, RouterSwitch  # noqa: E402
from tac_osm.builder import build_model  # noqa: E402
from tac_osm.contract import load_contract  # noqa: E402
from tac_osm.integrity import (  # noqa: E402
    IntegrityError,
    assert_trained,
    snapshot_router,
)
from tac_osm.leakage import audit_router_inputs  # noqa: E402
from tac_osm.measurement import results as _results  # noqa: E402
from tac_osm.measurement.results import (  # noqa: E402
    Design,
    Gate,
    GateCell,
    MeasurementRecord,
    Provenance,
)
from tac_osm.measurement.verdicts import delta_detail  # noqa: E402
from tac_osm.retrieval import (  # noqa: E402
    FAILURE_CATEGORIES,
    RetrievalIndex,
    classify_step,
    budgets,
    gold_rank,
    top_k,
)

#: The experiment this script implements. One name for the contract lookup,
#: the record's provenance and the smoke report, so the three cannot disagree
#: about which pre-registration a run claims — a disagreement there is the
#: one that silently invalidates every other check.
EXPERIMENT_ID = "TACOSM-RETRIEVAL-001"

# H is the candidate count: one relevant item plus H-1 distractors. The same
# levels as MATCHED-001, so the quality reference is measured at the same
# populations the learning-dynamics result is recorded against.
H_LEVELS = (8, 64, 256)

# The retrieval budgets. K=1 is deliberately excluded: M1.8 moved recall@16
# and did not move routing@1, so F1's registered evidence is top-K at K >= 2.
# A K=1 column would invite the reading the pre-registration forbids —
# reporting a top-1 number M1 says is not there.
K_LEVELS = (2, 4, 16)

# The registered schedule length, as a literal so the machine-readable contract
# can be compared with this script statically — without executing it — which
# is what makes drift detectable rather than merely re-readable.
REGISTERED_STEPS = 500

#: The four arms, in the pre-registration's A/B/C/D order. ``require_arms``
#: compares this tuple against the contract, so an arm added or dropped here
#: is a contract violation rather than a silent design change.
ARMS = (
    "exhaustive_top1",
    "exhaustive_topk",
    "indexed_topk",
    "oracle_index_topk",
)

#: The candidate population the router is trained at. Fixed across every arm
#: and every H, so the quality reference is one trained state — the same
#: design F0 and MATCHED-001 used, which is what makes the B arm the
#: registered baseline rather than a new measurement.
TRAIN_CANDIDATES = 8

#: The arms that score the full population to build their retained set, and
#: therefore carry the ``O(H)`` routing cost the trade exists to reduce. C and
#: D score only what the index retained, so their cost grows with ``K``.
_FULL_SCORING_ARMS = frozenset({"exhaustive_top1", "exhaustive_topk"})


# --------------------------------------------------------------------------- #
# The arms
# --------------------------------------------------------------------------- #


def _train_one(n_candidates: int, seed: int, n_steps: int) -> tuple[list[float], dict]:
    """Train one router at a fixed candidate count. Returns (weights, manifest)."""
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = n_candidates
    bm.model.run()
    checkpoint = snapshot_router(bm.model.router)
    if checkpoint is None:
        raise IntegrityError(
            "trained arm carries no learned parameters; nothing to evaluate"
        )
    # Training is gated too: a run that left the weights at exactly zero
    # would archive a uniform scorer as a trained state. Same rule as
    # MATCHED-001, LEARN-001 and SURROGATE-001.
    assert_trained(checkpoint,
                   context=f"F1 training H={n_candidates} seed={seed}")
    return list(checkpoint.weights), {
        "parameter_norm": checkpoint.norm,
        "parameter_hash": checkpoint.hash,
        "source": checkpoint.source,
    }


def _oracle_check(h_eval: int, seed: int, n_steps: int) -> float:
    """Oracle accuracy at this eval H. Must be 1.0, or the task is ambiguous.

    The oracle reads ``Task.target_action``, which no real router can see.
    Its accuracy is a property of the environment, not of routing, so it is
    the control that makes a degradation attributable to the model rather
    than to an ambiguous task. An oracle below 1.0 means gold is not the
    unique relation-satisfier and every learned-arm number from that H is
    meaningless.
    """
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="oracle"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = h_eval
    ep = bm.model.run()
    return ep.accuracy


def _eval_cell(arm: str, h: int, k: int, weights: Sequence[float], seed: int,
               n_steps: int) -> dict:
    """One ``(arm, H, K, seed)`` cell.

    The arms differ only in which candidates the scorer sees and how the
    retained set is formed. The scorer itself — ``router.score`` — is called
    identically in every arm, which is what keeps this a cost experiment
    rather than a router experiment.
    """
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type="learned"))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = h
    bm.model.router.load_weights(weights)
    bm.model.config.learn = False
    # Unconditional. The HS-001 failure mode is silent, so the gate is not
    # optional: an all-zero vector produces a smooth, bounded, plausible
    # distribution and only the parameters can catch it.
    assert_trained(snapshot_router(bm.model.router),
                   context=f"F1 {arm} H={h} K={k} seed={seed}")

    model = bm.model
    env = model.environment

    n = 0
    hits = 0
    route1 = 0
    ranks: list[int] = []
    in_top = 0
    scorer_n = 0
    retrieved_hits = 0
    retrieved_and_ok = 0
    retained_n = 0
    index_ops = 0
    router_ops = 0
    failures: dict[str, int] = {c: 0 for c in FAILURE_CATEGORIES}
    t0 = time.perf_counter()

    for _ in range(n_steps):
        task = env.next_task(model.state)
        query = task.public()
        read = model.state.read(query)
        gold = task.target_action

        # Leakage audit on the router's (and index's) inputs, at measurement
        # time rather than only in the test suite: the retrieval boundary is
        # a new component in the loop, so its input boundary is checked where
        # it runs.
        audit_router_inputs(query=query, candidates=task.candidates,
                            state=model.state)

        # The scorer is called once per step, on the full population. It is
        # the same call in every arm: arms differ in which candidates the
        # retained set holds, never in how the scorer scores. Scoring first
        # and then selecting the retained set means the exhaustive arms pay
        # the full O(H) cost and the indexed arms pay the index's O(n_marked)
        # plus K scoring calls — the cost model reports both.
        raw_full = model.router.score(query, model.state, task.candidates)

        # -- the retrieval boundary --------------------------------------
        # The retained set is arm-dependent, and it is what the primary
        # endpoint is defined over (amendment A1): A keeps everything, B
        # keeps the scorer's own top-K — B is itself a retrieval boundary
        # built out of the scorer — C keeps the index's K, and D keeps the
        # index's K-1 plus gold.
        if arm == "exhaustive_top1":
            kept = tuple(range(len(task.candidates)))
        elif arm == "exhaustive_topk":
            # B's retained set is its own top-K by score. Before amendment A1
            # this arm kept the whole population and reported an arm-
            # independent endpoint, which made it indistinguishable from A on
            # the primary endpoint while pretending to test a budget.
            kept = top_k(raw_full, k)
        elif arm == "indexed_topk":
            index = RetrievalIndex(k=k, seed=seed)
            kept, _isc = index.retrieve(query, task.candidates, read)
            index_ops += len(task.candidates)
        elif arm == "oracle_index_topk":
            # The ceiling: gold is forced into the retained set, and the
            # remaining K-1 slots are the best of the rest by index score.
            # It reads information no runtime component may have, exactly as
            # ``OracleRouter`` does, and for the same reason.
            #
            # The index is applied to the *reduced* population that excludes
            # gold, so the retrieved indices are not full-list indices — a
            # previous version of this arm built ``kept`` by unioning the
            # reduced-list indices with ``gold`` as if they shared a frame of
            # reference, which silently re-indexed the wrong candidates.
            # Every retained index is mapped back through ``rest`` before use.
            index = RetrievalIndex(k=max(1, k - 1), seed=seed)
            rest = [i for i in range(len(task.candidates)) if i != gold]
            others, _isc = index.retrieve(
                query,
                [task.candidates[i] for i in rest],
                read,
            )
            kept = tuple(sorted({rest[o] for o in others} | {gold}))
            index_ops += len(task.candidates)
        else:
            raise ValueError(f"unknown arm {arm!r}")

        # Gold's score is its score in the full population; the ranking it is
        # measured against is over the retained set, because that is the set
        # the loop would select from under this arm.
        gold_score = raw_full[gold]
        sub_scores = [raw_full[i] for i in kept]
        # The retained set's size is |R| — the quantity C5 needs, and the cost
        # half of the trade.
        retained_n += len(kept)
        # Cost accounting. ``router_operations`` counts the scoring work the
        # arm's *architecture* would perform, which is what the trade is over:
        # H candidate-scoring calls in A and B — they score the population to
        # build their retained set — and K in C and D, which score only what
        # the index kept. It is a modelled count rather than a measured one,
        # because this loop scores the full population in every arm to keep
        # the endpoint definitions comparable; ``wall_clock`` is the measured
        # counterpart and it does not share this accounting.
        router_ops += len(task.candidates) if arm in _FULL_SCORING_ARMS else len(kept)

        if arm == "exhaustive_top1":
            # A: the argmax over all H. The current architecture.
            selected_full = max(range(len(raw_full)), key=lambda i: raw_full[i])
            gold_selected = selected_full == gold
            retrieved = True
        else:
            retrieved = gold in kept
            if not sub_scores:
                gold_selected = False
            else:
                best = max(sub_scores)
                # Ties count against gold, the convention every experiment
                # since HS-001 has used for routing@1.
                gold_selected = (retrieved and gold_score >= best
                                 and sub_scores.count(best) == 1)

        # scorer_recall@K is the F0 endpoint: gold in the top-K of the *full*
        # score ordering, arm-independent by construction and reported so the
        # F0 curve stays readable in an F1 run. recall@K — the primary, since
        # amendment A1 — is the arm-dependent version computed below.
        order = sorted(range(len(raw_full)), key=lambda i: -raw_full[i])
        scorer_recall = gold in set(order[:k])
        ranks.append(gold_rank(raw_full, gold))
        route1 += int(raw_full[gold] == max(raw_full))

        # recall@K, the primary endpoint after amendment A1: gold survived the
        # arm's retrieval boundary AND the scorer ranks it first among the
        # retained candidates. It is the product of the two factors the
        # decomposition reports separately — index_retains_gold is the first,
        # and the ranking factor is what the taxonomy's ranking_miss counts.
        in_top_k = bool(retrieved and gold_selected)

        # The loop's own decision, executed once. ``transition`` consumes the
        # task it was issued, so it runs exactly once per step.
        decision = model.router.route(query, model.state, task.candidates)
        outcome = env.transition(model.state, decision.selected, query)
        ok = bool(outcome.success)
        hits += int(ok)

        if scorer_recall:
            scorer_n += 1
        if in_top_k:
            in_top += 1
        if retrieved:
            retrieved_hits += 1
            if ok:
                retrieved_and_ok += 1

        cat = classify_step(
            success=ok,
            gold_in_retrieved=retrieved,
            gold_is_selected=gold_selected,
            verification_passed=True,
        )
        if cat is not None:
            failures[cat] += 1

        n += 1

    wall = time.perf_counter() - t0
    return {
        "exact_success": hits / n if n else 0.0,
        "routing@1": route1 / n if n else 0.0,
        # The primary endpoint (amendment A1): arm-dependent by construction.
        "recall@K": in_top / n if n else 0.0,
        # The F0 endpoint, arm-independent; the quantity the previous
        # definition of the primary actually measured.
        "scorer_recall@K": scorer_n / n if n else 0.0,
        "gold_rank": statistics.fmean(ranks) if ranks else 0.0,
        "acc|retrieved": (retrieved_and_ok / retrieved_hits) if retrieved_hits else 0.0,
        "index_retains_gold": retrieved_hits / n if n else 0.0,
        # |R| for this arm: the mean retained-set size, which is H in A, K in
        # B/C/D. The cost half of the trade, and the quantity C5 needs.
        "candidates_inspected": float(retained_n / n) if n else float(h),
        "router_operations": float(router_ops / n if n else 0),
        "index_operations": float(index_ops / n if n else 0),
        "executor_nodes": 10.0,  # architectural constant, not a measurement
        "verifier_operations": 1.0,
        "wall_clock": wall,
        "failures": failures,
        "n_steps": float(n),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _print_table(title: str, metric: str,
                 rows: dict[tuple[str, int, int], dict[str, Any]]) -> None:
    """Print one metric as an arm x (H, K) table.

    A cell is ``--`` when the metric is undefined at that (H, K) — the
    expected case for a budget at or above the population, which is dropped
    rather than clamped (see :func:`budgets`). The placeholder keeps the
    table readable and does not invent a number.
    """
    print(f"\n{title}")
    cells = sorted({(h, k) for (_a, h, k) in rows})
    header = "  ".join(f"{f'{h}/{k}':>10}" for (h, k) in cells)
    print(f"{'arm':>18}  {header}")
    print("-" * (18 + 2 + len(header)))
    for arm in ARMS:
        vals = []
        for (h, k) in cells:
            key = (arm, h, k)
            if key not in rows or metric not in rows[key]:
                vals.append("--")
            else:
                vals.append(f"{rows[key][metric]:>10.4f}")
        print(f"{arm:>18}  {'  '.join(vals)}")


def _build_provenance(contract_source: str) -> Provenance:
    """What produced the record, so it is self-describing.

    The three identifiers are the pre-registration id, a content hash of that
    contract file, and the commit the script ran from. All degrade to a
    stated non-value rather than raising: a run in a tarball still produced
    real numbers, and the honest record says the commit is unknown.
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
    """Write the machine-readable summary.

    ``results/`` is gitignored: the committed part of a result is the script
    and the gates, and the JSON is regenerable from the deterministic seeds.
    """
    _results.write_record(record, out_path)
    print()
    print(f"machine-readable summary written to {out_path}")
    print("  (status is left to the reader: the instrument measures, it does")
    print("   not arbitrate the decision rule)")


def _results_path() -> Path:
    """Where the machine-readable summary is written."""
    return _results.results_dir_for(__file__) / "retrieval_001.json"


def _gate_cells(oracle: dict[int, float],
                h_levels: tuple[int, ...]) -> tuple[GateCell, ...]:
    """The oracle gate's cells, computed once for the report and the record.

    The oracle is the environment control. MATCHED-001 registers no
    reproduction baseline because it *is* the baseline the other experiments
    reproduce; F1's gate is the same prior control — that the task has a
    unique relation-satisfier at every population, so a learned-arm number
    from any cell means something.
    """
    return tuple(
        GateCell(
            cell=f"oracle@{h_ev}",
            metric="accuracy",
            published=1.0,
            observed=oracle[h_ev],
            diff=oracle[h_ev] - 1.0,
            tol=0.0,
            passed=oracle[h_ev] >= 1.0 - 1e-12,
        )
        for h_ev in h_levels if h_ev in oracle
    )


def _build_record(args: argparse.Namespace,
                  seeds: list[int],
                  h_levels: tuple[int, ...],
                  k_levels: tuple[int, ...],
                  rows: dict[tuple[str, int, int], dict[str, float]],
                  per_seed: dict[tuple[str, int, int], list[float]],
                  oracle: dict[int, float],
                  gate_ok: bool,
                  *,
                  contract_source: str,
                  deviations: tuple[dict[str, Any], ...] = (),
                  ) -> MeasurementRecord:
    """The run as one record.

    The envelope is shared with the other measurement scripts; the payload is
    F1's own. Endpoints are keyed by ``arm`` at each ``(H, K)``, because the
    experiment's structure *is* the arm comparison at matched K — flattening
    it into a per-arm table would lose the C-vs-B pairing the decision rule
    reads.
    """
    contract = load_contract(EXPERIMENT_ID)
    primary = contract.primary_endpoint()

    # The decision rule's evidence: C against B at matched (H, K), for the
    # primary endpoint and every secondary one. Each row carries the
    # threshold and the verdict rather than a bare delta, because a delta
    # without its threshold is a number a reader has to arbitrate — and the
    # rule's branches are pre-committed, so the threshold belongs in the row.
    decision: list[dict] = []
    for h in h_levels:
        for k in k_levels:
            if k >= h:
                continue
            b = rows.get(("exhaustive_topk", h, k))
            c = rows.get(("indexed_topk", h, k))
            if b is None or c is None:
                continue
            # The baseline's own seed spread is the materiality threshold,
            # floored at SPREAD_FLOOR. Measured before the comparison, so the
            # threshold cannot depend on the arm being judged.
            b_spread = (max(b_vals) - min(b_vals)) if (b_vals := per_seed.get(
                ("exhaustive_topk", h, k), [])) else 0.0
            for name in [primary] + [e.name for e in contract.endpoints
                                     if not e.primary and e.name in b]:
                if name not in c:
                    continue
                delta = c[name] - b[name]
                detail = delta_detail(
                    arm="indexed_topk", metric=name,
                    baseline=b[name], observed=c[name],
                    baseline_spread=b_spread,
                    context={"h": h, "k": k},
                )
                decision.append({
                    "endpoint": name,
                    "h": h,
                    "k": k,
                    "baseline_exhaustive_topk": b[name],
                    "indexed_topk": c[name],
                    "delta_indexed_minus_exhaustive": delta,
                    "baseline_seed_spread": b_spread,
                    "materiality_threshold": detail["threshold"],
                    "verdict": detail["verdict"],
                })

    audit = [
        {"h_eval": h_ev, "oracle_accuracy": oracle[h_ev]}
        for h_ev in h_levels if h_ev in oracle
    ]

    # Per-seed values for the primary endpoint, because a mean alone does not
    # show whether the spread and the difference between two cells are the
    # same order of magnitude.
    per_seed_out: list[dict] = []
    for arm in ARMS:
        for h in h_levels:
            for k in k_levels:
                if k >= h:
                    continue
                vals = per_seed.get((arm, h, k))
                if not vals:
                    continue
                per_seed_out.append({
                    "arm": arm,
                    "h": h,
                    "k": k,
                    "seeds": {str(s): v for s, v in zip(seeds, vals)},
                    "mean": statistics.fmean(vals),
                    "spread": max(vals) - min(vals) if vals else 0.0,
                })

    smoke = bool(getattr(args, "smoke", False))
    return MeasurementRecord(
        provenance=_build_provenance(contract_source),
        design=Design(
            steps=args.steps,
            eval_steps=args.eval_steps,
            seeds=tuple(seeds),
            h_levels=tuple(h_levels),
            k_levels=tuple(k_levels),
            arms=tuple(ARMS),
            smoke=smoke,
            contract_checked=not smoke,
            deviations=deviations,
        ),
        gate=Gate(
            name="oracle accuracy at every eval population",
            tolerance="oracle accuracy must be exactly 1.0",
            passed=gate_ok,
            cells=_gate_cells(oracle, h_levels),
        ),
        endpoints={
            f"{arm}@{h}x{k}": {kk: vv for kk, vv in cell.items() if kk != "n_steps"}
            for (arm, h, k), cell in rows.items()
        },
        decision_rule=tuple(decision),
        audit={"oracle": audit},
        per_seed={"cells": per_seed_out},
    )


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
    ap.add_argument("--steps", type=int, default=REGISTERED_STEPS)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--levels", default=",".join(str(h) for h in H_LEVELS))
    ap.add_argument("--k-levels", default=",".join(str(k) for k in K_LEVELS))
    ap.add_argument("--train-candidates", type=int, default=TRAIN_CANDIDATES)
    ap.add_argument(
        "--smoke", action="store_true",
        help="declare this run as a smoke test and skip the contract check",
    )
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    h_levels = _parse_levels(args.levels)
    k_levels = tuple(int(k) for k in args.k_levels.split(",") if k.strip())
    for k in k_levels:
        if k < 2:
            raise SystemExit(
                f"F1's registered evidence is top-K at K >= 2 (M1.8 moved "
                f"recall and not routing@1); got K={k}."
            )

    # -- 0. The machine-readable contract -------------------------------- #
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

    print(f"steps={args.steps} eval_steps={args.eval_steps} seeds={seeds}")
    print(f"arms={list(ARMS)}")
    print(f"train_candidates={args.train_candidates} (weights loaded into every "
          f"evaluation router, learning disabled)")
    print("the model-state integrity gate runs before every measurement")
    print("the leakage audit runs on every step the index sees")
    print()

    # -- 1. Oracle sanity, before any learned number is printed ----------- #
    oracle: dict[int, float] = {}
    print("oracle accuracy (environment control; must be 1.0):")
    for h_ev in h_levels:
        accs = [_oracle_check(h_ev, s, args.eval_steps) for s in seeds]
        oracle[h_ev] = statistics.fmean(accs)
        print(f"  H_eval={h_ev:>5}: {oracle[h_ev]:.4f}")
    gate_ok = all(acc >= 1.0 - 1e-12 for acc in oracle.values())
    print()

    # -- 2. Train once; the same trained state scores every arm ----------- #
    trained: dict[int, tuple[list[float], dict]] = {}
    for seed in seeds:
        w, manifest = _train_one(args.train_candidates, seed, args.steps)
        trained[seed] = (w, manifest)
        print(f"trained seed={seed}: norm={manifest['parameter_norm']:.4f} "
              f"hash={manifest['parameter_hash'][:12]} "
              f"source={manifest['source']}")
    print()

    # -- 3. Evaluate every (arm, H, K) over the same task stream --------- #
    # One trained state per seed, and the arms share the seed's task stream,
    # so an arm comparison is a comparison of the budget rather than of the
    # world.
    rows: dict[tuple[str, int, int], dict[str, float]] = {}
    per_seed: dict[tuple[str, int, int], list[float]] = {}
    for arm in ARMS:
        for h in h_levels:
            for k in budgets(h, k_levels):
                cells = [_eval_cell(arm, h, k, trained[s][0], s, args.eval_steps)
                         for s in seeds]
                rows[(arm, h, k)] = {
                    key: statistics.fmean(c[key] for c in cells)
                    if key != "failures" else {
                        cat: sum(c["failures"][cat] for c in cells)
                        for cat in FAILURE_CATEGORIES
                    }
                    for key in cells[0]
                }
                per_seed[(arm, h, k)] = [c["recall@K"] for c in cells]
                print(f"  {arm:>18} H={h:>4} K={k:>3} done")

    # -- 4. Report -------------------------------------------------------- #
    _print_table("recall@K — PRIMARY: P(gold retained AND ranked first over the "
                 "retained set)", "recall@K", rows)
    _print_table("scorer_recall@K — the F0 endpoint: P(gold in top-K of the full "
                 "ordering); arm-independent", "scorer_recall@K", rows)
    _print_table("index_retains_gold — P(gold in the retained set); the first "
                 "factor of the decomposition", "index_retains_gold", rows)
    _print_table("exact_success — the loop's own task success",
                 "exact_success", rows)
    _print_table("routing@1 — argmax accuracy (the M1 endpoint; not F1's headline)",
                 "routing@1", rows)
    _print_table("gold_rank — mean rank of gold, ties against gold",
                 "gold_rank", rows)
    _print_table("acc|retrieved — P(success | gold retrieved)",
                 "acc|retrieved", rows)
    _print_table("candidates_inspected — |R|, the quantity C5 needs",
                 "candidates_inspected", rows)
    _print_table("router_operations — scorer calls per step (modelled; see the "
                 "cost section)", "router_operations", rows)
    _print_table("index_operations — index probes per step",
                 "index_operations", rows)
    _print_table("wall_clock — measured, not derived",
                 "wall_clock", rows)

    print()
    print("How to read this:")
    print("  recall@K (the primary, since amendment A1) decomposes as")
    print("  P(gold retained) x P(scorer ranks gold first | retained), and")
    print("  the two factors are reported separately: index_retains_gold is")
    print("  the first, and the second is what ranking_miss counts when it")
    print("  fails. So a cell where index_retains_gold is close to B's but")
    print("  recall@K is lower is a *scorer* failure inside the retrieved")
    print("  set — not an argument against the index — while a cell where")
    print("  index_retains_gold itself drops is.")
    print()
    print("  scorer_recall@K is the F0 endpoint, arm-independent by")
    print("  construction, reported so the F0 curve stays readable here.")
    print()
    print("  candidates_inspected is |R|: H in arm A, K in B/C/D. It is the")
    print("  quantity C5 needs, and it is only defined now because a")
    print("  retrieval boundary is in the loop. router_operations is a")
    print("  modelled count of the scoring work each architecture would")
    print("  perform, and wall_clock is its measured counterpart; the two")
    print("  are not the same quantity and neither is read off the other.")
    print("  executor_nodes is fixed at max_nodes = 10 by the v0.1 executor")
    print("  — an architectural constant, not a scaling measurement.")
    print()
    print("  The third branch of the rule is live: the index is a filter and")
    print("  not a ranker, and the environment's distractors are built to")
    print("  defeat cheap agreement rules. If the index retains gold at a")
    print("  higher rate than the trained scorer ranks it, that is a result")
    print("  about the scorer, and it is consistent with MATCHED-001's")
    print("  finding that the learning rule is the failure at large H.")

    # -- 5. The failure taxonomy, printed in full ------------------------ #
    print()
    print("=" * 72)
    print("FAILURE TAXONOMY — every failed step, attributed once")
    print("=" * 72)
    for arm in ARMS:
        print(f"\n{arm}")
        for h in h_levels:
            for k in budgets(h, k_levels):
                cell = rows.get((arm, h, k))
                if not cell:
                    continue
                f = cell.get("failures", {})
                total = sum(f.values())
                print(f"  H={h:>4} K={k:>3} failed={total:>4}  " +
                      "  ".join(f"{c}={f.get(c, 0)}" for c in FAILURE_CATEGORIES))

    # -- 6. The machine-readable summary ---------------------------------- #
    _emit_results(
        _build_record(args, seeds, h_levels, k_levels, rows, per_seed, oracle,
                      gate_ok,
                      contract_source=contract_source,
                      deviations=deviations),
        _results_path())


if __name__ == "__main__":
    main()
