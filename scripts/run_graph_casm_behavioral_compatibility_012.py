#!/usr/bin/env python3
"""G-CASM-012: query-conditioned behavioral compatibility routing.

The learned router scores whether a candidate executable program is compatible
with each observed support row. It is trained on public training-program
behavior, never on evaluation targets or verifier-only rows.

This experiment keeps the G-CASM-010 benchmark family and exact executor but
changes the learning target from "which exact program is this?" to
"does this candidate explain the observed behavior?".
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import random
import resource
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import run_graph_casm_selective_010 as g010

from tac_osm.behavioral_compatibility_router import (
    BehavioralCompatibilityRouter,
    BehavioralRouterConfig,
)

SEEDS = (0, 1, 2, 3, 4)
M_LEVELS = (32, 64, 128, 256, 512)
BUDGETS = (1, 2, 4, 8)
SUPPORT_SIZES = (4, 8, 12)
PRIMARY_SUPPORT_SIZE = 4
TASKS_PER_M = 32
SMOKE_M_LEVELS = (32, 128)
SMOKE_TASKS_PER_M = 4
TRAIN_PROGRAMS = 256
ROUTER_STEPS = 1200
ROUTER_BATCH = 32
TRAIN_SUPPORT_SIZE = 4
MAX_NODES = g010.MAX_NODES
INPUT_COUNT = g010.INPUT_COUNT
FEATURE_DIM = g010.FEATURE_DIM
GENERATOR_COMMIT = "c31554413301e3c9d3e6b3f8c8c6be572a74a748"
EXPERIMENT_ID = "TACOSM-GRAPH-CASM-BEHAVIORAL-COMPATIBILITY-012"
INPUT_ROWS = tuple(itertools.product((0, 1), repeat=INPUT_COUNT))


def load_contract():
    from tac_osm.contract import load_contract
    return load_contract(EXPERIMENT_ID)


def split_digest(train_structures, train_truths) -> str:
    payload = repr(sorted(map(repr, train_structures))) + repr(sorted(map(repr, train_truths)))
    return hashlib.sha256(payload.encode()).hexdigest()


def support_permutation(seed: int, m: int, task_i: int, heldout: bool) -> tuple[int, ...]:
    rng = random.Random(
        seed * 1000003
        + m * 7919
        + task_i * 104729
        + (500000 if heldout else 0)
        + 1207
    )
    rows = list(range(len(INPUT_ROWS)))
    rng.shuffle(rows)
    return tuple(rows)


def build_task(
    seed: int,
    m: int,
    target,
    train_structures,
    train_truths,
    task_i: int,
    support_size: int,
    *,
    heldout: bool,
):
    base_task, candidates = g010.make_task(
        seed,
        m,
        target,
        train_structures,
        train_truths,
        heldout=heldout,
    )
    order = support_permutation(seed, m, task_i, heldout)
    chosen = tuple(order[:support_size])
    support = tuple(
        (INPUT_ROWS[k], int(target.truth_table[INPUT_ROWS[k]])) for k in chosen
    )
    verify = tuple(
        (INPUT_ROWS[k], int(target.truth_table[INPUT_ROWS[k]]))
        for k in order[support_size:]
    )
    digest = hashlib.sha256(
        repr((support, verify, task_i, m, heldout)).encode()
    ).hexdigest()[:16]
    task = replace(
        base_task,
        task_id=f"g12:{seed}:{m}:{task_i}:{support_size}:{int(heldout)}:{digest}",
        support=support,
        verify=verify,
    )
    return task, candidates


def node_features(ep) -> list[float]:
    return g010.node_features(ep)


def candidate_features(candidates, include_wiring: bool) -> torch.Tensor:
    return torch.tensor(
        [g010.features(ep, include_wiring) for ep in candidates],
        dtype=torch.float32,
    )


def support_rows_tensor(support) -> torch.Tensor:
    return torch.tensor(
        [
            [float(x) for x in (*tuple(bits), int(y))]
            for bits, y in support
        ],
        dtype=torch.float32,
    )


def compatibility_labels(candidates, support) -> torch.Tensor:
    return torch.tensor(
        [
            [
                float(int(ep.truth_table[tuple(bits)]) == int(y))
                for bits, y in support
            ]
            for ep in candidates
        ],
        dtype=torch.float32,
    )


def training_batch(
    training,
    rng: random.Random,
    batch_size: int,
    support_size: int,
):
    candidate_batches = []
    row_batches = []
    label_batches = []
    for _ in range(batch_size):
        target_idx = rng.randrange(len(training))
        target = training[target_idx]
        other_indices = [
            i for i in rng.sample(
                [i for i in range(len(training)) if i != target_idx],
                g010.ROUTER_BATCH if False else 7,
            )
        ]
        candidate_indices = other_indices + [target_idx]
        rng.shuffle(candidate_indices)
        candidates = [training[i] for i in candidate_indices]

        row_rng = random.Random(
            rng.randrange(2**31)
            + target_idx * 7919
            + support_size * 101
        )
        row_keys = row_rng.sample(sorted(target.truth_table), support_size)
        support = tuple(
            (tuple(k), int(target.truth_table[k])) for k in row_keys
        )

        candidate_batches.append(
            candidate_features(candidates, include_wiring=True)
        )
        row_batches.append(support_rows_tensor(support))
        label_batches.append(compatibility_labels(candidates, support))

    return (
        torch.stack(candidate_batches),
        torch.stack(row_batches),
        torch.stack(label_batches),
    )


def fit_behavioral_router(
    seed: int,
    training: Sequence,
    representation: str,
) -> BehavioralCompatibilityRouter:
    model = BehavioralCompatibilityRouter(
        BehavioralRouterConfig(
            candidate_dim=FEATURE_DIM,
            hidden_dim=64,
            latent_dim=32,
            pair_hidden_dim=64,
            learning_rate=1e-3,
            aggregate_temperature=0.25,
            pair_loss_weight=1.0,
            set_loss_weight=1.0,
            steps=ROUTER_STEPS,
            batch_size=ROUTER_BATCH,
            candidates_per_task=8,
            seed=seed + (101 if representation == "summary" else 0),
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model.config.learning_rate
    )
    rng = random.Random(seed * 31337 + (17 if representation == "summary" else 11))
    model.train()
    for _ in range(ROUTER_STEPS):
        features, rows, labels = training_batch(
            training, rng, ROUTER_BATCH, TRAIN_SUPPORT_SIZE
        )
        if representation == "summary":
            # g010.features always returns the same 290-width vector; zeroing
            # the 200 wiring coordinates reproduces the registered summary arm.
            features = features.clone()
            features[:, :, -200:] = 0.0

        loss = model.training_loss(features, rows, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    return model


def exact_support_consistent(candidates, support) -> list[int]:
    wanted = {(tuple(bits), int(y)) for bits, y in support}
    return [
        i
        for i, ep in enumerate(candidates)
        if all((bits, int(ep.truth_table[bits])) in wanted for bits, _ in support)
    ]


def rank_candidates(
    model: BehavioralCompatibilityRouter,
    candidates,
    support,
    representation: str,
):
    features = candidate_features(candidates, include_wiring=True)
    if representation == "summary":
        features[:, -200:] = 0.0
    rows = support_rows_tensor(support)
    with torch.no_grad():
        scores = model.compatibility_scores(features, rows).tolist()
        pair_logits = model.pair_logits(features, rows).tolist()
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return order, scores, pair_logits


def execute_subset(candidates, indices, verify_rows):
    start = time.perf_counter()
    ok_any = False
    total_work = 0
    for idx in indices:
        candidate = candidates[idx]
        ok = True
        for bits, expected in verify_rows:
            got, work = g010.execute_exact(candidate, bits)
            total_work += work.total
            if got != int(expected):
                ok = False
                break
        if ok and verify_rows:
            ok_any = True
            break
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ok_any, total_work, elapsed_ms


def evaluate_behavioral(model, representation, task, candidates):
    order, scores, pair_logits = rank_candidates(
        model, candidates, task.support, representation
    )
    relevant = set(exact_support_consistent(candidates, task.support))
    if task.target_index not in relevant:
        raise RuntimeError("target omitted from exact support-consistent set")
    exhaustive_ok, exhaustive_work, exhaustive_ms = execute_subset(
        candidates, range(len(candidates)), task.verify
    )
    if not exhaustive_ok:
        raise RuntimeError("exhaustive exact verification failed for target task")

    budgets = {}
    for budget in BUDGETS:
        selected = order[:budget]
        fixed_ok, fixed_work, fixed_ms = execute_subset(
            candidates, selected, task.verify
        )
        adaptive_ok, adaptive_work, adaptive_ms = execute_subset(
            candidates, selected, task.verify
        )
        budgets[str(budget)] = {
            "target_in_budget": float(task.target_index in selected),
            "routing_recall": float(task.target_index in selected),
            "support_set_recall": (
                len(set(selected) & relevant) / max(1, len(relevant))
            ),
            "support_set_hit": float(bool(set(selected) & relevant)),
            "semantic_success": float(fixed_ok),
            "adaptive_semantic_success": float(adaptive_ok),
            "fixed_budget_executed_candidates": budget,
            "adaptive_executed_candidates": (
                budget if not adaptive_ok else next(
                    j + 1 for j, idx in enumerate(selected)
                    if execute_subset(candidates, [idx], task.verify)[0]
                )
            ),
            "fixed_budget_execution_work_units": fixed_work,
            "adaptive_execution_work_units": adaptive_work,
            "fixed_budget_execution_work_fraction": fixed_work / max(1, exhaustive_work),
            "adaptive_execution_work_fraction": adaptive_work / max(1, exhaustive_work),
            "fixed_budget_execution_ms": fixed_ms,
            "adaptive_execution_ms": adaptive_ms,
        }

    return {
        "rank": order.index(task.target_index) + 1,
        "support_consistent_count": len(relevant),
        "target_unique": int(len(relevant) == 1),
        "exhaustive": {
            "semantic_success": float(exhaustive_ok),
            "execution_work_units": exhaustive_work,
            "execution_ms": exhaustive_ms,
        },
        "budgets": budgets,
        "scores": scores,
        "pair_logits": pair_logits,
    }


def legacy_evaluate(router, task, candidates):
    memory = g010.Memory()
    memory.write(task.task_id, task.support)
    return g010.evaluate(router, "graph", task, candidates, memory)


def deterministic_control(candidates, task, kind: str, seed: int):
    rng = random.Random(seed)
    if kind == "constant":
        order = list(range(len(candidates)))
    elif kind == "random":
        order = list(range(len(candidates)))
        rng.shuffle(order)
    else:
        raise ValueError(kind)
    relevant = set(exact_support_consistent(candidates, task.support))
    metrics = {}
    exhaustive_ok, exhaustive_work, _ = execute_subset(
        candidates, range(len(candidates)), task.verify
    )
    if not exhaustive_ok:
        raise RuntimeError("control exhaustive verifier failed")
    for budget in BUDGETS:
        selected = order[:budget]
        ok, work, _ = execute_subset(candidates, selected, task.verify)
        metrics[str(budget)] = {
            "routing_recall": float(task.target_index in selected),
            "support_set_recall": len(set(selected) & relevant) / max(1, len(relevant)),
            "semantic_success": float(ok),
            "execution_work_fraction": work / max(1, exhaustive_work),
        }
    return metrics


def seed_bootstrap(values: Sequence[float], seed: int, rounds: int = 4000):
    if len(values) <= 1:
        x = float(values[0]) if values else 0.0
        return [x, x]
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(rounds)
    )
    return [
        float(draws[int(0.025 * (rounds - 1))]),
        float(draws[int(0.975 * (rounds - 1))]),
    ]


def summarize_rep(seed_rows):
    summary = {}
    for support_size in SUPPORT_SIZES:
        summary[str(support_size)] = {}
        for m in M_LEVELS:
            blocks = [
                seed_rows[str(seed)][str(m)][str(support_size)]
                for seed in SEEDS
                if str(seed) in seed_rows
                and str(m) in seed_rows[str(seed)]
                and str(support_size) in seed_rows[str(seed)][str(m)]
            ]
            trials = [t for block in blocks for t in block["trials"]]
            if not trials:
                continue
            row = {
                "trials": len(trials),
                "support_consistent_count_mean": statistics.fmean(
                    t["support_consistent_count"] for t in trials
                ),
                "target_unique_fraction": statistics.fmean(
                    t["target_unique"] for t in trials
                ),
                "exhaustive_success": statistics.fmean(
                    t["exhaustive"]["semantic_success"] for t in trials
                ),
                "budgets": {},
            }
            for b in BUDGETS:
                vals = [
                    t["budgets"][str(b)]["semantic_success"] for t in trials
                ]
                recalls = [
                    t["budgets"][str(b)]["routing_recall"] for t in trials
                ]
                setrec = [
                    t["budgets"][str(b)]["support_set_recall"] for t in trials
                ]
                work = [
                    t["budgets"][str(b)]["adaptive_execution_work_fraction"]
                    for t in trials
                ]
                ex = row["exhaustive_success"]
                sem = statistics.fmean(vals)
                row["budgets"][str(b)] = {
                    "semantic_success": sem,
                    "adaptive_semantic_success": statistics.fmean(
                        t["budgets"][str(b)]["adaptive_semantic_success"]
                        for t in trials
                    ),
                    "routing_recall": statistics.fmean(recalls),
                    "support_set_recall": statistics.fmean(setrec),
                    "capability_retention": sem / ex if ex > 0 else None,
                    "adaptive_capability_retention": (
                        statistics.fmean(
                            t["budgets"][str(b)]["adaptive_semantic_success"]
                            for t in trials
                        ) / ex if ex > 0 else None
                    ),
                    "adaptive_execution_work_fraction": statistics.fmean(work),
                    "seed_bootstrap_95ci_semantic_success": seed_bootstrap(
                        [
                            statistics.fmean(
                                t["budgets"][str(b)]["semantic_success"]
                                for t in block["trials"]
                            )
                            for block in blocks
                        ],
                        7000 + m * 31 + support_size * 7 + b,
                    ),
                    "seed_bootstrap_95ci_work_fraction": seed_bootstrap(
                        [
                            statistics.fmean(
                                t["budgets"][str(b)]["adaptive_execution_work_fraction"]
                                for t in block["trials"]
                            )
                            for block in blocks
                        ],
                        7100 + m * 31 + support_size * 7 + b,
                    ),
                }

            eligible = [
                (b, v)
                for b, v in row["budgets"].items()
                if (
                    row["exhaustive_success"] >= 0.80
                    and v["capability_retention"] is not None
                    and v["capability_retention"] >= 0.80
                )
            ]
            row["primary_endpoint"] = (
                {
                    "eligible": True,
                    "min_adaptive_execution_work_fraction": min(
                        v["adaptive_execution_work_fraction"] for _, v in eligible
                    ),
                }
                if support_size == PRIMARY_SUPPORT_SIZE and eligible
                else {
                    "eligible": False,
                    "reason": (
                        "exhaustive_ceiling_below_0.80"
                        if row["exhaustive_success"] < 0.80
                        else "no_budget_reached_capability_floor"
                    ),
                }
            )
            summary[str(support_size)][str(m)] = row
    return summary


def primary_bootstrap(seed_rows, m: int, rounds: int = 4000):
    available = [
        str(s)
        for s in SEEDS
        if str(s) in seed_rows and str(m) in seed_rows[str(s)]
        and str(PRIMARY_SUPPORT_SIZE) in seed_rows[str(s)][str(m)]
    ]
    if len(available) <= 1:
        return {"ci95": None, "valid_fraction": 0.0}
    rng = random.Random(90000 + m)
    selected = []
    for _ in range(rounds):
        sampled = [rng.choice(available) for _ in available]
        trials = []
        for sid in sampled:
            trials.extend(
                seed_rows[sid][str(m)][str(PRIMARY_SUPPORT_SIZE)]["trials"]
            )
        ex = statistics.fmean(
            t["exhaustive"]["semantic_success"] for t in trials
        )
        eligible = []
        for b in BUDGETS:
            sem = statistics.fmean(
                t["budgets"][str(b)]["semantic_success"] for t in trials
            )
            work = statistics.fmean(
                t["budgets"][str(b)]["adaptive_execution_work_fraction"]
                for t in trials
            )
            retention = sem / ex if ex > 0 else None
            if ex >= 0.80 and retention is not None and retention >= 0.80:
                eligible.append(work)
        if eligible:
            selected.append(min(eligible))
    if not selected:
        return {"ci95": None, "valid_fraction": 0.0}
    selected.sort()
    return {
        "ci95": [
            float(selected[int(0.025 * (len(selected) - 1))]),
            float(selected[int(0.975 * (len(selected) - 1))]),
        ],
        "valid_fraction": float(len(selected) / rounds),
    }


def run(smoke: bool = False):
    contract = load_contract()
    full = contract
    if smoke:
        seeds = (0,)
        m_levels = SMOKE_M_LEVELS
        tasks_per_m = SMOKE_TASKS_PER_M
    else:
        seeds = SEEDS
        m_levels = M_LEVELS
        tasks_per_m = TASKS_PER_M

    # The contract is checked against the registered full grid before a run.
    contract.require_levels(M_LEVELS if not smoke else SMOKE_M_LEVELS, strict=not smoke)
    contract.require_seeds(SEEDS if not smoke else (0,), strict=not smoke)
    contract.require_steps(ROUTER_STEPS)
    contract.require_eval_steps(TASKS_PER_M if not smoke else SMOKE_TASKS_PER_M)
    contract.require_k_levels(SUPPORT_SIZES)
    contract.require_arms(
        ["behavioral_graph", "legacy_graph", "behavioral_summary", "constant", "random"]
    )

    by_arm = {arm: {} for arm in contract.to_dict()["arms"][0:0]}  # never used
    results = {
        "behavioral_graph": {},
        "behavioral_summary": {},
        "legacy_graph": {},
        "constant": {},
        "random": {},
    }
    checks = {}

    for seed in seeds:
        training = g010.generate(seed + 100, TRAIN_PROGRAMS)
        train_s = {g010.structure_key(ep) for ep in training}
        train_t = {g010.truth_signature(ep) for ep in training}
        manifest = g010.build_manifest(
            seed, m_levels, tasks_per_m,
            exclude_structures=train_s,
            exclude_truths=train_t,
        )
        split_failures = []
        for target in manifest.values():
            if g010.structure_key(target) in train_s:
                split_failures.append("training/evaluation structure overlap")
            if g010.truth_signature(target) in train_t:
                split_failures.append("training/evaluation truth overlap")
        if split_failures:
            raise RuntimeError("; ".join(sorted(set(split_failures))))

        checks[str(seed)] = {
            "train_programs": TRAIN_PROGRAMS,
            "train_structure_count": len(train_s),
            "train_truth_count": len(train_t),
            "split_integrity": True,
            "split_digest": split_digest(train_s, train_t),
            "executor": g010.executor_check(
                training[:32] + list(manifest.values())
            ),
        }
        if not checks[str(seed)]["executor"]["pass"]:
            raise RuntimeError("exact executor check failed")

        behavioral_graph = fit_behavioral_router(seed, training, "graph")
        behavioral_summary = fit_behavioral_router(seed, training, "summary")
        legacy = g010.fit_router(seed, training, "graph")

        for arm in results:
            results[arm][str(seed)] = {}

        for m in m_levels:
            for support_size in SUPPORT_SIZES:
                trials = []
                constants = []
                randoms = []
                for task_i in range(tasks_per_m):
                    target = manifest[(m, task_i)]
                    task, candidates = build_task(
                        seed * 100 + task_i,
                        m,
                        target,
                        train_s,
                        train_t,
                        task_i,
                        support_size,
                        heldout=True,
                    )

                    btrial = evaluate_behavioral(
                        behavioral_graph, "graph", task, candidates
                    )
                    strial = evaluate_behavioral(
                        behavioral_summary, "summary", task, candidates
                    )
                    ctrial = deterministic_control(
                        candidates, task, "constant", seed * 1000003 + m + task_i
                    )
                    rtrial = deterministic_control(
                        candidates, task, "random", seed * 2000003 + m + task_i
                    )

                    constants.append({
                        "task": task_i,
                        "target_index": task.target_index,
                        "support_consistent_count": len(
                            exact_support_consistent(candidates, task.support)
                        ),
                        "budgets": ctrial,
                    })
                    randoms.append({
                        "task": task_i,
                        "target_index": task.target_index,
                        "support_consistent_count": len(
                            exact_support_consistent(candidates, task.support)
                        ),
                        "budgets": rtrial,
                    })

                    if support_size == PRIMARY_SUPPORT_SIZE:
                        ltrial = legacy_evaluate(legacy, task, candidates)
                        results["legacy_graph"][str(seed)].setdefault(str(m), {}).setdefault(
                            str(support_size), {"trials": []}
                        )
                        results["legacy_graph"][str(seed)][str(m)][str(support_size)]["trials"].append(
                            {
                                "task": task_i,
                                "target_index": task.target_index,
                                "support_consistent_count": len(
                                    exact_support_consistent(candidates, task.support)
                                ),
                                **ltrial,
                            }
                        )

                    trials.append({
                        "task": task_i,
                        "target_index": task.target_index,
                        "support": [[list(bits), int(y)] for bits, y in task.support],
                        "verify_row_count": len(task.verify),
                        **btrial,
                    })
                    results["behavioral_summary"][str(seed)].setdefault(str(m), {}).setdefault(
                        str(support_size), {"trials": []}
                    )["trials"].append({
                        "task": task_i,
                        "target_index": task.target_index,
                        "support": [[list(bits), int(y)] for bits, y in task.support],
                        "verify_row_count": len(task.verify),
                        **strial,
                    })

                results["behavioral_graph"][str(seed)].setdefault(str(m), {})[
                    str(support_size)
                ] = {"trials": trials}
                results["constant"][str(seed)].setdefault(str(m), {})[
                    str(support_size)
                ] = {"trials": constants}
                results["random"][str(seed)].setdefault(str(m), {})[
                    str(support_size)
                ] = {"trials": randoms}

        output = Path("artifacts") / f"{EXPERIMENT_ID}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "experiment_id": EXPERIMENT_ID,
            "status": "checkpoint",
            "provenance": {
                "tacosm_commit": os.environ.get("GITHUB_SHA", "local"),
                "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                "generator_commit": GENERATOR_COMMIT,
                "torch": torch.__version__,
                "python": sys.version,
            },
            "protocol": {
                "seeds": list(seeds),
                "M_levels": list(m_levels),
                "budgets": list(BUDGETS),
                "support_sizes": list(SUPPORT_SIZES),
                "primary_support_size": PRIMARY_SUPPORT_SIZE,
                "train_support_size": TRAIN_SUPPORT_SIZE,
                "train_programs": TRAIN_PROGRAMS,
                "router_steps": ROUTER_STEPS,
                "candidate_feature_dimension": FEATURE_DIM,
            },
            "checks": checks,
            "results": results,
        }
        output.write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summaries = {
        "behavioral_graph": summarize_rep(results["behavioral_graph"]),
        "behavioral_summary": summarize_rep(results["behavioral_summary"]),
        "legacy_graph": summarize_rep(results["legacy_graph"]),
    }
    primary_ci = {
        "behavioral_graph": {
            str(m): primary_bootstrap(results["behavioral_graph"], m)
            for m in m_levels
        },
        "behavioral_summary": {
            str(m): primary_bootstrap(results["behavioral_summary"], m)
            for m in m_levels
        },
    }

    # Controls are descriptive; no model-selection decision is made from them.
    control_summaries = {}
    for arm in ("constant", "random"):
        control_summaries[arm] = {}
        for support_size in SUPPORT_SIZES:
            control_summaries[arm][str(support_size)] = {}
            for m in m_levels:
                trials = [
                    t
                    for seed in seeds
                    for t in results[arm][str(seed)][str(m)][str(support_size)]["trials"]
                ]
                control_summaries[arm][str(support_size)][str(m)] = {
                    "trials": len(trials),
                    "budgets": {
                        str(b): {
                            "routing_recall": statistics.fmean(
                                t["budgets"][str(b)]["routing_recall"] for t in trials
                            ),
                            "semantic_success": statistics.fmean(
                                t["budgets"][str(b)]["semantic_success"] for t in trials
                            ),
                            "support_set_recall": statistics.fmean(
                                t["budgets"][str(b)]["support_set_recall"] for t in trials
                            ),
                        }
                        for b in BUDGETS
                    },
                }

    final = {
        "experiment_id": EXPERIMENT_ID,
        "status": "measured",
        "provenance": {
            "tacosm_commit": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "generator_commit": GENERATOR_COMMIT,
            "python": sys.version,
            "torch": torch.__version__,
        },
        "protocol": {
            "seeds": list(seeds),
            "M_levels": list(m_levels),
            "tasks_per_seed_M": tasks_per_m,
            "support_sizes": list(SUPPORT_SIZES),
            "primary_support_size": PRIMARY_SUPPORT_SIZE,
            "train_support_size": TRAIN_SUPPORT_SIZE,
            "budgets": list(BUDGETS),
            "train_programs": TRAIN_PROGRAMS,
            "router_steps": ROUTER_STEPS,
            "candidate_feature_dimension": FEATURE_DIM,
            "graph_wiring_public": True,
            "summary_wiring_removed": True,
            "training_objective": "row-level compatibility plus support-set compatibility",
            "legacy_baseline": "TACOSM-GRAPH-CASM-SELECTIVE-010 graph dual encoder",
        },
        "checks": checks,
        "summary": summaries,
        "primary_endpoint_bootstrap": primary_ci,
        "controls": control_summaries,
        "results": results,
        "scope": {
            "primary": "capability-constrained selective execution under public executable graph representation",
            "target_identity_training": False,
            "verifier_rows_in_router": False,
            "asymptotic_claim": False,
            "hardware_speedup_claim": False,
            "language_semantics": False,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    output = Path("artifacts") / f"{EXPERIMENT_ID}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final["summary"], indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)


if __name__ == "__main__":
    main()
