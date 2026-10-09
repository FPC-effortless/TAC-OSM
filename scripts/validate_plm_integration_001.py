#!/usr/bin/env python3
"""Independent conservative validation for TACOSM-PLM-INTEGRATION-001.

This file never trains, makes predictions, or silently fills missing results.
Expected artifact: {"experiment_id": ..., "seeds": [...], "episodes": [...],
"source_sha256": ..., "contract_sha256": ..., "instrumentation": {...}}.
Each episode carries seed, H, D, K, interference_strength, pair_id, variant
(0 or 1), target_bit, query_id, current_observation, target_identity,
target_slot, histories (list of identity/payload rows), predictions (arm -> bit),
recall_at_k (address -> bool), work (execution arm -> component -> nonnegative
number). All fields must reflect actual instrumented episodes; fabricated
episode records are not valid scientific evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "TACOSM-PLM-INTEGRATION-001.json"
REQUIRED_COSTS = (
    "index_build", "index_refresh", "memory_writes", "query_address",
    "candidate_retrieval", "execution", "verification",
)
MEMORIES = ("persistent", "reset", "paired_swap", "no_write")
ADDRESSES = ("learned_condition_blind", "raw_dot", "random", "oracle")
EXECUTIONS = ("sparse_k", "exhaustive_exact")


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def validate(artifact: dict, contract: dict) -> dict:
    problems = []
    def gate(ok, msg):
        if not ok:
            problems.append(msg)

    gate(artifact.get("experiment_id") == contract["experiment_id"], "experiment id mismatch")
    gate(artifact.get("contract_sha256") == hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
         "contract hash mismatch")
    gate(bool(artifact.get("source_sha256")), "missing source fingerprint")
    inst = artifact.get("instrumentation", {})
    for key in ("no_full_scan", "weights_changed", "nonzero_write_gradients",
                "key_family_disjoint", "no_oracle_input", "actual_write_counts_verified"):
        gate(inst.get(key) is True, f"unverified integrity evidence: {key}")
    rows = artifact.get("episodes", [])
    gate(isinstance(rows, list) and bool(rows), "missing episode records")
    if not isinstance(rows, list) or not rows:
        return {"decision": "VOID", "problems": problems}
    ev = contract["evaluation"]
    cells = defaultdict(list)
    pairs = defaultdict(list)
    costs = defaultdict(list)
    for i, r in enumerate(rows):
        try:
            seed, h, d, k, noise = (r[x] for x in
                                     ("seed", "H", "D", "K", "interference_strength"))
            c = (seed, h, d, k, noise)
            gate(seed in ev["seeds"] and h in ev["history_lengths"]
                 and d in ev["intervening_writes"] and k in ev["retrieval_budgets"]
                 and noise in ev["interference_strengths"], f"bad cell {i}")
            gate(r["variant"] in (0, 1), f"bad variant {i}")
            gate(r["target_bit"] in (0, 1), f"bad target {i}")
            history = r["histories"]
            gate(len(history) == h, f"history length mismatch {i}")
            hits = [j for j, x in enumerate(history) if x["identity"] == r["target_identity"]]
            gate(len(hits) == 1 and hits[0] == r["target_slot"], f"nonunique satisfier {i}")
            gate(all("target_bit" not in x and "role" not in x and "phase" not in x
                     for x in history), f"leaked field in history {i}")
            p = r["predictions"]
            for memory in MEMORIES:
                for address in ADDRESSES:
                    for execution in EXECUTIONS:
                        key = f"{memory}/{address}/{execution}"
                        gate(p[key] in (0, 1), f"invalid prediction {key} row {i}")
            for address in ADDRESSES:
                gate(isinstance(r["recall_at_k"][address], bool),
                     f"missing recall indicator {address} row {i}")
            for exe in EXECUTIONS:
                component_costs = r["work"][exe]
                for x in REQUIRED_COSTS:
                    val = component_costs[x]
                    gate(isinstance(val, (int, float)) and math.isfinite(val) and val >= 0,
                         f"invalid cost {exe}/{x} row {i}")
                costs[(c, exe)].append(sum(component_costs[x] for x in REQUIRED_COSTS))
            cells[c].append(r)
            pairs[(c, r["pair_id"])].append(r)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            gate(False, f"invalid row {i}: {exc}")
    if problems:
        return {"decision": "VOID", "problems": problems[:60]}
    expected = len(ev["seeds"]) * len(ev["history_lengths"]) * len(
        ev["intervening_writes"]) * len(ev["retrieval_budgets"]) * len(
        ev["interference_strengths"])
    gate(len(cells) == expected, f"missing cells: {len(cells)}/{expected}")
    for c, group in cells.items():
        gate(len(group) == 2 * ev["pairs_per_cell_per_seed"],
             f"cell episode count {c}: {len(group)}")
        gate(sum(x["target_bit"] for x in group) * 2 == len(group),
             f"unbalanced labels {c}")
    for key, pair in pairs.items():
        gate(len(pair) == 2 and set(x["variant"] for x in pair) == {0, 1},
             f"unpaired observations {key}")
        if len(pair) != 2:
            continue
        a, b = pair
        gate(a["current_observation"] == b["current_observation"]
             and a["query_id"] == b["query_id"], f"pair input leakage {key}")
        gate(a["target_bit"] != b["target_bit"], f"no opposite target {key}")
        gate(a["target_identity"] == b["target_identity"], f"target identity shifts {key}")
        gate(a["target_slot"] == b["target_slot"], f"target location shifts {key}")
        gate(all(a["histories"][j] == b["histories"][j]
                 for j in range(len(a["histories"])) if j != a["target_slot"]),
             f"non-target event differs {key}")
        gate(a["predictions"]["reset/learned_condition_blind/sparse_k"]
             == b["predictions"]["reset/learned_condition_blind/sparse_k"],
             f"no-memory collision violated {key}")
    if problems:
        return {"decision": "VOID", "problems": problems[:60]}
    t = contract["thresholds"]
    failures = []
    acc = {}
    for c, group in cells.items():
        def accuracy(arm):
            return sum(x["predictions"][arm] == x["target_bit"] for x in group) / len(group)
        persistent = accuracy("persistent/learned_condition_blind/sparse_k")
        exhaustive = accuracy("persistent/oracle/exhaustive_exact")
        reset = accuracy("reset/learned_condition_blind/sparse_k")
        swapped = accuracy("paired_swap/learned_condition_blind/sparse_k")
        recall = sum(x["recall_at_k"]["learned_condition_blind"] for x in group) / len(group)
        acc[str(c)] = {"persistent": persistent, "exhaustive_oracle": exhaustive,
                       "reset": reset, "paired_swap": swapped, "recall_at_k": recall}
        if persistent < t["minimum_seed_cell_accuracy"]: failures.append(f"cell accuracy {c}")
        if exhaustive - persistent > t["maximum_parity_gap_vs_exhaustive"]:
            failures.append(f"capability gap {c}")
        if recall < t["minimum_recall_at_k"]: failures.append(f"recall {c}")
    pooled = sum(x["predictions"]["persistent/learned_condition_blind/sparse_k"] ==
                 x["target_bit"] for x in rows) / len(rows)
    reset = sum(x["predictions"]["reset/learned_condition_blind/sparse_k"] ==
                x["target_bit"] for x in rows) / len(rows)
    swapped = sum(x["predictions"]["paired_swap/learned_condition_blind/sparse_k"] ==
                  x["target_bit"] for x in rows) / len(rows)
    if pooled < t["minimum_persistent_accuracy"]: failures.append("pooled accuracy")
    if pooled - reset < t["minimum_gain_over_reset"]: failures.append("reset gain")
    if pooled - swapped < t["minimum_gain_over_swapped"]: failures.append("swapped gain")
    # Pair-specific opposite-history predictions.
    disagreement = sum(
        p[0]["predictions"]["persistent/learned_condition_blind/sparse_k"] !=
        p[1]["predictions"]["persistent/learned_condition_blind/sparse_k"]
        for p in pairs.values()) / len(pairs)
    if disagreement < t["minimum_paired_history_disagreement"]:
        failures.append("paired-history disagreement")
    # Every cost includes the entire recorded work; compare same cell only.
    ratios = []
    xpoints = []
    for c in cells:
        def mean_cost(exe):
            z = costs[(c, exe)]
            return sum(z) / len(z)
        baseline = mean_cost("exhaustive_exact")
        sparse = mean_cost("sparse_k")
        if baseline <= 0:
            return {"decision": "VOID", "problems": [f"zero exhaustive work {c}"]}
        if c[1] == 512 and c[3] == 8:
            ratios.append(sparse / baseline)
        if c[3] == 8:
            xpoints.append((math.log2(c[1]), math.log2(max(sparse, 1e-12))))
    if sum(ratios) / len(ratios) > t["maximum_avg_total_work_ratio_at_h512_k8"]:
        failures.append("total work ratio")
    mx = sum(x for x, _ in xpoints) / len(xpoints)
    my = sum(y for _, y in xpoints) / len(xpoints)
    slope = sum((x-mx)*(y-my) for x,y in xpoints) / sum(
        (x-mx)**2 for x,_ in xpoints)
    if slope > t["maximum_log2_h_work_slope"]: failures.append("work slope")
    # Conservative additional limit: seed-level positivity is not inferred from mean.
    positive = sum(
        all(acc[str(c)]["persistent"] >= t["minimum_seed_cell_accuracy"]
            for c in cells if c[0] == seed)
        for seed in ev["seeds"])
    if positive < t["minimum_positive_seeds"]: failures.append("positive seeds")
    return {"decision": "FAIL" if failures else "PASS", "failures": failures[:100],
            "episodes": len(rows), "cells": len(cells), "pooled_accuracy": pooled,
            "reset_accuracy": reset, "swapped_accuracy": swapped,
            "paired_disagreement": disagreement, "total_work_ratio_h512_k8":
            sum(ratios)/len(ratios), "work_log2_slope": slope,
            "positive_seeds": positive, "per_cell": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    result = validate(json.loads(args.artifact.read_text()),
                      json.loads(CONTRACT.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["decision"] == "PASS" else 1)


if __name__ == "__main__":
    main()
