"""DEV-017 independent stdlib target/cost/sensing audit and success decision.

Does not perform neural replay or SGD replication. The CPU runner separately
reloads every trained checkpoint and regenerates its complete inference rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.development.schema_maintenance import (  # noqa: E402
    make_pair, execute, exact_public_targets, FAILURE_COST, SERVICE_COST, INSPECTION_COST,
)

ARMS = ("schema41", "schema41_encoder", "tac_reader", "tac_encoder", "tac_memory", "tac_joint")


def classify(logits):
    if len(logits) != 2 or not all(math.isfinite(x) for x in logits):
        raise AssertionError("invalid saved logits")
    return max(range(2), key=lambda i: logits[i])


def mean(values):
    return sum(values) / len(values) if values else None


def metrics(rows):
    mature = [r for r in rows if r["condition"]["deadline"] != "before"]
    correct = lambda r, field: r[field][r["target_entity"]][0] == r["wait_targets"][r["target_entity"]]
    long = [r for r in mature if r["condition"]["history"] == 512 and
            r["condition"]["due_delay"] >= 64 and "cancel" in r["condition"]["role"]]
    h512 = [r for r in rows if r["condition"]["history"] == 512]
    out = {"episodes": len(rows), "primary_future_accuracy": mean([correct(r, "predictions") for r in mature]),
           "reset_primary_accuracy": mean([correct(r, "reset_predictions") for r in mature]),
           "swap_primary_accuracy": mean([correct(r, "swapped_predictions") for r in mature]),
           "H512_long_pending_accuracy": mean([correct(r, "predictions") for r in long]),
           "before_due_accuracy": mean([correct(r, "predictions") for r in rows
                                        if r["condition"]["deadline"] == "before"])}
    for role in ("retention", "acquisition", "pending_cancel", "reissued_cancel"):
        out[role + "_accuracy"] = mean([correct(r, "predictions") for r in mature
                                        if r["condition"]["role"] == role])
    for mode in ("none", "selected", "matched_random"):
        for field, divisor in (("total_cost", 8), ("failures", 8), ("unnecessary_services", 8),
                               ("inspections", 1), ("services", 8)):
            numeric = lambda r: len(r["controls"][mode][field]) if field == "services" else r["controls"][mode][field]
            out[mode + "_" + field] = mean([numeric(r) / divisor for r in rows])
            out["H512_" + mode + "_" + field] = mean([numeric(r) / divisor for r in h512])
    return out


def audit(folder, spec):
    report = json.loads((folder / "report.json").read_text())
    seed = report["seed"]
    if (report["study"] != "DEV-017" or report["smoke"] or seed not in spec["seeds"] or
            report["status"] != "DEVELOPMENT_ONLY_NOT_CONFIRMATORY"):
        raise AssertionError("wrong study, smoke, seed or promoted status")
    for source, digest in report["source_sha256"].items():
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
            raise AssertionError("current source differs from experiment")
    training = report["training"]
    if (training["steps"] != spec["steps"] or len(set(training["train_fingerprints"])) != spec["steps"] * spec["batch_size"] or
            len(training["training_sensing_receipt_sha256"]) != spec["steps"] * spec["batch_size"]):
        raise AssertionError("training budget mismatch")
    for group in (ARMS[:2], ARMS[2:]):
        if len({training["initial_hashes"][a] for a in group}) != 1:
            raise AssertionError("control initialization mismatch")
    for arm in ARMS:
        for part, trainable in (("memory", {"schema41", "tac_memory", "tac_joint"}),
                                ("encoder", {"schema41", "schema41_encoder", "tac_encoder", "tac_joint"})):
            changed = training["initial_components"][arm][part] != training["final_components"][arm][part]
            if changed != (arm in trainable) or training["gradients"][arm][part] != (arm in trainable):
                raise AssertionError("memory/encoder intervention integrity failed")
    baseline_fps = None
    expected_n = len(spec["h_levels"]) * len(spec["eval_due_delays"]) * 3 * 4 * spec["eval_steps"] * 2
    cases_cache = {}
    audited = {}
    for arm in ARMS:
        path = folder / f"{arm}-seed{seed}.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["models"][arm]["sha256"]:
            raise AssertionError("checkpoint file altered")
        rows = report["rows"][arm]
        fps = [r["fingerprint"] for r in rows]
        if len(rows) != expected_n or len(set(fps)) != expected_n or set(fps) & set(training["train_fingerprints"]):
            raise AssertionError("incomplete, duplicate or overlapping evaluation")
        if baseline_fps is not None and baseline_fps != fps:
            raise AssertionError("different public observations across arms")
        baseline_fps = fps
        for row in rows:
            c = row["condition"]
            key = tuple(c[k] for k in ("history", "due_delay", "role", "deadline", "repetition"))
            if key not in cases_cache:
                cases_cache[key] = make_pair((seed + 17000) * 10000 + c["repetition"],
                                             c["history"], c["due_delay"], c["role"], c["deadline"])
            ep = cases_cache[key][c["partner"]]
            if (row["fingerprint"] != ep.fingerprint or row["target_entity"] != ep.target_entity or
                    row["wait_targets"] != list(ep.wait_targets) or row["current_targets"] != list(ep.current_targets) or
                    row["service_targets"] != list(ep.service_targets) or row["pre_sensing_observations"] != c["history"] or
                    row["state_scalars"] != (41 if arm.startswith("schema") else 768)):
                raise AssertionError("public length, generator, targets or memory budget mismatch")
            if exact_public_targets(ep.observations, ep.tick, ep.horizon) != ep.wait_targets:
                raise AssertionError("public sufficient statistic fails")
            prediction = [[classify(candidate) for candidate in entity] for entity in row["logits"]]
            if prediction != row["predictions"]:
                raise AssertionError("pre-sensing prediction not derived from logits")
            selected_count = row["controls"]["selected"]["inspections"]
            for mode, outcome in row["controls"].items():
                entities = outcome["inspected_entities"]
                if len(entities) != outcome["inspections"] or len(set(entities)) != len(entities) or len(entities) > 2:
                    raise AssertionError("inspection count/budget changed")
                if mode == "none" and entities:
                    raise AssertionError("no-sensing policy received inspection")
                if mode == "matched_random":
                    order = list(range(8))
                    random.Random("dev017-random-sensing:" + ep.fingerprint).shuffle(order)
                    if entities != order[:selected_count]:
                        raise AssertionError("random sensing not budget matched")
                if mode == "selected":
                    chosen = []
                    for trace in outcome["sensing_trace"]:
                        if trace["already_inspected"] != chosen:
                            raise AssertionError("sensing trace state inconsistent")
                        scores = [(FAILURE_COST * p * (1 - p), e) for e, p in enumerate(trace["wait_probabilities"])
                                  if e not in chosen]
                        score, e = max(scores, key=lambda x: (x[0], -x[1]))
                        expected_entity = e if score > INSPECTION_COST else None
                        if trace["chosen_entity"] != expected_entity:
                            raise AssertionError("sensing choice not determined by saved learned beliefs")
                        if expected_entity is not None:
                            chosen.append(expected_entity)
                    if chosen != entities:
                        raise AssertionError("sensing trace differs from executed inspections")
                if len(outcome["inspection_receipts"]) != len(entities):
                    raise AssertionError("missing charged receipts")
                for e, receipt in zip(entities, outcome["inspection_receipts"]):
                    if (receipt["entity"] != e or receipt["kind"] != "inspect" or receipt["tick"] != ep.tick or
                            receipt["value"] != ep.current_targets[e] or receipt["due"] != -1):
                        raise AssertionError("not a public action-selected inspection receipt")
                actual = execute(ep.world, ep.horizon, tuple(outcome["services"]), len(entities))
                if any(outcome[k] != value for k, value in actual.items()):
                    raise AssertionError("causal outcome or cost arithmetic changed")
                probabilities = outcome["decision_probabilities"]
                services = [e for e, p in enumerate(probabilities) if FAILURE_COST * p[0] > SERVICE_COST + FAILURE_COST * p[1]]
                if services != outcome["services"]:
                    raise AssertionError("service actions not derived from charged predictive cost")
                committed = [classify(outcome["decision_logits"][e][1 if e in services else 0]) for e in range(8)]
                if (committed != outcome["committed_forecast"] or
                        outcome["verification"] != [a == b for a, b in zip(committed, actual["outcomes"])]):
                    raise AssertionError("post-action verification not derived from committed prediction")
        derived = metrics(rows)
        if derived != report["summaries"][arm] or derived["reset_primary_accuracy"] != .5:
            raise AssertionError("metric derivation or reset gate failed")
        audited[arm] = derived
    return {"seed": seed, "metrics": audited, "sources": report["source_sha256"], "models": report["models"]}


def interval(deltas):
    m = mean(deltas)
    half = 4.302652729749 * statistics.stdev(deltas) / math.sqrt(3)
    return {"mean": m, "exploratory_95_t_interval": [m - half, m + half], "seed_deltas": deltas}


def aggregate(folders):
    spec = json.loads((ROOT / "contracts/DEV-017.json").read_text())
    records = sorted((audit(f, spec) for f in folders), key=lambda r: r["seed"])
    if [r["seed"] for r in records] != spec["seeds"] or any(r["sources"] != records[0]["sources"] for r in records):
        raise AssertionError("wrong three-seed coverage or inconsistent sources")
    means = {a: {k: mean([r["metrics"][a][k] for r in records]) for k in records[0]["metrics"][a]}
             for a in ARMS}
    contrasts = {}
    for a, b in (("schema41", "tac_joint"), ("schema41", "schema41_encoder"),
                 ("tac_joint", "tac_encoder"), ("tac_memory", "tac_reader")):
        contrasts[a + "_minus_" + b] = interval([r["metrics"][a]["primary_future_accuracy"] -
                                                 r["metrics"][b]["primary_future_accuracy"] for r in records])
    thresholds = spec["success_thresholds"]
    reliability = {}
    for a in ARMS:
        seed_pass = all(r["metrics"][a]["primary_future_accuracy"] >= thresholds["primary_each_seed"] and
                        r["metrics"][a]["H512_long_pending_accuracy"] >= thresholds["H512_long_pending_each_seed"] and
                        r["metrics"][a]["H512_selected_failures"] <= thresholds["H512_active_failure_each_seed_max"] and
                        r["metrics"][a]["H512_selected_unnecessary_services"] <= thresholds["H512_active_unnecessary_each_seed_max"] and
                        r["metrics"][a]["H512_selected_total_cost"] <= thresholds["H512_active_cost_each_seed_max"]
                        for r in records)
        sensing_gain = means[a]["selected_total_cost"] - means[a]["none_total_cost"]
        reliability[a] = seed_pass and sensing_gain <= thresholds["selected_minus_none_cost_mean_max"]
    causal = all(contrasts[k]["mean"] >= thresholds["memory_control_mean_gain"] and
                 contrasts[k]["exploratory_95_t_interval"][0] > thresholds["memory_control_interval_lower_min"]
                 for k in ("tac_joint_minus_tac_encoder", "tac_memory_minus_tac_reader"))
    # Reset chance is a validity gate; counterfactual state-swap should also
    # damage the reliable joint forecast before claiming past-state causality.
    causal = causal and means["tac_joint"]["primary_future_accuracy"] - means["tac_joint"]["swap_primary_accuracy"] >= thresholds["state_swap_primary_drop_min"]
    return {"study": "DEV-017", "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY", "seeds": spec["seeds"],
            "checkpoints_audited": 18, "model_episodes_audited": sum(r["metrics"][a]["episodes"] for r in records for a in ARMS),
            "means": means, "contrasts": contrasts, "by_seed": records,
            "reliability_and_cost_gate": reliability,
            "TAC_memory_causality_gate": causal,
            "DEV017_TAC_success": reliability["tac_joint"] and causal,
            "warning": "Adaptive development, exploratory three-seed intervals, no SGD or independent scientific replication."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=Path, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = aggregate(args.folders)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({k: analysis[k] for k in ("means", "contrasts", "reliability_and_cost_gate",
                                              "TAC_memory_causality_gate", "DEV017_TAC_success")}, indent=2))
