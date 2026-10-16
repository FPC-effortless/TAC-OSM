"""DEV-018 standard-library audit, seed-level analysis and accessible CSVs.

Independent target/action/cost derivation; neural replay is a separate gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.development.schema_maintenance import make_pair, execute, exact_public_targets  # noqa: E402

ARMS = ("memoryless", "gru768", "delta768", "schema41", "schema41_frozen",
        "tac_joint", "tac_encoder", "tac_memory", "tac_reader", "transformer")


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def classify(logits):
    if len(logits) != 2 or not all(math.isfinite(x) for x in logits):
        raise AssertionError("invalid logits")
    return int(logits[1] > logits[0])


def probability(logits):
    difference = max(-700., min(700., logits[0] - logits[1]))
    return 1 / (1 + math.exp(difference))


def primary_rows(rows):
    long = [r for r in rows if r["condition"]["history"] == 512 and
            r["condition"]["deadline"] != "before" and "cancel" in r["condition"]["role"]]
    return long or [r for r in rows if "cancel" in r["condition"]["role"] and
                    r["condition"]["deadline"] != "before"]  # smoke only


def accuracy(rows, field="predictions"):
    return mean([r[field][r["target_entity"]][0] == r["wait_targets"][r["target_entity"]] for r in rows])


def metrics(rows):
    primary = primary_rows(rows)
    counts = Counter(r["wait_targets"][r["target_entity"]] for r in primary)
    controls = [r for r in rows if "controls" in r]
    out = {"episodes": len(rows), "primary_n": len(primary), "primary": accuracy(primary),
           "majority_baseline": max(counts.values()) / len(primary),
           "reset_primary": accuracy(primary, "reset_predictions"),
           "swap_primary": accuracy(primary, "swapped_predictions"),
           "all_mature_accuracy": accuracy([r for r in rows if r["condition"]["deadline"] != "before"]),
           "primary_brier": mean([(probability(r["logits"][r["target_entity"]][0]) -
                                    r["wait_targets"][r["target_entity"]]) ** 2 for r in primary])}
    for mode in ("none", "selected", "matched_random", "do_nothing", "always_service", "public_controller"):
        for field, divisor in (("total_cost", 8), ("failures", 8), ("unnecessary_services", 8),
                               ("services", 8), ("inspections", 1)):
            values = [r["controls"][mode][field] for r in controls]
            out[mode + "_" + field] = mean([(len(v) if isinstance(v, list) else v) / divisor for v in values])
    return out


def audit(folder, spec):
    report = json.loads((folder / "report.json").read_text())
    seed, regime = report["seed"], report["regime"]
    if (report["study"] != "DEV-018" or report["smoke"] or seed not in spec["seeds"] or
            regime not in spec["regimes"] or report["status"] != "ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY"):
        raise AssertionError("provenance mismatch")
    for p, sha in report["source_sha256"].items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha:
            raise AssertionError("source changed")
    expected_n = len(spec["h_levels"]) * len(spec["eval_due_delays"]) * 3 * 4 * spec["eval_steps"] * 2
    cfg = spec["regimes"][regime]
    reference_fps = reference_train = None
    cache = {}
    for arm in ARMS:
        if hashlib.sha256((folder / f"{arm}.pt").read_bytes()).hexdigest() != report["models"][arm]["sha256"]:
            raise AssertionError("checkpoint changed")
        tr = report["training"][arm]
        if (tr["steps"] != spec["steps"] or tr["history"] != cfg["history"] or
                tr["batch_size"] != cfg["batch_size"] or
                tr["observed_events"] != spec["steps"] * 1024 or
                tr["training_receipts"] != spec["steps"] * cfg["batch_size"] or
                len(set(tr["fingerprints"])) != spec["steps"] * cfg["batch_size"]):
            raise AssertionError("training exposure mismatch")
        if reference_train is not None and reference_train != tr["fingerprints"]:
            raise AssertionError("unmatched training evidence")
        reference_train = tr["fingerprints"]
        for prefix in ("event_encoder.", "memory."):
            expected = (arm not in ("tac_memory", "tac_reader") if prefix == "event_encoder." else
                        arm not in ("memoryless", "schema41_frozen", "tac_encoder", "tac_reader"))
            if ((tr["before_components"][prefix] != tr["after_components"][prefix]) != expected or
                    tr["gradients"][prefix] != expected):
                raise AssertionError("parameter intervention failed")
        rows = report["rows"][arm]
        fps = [r["fingerprint"] for r in rows]
        if (len(rows) != expected_n or len(set(fps)) != expected_n or
                set(fps) & set(tr["fingerprints"]) or (reference_fps is not None and reference_fps != fps)):
            raise AssertionError("evaluation coverage, matching or separation failed")
        reference_fps = fps
        for row in rows:
            c = row["condition"]
            key = tuple(c[k] for k in ("history", "due_delay", "role", "deadline", "repetition"))
            if key not in cache:
                cache[key] = make_pair((seed + 18000) * 10000 + c["repetition"], c["history"], c["due_delay"],
                                      c["role"], c["deadline"], domain="dev018-eval")
            ep = cache[key][c["partner"]]
            expected_state = None if arm == "transformer" else (0 if arm == "memoryless" else
                             41 if arm.startswith("schema") else 768)
            expected_stored = c["history"] * 12 if arm == "transformer" else (12 if arm == "memoryless" else expected_state)
            if (row["fingerprint"] != ep.fingerprint or row["wait_targets"] != list(ep.wait_targets) or
                    row["service_targets"] != list(ep.service_targets) or row["current_targets"] != list(ep.current_targets) or
                    row["target_entity"] != ep.target_entity or row["state_scalars"] != expected_state or
                    row["stored_feature_scalars"] != expected_stored or
                    exact_public_targets(ep.observations, ep.tick, ep.horizon) != ep.wait_targets):
                raise AssertionError("public boundary, state budget or target mismatch")
            if row["predictions"] != [[classify(z) for z in e] for e in row["logits"]]:
                raise AssertionError("saved prediction not determined by logits")
            if ("controls" in row) != (c["history"] == 512):
                raise AssertionError("control coverage changed")
            if "controls" not in row:
                continue
            for mode, outcome in row["controls"].items():
                n = outcome["inspections"]
                if n not in (0, 1):
                    raise AssertionError("inspection budget exceeded")
                actual = execute(ep.world, ep.horizon, tuple(outcome["services"]), n)
                if any(outcome[k] != v for k, v in actual.items()):
                    raise AssertionError("causal outcome/cost arithmetic differs")
                if mode == "do_nothing":
                    expected_services = []
                elif mode == "always_service":
                    expected_services = list(range(8))
                elif mode == "public_controller":
                    expected_services = [e for e in range(8) if ep.wait_targets[e]]
                else:
                    probs = outcome["decision_probabilities"]
                    if any(not math.isfinite(p) or not 0 <= p <= 1 for entity in probs for p in entity):
                        raise AssertionError("invalid decision probability")
                    expected_services = [e for e, p in enumerate(probs) if p[0] - p[1] > .25]
                if outcome["services"] != expected_services:
                    raise AssertionError("policy actions changed")
                if mode in ("selected", "matched_random"):
                    entity, receipt = outcome["entity"], outcome["receipt"]
                    if n != int(entity is not None) or (receipt is None) != (entity is None):
                        raise AssertionError("receipt/cost mismatch")
                    if entity is not None and receipt != {"tick": ep.tick, "entity": entity, "kind": "inspect",
                                                          "value": ep.current_targets[entity], "due": -1}:
                        raise AssertionError("invalid chosen public receipt")
                    if mode == "selected":
                        pre = row["controls"]["none"]["decision_probabilities"]
                        base = sum(min(4 * p[0], 1 + 4 * p[1]) for p in pre)
                        gains = [base - ((1 - pre[e][2]) * costs[0] + pre[e][2] * costs[1]) - .1
                                 for e, costs in enumerate(outcome["hypothetical_costs"])]
                        if gains != outcome["estimated_gains"]:
                            raise AssertionError("value-of-observation arithmetic changed")
                        best = max(range(8), key=lambda e: (gains[e], -e))
                        if entity != (best if gains[best] > 0 else None):
                            raise AssertionError("inspection selected using wrong information")
                    else:
                        expected = random.Random("dev018-sensing:" + ep.fingerprint).randrange(8) if row["controls"]["selected"]["inspections"] else None
                        if entity != expected:
                            raise AssertionError("random control not budget matched")
            probs = row["controls"]["none"]["decision_probabilities"]
            for threshold in spec["frontier_thresholds"]:
                services = tuple(e for e, p in enumerate(probs) if p[0] - p[1] > threshold)
                if row["frontier"][str(threshold)] != execute(ep.world, ep.horizon, services, 0):
                    raise AssertionError("frontier/cost changed")
        if metrics(rows) != report["summaries"][arm] or metrics(rows)["reset_primary"] != .5:
            raise AssertionError("summary or paired reset leakage gate failed")
    # Verify byte-identical initial native-writer models and common components.
    native = [report["training"][a] for a in ("tac_joint", "tac_encoder", "tac_memory", "tac_reader")]
    if len({r["initial_hash"] for r in native}) != 1:
        raise AssertionError("native intervention initial tensors differ")
    if len({report["training"][a]["before_components"]["event_encoder."] for a in ARMS}) != 1:
        raise AssertionError("shared encoder initialization differs")
    return report


def interval(values):
    half = 4.302652729749 * statistics.stdev(values) / math.sqrt(3)
    m = mean(values)
    return {"mean": m, "exploratory_95_t_df2": [m - half, m + half], "seed_deltas": values}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(folders, output):
    spec = json.loads((ROOT / "contracts/DEV-018.json").read_text())
    reports = sorted([audit(f, spec) for f in folders], key=lambda r: (r["regime"], r["seed"]))
    expected = {(g, s) for g in spec["regimes"] for s in spec["seeds"]}
    if len(reports) != len(expected) or {(r["regime"], r["seed"]) for r in reports} != expected:
        raise AssertionError("incomplete seed/regime matrix")
    if any(r["source_sha256"] != reports[0]["source_sha256"] for r in reports):
        raise AssertionError("mixed sources")
    means, contrasts, reliability = {}, {}, {}
    seed_csv, curve_csv, frontier_csv = [], [], []
    for regime in spec["regimes"]:
        group = [r for r in reports if r["regime"] == regime]
        means[regime] = {a: {k: mean([r["summaries"][a][k] for r in group])
                             for k in group[0]["summaries"][a]} for a in ARMS}
        contrasts[regime] = {}
        for a, b in (("tac_joint", "tac_encoder"), ("tac_memory", "tac_reader"),
                     ("tac_joint", "gru768"), ("tac_joint", "delta768"),
                     ("tac_joint", "memoryless"), ("schema41", "schema41_frozen")):
            contrasts[regime][a + "_minus_" + b] = interval([r["summaries"][a]["primary"] -
                                                             r["summaries"][b]["primary"] for r in group])
        reliability[regime] = {}
        for a in ARMS:
            reliability[regime][a] = all(r["summaries"][a]["primary"] >= .8 and
                r["summaries"][a]["selected_failures"] <= .05 and
                r["summaries"][a]["selected_unnecessary_services"] <= .10 and
                r["summaries"][a]["selected_total_cost"] <= .8 for r in group) and (
                means[regime][a]["selected_total_cost"] <= means[regime][a]["none_total_cost"])
    horizon = {a: interval([next(r for r in reports if r["regime"] == "long" and r["seed"] == s)["summaries"][a]["primary"] -
                           next(r for r in reports if r["regime"] == "short" and r["seed"] == s)["summaries"][a]["primary"]
                           for s in spec["seeds"]]) for a in ARMS}
    causal = all(contrasts["long"][key]["mean"] >= .05 and contrasts["long"][key]["exploratory_95_t_df2"][0] > 0
                 for key in ("tac_joint_minus_tac_encoder", "tac_memory_minus_tac_reader")) and (
                 means["long"]["tac_joint"]["primary"] - means["long"]["tac_joint"]["swap_primary"] >= .1)
    for r in reports:
        for arm in ARMS:
            seed_csv.append(dict(seed=r["seed"], regime=r["regime"], arm=arm, **r["summaries"][arm]))
            buckets = defaultdict(list)
            for row in r["rows"][arm]:
                c = row["condition"]
                buckets[(c["history"], c["due_delay"], c["deadline"], c["role"])].append(row)
            for (h, d, deadline, role), rows in buckets.items():
                counts = Counter(x["wait_targets"][x["target_entity"]] for x in rows)
                curve_csv.append(dict(seed=r["seed"], regime=r["regime"], arm=arm, history=h, due_delay=d,
                    deadline=deadline, role=role, n=len(rows), accuracy=accuracy(rows),
                    majority_baseline=max(counts.values()) / len(rows)))
            controlled = [x for x in r["rows"][arm] if "frontier" in x]
            for t in spec["frontier_thresholds"]:
                outcomes = [x["frontier"][str(t)] for x in controlled]
                frontier_csv.append(dict(seed=r["seed"], regime=r["regime"], arm=arm, threshold=t,
                    failures=mean([x["failures"] / 8 for x in outcomes]),
                    services=mean([len(x["services"]) / 8 for x in outcomes]),
                    cost=mean([x["cost_per_entity"] for x in outcomes])))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per-seed.csv", seed_csv)
    write_csv(output / "accuracy-by-history-delay.csv", curve_csv)
    write_csv(output / "failure-service-frontier.csv", frontier_csv)
    result = {"study": "DEV-018", "status": "ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY", "means": means,
              "contrasts": contrasts, "long_minus_short": horizon, "reliability_and_cost": reliability,
              "TAC_memory_attribution": causal,
              "TAC_success": reliability["long"]["tac_joint"] and causal,
              "checkpoints": len(reports) * len(ARMS),
              "model_episodes": sum(len(r["rows"][a]) for r in reports for a in ARMS),
              "source_sha256": reports[0]["source_sha256"],
              "by_seed": [{k: r[k] for k in ("seed", "regime", "summaries", "models", "training")} for r in reports],
              "warning": "Three seed exploratory intervals, correlated paired rows, no external replication; match recurrent scalars not model FLOPs."}
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print("AUDITED " + json.dumps({k: result[k] for k in ("means", "contrasts", "long_minus_short", "reliability_and_cost", "TAC_memory_attribution", "TAC_success", "checkpoints", "model_episodes")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", type=Path, nargs=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.folders, args.output)
