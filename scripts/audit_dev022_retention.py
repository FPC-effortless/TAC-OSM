"""DEV-022 stdlib evidence audit; independent public histories, decisions and costs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tac_osm.development.schema_maintenance import make_pair, execute, exact_public_targets, ROLES, DEADLINES  # noqa: E402
from audit_memory_controls_dev import metrics, classify, probability, interval, mean  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_and_sources():
    spec = json.loads((ROOT / "contracts/DEV-018.json").read_text())
    spec.update(json.loads((ROOT / "contracts/DEV-022.json").read_text()))
    return spec


def validate_rows(rows, seed, arm, spec):
    expected = []
    for h in spec["h_levels"]:
        for d in spec["eval_due_delays"]:
            for deadline in DEADLINES:
                for role in ROLES:
                    for rep in range(spec["eval_steps"]):
                        for partner in (0, 1):
                            expected.append(dict(history=h, due_delay=d, deadline=deadline,
                                                 role=role, repetition=rep, partner=partner))
    assert [r["condition"] for r in rows] == expected, "evaluation grid/order mismatch"
    assert len({r["fingerprint"] for r in rows}) == len(expected)
    cache = {}
    for row in rows:
        c = row["condition"]
        key = tuple(c[k] for k in ("history", "due_delay", "role", "deadline", "repetition"))
        if key not in cache:
            cache[key] = make_pair((seed + 22000) * 10000 + c["repetition"], c["history"], c["due_delay"],
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
        for entity, logits in enumerate(row["logits"]):
            for action, z in enumerate(logits):
                probability(z)  # finite two-class validation is also above
                if "controls" in row:
                    actual_p = row["controls"]["none"]["decision_probabilities"][entity][action]
                    assert abs(actual_p - probability(z)) < 2e-6
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


def audit(folder, spec):
    path = folder / "report.json"
    r = json.loads(path.read_text())
    assert r["study"] == "DEV-022" and not r["smoke"]
    assert r["status"] == "ADAPTIVE_TRAINED_INTERVENTION_NOT_CONFIRMATORY"
    seed, regime, arm = r["seed"], r["regime"], r["arm"]
    assert seed in spec["seeds"] and regime in spec["regimes"] and arm in spec["arms"]
    required = {"scripts/run_dev022_retention.py", "scripts/audit_dev022_retention.py",
                "contracts/DEV-022.json", "docs/DEV-022-RETENTION-PROTOCOL.md",
                "src/tac_osm/development/retention_controls.py", "scripts/run_memory_controls_dev.py",
                "scripts/run_learned_schema_dev.py", "scripts/audit_memory_controls_dev.py",
                "contracts/DEV-018.json", "src/tac_osm/development/memory_baselines.py",
                "src/tac_osm/development/noisy_memory.py", "src/tac_osm/development/schema_maintenance.py",
                "src/tac_osm/development/delayed_maintenance.py"}
    assert set(r["source_sha256"]) == required
    assert all(digest(ROOT / p) == sha for p, sha in r["source_sha256"].items())
    checkpoint = folder / "model.pt"
    assert digest(checkpoint) == r["model"]["sha256"]
    replay = json.loads((folder / "replay.json").read_text())
    assert replay == {"verified": True, "checkpoint_sha256": digest(checkpoint),
                      "report_sha256": digest(path), "source_sha256": r["source_sha256"]}
    cfg = spec["regimes"][regime]
    tr = r["training"]
    assert (tr["steps"], tr["history"], tr["batch_size"], tr["observed_events"], tr["training_receipts"]) == (
        spec["steps"], cfg["history"], cfg["batch_size"], spec["steps"] * 1024, spec["steps"] * cfg["batch_size"])
    expected_fps = []
    for step in range(spec["steps"]):
        for j in range(cfg["batch_size"] // 2):
            pair = make_pair((seed + 4000) * 1000000 + step * 4 + j, cfg["history"],
                             spec["train_due_delays"][(step // 4) % 3], ROLES[(step + j) % 4],
                             DEADLINES[(step // 12) % 3], domain="dev018-train")
            expected_fps.extend(c.fingerprint for c in pair)
    assert tr["fingerprints"] == expected_fps and len(set(expected_fps)) == len(expected_fps)
    assert not set(expected_fps) & {row["fingerprint"] for row in r["rows"]}
    for prefix in ("event_encoder.", "memory."):
        active = prefix == "event_encoder." or arm != "memoryless"
        assert (tr["before_components"][prefix] != tr["after_components"][prefix]) == active
        assert tr["gradients"][prefix] == active
    assert tr["initial_hash"] != tr["final_hash"]
    assert math.isfinite(tr["final_loss"])
    assert 0 < r["model"]["trainable_parameters"] <= r["model"]["parameters"]
    validate_rows(r["rows"], seed, arm, spec)
    assert metrics(r["rows"]) == r["summary"]
    assert r["summary"]["primary_n"] == 96 and r["summary"]["reset_primary"] == .5
    return r


def summarize(records, spec):
    expected = {(s, g, a) for s in spec["seeds"] for g in spec["regimes"] for a in spec["arms"]}
    by = {(r["seed"], r["regime"], r["arm"]): r for r in records}
    assert len(by) == len(records) and set(by) == expected
    assert all(r["source_sha256"] == records[0]["source_sha256"] for r in records)
    for s in spec["seeds"]:
        for g in spec["regimes"]:
            group = [by[s, g, a] for a in spec["arms"]]
            assert len({r["training"]["before_components"]["event_encoder."] for r in group}) == 1
            assert all(r["training"]["initial_common_components"] == group[0]["training"]["initial_common_components"] for r in group)
            native = [r for r in group if r["arm"] not in ("schema41", "memoryless")]
            assert len({r["training"]["initial_hash"] for r in native}) == 1
            assert len({r["model"]["parameters"] for r in native}) == 1
            assert all(r["training"]["fingerprints"] == group[0]["training"]["fingerprints"] for r in group)
            assert all([x["fingerprint"] for x in r["rows"]] == [x["fingerprint"] for x in group[0]["rows"]] for r in group)
    result = {"study": "DEV-022", "status": "ADAPTIVE_NOT_CONFIRMATORY", "cells": len(records), "by_regime": {},
              "official_TAC_overall_success": False, "entity_count_generalization_tested": False}
    gate = spec["reliability"]
    for g in spec["regimes"]:
        out = {"arms": {}, "retention_contrasts": {}}
        for a in spec["arms"]:
            rs = [by[s, g, a] for s in spec["seeds"]]
            ms = [r["summary"] for r in rs]
            prediction = all(m["primary"] >= gate["min_seed_accuracy"] and
                             m["primary"] - m["swap_primary"] >= spec["retention_gain"]["min_seed_swap_drop"] for m in ms)
            physical = all(m["selected_failures"] <= gate["max_failure_rate"] and
                           m["selected_unnecessary_services"] <= gate["max_unnecessary_service_rate"] and
                           m["selected_total_cost"] <= gate["max_cost_per_entity"] for m in ms)
            physical &= mean([m["selected_total_cost"] for m in ms]) <= mean([m["none_total_cost"] for m in ms])
            strata = {}
            for h in spec["h_levels"]:
                for role in ROLES:
                    selected = [[x for x in r["rows"] if x["condition"]["history"] == h and
                                 x["condition"]["role"] == role and x["condition"]["deadline"] != "before"] for r in rs]
                    strata[f"H{h}:{role}"] = [mean([x["predictions"][x["target_entity"]][0] == x["wait_targets"][x["target_entity"]] for x in xs]) for xs in selected]
            out["arms"][a] = {"per_seed": ms, "means": {k: mean([m[k] for m in ms]) for k in ms[0]},
                              "prediction_and_swap_gate": prediction, "physical_gate": physical,
                              "combined_development_gate": prediction and physical, "mature_strata": strata,
                              "trainable_parameters": [r["model"]["trainable_parameters"] for r in rs]}
        for treatment, control in spec["retention_contrasts"]:
            delta = [by[s, g, treatment]["summary"]["primary"] - by[s, g, control]["summary"]["primary"] for s in spec["seeds"]]
            contrast = interval(delta)
            contrast["passes_gain_gate"] = (contrast["mean"] >= spec["retention_gain"]["min_mean_accuracy_gain"] and
                                             contrast["exploratory_95_t_df2"][0] > 0)
            out["retention_contrasts"][treatment + "_minus_" + control] = contrast
        interaction = [by[s, g, "neither"]["summary"]["primary"] - by[s, g, "no_erase"]["summary"]["primary"] -
                       by[s, g, "no_decay"]["summary"]["primary"] + by[s, g, "native"]["summary"]["primary"] for s in spec["seeds"]]
        out["decay_erase_accuracy_interaction"] = interval(interaction)
        out["primary_retention_success"] = (out["retention_contrasts"]["no_decay_minus_native"]["passes_gain_gate"] and
                                             out["arms"]["no_decay"]["combined_development_gate"])
        result["by_regime"][g] = out
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    spec = spec_and_sources()
    records = [audit(folder, spec) for folder in a.folders]
    result = summarize(records, spec)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print("DEV022_AUDIT_PASSED cells=" + str(len(records)), flush=True)
