"""Run/replay DEV-016 environment validity; no learned-model claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.development.delayed_maintenance import (  # noqa: E402
    Action, MaintenanceWorld, SchemaReference, make_pair,
)

SOURCES = (
    "src/tac_osm/development/delayed_maintenance.py",
    "scripts/run_delayed_maintenance_dev.py",
    "contracts/DEV-016.json",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    contract = json.loads((ROOT / "contracts/DEV-016.json").read_text())
    rows = []
    fingerprints = set()
    for seed in contract["audit_seeds"]:
        for history in contract["histories"]:
            for delay in contract["delays"]:
                for role in ("retention", "acquisition", "cancelled_service"):
                    for repetition in range(contract["pairs_per_cell"]):
                        pair = make_pair(seed * 10000 + repetition, history, delay,
                                         contract["entities"], role=role)
                        if pair[0].expected == pair[1].expected:
                            raise AssertionError("counterfactual answers collapsed")
                        if (pair[0].candidate != pair[1].candidate or
                                pair[0].query_entity != pair[1].query_entity):
                            raise AssertionError("current query leaks partner identity")
                        for partner, case in enumerate(pair):
                            if case.fingerprint in fingerprints:
                                raise AssertionError("duplicate episode")
                            fingerprints.add(case.fingerprint)
                            model = SchemaReference(contract["entities"])
                            for obs in case.observations:
                                model.observe(obs)
                            prediction = model.predict(case.query_entity, case.candidate)
                            # Both actions use the public bounded statistic;
                            # diagnostic service has a known no-change horizon.
                            serviced = model.predict(case.query_entity,
                                                     Action("service", case.query_entity, 0, delay))
                            rows.append({
                                "seed": seed, "history": history, "delay": delay,
                                "role": role, "repetition": repetition, "partner": partner,
                                "fingerprint": case.fingerprint, "target": case.expected,
                                "reference_prediction": prediction,
                                "reset_prediction": 0,
                                "service_prediction": serviced,
                                "service_target": case.service_expected,
                                "optimal_action": "service" if case.expected else "wait",
                                "reference_action": "service" if prediction else "wait",
                                "post_action_failure": case.service_expected if prediction else case.expected,
                                "state_scalars": model.state_scalars,
                                "observation_count": len(case.observations),
                            })
    accuracy = lambda key: sum(r[key] == r["target"] for r in rows) / len(rows)
    summary = {"episodes": len(rows), "schema_accuracy": accuracy("reference_prediction"),
               "reset_accuracy": accuracy("reset_prediction"),
               "service_future_accuracy": sum(r["service_prediction"] == r["service_target"]
                                              for r in rows) / len(rows),
               "reference_action_accuracy": sum(r["reference_action"] == r["optimal_action"]
                                                for r in rows) / len(rows),
               "reference_post_action_failure_rate": sum(r["post_action_failure"] for r in rows) / len(rows),
               "state_scalars": max(r["state_scalars"] for r in rows),
               "by_role": {role: sum(r["reference_prediction"] == r["target"]
                                      for r in rows if r["role"] == role) /
                           sum(r["role"] == role for r in rows)
                           for role in ("retention", "acquisition", "cancelled_service")}}
    if summary["schema_accuracy"] != 1 or summary["reset_accuracy"] != .5:
        raise AssertionError("validity gates failed")
    if summary["state_scalars"] > contract["memory_scalar_limit"]:
        raise AssertionError("reference capacity exceeded")
    # Test actual public verifier on correct and incorrect committed claims.
    world = MaintenanceWorld((0, 1))
    if not world.inspect_receipt(0, 0).accepted or world.inspect_receipt(0, 1).accepted:
        raise AssertionError("verifier accepts contradictions")
    return {"study": "DEV-016", "status": "BENCHMARK_VALIDITY_ONLY",
            "parent_commit": contract["parent_commit"],
            "source_sha256": {p: digest(ROOT / p) for p in SOURCES},
            "summary": summary, "rows": rows,
            "learned_models_evaluated": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/DEV-016-validity")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    report = run()
    path = args.output / "report.json"
    if args.replay:
        saved = json.loads(path.read_text())
        if saved != report:
            raise AssertionError("source, grid or evaluation replay changed")
        print("Exact replay passed:", report["summary"]["episodes"], "episodes")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
