"""Separate-process DEV-052B training, calibration, policy and native audit.

Native verification below replays recorded primitives without the rollout's
sensor/action wrapper and independently accounts paid costs and path constraints.
This is a deterministic reproduction audit by the same implementation author,
not an independent human review or an independent simulator implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import torch
import numpy as np
from investigation_world.operational.models import EpisodeSubmission
from investigation_world.operational.runtime import OperationalRuntime
from tac_osm.development.learned_belief_world import (
    NEURAL, ARMS, BASE, NativeWorld, make_world, collect, estimate,
    tensorize, summarize, strip_timing, bellman,
)
from run_learned_belief_dev052b import (
    PROTOCOL, digest, write, setup, specifications, relabel, fit_arm,
    evaluate, choose_calibration,
)


def native_replay(row, shifted):
    episode, params, _ = make_world(row["seed"], row["index"], shifted)
    runtime = OperationalRuntime(episode)
    reads = [0, 0]
    paid = path_violations = 0
    for call in row["calls"]:
        name, args = call["name"], call["parameters"]
        entity = args["entity"]
        result = dict(runtime.act(name, **args))
        paid += {"inspect": 1, "isolate": 2, "service": 10}[name]
        if name == "inspect":
            # Independent spelling of the registered stream, no NativeWorld.act.
            rng = random.Random(f"tac-dev052b-v1:sensor:{row['seed']}:{row['index']}:{entity}:{reads[entity]}")
            bit = int(runtime.state[f"e{entity}.fault"])
            result["fault"] = bit if rng.random() < params[0] else 1-bit
            reads[entity] += 1
        assert result == call["receipt"], (row["world"], name, "receipt drift")
        path_violations += int(runtime.state["plant.collateral"] != 0)
    faults = sum(int(runtime.state[f"e{e}.fault"]) for e in (0, 1))
    path_violations += int(runtime.state["plant.slot"] != -1)
    verified = runtime.submit(EpisodeSubmission(conclusion="DEV-052B public predictive repair"))
    assert verified.model_dump(mode="json") == row["verification"]
    assert faults == row["failures"]
    assert paid == runtime.budget.spent == row["native_spent"]
    assert paid/10+4*faults == row["cost"]
    assert int(verified.state == 1) == row["complete"]
    assert sum(event.blocked for event in runtime.events) == row["blocked"]
    assert path_violations == row["violations"]
    return len(row["calls"])


def paired_interval(reports, common, arm, cohort):
    baseline = common["baselines"]["estimated_bayes"][cohort]["rows"]
    rows = [report["arms"][arm]["evaluation"][cohort]["rows"] for report in reports]
    delta = np.array([[r["cost"]/2-baseline[i]["cost"]/2 for i, r in enumerate(seed_rows)]
                      for seed_rows in rows]).mean(0)
    failures = np.array([[r["failures"]-baseline[i]["failures"] for i, r in enumerate(seed_rows)]
                         for seed_rows in rows]).mean(0)
    rng = np.random.default_rng(5521999)
    n = len(delta)
    if cohort == "shifted":
        groups = [np.arange(k, n, 4) for k in range(4)]
        ids = np.concatenate([rng.choice(g, size=(2000, len(g))) for g in groups], axis=1)
    else:
        ids = rng.integers(n, size=(2000, n))
    return {"paired_mean_cost_delta": float(delta.mean()),
            "cost_delta_95pct_descriptive_interval": np.quantile(delta[ids].mean(1), [.025, .975]).tolist(),
            "paired_mean_fault_delta_per_world": float(failures.mean()),
            "fault_delta_95pct_descriptive_interval": np.quantile(failures[ids].mean(1), [.025, .975]).tolist(),
            "independent_worlds": n,
            "note": "Model seeds averaged within worlds; shifted resampling stratified by family. Adaptive development, not confirmation."}


def audit(directory, veritas_root, smoke=False):
    started = time.monotonic()
    proto = setup(veritas_root, dirty=smoke)
    folder = Path(directory)
    common = json.loads((folder/"common.json").read_text())
    assert common["study"] == "DEV-052B" and common["smoke"] == smoke
    assert common["protocol_sha256"] == digest(PROTOCOL)
    manifest = json.loads((folder/"manifest.json").read_text())
    for path, expected in manifest.items():
        assert digest(folder/path) == expected, (path, "artifact digest mismatch")
    specs = specifications(proto, smoke)
    ledger = json.loads((folder/"split-ledger.json").read_text())
    ids = [identity for group in ledger.values() for identity in group]
    assert len(ids) == len(set(ids))
    for name, spec in specs.items():
        expected = [NativeWorld(spec["seed"], i, spec["shifted"]).episode.episode_id
                    for i in range(spec["worlds"])]
        assert ledger[name] == expected
    sequences, sensors = collect(specs["train"])
    params, table = estimate(sequences, sensors)
    relabel(sequences, params)
    validation, _ = collect(specs["validation"], params)
    calibration, _ = collect(specs["calibration"], params)
    reconstructed = {"training": sequences, "sensor_correctness": sensors, "validation": validation,
                     "calibration": calibration, "estimated_parameters": params, "event_table": table}
    old_data = json.loads((folder/"training-data.json").read_text())
    assert json.loads(json.dumps(reconstructed)) == old_data
    native_calls = native_worlds = 0
    source = common["source"]
    baseline_cal = evaluate(specs["calibration"], "estimated_bayes", params)["summary"]
    assert strip_timing(baseline_cal) == strip_timing(common["calibration_baseline"])
    for arm in ARMS:
        if arm in NEURAL:
            continue
        print(f"audit baseline {arm}", flush=True)
        for cohort in ("in_regime", "shifted"):
            expected = common["baselines"][arm][cohort]
            again = evaluate(specs[cohort], arm, params, table=table)
            assert strip_timing(again) == strip_timing(expected)
            for row in expected["rows"]:
                native_calls += native_replay(row, cohort == "shifted")
                native_worlds += 1
    seeds = proto["model_seeds"][:1] if smoke else proto["model_seeds"]
    updates = 4 if smoke else proto["training"]["updates_per_arm"]
    reports, seed_guards = [], {}
    cal_data = tensorize(calibration)
    student_targets = student_calls = 0
    all_summaries = {k: {} for k in ("in_regime", "shifted")}
    for seed in seeds:
        report = json.loads((folder/f"seed{seed}/report.json").read_text())
        assert report["source"] == source and report["protocol_sha256"] == digest(PROTOCOL)
        assert set(report["arms"]) == set(NEURAL)
        reports.append(report)
        seed_guards[str(seed)] = {}
        for arm in NEURAL:
            print(f"audit seed {seed} arm {arm}: reconstruct training", flush=True)
            saved = report["arms"][arm]
            path = folder/f"seed{seed}/{arm}.pt"
            assert digest(path) == saved["checkpoint_sha256"]
            model, training, student = fit_arm(seed, arm, sequences, params, specs["train"], updates)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            for key, tensor in model.state_dict().items():
                assert torch.equal(tensor, checkpoint[key]), (seed, arm, key, "training reconstruction")
            assert strip_timing(training) == strip_timing(saved["training"])
            if student:
                assert student == json.loads((folder/f"seed{seed}/student-data.json").read_text())
                student_events = sum(map(len, student))
                student_calls += student_events-len(student)
                student_targets += 4*student_events
            cfg, trials = choose_calibration(model, arm, cal_data, specs["calibration"], params,
                                              table, baseline_cal, proto)
            assert cfg == saved["calibration"]
            assert strip_timing(trials) == strip_timing(saved["calibration_trials"])
            seed_guards[str(seed)][arm] = {}
            for cohort in ("in_regime", "shifted"):
                expected = saved["evaluation"][cohort]
                again = evaluate(specs[cohort], arm, params, model, cfg, table)
                assert strip_timing(again) == strip_timing(expected), (seed, arm, cohort, "policy replay")
                for row in expected["rows"]:
                    native_calls += native_replay(row, cohort == "shifted")
                    native_worlds += 1
                summary = expected["summary"]
                assert strip_timing(summary) == strip_timing(summarize(expected["rows"]))
                ref = common["baselines"]["estimated_bayes"][cohort]["summary"]
                passed = (summary["residual_faults"] <= ref["residual_faults"] and
                          summary["blocked_actions"] <= ref["blocked_actions"] and
                          summary["trajectory_violations"] <= ref["trajectory_violations"] and
                          summary["cost_per_entity"] < ref["cost_per_entity"])
                seed_guards[str(seed)][arm][cohort] = passed
    for cohort in ("in_regime", "shifted"):
        all_summaries[cohort] = {arm: data[cohort]["summary"] for arm, data in common["baselines"].items()}
        for arm in NEURAL:
            summaries = [r["arms"][arm]["evaluation"][cohort]["summary"] for r in reports]
            numeric = {key: sum(s[key] for s in summaries)/len(summaries) for key in summaries[0]}
            all_summaries[cohort][arm] = {"mean_across_model_seeds": numeric,
                                         "per_seed": {str(s): summary for s, summary in zip(seeds, summaries)},
                                         "paired_uncertainty": paired_interval(reports, common, arm, cohort)}
    guard = {arm: all(seed_guards[str(seed)][arm][cohort] for seed in seeds
                      for cohort in ("in_regime", "shifted")) for arm in NEURAL}
    result = {"study": "DEV-052B", "status": "AUDITED_ADAPTIVE_DEVELOPMENT" if not smoke else "AUDITED_SMOKE_ONLY",
              "source": source, "protocol_sha256": digest(PROTOCOL),
              "all_checkpoint_tensors_reconstructed": True,
              "all_calibration_choices_reconstructed": True,
              "all_policy_actions_receipts_and_verifiers_reproduced": True,
              "primitive_native_replay_worlds": native_worlds, "primitive_native_replay_calls": native_calls,
              "public_only_controller_inputs": True, "split_ids_disjoint": True,
              "review_independence": "Separate process and independently implemented native action accounting; same author, shared generator/model code; not independent third-party scientific review",
              "seeds": seeds, "seed_guards": seed_guards,
              "passed_both_cohort_all_seed_cost_and_failure_guard": guard,
              "summary": all_summaries, "original_compute": common["compute"],
              "student_native_calls": student_calls, "student_target_queries": student_targets,
              "known_bayes_initial_inspection_value": bellman(.5, 0, 4, BASE)[0]-bellman(.5, 0, 0, BASE)[0],
              "audit_seconds": time.monotonic()-started,
              "no_confirmation_or_general_architectural_superiority_claim": True}
    write(folder/"audit.json", result)
    print(json.dumps({"status": result["status"], "guards": guard, "audit_seconds": result["audit_seconds"],
                      "native_worlds": native_worlds, "native_calls": native_calls}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--veritas-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    audit(args.directory, args.veritas_root, args.smoke)
