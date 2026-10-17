"""Frozen DEV-052B matrix: train -> calibrate -> native end-to-end evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from tac_osm.development.learned_belief_world import (
    ARMS, NEURAL, HIDDEN, NativeWorld, BeliefModel,
    bellman, posterior, collect, estimate, tensorize, fit,
    score_predictions, rollout, summarize,
)

PROTOCOL = ROOT / "research/dev052b/execution-protocol.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def setup(veritas_root, dirty=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    import investigation_world
    vr = Path(veritas_root).resolve()
    assert Path(investigation_world.__file__).resolve().is_relative_to(vr)
    proto = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vr, text=True).strip() == proto["veritas_commit"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=vr, text=True).strip()
    if not dirty:
        assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip(), "commit code before research"
    return proto


def specifications(proto, smoke=False):
    return {name: {**s, "kind": name, "shifted": name == "shifted",
                   "worlds": min(s["worlds"], 4) if smoke else s["worlds"]}
            for name, s in proto["splits"].items()}


def relabel(sequences, params):
    for seq in sequences:
        p = .5
        for row in seq:
            x = row["x"]
            if x[1]:
                p = posterior(p, int(x[4]), params[0])
            elif x[3]:
                p *= 1-params[1][row["attempts"]-1]
            row["teacher"] = bellman(p, row["attempts"], row["senses"], params)[1]


def evaluate(spec, arm, params, model=None, calibration=None, table=None):
    rows = [rollout(spec, i, arm, params, model, calibration, table) for i in range(spec["worlds"])]
    return {"summary": summarize(rows), "rows": rows}


def choose_calibration(model, arm, data, spec, params, table, baseline, proto):
    if arm in ("policy_bc", "policy_dagger"):
        cfg = {"temperature": 1., "risk": 1.}
        result = evaluate(spec, arm, params, model, cfg, table)
        return cfg, [{"configuration": cfg, "summary": result["summary"]}]
    candidates = [(score_predictions(model, data, t)["nll"], t)
                  for t in proto["calibration"]["fault_temperatures"]]
    temperature = min(candidates, key=lambda pair: pair[0])[1]
    trials = []
    for risk in proto["calibration"]["risk_multipliers"]:
        cfg = {"temperature": temperature, "risk": risk}
        result = evaluate(spec, arm, params, model, cfg, table)
        trials.append({"configuration": cfg, "summary": result["summary"]})
    def eligible(trial):
        s = trial["summary"]
        return (s["residual_faults"] <= baseline["residual_faults"] and
                s["blocked_actions"] <= baseline["blocked_actions"] and
                s["trajectory_violations"] <= baseline["trajectory_violations"])
    good = [trial for trial in trials if eligible(trial)]
    chosen = min(good, key=lambda tr: tr["summary"]["cost_per_entity"]) if good else min(
        trials, key=lambda tr: (tr["summary"]["residual_faults"], tr["summary"]["cost_per_entity"]))
    return chosen["configuration"], trials


def fit_arm(seed, arm, initial, params, spec, updates):
    model = BeliefModel(seed, arm)
    initial_writer = {k: v.clone() for k, v in model.writer.state_dict().items()}
    start = time.monotonic()
    curves, student = [], []
    if arm == "policy_dagger":
        first = updates//2
        opt, curve = fit(model, tensorize(initial), params[0], first, seed)
        curves += curve
        student, _ = collect(spec, params, model)
        relabel(student, params)
        _, curve = fit(model, tensorize(initial+student), params[0], updates-first,
                       seed, opt, offset=first)
        curves += curve
    else:
        _, curves = fit(model, tensorize(initial), params[0], updates, seed)
    unchanged = all(torch.equal(v, model.writer.state_dict()[k]) for k, v in initial_writer.items())
    assert unchanged == (arm in ("frozen_writer", "last_event"))
    model.eval()
    return model, {"training_seconds": time.monotonic()-start, "curve": curves,
                   "updates": updates, "student_entities": len(student),
                   "student_events": sum(map(len, student)),
                   "writer_unchanged": unchanged,
                   "parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
                   "persistent_neural_state_bytes": 2*HIDDEN*4,
                   "public_counter_state": "two PublicState records, at most four read pairs per entity; Python allocator overhead excluded"}, student


def run(output, veritas_root, smoke=False):
    proto = setup(veritas_root, dirty=smoke)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    specs = specifications(proto, smoke)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    start = time.monotonic()
    split_ids = {name: [NativeWorld(s["seed"], i, s["shifted"]).episode.episode_id
                        for i in range(s["worlds"])] for name, s in specs.items()}
    all_ids = [identity for values in split_ids.values() for identity in values]
    assert len(all_ids) == len(set(all_ids))
    write(out/"split-ledger.json", split_ids)
    sequences, sensor = collect(specs["train"])
    params, table = estimate(sequences, sensor)
    relabel(sequences, params)
    validation, _ = collect(specs["validation"], params)
    calibration, _ = collect(specs["calibration"], params)
    data = {"training": sequences, "sensor_correctness": sensor, "validation": validation,
            "calibration": calibration, "estimated_parameters": params, "event_table": table}
    write(out/"training-data.json", data)
    common = {"study": "DEV-052B", "smoke": smoke, "source": source,
              "protocol_sha256": digest(PROTOCOL), "veritas_commit": proto["veritas_commit"],
              "training_data_sha256": digest(out/"training-data.json"),
              "split_ledger_sha256": digest(out/"split-ledger.json"),
              "python": platform.python_version(), "torch": torch.__version__,
              "estimated_parameters": params, "table_entries": len(table),
              "baselines": {}, "calibration_baseline": {}, "seeds": []}
    # All non-neural controllers evaluated once; model seeds reuse these worlds.
    for arm in ARMS:
        if arm in NEURAL:
            continue
        print(f"baseline {arm}", flush=True)
        common["baselines"][arm] = {name: evaluate(specs[name], arm, params, table=table)
                                    for name in ("in_regime", "shifted")}
    common["calibration_baseline"] = evaluate(specs["calibration"], "estimated_bayes", params)["summary"]
    write(out/"common.json", common)
    val_tensor, cal_tensor = tensorize(validation), tensorize(calibration)
    seeds = proto["model_seeds"][:1] if smoke else proto["model_seeds"]
    updates = 4 if smoke else proto["training"]["updates_per_arm"]
    for seed in seeds:
        seed_dir = out/f"seed{seed}"
        seed_dir.mkdir()
        report = {"seed": seed, "source": source, "smoke": smoke,
                  "protocol_sha256": digest(PROTOCOL), "arms": {}}
        for arm in NEURAL:
            print(f"seed {seed} arm {arm}: training", flush=True)
            model, training, student = fit_arm(seed, arm, sequences, params, specs["train"], updates)
            checkpoint = seed_dir/f"{arm}.pt"
            torch.save(model.state_dict(), checkpoint)
            if student:
                write(seed_dir/"student-data.json", student)
            cfg, trials = choose_calibration(model, arm, cal_tensor, specs["calibration"], params,
                                              table, common["calibration_baseline"], proto)
            # Persist the choice before opening either held-out cohort for this arm.
            write(seed_dir/f"{arm}-calibration.json", {"configuration": cfg, "trials": trials})
            print(f"seed {seed} arm {arm}: evaluate {cfg}", flush=True)
            report["arms"][arm] = {"checkpoint_sha256": digest(checkpoint),
                                   "training": training, "calibration": cfg,
                                   "calibration_trials": trials,
                                   "validation": score_predictions(model, val_tensor),
                                   "evaluation": {name: evaluate(specs[name], arm, params, model, cfg, table)
                                                  for name in ("in_regime", "shifted")}}
            write(seed_dir/"report.json", report)
        common["seeds"].append(seed)
        write(out/"common.json", common)
    events = sum(map(len, sequences))
    common["compute"] = {"total_seconds": time.monotonic()-start,
                          "initial_collection_worlds": specs["train"]["worlds"],
                          "training_events": events,
                          "training_target_oracle_queries": 4*events+len(sensor),
                          "common_collection_native_actions": 20*sum(specs[k]["worlds"] for k in ("train", "validation", "calibration")),
                          "optimizer_updates": len(seeds)*len(NEURAL)*updates,
                          "model_fits": len(seeds)*len(NEURAL),
                          "note": "Exact native and student target counts recomputed by auditor; wall clock includes calibration and evaluation. Repeated seed worlds are not independent samples."}
    common["status"] = "COMPLETED_UNAUDITED_DEVELOPMENT" if not smoke else "SMOKE_ONLY"
    write(out/"common.json", common)
    manifest = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write(out/"manifest.json", manifest)
    print(json.dumps({"status": common["status"], "seeds": seeds, "compute": common["compute"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--veritas-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output, args.veritas_root, args.smoke)
