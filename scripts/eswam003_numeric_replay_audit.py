"""Read-only original ESWAM003 cross-runtime score-replay audit. NO training.

Do not change original scores, checkpoint bytes, original numerical tolerance
or announce a strict replay PASS when even one field breaches it.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from scripts.eswam003_train_real import (
    ROOT, OUT, SEEDS, UPDATES, PARAMETERS, STUDY,
    load_original, evaluation_origins, score, sha, preflight, write,
)
from tac_osm.variants.selective_exchange_world import SelectiveExchangeWorldModel


SCALAR_FIELDS = (
    "physical_mae_model",
    "physical_mae_persistence",
    "physical_mae_fit_servo_lag1",
    "normalized_laplace_nll",
    "movement_probability",
    "movement_brier",
    "next_recorded_command_imitation_l1",
    "repeat_last_recorded_command_l1",
    "physical_movement_mean",
)
VECTOR_FIELDS = (
    "measured_next_state_normalized",
    "forecast_mean_normalized",
    "forecast_scale_normalized",
)
RTOL = 1e-6
ATOL = 1e-6


def audit(cache):
    preflight()
    torch.set_num_threads(1)
    _, _, report, identified, source = load_original(cache)
    frozen_source = json.loads((ROOT / "SOURCE_QUALIFICATION.json").read_text())
    if frozen_source != source:
        raise RuntimeError("original full-source identity changed")
    queries, _ = evaluation_origins(report, identified)
    all_differences = []
    for seed in SEEDS:
        folder = OUT / f"seed-{seed}"
        registered = json.loads((folder / "RESULTS.json").read_text())
        if sha(folder / "final.pt") != registered["original_checkpoint_sha256"]:
            raise RuntimeError("original Git model checkpoint SHA mismatch")
        original = json.loads((folder / "ORIGINAL_REAL_PREDICTIONS.json").read_text())
        if sha(folder / "ORIGINAL_REAL_PREDICTIONS.json") != registered["original_prediction_sha256"]:
            raise RuntimeError("original predictions changed")
        saved = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
        model = SelectiveExchangeWorldModel(14, 14)
        model.load_state_dict(saved["weights"], strict=True)
        if saved["seed"] != seed or saved["original_live_training_steps"] != UPDATES:
            raise RuntimeError("original fitted model identity mismatch")
        regenerated, _ = score(model, report, identified, queries)
        if len(regenerated) != len(original):
            raise RuntimeError("case count changed")
        values = defaultdict(lambda: {"total": 0, "violations": 0, "max_abs": 0.0,
                                      "worst_id": None, "original": None, "replayed": None})
        for a, b in zip(original, regenerated):
            identity = [a["task"], a["episode_index"], a["frame_index"]]
            if identity != [b["task"], b["episode_index"], b["frame_index"]]:
                raise RuntimeError("mismatched original source case identity")
            if a["motion_rich_threshold_0_025"] != b["motion_rich_threshold_0_025"]:
                raise RuntimeError("source future-motion label changed")
            for field in SCALAR_FIELDS + VECTOR_FIELDS:
                aa, bb = a[field], b[field]
                if field in VECTOR_FIELDS:
                    pairs = zip(aa, bb)
                else:
                    pairs = [(aa, bb)]
                for k, (x, y) in enumerate(pairs):
                    if not (math.isfinite(x) and math.isfinite(y)):
                        raise RuntimeError("nonfinite original or replay observation")
                    record = values[field]
                    record["total"] += 1
                    diff = abs(x - y)
                    if diff > record["max_abs"]:
                        record["max_abs"] = diff
                        record["worst_id"] = identity + ([k] if field in VECTOR_FIELDS else [])
                        record["original"] = x
                        record["replayed"] = y
                    if not math.isclose(x, y, rel_tol=RTOL, abs_tol=ATOL):
                        record["violations"] += 1
        all_differences.append({"seed":seed,"original_real_cases":len(original),
                                "original_checkpoint_sha256":registered["original_checkpoint_sha256"],
                                "fields":dict(values)})
    total_violations=sum(x["violations"] for r in all_differences for x in r["fields"].values())
    audit_result={"study":STUDY,"original_fit_modified":False,"original_checkpoints_modified":False,
                  "registered_rtol":RTOL,"registered_atol":ATOL,
                  "strict_original_replay":"PASS" if total_violations==0 else "FAIL",
                  "total_violation_count":total_violations,
                  "only_authentic_prior_exposed_DEV":True,
                  "all_seeds":all_differences}
    dest=OUT/"CROSS_CPU_REPLAY_AUDIT.json"
    write(dest, audit_result)
    print("ESWAM003_INDEPENDENT_NUMERIC_AUDIT="+json.dumps({
        "strict_original_replay":audit_result["strict_original_replay"],
        "total_violations":total_violations,
        "seed_failures":{r["seed"]:{name:rec["violations"]
                for name,rec in r["fields"].items() if rec["violations"]}
                 for r in all_differences}},sort_keys=True),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    args=parser.parse_args()
    audit(args.cache)
