"""Replay saved DEV-008 evaluations from final checkpoints (not training replay).

Fail closed on source/checkpoint hashes, missing conditions, and altered raw rows.
This does NOT establish prospective independence or reproduce training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.development.ced_hardened_router import QueryOnlyCED  # noqa: E402
from run_ced_hardened_dev import evaluate_loop, _hash_weights  # noqa: E402


def validate(output: Path):
    report = json.loads((output / "report.json").read_text())
    if report.get("study") != "DEV-008" or report.get("classification") != "DEVELOPMENT_SYNTHETIC_ONLY":
        raise AssertionError("wrong experiment identity or classification")
    for path, original in report["source_sha256"].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != original:
            raise AssertionError(f"source hash mismatch: {path}")
    if not report["records"]:
        raise AssertionError("missing trained seeds")
    checks = 0
    rows = 0
    for record in report["records"]:
        seed = record["seed"]
        if set(record["arms"]) != {"shared", "sparse"}:
            raise AssertionError("incomplete model arms")
        if record["arms"]["shared"]["initial_weights_sha256"] != (
                record["arms"]["sparse"]["initial_weights_sha256"]):
            raise AssertionError("initial conditions do not match")
        for mode, arm in record["arms"].items():
            filename = arm["checkpoint_file"]
            if Path(filename).name != filename:
                raise AssertionError("checkpoint path escapes archive")
            path = output / filename
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != arm["checkpoint_sha256"]:
                raise AssertionError(f"modified checkpoint {path.name}")
            model = QueryOnlyCED(mode=mode, topk=4)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            if _hash_weights(model) != arm["final_weights_sha256"]:
                raise AssertionError("checkpoint tensors differ from reported model")
            if _hash_weights(model) == arm["initial_weights_sha256"]:
                raise AssertionError("checkpoint identical to initialization")
            cells = arm["measurements"]
            expected = {(report["train_candidate_count"], 0),
                        (report["train_candidate_count"], 4), (32, 0), (32, 4)}
            if {(c["candidates"],c["delay"]) for c in cells} != expected or len(cells) != 4:
                raise AssertionError("missing or duplicate evaluation cell")
            for c in cells:
                if len(c["raw"]) != report["samples_per_cell"]:
                    raise AssertionError("missing evaluation rows")
                regenerated = evaluate_loop(model, seed=seed + 987,
                    n=c["candidates"],delay=c["delay"],
                    samples=report["samples_per_cell"])
                if regenerated != c:
                    raise AssertionError(f"evaluation differs seed={seed} mode={mode} "
                                         f"n={c['candidates']} delay={c['delay']}")
                checks += 1
                rows += len(c["raw"])
    return {"status":"REPLAY_VALIDATED_DEVELOPMENT_ONLY",
            "checkpoints":len(report["records"])*2,"cells":checks,"episodes":rows,
            "warning":"Replays evaluation, not optimization/training."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(validate(args.output), indent=2))
