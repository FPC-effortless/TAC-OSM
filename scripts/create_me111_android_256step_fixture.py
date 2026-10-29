#!/usr/bin/env python3
"""Generate unchanged ORIGINAL frozen ME111 2x256-step Android JNI oracle fixtures.

Constructed observation streams only. Neither forecasts nor memory scores are
evidence of authentic real-world utility or long-history retrieval advantage.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from verify_me111_256step_cpu import event, ORIGINAL_SHA, ONNX_SHA, digest

CHECK_STEPS = frozenset((0, 1, 7, 15, 31, 63, 95, 127, 159, 191, 223, 255))


def generate(source: Path, output: Path):
    import numpy as np
    import torch

    original = (source / "research/replay-bundles/ME111/conditional_belief"
                / "seed-111001/final.pt")
    if digest(original) != ORIGINAL_SHA:
        raise ValueError("original trained checkpoint bytes changed")
    sys.path[:0] = [str(source / "src"), str(source)]
    from tac_osm.variants.conditional_belief_sensor_world import ConditionalBeliefSensorWorld

    saved = torch.load(original, map_location="cpu", weights_only=False)
    if saved.get("seed") != 111001 or saved.get("steps") != 1536:
        raise ValueError("original model training provenance changed")
    torch.set_num_threads(1)
    model = ConditionalBeliefSensorWorld(torch.ones(6, dtype=torch.float32))
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != 104663:
        raise ValueError("frozen trained parameters changed")

    streams = []
    with torch.inference_mode():
        for owner in (0, 5):
            owners = torch.tensor([owner], dtype=torch.int64)
            state = replace(model.initial_state(owners),
                            last_hour=torch.tensor([99], dtype=torch.int64))
            hour = 99
            records = []
            observed_updates = 0
            for step in range(256):
                hour += 3 if step % 11 == 0 else 1
                x = event(np, owner, step, hour)
                observed_updates += int(x["observed"].any())
                prediction, state = model(
                    torch.from_numpy(x["values"]),
                    torch.from_numpy(x["observed"]),
                    torch.from_numpy(x["ages"]),
                    torch.from_numpy(x["clock"]),
                    torch.from_numpy(x["owners"]),
                    torch.from_numpy(x["hours"]),
                    state=state
                )
                rec = {
                    "step": step,
                    "hour": hour,
                    "values": x["values"][0, 0].tolist(),
                    "observed": x["observed"][0, 0].tolist(),
                    "ages": x["ages"][0, 0].tolist(),
                    "clock": x["clock"][0, 0].tolist()
                }
                if step in CHECK_STEPS:
                    rec["reference"] = {
                        "log_location": prediction.log_location[0].tolist(),
                        "left_scale": prediction.left_scale[0].tolist(),
                        "right_scale": prediction.right_scale[0].tolist(),
                        "source_sum": state.source_sum[0].tolist(),
                        "source_mass": state.source_mass[0].tolist(),
                        "evidence_sum": state.evidence_sum[0].tolist(),
                        "memory": state.memory[0].tolist(),
                        "count": int(state.count[0, 0]),
                        "last_hour": int(state.last_hour[0]),
                    }
                records.append(rec)
            if int(state.count[0, 0]) != observed_updates:
                raise AssertionError("source state count unexpectedly differs")
            streams.append({
                "owner": owner, "count": observed_updates,
                "final_hour": hour, "records": records
            })
    result = {
        "status": "ORIGINAL_FROZEN_PYTORCH_256_STEP_CONSTRUCTED_ANDROID_FIXTURE",
        "not_authentic_source_evaluation": True,
        "original_scientific_gate": "ME111_FAILED_MIXED",
        "original_checkpoint_sha256": ORIGINAL_SHA,
        "onnx_model_sha256": ONNX_SHA,
        "per_owner_steps": 256,
        "total_steps": 512,
        "expected_steps_with_model_outputs": sorted(CHECK_STEPS),
        "streams": streams,
    }
    if output.exists():
        raise ValueError("refusing to overwrite pinned fixture")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, allow_nan=False,
                                 separators=(",", ":")) + "\n", encoding="utf-8")
    print("ME111_ANDROID_ORIGINAL_PYTORCH_2_BY_256_CONSTRUCTED_FIXTURE=GENERATED")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    generate(args.source_root.resolve(), args.output.resolve())
