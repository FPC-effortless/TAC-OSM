#!/usr/bin/env python3
"""Constructed ME111 Android runtime reference vectors from ORIGINAL frozen torch.

NOT real-world data, no fitting, no evaluation labels, no superiority claim.
Use for Android JNI/CPU operator correctness and recurrent-state replay only.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ORIGINAL_CHECKPOINT_SHA = "337b882e9a10631dc85700846ef2b15ca8908699499668186ad3a248dc108449"
MODEL_SHA = "dd65519a9e4f3388966538af53119b56e9292b46a9209d63ad4927b87f0a176b"


def run(source, output):
    import torch
    sys.path[:0] = [str(source / "src"), str(source)]
    from tac_osm.variants.conditional_belief_sensor_world import ConditionalBeliefSensorWorld

    checkpoint = (source / "research/replay-bundles/ME111/conditional_belief"
                  / "seed-111001/final.pt")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != ORIGINAL_CHECKPOINT_SHA:
        raise ValueError("original trained checkpoint identity failure")
    torch.set_num_threads(1)
    model = ConditionalBeliefSensorWorld(torch.ones(6, dtype=torch.float32))
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state_dict.get("seed") != 111001 or state_dict.get("steps") != 1536:
        raise ValueError("wrong original trained checkpoint")
    model.load_state_dict(state_dict["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    if sum(x.numel() for x in model.parameters()) != 104663:
        raise ValueError("original model parameter budget changed")
    owners = torch.tensor([0], dtype=torch.int64)
    state = model.initial_state(owners)
    state = replace(state, last_hour=torch.tensor([99], dtype=torch.int64))
    records = []
    with torch.no_grad():
        for hour, value in [(100, 0.05), (101, 0.10)]:
            features = torch.zeros(1, 1, 13, dtype=torch.float32)
            features[0, 0, 0] = value
            observed = torch.zeros(1, 1, 13, dtype=torch.bool)
            observed[0, 0, 0] = True
            ages = torch.zeros_like(features)
            clock = torch.zeros(1, 1, 6, dtype=torch.float32)
            time = torch.tensor([[hour]], dtype=torch.int64)
            prediction, state = model(features, observed, ages, clock, owners,
                                      time, state=state)
            records.append({
                "owner": 0, "hour": hour, "input_value_first": value,
                "log_location": prediction.log_location[0].tolist(),
                "left_scale": prediction.left_scale[0].tolist(),
                "right_scale": prediction.right_scale[0].tolist(),
                "source_sum": state.source_sum[0].tolist(),
                "source_mass": state.source_mass[0].tolist(),
                "evidence_sum": state.evidence_sum[0].tolist(),
                "memory": state.memory[0].tolist(),
                "count": int(state.count[0, 0]),
                "last_hour": int(state.last_hour[0]),
            })
    fixture = {
        "status": "CONSTRUCTED_SOFTWARE_PARITY_ONLY",
        "real_world_scientific_claim": "NONE",
        "original_scientific_result": "ME111_FULL_GATE_FAIL_MIXED",
        "original_checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA,
        "exported_onnx_sha256": MODEL_SHA,
        "records": records
    }
    if output.exists():
        raise ValueError("refuse to replace existing frozen software fixture")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=2, allow_nan=False, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("ME111_ORIGINAL_TORCH_CONSTRUCTED_ANDROID_REFERENCE=GENERATED")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.source_root.resolve(), args.output.resolve())
