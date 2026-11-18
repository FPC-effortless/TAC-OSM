#!/usr/bin/env python3
"""Authentic ORIGINAL FIT input (2013) -> original ME111 PyTorch oracle.

No scoring/targets, sealed sources, model updates or model-skill claims.
Attribution: Chen (2017), UCI Beijing Multi-Site Air Quality, CC BY 4.0.
Only 256 original published station-0 rows are extracted into the Android
INSTRUMENTATION TEST APK (never the customer APK).
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

SOURCE_SHA = "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
CHECKPOINT_SHA = "337b882e9a10631dc85700846ef2b15ca8908699499668186ad3a248dc108449"
GRAPH_SHA = "dd65519a9e4f3388966538af53119b56e9292b46a9209d63ad4927b87f0a176b"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(source, csv_output, reference_output):
    if csv_output.exists() or reference_output.exists():
        raise RuntimeError("refuse overwrite original source FIT parity evidence")
    import numpy as np
    import torch
    sys.path[:0] = [str(source/"src"), str(source)]
    from scripts import me100_sensor_data as data
    from tac_osm.variants.conditional_belief_sensor_world import ConditionalBeliefSensorWorld

    if data.SHA != SOURCE_SHA or digest(data.ARCHIVE) != SOURCE_SHA:
        raise RuntimeError("original public source archive missing or changed")
    if data.SITES[0] != "Aotizhongxin":
        raise RuntimeError("station index contract changed")
    d = data.load("fit")
    if d["phase"] != "fit" or int(d["hours"][0, 255]) != 255:
        raise RuntimeError("FIT station source chronology changed")
    with zipfile.ZipFile(data.ARCHIVE) as z:
        exact = [p for p in z.namelist()
                 if p.endswith("PRSA_Data_Aotizhongxin_20130301-20170228.csv")]
        if len(exact) != 1:
            raise RuntimeError("publisher original station source missing/ambiguous")
        with io.TextIOWrapper(z.open(exact[0]), encoding="utf-8-sig") as rows:
            content = [next(rows).rstrip("\n\r")]
            for k in range(256):
                row = next(rows).rstrip("\n\r")
                if "Aotizhongxin" not in row:
                    raise RuntimeError("source station label changed")
                content.append(row)
    cp = (source/"research/replay-bundles/ME111/conditional_belief"
          /"seed-111001/final.pt")
    if digest(cp) != CHECKPOINT_SHA:
        raise RuntimeError("original ME111 checkpoint changed")
    weights = torch.load(cp, map_location="cpu", weights_only=False)
    if weights.get("seed") != 111001 or weights.get("steps") != 1536:
        raise RuntimeError("trained source identity changed")
    torch.set_num_threads(1)
    model = ConditionalBeliefSensorWorld(torch.ones(6, dtype=torch.float32))
    model.load_state_dict(weights["model"], strict=True)
    model.eval().requires_grad_(False)
    if sum(x.numel() for x in model.parameters()) != 104663:
        raise RuntimeError("original trained model size changed")

    state = replace(model.initial_state(torch.tensor([0], dtype=torch.int64)),
                    last_hour=torch.tensor([-1], dtype=torch.int64))
    with torch.inference_mode():
        for t in range(256):
            v = torch.from_numpy(d["values"][0, t].reshape(1, 1, 13).copy())
            obs = torch.from_numpy(d["observed"][0, t].reshape(1, 1, 13).copy())
            age = torch.from_numpy(d["ages"][0, t].reshape(1, 1, 13).copy())
            clock = torch.from_numpy(d["clock"][0, t].reshape(1, 1, 6).copy())
            owner = torch.tensor([0], dtype=torch.int64)
            hours = torch.tensor([[int(d["hours"][0,t])]], dtype=torch.int64)
            forecast, state = model(v, obs, age, clock, owner, hours, state=state)
    if int(state.last_hour[0]) != 255:
        raise RuntimeError("256 original source event carry failed")
    assert np.all(np.isfinite(forecast.log_location.detach().numpy()))
    reference = dict(
        status="ORIGINAL_AUTHENTIC_FIT_INPUT_256_EVENT_PYTORCH_SOFTWARE_ORACLE",
        training_performed=False,
        original_publisher_license="CC BY 4.0 Chen 2017 UCI",
        original_source_sha256=SOURCE_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,
        onnx_sha256=GRAPH_SHA,
        source_station="Aotizhongxin",
        start_hour=0,
        final_hour=255,
        steps=256,
        log_location=forecast.log_location[0].tolist(),
        left_scale=forecast.left_scale[0].tolist(),
        right_scale=forecast.right_scale[0].tolist(),
        count=int(state.count[0,0]),
        source_sum=state.source_sum[0].tolist(),
        source_mass=state.source_mass[0].tolist(),
        evidence_sum=state.evidence_sum[0].tolist(),
        memory=state.memory[0].tolist(),
        authentic_forecast_accuracy="NOT_EVALUATED",
        original_scientific_gate="FAILED_MIXED_TRADEOFF",
        phone_qualified=False,
    )
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    reference_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.write_text("\n".join(content) + "\n", encoding="utf-8")
    reference_output.write_text(
        json.dumps(reference, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print("ME111_256_AUTHENTIC_FIT_SOURCE_INPUT_ORIGINAL_TORCH_REFERENCE=GENERATED_NOT_SCIENTIFIC_WIN")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--csv-output",type=Path,required=True)
    p.add_argument("--reference-output",type=Path,required=True)
    args=p.parse_args()
    generate(args.source_root.resolve(),
             args.csv_output.resolve(), args.reference_output.resolve())
