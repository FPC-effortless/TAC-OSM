"""Run a saved DEV-052B checkpoint end to end against native Veritas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import torch
from tac_osm.development.learned_belief_world import BeliefModel, NEURAL, rollout
from run_learned_belief_dev052b import setup, specifications, digest


def run(evidence, veritas_root, seed, arm, index, shifted, output=None):
    proto = setup(veritas_root)
    folder = Path(evidence)
    common = json.loads((folder/"common.json").read_text())
    report = json.loads((folder/f"seed{seed}/report.json").read_text())
    data = json.loads((folder/"training-data.json").read_text())
    saved = report["arms"][arm]
    checkpoint = folder/f"seed{seed}/{arm}.pt"
    assert digest(checkpoint) == saved["checkpoint_sha256"]
    model = BeliefModel(seed, arm)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    spec = specifications(proto)["shifted" if shifted else "in_regime"]
    assert 0 <= index < spec["worlds"]
    row = rollout(spec, index, arm, data["estimated_parameters"], model,
                  saved["calibration"], data["event_table"])
    print(f"Checkpoint: {arm}, model seed {seed}, source {common['source']}")
    print(f"World: {row['world']} ({'shifted' if shifted else 'in regime'})")
    cursor = 0
    for decision in row["decisions"]:
        print(f"entity={decision['entity']} P(fault)={decision['p_fault']:.4f} "
              f"repair risks={[round(p, 4) for p in decision['repair_risk']]} -> {decision['command']}")
        count = {"done": 0, "inspect": 1, "service": 2}[decision["command"]]
        for call in row["calls"][cursor:cursor+count]:
            print(f"  native {call['name']} {call['parameters']} -> {call['receipt']}")
        cursor += count
    print(json.dumps({key: row[key] for key in ("cost", "failures", "complete", "blocked", "violations", "verification")}, indent=2))
    if output:
        Path(output).write_text(json.dumps(row, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--veritas-root", required=True)
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--arm", choices=NEURAL, default="joint")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--shifted", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    run(args.evidence, args.veritas_root, args.seed, args.arm, args.index, args.shifted, args.output)
