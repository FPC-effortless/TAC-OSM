"""Separate DEV-045B checkpoint inference, calibration and physical replay audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
import torch
from tac_osm.development.frozen_action_heads import FrozenActionHead
from tac_osm.development.schema_maintenance import execute
from tac_osm.development.noisy_memory import parameter_hash
from run_frozen_policy_dev045 import cohort, gather, digest, threshold_for, rows_for, PROTOCOL, HEADS
from run_reservoir_scaling_dev044 import build


def audit(folder, parent_root):
    folder = Path(folder)
    r = json.loads((folder/"report.json").read_text())
    assert r["study"] == "DEV-045B" and not r["smoke"]
    assert r["protocol_sha256"] == digest(PROTOCOL)
    cs = {s:cohort(r["seed"],s,2 if s=="evaluation" else 1) for s in ("fit","calibration","evaluation")}
    for s,(cases,_) in cs.items():
        assert r["split_fingerprints"][s] == [c.fingerprint for c in cases]
    sets = [set(v) for v in r["split_fingerprints"].values()]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    summary = {}
    for memory,data in r["memories"].items():
        ck = Path(parent_root)/str(r["seed"]-100)/(data["parent_arm"]+".pt")
        assert digest(ck) == data["parent_checkpoint_sha256"]
        model = build(data["parent_arm"],r["seed"]-100)
        model.load_state_dict(torch.load(ck,weights_only=True,map_location="cpu")["model"])
        model.eval().requires_grad_(False)
        assert parameter_hash(model) == data["frozen_model_hash"]
        feature_file = folder/(memory+"-public-features.pt")
        assert digest(feature_file) == data["feature_file_sha256"]
        saved = torch.load(feature_file,weights_only=True,map_location="cpu")
        for split,(cases,_) in cs.items():
            features,original,_ = gather(model,cases)
            assert torch.equal(features,saved[split]["features"])
            assert torch.equal(original,saved[split]["original"])
        summary[memory] = {}
        for name,head_data in data["heads"].items():
            predictions = {}
            if name in HEADS:
                ck = folder/(memory+"-"+name+".pt")
                assert digest(ck) == head_data["checkpoint_sha256"]
                kind = "original" if name=="probability_joint" else "features"
                head = FrozenActionHead(saved["fit"][kind].shape[-1])
                head.load_state_dict(torch.load(ck,weights_only=True,map_location="cpu"))
                with torch.no_grad():
                    predictions = {s:head(saved[s][kind]).sigmoid() for s in ("calibration","evaluation")}
                assert head_data["training"]["curve"][-1]["step"] == 240
            else:
                predictions = {s:saved[s]["original"] for s in ("calibration","evaluation")}
            cal_rows = rows_for(*cs["calibration"], predictions["calibration"])
            assert threshold_for(cal_rows) == head_data["threshold"]
            if name in HEADS:
                assert cal_rows == head_data["calibration_rows"]
            rows = head_data["evaluation"]["rows"]
            assert len(rows) == len(cs["evaluation"][0]) == 144
            costs = dict(failures=0,services=0,unnecessary_services=0,total_cost=0.)
            for i,(case,condition) in enumerate(zip(*cs["evaluation"])):
                row = rows[i]
                assert all(row[k] == v for k,v in condition.items())
                p = predictions["evaluation"][i].tolist()
                assert row["probabilities"] == p
                services = tuple(e for e,q in enumerate(p) if q[0]-q[1] > head_data["threshold"])
                physical = execute(case.world,case.horizon,services,0)
                assert list(services) == row["services"] and row["physical"] == physical
                for k in costs:
                    costs[k] += len(services) if k=="services" else physical[k]
            for k,v in costs.items():
                assert head_data["evaluation"]["physical"][k] == v
            summary[memory][name] = dict(**costs,cost_per_entity=costs["total_cost"]/(8*len(rows)))
    return dict(seed=r["seed"],source=r["source"],report_sha256=digest(folder/"report.json"),memories=summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs=3)
    ap.add_argument("--parent-root",required=True)
    ap.add_argument("--output",required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    results = [audit(f,a.parent_root) for f in a.folders]
    assert {r["seed"] for r in results} == {4501,4502,4503}
    assert len({r["source"] for r in results}) == 1
    Path(a.output).write_text(json.dumps(dict(study="DEV-045B",status="CHECKPOINT_AND_PHYSICAL_REPLAY_PASSED_DEVELOPMENT_ONLY",
        per_seed=results,confirmatory_claim=False),indent=2)+"\n")
    print("DEV-045B checkpoint, calibration, and physical replay passed for all three seeds")
