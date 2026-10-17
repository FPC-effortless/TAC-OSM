"""Additional DEV-044A audit: reproduce calibrated heads from checkpoint tensors.

The original published audit verifies supplied probabilities and simulator
costs. This extension tests the checkpoint-to-probability link as well.
Batched inference permits 1e-5 floating-point drift; boundary-sensitive actions
are checked again using the original paired-batch shape.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from run_reservoir_scaling_dev044 import build, cases_for, sha, select_threshold, physical, ARM_NAMES
from run_learned_schema_dev import public_stream_tensors, public_queries


@torch.no_grad()
def predict_rows(model,split,repetitions):
    items = [(c,cond) for pair,conditions in cases_for(split,repetitions) for c,cond in zip(pair,conditions)]
    rows = []
    for offset in range(0,len(items),16):
        batch = items[offset:offset+16]
        cases = [c for c,_ in batch]
        ev = public_stream_tensors([c.observations for c in cases])
        q = public_queries([c.tick for c in cases],[c.horizon for c in cases])
        p = model.decode(model.encode(*ev),q).softmax(-1)[...,1].tolist()
        for (c,cond),prob in zip(batch,p):
            rows.append(dict(**cond,probabilities=prob,labels=[[c.wait_targets[e],c.service_targets[e],c.current_targets[e]] for e in range(8)]))
    return rows,items


@torch.no_grad()
def original_pair(model,items,offset):
    cases = [c for c,_ in items[offset:offset+2]]
    p = model.decode(model.encode(*public_stream_tensors([c.observations for c in cases])),
                     public_queries([c.tick for c in cases],[c.horizon for c in cases])).softmax(-1)[...,1].tolist()
    return p


def audit(folder):
    folder = Path(folder)
    r = json.loads((folder/"report.json").read_text())
    assert r["study"]=="DEV-044A" and not r["smoke"]
    result = {}
    for arm in ARM_NAMES:
        data = r["arms"][arm]
        ck = folder/(arm+".pt")
        assert sha(ck)==data["checkpoint_sha256"]
        model = build(arm,r["seed"])
        model.load_state_dict(torch.load(ck,weights_only=True,map_location="cpu")["model"])
        model.eval()
        calibration,items = predict_rows(model,"calibration",2)
        threshold = select_threshold(calibration)
        if threshold!=data["threshold"] or physical(calibration,threshold)!=data["calibration_cost"]:
            for offset in range(0,len(calibration),2):
                p = original_pair(model,items,offset)
                for row,prob in zip(calibration[offset:offset+2],p):
                    row["probabilities"] = prob
            threshold = select_threshold(calibration)
        assert threshold==data["threshold"] and physical(calibration,threshold)==data["calibration_cost"]
        evaluation,items = predict_rows(model,"evaluation",8)
        saved = data["results"]["rows"]
        assert len(evaluation)==len(saved)==576
        repairs = 0
        for offset in range(0,len(evaluation),2):
            pp = torch.tensor([row["probabilities"] for row in evaluation[offset:offset+2]])
            sp = torch.tensor([row["probabilities"] for row in saved[offset:offset+2]])
            assert torch.allclose(pp,sp,atol=1e-5,rtol=1e-5)
            if not torch.equal(pp[...,0]-pp[...,1]>threshold,sp[...,0]-sp[...,1]>threshold):
                exact = original_pair(model,items,offset)
                for row,prob in zip(evaluation[offset:offset+2],exact):
                    row["probabilities"] = prob
                repairs += 1
        for row,original in zip(evaluation,saved):
            assert row["fingerprint"]==original["fingerprint"] and row["labels"]==original["labels"]
        assert physical(evaluation,threshold)==data["results"]["evaluation"]["physical"]
        result[arm] = dict(checkpoint_sha256=sha(ck),threshold=threshold,
                           replayed_cost=physical(evaluation,threshold),boundary_pair_rechecks=repairs)
    return dict(seed=r["seed"],source=r["source"],report_sha256=sha(folder/"report.json"),arms=result)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders",nargs=3)
    ap.add_argument("--output",required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    results = [audit(folder) for folder in a.folders]
    assert {r["seed"] for r in results}=={4401,4402,4403}
    assert len({r["source"] for r in results})==1
    Path(a.output).write_text(json.dumps(dict(study="DEV-044A",status="ADDITIONAL_CHECKPOINT_REPLAY_PASSED",
        per_seed=results,confirmatory_claim=False),indent=2)+"\n")
    print("DEV-044A checkpoint-to-prediction and calibration replay passed for all 27 checkpoints")
