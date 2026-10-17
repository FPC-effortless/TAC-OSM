"""Independent reconstruction and physical replay for DEV-046B."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from run_joint_training_dev046 import build, writer_hash, predict, study_cohort, ARMS, PROTOCOL, training_cases
from run_frozen_policy_dev045 import digest, rows_for, threshold_for
from tac_osm.development.schema_maintenance import execute


def audit(folder):
    folder = Path(folder)
    r = json.loads((folder/"report.json").read_text())
    assert not r["smoke"] and r["study"]=="DEV-046B"
    assert r["protocol_sha256"]==digest(PROTOCOL)
    assert set(r["arms"])==set(ARMS)
    cohorts = {s:study_cohort(r["seed"],s) for s in ("calibration","evaluation")}
    expected_train = [c.fingerprint for step in range(72) for c in training_cases(r["seed"],step,False)]
    summary = {}
    for arm,data in r["arms"].items():
        model = build(arm,r["seed"])
        assert writer_hash(model)==data["initial_writer_hash"]
        ck = folder/(arm+".pt")
        assert digest(ck)==data["checkpoint_sha256"]
        model.load_state_dict(torch.load(ck,weights_only=True,map_location="cpu"))
        model.eval()
        assert writer_hash(model)==data["final_writer_hash"]
        assert (data["initial_writer_hash"]==data["final_writer_hash"])==arm.endswith("_frozen")
        assert data["train_fingerprints"]==expected_train and len(set(expected_train))==144
        for s,(cases,conditions) in cohorts.items():
            assert r["split_fingerprints"][s]==[c.fingerprint for c in cases]
            assert not set(expected_train)&set(r["split_fingerprints"][s])
            probabilities,_ = predict(model,cases)
            rows = rows_for(cases,conditions,probabilities)
            if s=="calibration":
                assert rows==data["calibration_rows"]
                assert threshold_for(rows)==data["threshold"]
                continue
            totals = dict(failures=0,services=0,unnecessary_services=0,total_cost=0.)
            for case,row,saved in zip(cases,rows,data["evaluation"]["rows"]):
                assert all(saved[k]==v for k,v in row.items())
                services = tuple(e for e,p in enumerate(row["probabilities"]) if p[0]-p[1]>data["threshold"])
                physical = execute(case.world,case.horizon,services,0)
                assert physical==saved["physical"] and list(services)==saved["services"]
                for k in totals:
                    totals[k] += len(services) if k=="services" else physical[k]
            for k,v in totals.items():
                assert data["evaluation"]["physical"][k]==v
            summary[arm] = dict(**totals,cost_per_entity=totals["total_cost"]/(8*len(cases)))
    return dict(seed=r["seed"],source=r["source"],report_sha256=digest(folder/"report.json"),arms=summary)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders",nargs=3)
    ap.add_argument("--output",required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    results = [audit(f) for f in a.folders]
    assert {r["seed"] for r in results}=={4601,4602,4603}
    assert len({r["source"] for r in results})==1
    Path(a.output).write_text(json.dumps(dict(study="DEV-046B",status="CHECKPOINT_AND_PHYSICAL_REPLAY_PASSED_DEVELOPMENT_ONLY",
        per_seed=results,confirmatory_claim=False),indent=2)+"\n")
    print("DEV-046B checkpoint and physical replay passed for all three seeds")
