"""DEV-044B0 hash-group, checkpoint prediction and physical replay audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from run_writer_reader_factorial_dev044b import build,group_hash,ARMS,PROTOCOL,study_cohort,training_cases,predict
from run_frozen_policy_dev045 import rows_for,threshold_for,digest
from tac_osm.development.schema_maintenance import execute


def audit(folder):
    folder = Path(folder)
    r = json.loads((folder/"report.json").read_text())
    assert r["study"]=="DEV-044B0" and not r["smoke"] and r["protocol_sha256"]==digest(PROTOCOL)
    assert set(r["arms"])==set(ARMS)
    train = [c.fingerprint for step in range(108) for c in training_cases(r["seed"],step)]
    summary = {}
    for arm,data in r["arms"].items():
        model = build(arm,r["seed"])
        assert {k:group_hash(model,v) for k,v in (("writer",False),("reader",True))}==data["before_hashes"]
        ck = folder/(arm+".pt")
        assert digest(ck)==data["checkpoint_sha256"]
        model.load_state_dict(torch.load(ck,weights_only=True,map_location="cpu"))
        assert {k:group_hash(model,v) for k,v in (("writer",False),("reader",True))}==data["after_hashes"]
        for key,expected in (("writer",arm.startswith("writer_train_")),("reader",arm.endswith("_reader_train"))):
            assert (data["before_hashes"][key]!=data["after_hashes"][key])==expected==data["gradient_present"][key]
        assert train==data["train_fingerprints"] and data["curve"][-1]["step"]==108
        split_fps = [set(train)]
        for split in ("calibration","evaluation"):
            cases,conditions = study_cohort(r["seed"],split)
            split_fps.append({c.fingerprint for c in cases})
            p = predict(model,cases)
            rows = rows_for(cases,conditions,p)
            if split=="calibration":
                assert rows==data["calibration_rows"] and threshold_for(rows)==data["threshold"]
                continue
            total = dict(failures=0,services=0,unnecessary_services=0,total_cost=0)
            for c,row,saved in zip(cases,rows,data["evaluation"]["rows"]):
                assert all(saved[k]==v for k,v in row.items())
                services = tuple(e for e,p in enumerate(row["probabilities"]) if p[0]-p[1]>data["threshold"])
                replay = execute(c.world,c.horizon,services,0)
                assert replay==saved["physical"]
                for k in total:
                    total[k] += len(services) if k=="services" else replay[k]
            assert all(data["evaluation"]["physical"][k]==v for k,v in total.items())
            summary[arm] = dict(**total,cost_per_entity=total["total_cost"]/(8*len(cases)))
        assert all(not a&b for i,a in enumerate(split_fps) for b in split_fps[i+1:])
    return dict(seed=r["seed"],source=r["source"],report_sha256=digest(folder/"report.json"),arms=summary)


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders",nargs=3)
    ap.add_argument("--output",required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    results = [audit(f) for f in a.folders]
    assert {r["seed"] for r in results}=={4411,4412,4413} and len({r["source"] for r in results})==1
    Path(a.output).write_text(json.dumps(dict(study="DEV-044B0",status="FACTORIAL_CHECKPOINT_AND_PHYSICAL_REPLAY_PASSED_DEVELOPMENT_ONLY",per_seed=results,confirmatory_claim=False),indent=2)+"\n")
    print("DEV-044B0 separate writer/reader groups, forecasts and physical replay passed")
