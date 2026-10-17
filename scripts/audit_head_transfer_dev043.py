"""Separate-process DEV-043 audit: archived checkpoint, prediction, labels and physical cost."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from tac_osm.development.schema_maintenance import execute,exact_public_targets
from tac_osm.development.order_causal_cases import make_order_pair,last_observations
from run_head_transfer_dev043 import (ARMS,build,capture_split,calibrated,
    metrics,as_rows,sha)


def physical_oracle(rows,cases,threshold):
    assert len(rows)==len(cases)
    total=dict(failures=0,unnecessary_services=0,services=0,total_cost=0)
    for row,case in zip(rows,cases):
        assert row["fingerprint"]==case.fingerprint
        assert row["labels"]==[[case.wait_targets[e],case.service_targets[e],case.current_targets[e]] for e in range(8)]
        assert exact_public_targets(case.observations,case.tick,case.horizon)==case.wait_targets
        p=row["probabilities"]
        assert len(p)==8 and all(len(v)==3 for v in p)
        chosen=tuple(e for e in range(8) if p[e][0]-p[e][1]>threshold)
        verified=execute(case.world,case.horizon,chosen,0)
        for k in ("failures","unnecessary_services","total_cost"):
            total[k]+=verified[k]
        total["services"]+=len(chosen)
    total["cost_per_entity"]=total["total_cost"]/(len(cases)*8)
    return total


def audit(seed,folder,checkpoints):
    folder=Path(folder)
    report=json.loads((folder/"report.json").read_text())
    assert report["seed"]==seed and report["study"]=="DEV-043"
    assert report["status"]=="ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY"
    assert not report["smoke"]
    cases=[case for delay in (32,128,512) for deadline in ("before","at","after")
           for rep in range(16) for case in make_order_pair(
              430000+rep,512,delay,deadline,domain="dev043-order-eval-v1")]
    assert len(cases)==288
    fps=[c.fingerprint for c in cases]
    assert len(set(fps))==288 and fps==report["evaluation_fingerprints"]
    forbidden={c.fingerprint for d in (32,128,512) for deadline in ("before","at","after")
               for rep in range(16) for c in make_order_pair(420000+rep,512,d,deadline)}
    assert not forbidden&set(fps)
    for i in range(0,288,2):
        a,b=cases[i:i+2]
        assert last_observations(a.observations)==last_observations(b.observations)
        if a.wait_targets[a.target_entity]!=b.wait_targets[b.target_entity]:
            assert (a.wait_targets[a.target_entity],b.wait_targets[b.target_entity])==(1,0)
    details={}
    for arm in ARMS:
        src_dir=Path(checkpoints)/f"DEV041-RAW-seed{seed}-development"
        ckpath=src_dir/(arm+".pt")
        known=json.loads((src_dir/"report.json").read_text())
        assert sha(ckpath)==known["arms"][arm]["checkpoint_sha256"]==report["arms"][arm]["checkpoint_sha256"]
        blob=torch.load(folder/(arm+"-heads.pt"),map_location="cpu",weights_only=True)
        assert sha(folder/(arm+"-heads.pt"))==report["arms"][arm]["head_evidence_sha256"]
        ck=torch.load(ckpath,map_location="cpu",weights_only=True)
        for key,tensor in ck["model"].items():
            assert torch.equal(tensor,blob["source_checkpoint"][key])
        model=build(arm,seed);model.load_state_dict(ck["model"],strict=True)
        frozen={k:v.clone() for k,v in model.state_dict().items()}
        data=capture_split(model,"evaluation")
        assert [c["fingerprint"] for c in data["conditions"]]==fps
        variants={}
        for name,info in report["arms"][arm]["variants"].items():
            if name.startswith("refit_"):
                head=copy.deepcopy(model.head)
                head.load_state_dict(blob["head_refits"][name])
                with torch.no_grad(): logits=head(data["x"])
            else:
                logits=data["logits"]
            params=blob["calibration"][name]["params"]
            threshold=blob["calibration"][name]["threshold"]
            assert params==info["calibration_params"] and threshold==info["threshold"]
            with torch.no_grad():
                predictions=calibrated(logits,params)
            rows=as_rows(data,predictions)
            for row,known_p in zip(rows,info["probabilities"]):
                assert all(abs(x-y)<1e-7 for ps,pd in zip(row["probabilities"],known_p)
                           for x,y in zip(ps,pd))
            check=physical_oracle(rows,cases,threshold)
            for k,v in check.items():
                assert abs(v-info["physical"][k])<1e-9
            expected=metrics(rows)
            for k,v in expected.items():
                assert abs(v-info["metrics"][k])<1e-9
            variants[name]=dict(rank=info["metrics"]["pairwise_ranking"],
                                 hard=info["metrics"]["hard_mature_wait"],
                                 cost=info["physical"]["cost_per_entity"],
                                 failures=info["physical"]["failures"])
        assert all(torch.equal(t,model.state_dict()[k]) for k,t in frozen.items())
        details[arm]=variants
    return dict(seed=seed,arms=details,last_event=report["last_event"],
                reference=report["reference"]["metrics"]["pairwise_ranking"],
                source=report["source"])


def main(reports,checkpoints,output):
    r=[audit(seed,path,checkpoints) for seed,path in zip((4101,4102,4103),reports)]
    assert len({x["source"] for x in r})==1
    assert all(x["reference"]==1 for x in r)
    def values(arm,variant,key):
        return [x["arms"][arm][variant][key] for x in r]
    result=dict(study="DEV-043",status="AUDITED_ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY",
                per_seed=r,primary_order_ranking={
                 arm:{v:values(arm,v,"rank") for v in ("original_head","calibration_only","refit_0","refit_10","refit_50")}
                 for arm in ARMS},
                no_writer_updates=True,none_of_prior_DEV042_inspected_cases_used=True,
                confirms_reservoir_tuning=False,full_system_superiority=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("reports",nargs=3)
    p.add_argument("--checkpoints",required=True)
    p.add_argument("--output",required=True)
    x=p.parse_args()
    main(x.reports,x.checkpoints,x.output)
