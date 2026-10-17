"""DEV-044A independent audit: byte-linked checkpoints, paired labels and physical policy.

This does not independently repeat optimizer gradient steps; it verifies the
frozen interventions, dev-only selection and all reported action costs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from tac_osm.development.schema_maintenance import execute,exact_public_targets
from run_reservoir_scaling_dev044 import (ARM_NAMES,PROFILES,THRESHOLDS,sha,
    cases_for)


def all_cases(split,reps):
    results={}
    for pair,conditions in cases_for(split,reps):
        for case,c in zip(pair,conditions):
            assert case.fingerprint not in results
            assert exact_public_targets(case.observations,case.tick,case.horizon)==case.wait_targets
            results[case.fingerprint]=(case,c)
    return results


def replay_row(row,world,condition,threshold):
    assert row["fingerprint"]==condition["fingerprint"]
    assert all(row[k]==condition[k] for k in condition)
    assert row["labels"]==[[world.wait_targets[e],world.service_targets[e],
                            world.current_targets[e]] for e in range(8)]
    p=row["probabilities"]
    assert len(p)==8 and all(len(q)==3 and all(0<=float(v)<=1 for v in q) for q in p)
    chosen=tuple(e for e in range(8) if p[e][0]-p[e][1]>threshold)
    physical=execute(world.world,world.horizon,chosen,0)
    return dict(failures=physical["failures"],unnecessary_services=physical["unnecessary_services"],
                services=len(chosen),total_cost=physical["total_cost"])


def audit_one(folder):
    folder=Path(folder)
    report=json.loads((folder/"report.json").read_text())
    assert report["study"]=="DEV-044A" and not report["smoke"]
    assert report["status"]=="DEVELOPMENT_NOT_CONFIRMATORY"
    assert report["seed"] in (4401,4402,4403)
    assert report["protocol_sha256"]==sha(ROOT/"research/dev044/protocol.json")
    assert set(report["arms"])==set(ARM_NAMES)
    eval_cases=all_cases("evaluation",8)
    cal_cases=all_cases("calibration",2)
    assert len(eval_cases)==576 and len(cal_cases)==144
    assert not set(eval_cases)&set(cal_cases)
    by_arm={}
    for arm in ARM_NAMES:
        data=report["arms"][arm]
        ck=folder/(arm+".pt")
        assert sha(ck)==data["checkpoint_sha256"]
        blob=torch.load(ck,map_location="cpu",weights_only=True)
        assert blob["updates"]==108
        if arm.startswith("frozen_"):
            assert all(torch.equal(blob["model"]["memory."+k],v)
                       for k,v in blob["initial_writer"].items())
        if arm.startswith("trainable_"):
            assert any(not torch.equal(blob["model"]["memory."+k],v)
                       for k,v in blob["initial_writer"].items())
        assert data["train"]["steps"]==108
        train=set(data["train"]["fingerprints"])
        assert len(train)==len(data["train"]["fingerprints"])==216
        cal=set(data["calibration_fingerprints"])
        assert cal==set(cal_cases) and not cal&train and not set(eval_cases)&train
        rows=data["results"]["rows"]
        assert len(rows)==576
        assert {z["fingerprint"] for z in rows}==set(eval_cases)
        totals=dict(failures=0,unnecessary_services=0,services=0,total_cost=0)
        for row in rows:
            w,cond=eval_cases[row["fingerprint"]]
            x=replay_row(row,w,cond,data["threshold"])
            for k,v in x.items():totals[k]+=v
        expected=data["results"]["evaluation"]["physical"]
        for key in totals:assert totals[key]==expected[key]
        assert expected["cost_per_entity"]==totals["total_cost"]/(len(rows)*8)
        assert data["threshold"] in THRESHOLDS
        by_arm[arm]=dict(cost_per_entity=expected["cost_per_entity"],
                         failures=expected["failures"],unnecessary_services=expected["unnecessary_services"],
                         trainable_parameters=data["train"]["trainable_parameters"],
                         training_seconds=data["train"]["training_seconds"])
    for kind in ("trainable","frozen"):
        names=[kind+"_"+p for p in PROFILES]
        best=min(names,key=lambda a:(
            report["arms"][a]["calibration_cost"]["cost_per_entity"],
            report["arms"][a]["calibration_cost"]["failures"],a))
        assert best==report["selected"][kind]
        assert report["selected_results"][kind]["arm"]==best
    common=report["arms"][ARM_NAMES[0]]["results"]["rows"]
    for arm in ARM_NAMES[1:]:
        rows=report["arms"][arm]["results"]["rows"]
        assert [(r["fingerprint"],r["labels"]) for r in rows]==[
                  (r["fingerprint"],r["labels"]) for r in common]
        assert report["arms"][arm]["train"]["fingerprints"]==report["arms"][ARM_NAMES[0]]["train"]["fingerprints"]
    return dict(seed=report["seed"],source=report["source"],
                selected=report["selected"],arms=by_arm,
                last_event=report["last_event"]["summary"]["physical"])


def main(folders,outdir):
    results=[audit_one(x) for x in folders]
    assert {r["seed"] for r in results}=={4401,4402,4403}
    assert len({r["source"] for r in results})==1
    contrasts=[]
    for r in results:
        a=r["arms"][r["selected"]["trainable"]]["cost_per_entity"]
        b=r["arms"][r["selected"]["frozen"]]["cost_per_entity"]
        contrasts.append(a-b)
    output=dict(study="DEV-044A",status="AUDITED_DEVELOPMENT_NOT_CONFIRMATORY",
                per_seed=results,mean_selected_trainable_minus_frozen=statistics.mean(contrasts),
                per_seed_differences=contrasts,matched_update_budget_not_flops=True,
                confirmatory_success_claim=False,full_unified_architecture_validated=False)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"analysis.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(output,indent=2,allow_nan=False),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("seed_folders",nargs=3)
    a.add_argument("--output",required=True)
    x=a.parse_args()
    main(x.seed_folders,x.output)
