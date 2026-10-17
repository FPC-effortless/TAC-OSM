"""DEV-041 separate-process audit: public-target and physical cost reconstruction."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from tac_osm.development.schema_maintenance import execute, exact_public_targets
from run_frozen_writer_dev041 import ARMS, THRESHOLDS, cases_for, last_keys


def independent_physical(case, scores, threshold):
    chosen=tuple(e for e,ps in enumerate(scores) if ps[0]-ps[1]>threshold)
    actual=execute(case.world,case.horizon,chosen,0)
    return {k:actual[k] for k in ("failures","unnecessary_services","inspections","total_cost")},len(chosen)


def make_reference(case, table):
    scores=[table.get(k,[0.,0.,0.]) for k in last_keys(case)]
    return scores


def audit_one(directory):
    directory=Path(directory)
    report=json.loads((directory/"report.json").read_text())
    seed=report["seed"]; assert seed in (4101,4102,4103)
    assert report["study"]=="DEV-041" and not report["smoke"]
    conditions={}
    for worlds, cs in cases_for("eval",16):
        for c, condition in zip(worlds,cs):
            assert c.fingerprint not in conditions
            conditions[c.fingerprint]=(c,condition)
    assert len(conditions)==1152
    total_entity_count=len(conditions)*8
    census={}
    for arm in ARMS:
        entry=report["arms"][arm]
        checkpoint=directory/(arm+".pt")
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==entry["checkpoint_sha256"]
        stored=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if arm=="bounded_random_frozen":
            assert all(torch.equal(stored["model"]["memory."+k],v)
                       for k,v in stored["initial_writer"].items()),"frozen writer mutated"
        if arm=="bounded_trainable":
            assert any(not torch.equal(stored["model"]["memory."+k],v)
                       for k,v in stored["initial_writer"].items()),"trainable writer never updated"
        info=entry["results"]
        assert len(info["rows"])==len(conditions)
        observed=set();total=dict(failures=0,unnecessary_services=0,inspections=0,total_cost=0,services=0)
        for row in info["rows"]:
            fp=row["fingerprint"];assert fp not in observed;observed.add(fp)
            case,condition=conditions[fp]
            assert all(row[k]==condition[k] for k in condition)
            assert exact_public_targets(case.observations,case.tick,case.horizon)==case.wait_targets
            targets=[[case.wait_targets[e],case.service_targets[e],case.current_targets[e]]
                     for e in range(8)]
            assert row["labels"]==targets
            physical,n=independent_physical(case,row["probabilities"],info["threshold"])
            for key,value in physical.items():total[key]+=value
            total["services"]+=n
        assert observed==set(conditions)
        assert total==info["evaluation"]["physical"]
        assert math.isclose(info["evaluation"]["cost_per_entity"],total["total_cost"]/total_entity_count,abs_tol=1e-12)
        train=set(entry["training"]["fingerprints"])
        cal=set(info["calibration_fingerprints"])
        assert not train & observed and not cal & observed and not train & cal
        assert len(cal)==len(set(cal))==288
        census[arm]=dict(cost_per_entity=info["evaluation"]["cost_per_entity"],
                         failures=total["failures"],
                         unnecessary_services=total["unnecessary_services"],
                         mature_reissued_wait=info["evaluation"]["mature_reissued_wait"],
                         pending_wait=info["evaluation"]["pending_wait"],
                         service_all=info["evaluation"]["service_all"])
    # The three models must be evaluated on identical worlds, labels, and conditions.
    for other in ARMS[1:]:
        a=report["arms"][ARMS[0]]["results"]["rows"]
        b=report["arms"][other]["results"]["rows"]
        assert [(r["fingerprint"],r["labels"]) for r in a]==[(r["fingerprint"],r["labels"]) for r in b]
        assert report["arms"][ARMS[0]]["training"]["fingerprints"]==report["arms"][other]["training"]["fingerprints"]
    # Independently refit the last-event table using calibration cases alone.
    samples={}
    for worlds,_ in cases_for("calibration",4):
        for c in worlds:
            for k,t in zip(last_keys(c),zip(c.wait_targets,c.service_targets,c.current_targets)):
                item=samples.setdefault(k,[torch.zeros(3),0])
                item[0]+=torch.tensor(t,dtype=torch.float32)
                item[1]+=1
    table={k:(v/n).tolist() for k,(v,n) in samples.items()}
    reference_scores=[]
    for worlds,_ in cases_for("calibration",4):
        for c in worlds: reference_scores.append((c,make_reference(c,table)))
    best=min(THRESHOLDS,key=lambda t:(
        sum(independent_physical(c,s,t)[0]["total_cost"] for c,s in reference_scores),abs(t-.25),t))
    assert report["references"]["last_event_threshold"]==best
    last=dict(failures=0,unnecessary_services=0,inspections=0,total_cost=0,services=0)
    for c,_ in conditions.values():
        phys,n=independent_physical(c,make_reference(c,table),best)
        for k,v in phys.items():last[k]+=v
        last["services"]+=n
    assert last==report["references"]["last_event"]["physical"]
    return dict(seed=seed,arms=census,last_event_cost=last["total_cost"]/total_entity_count,
                last_event_failures=last["failures"],source=report["source"])


def main(inputs,output):
    rows=[audit_one(p) for p in inputs]
    assert {r["seed"] for r in rows}=={4101,4102,4103}
    assert len({r["source"] for r in rows})==1
    contrast=[r["arms"]["bounded_trainable"]["cost_per_entity"]-
              r["arms"]["bounded_random_frozen"]["cost_per_entity"] for r in rows]
    failures=[r["arms"]["bounded_trainable"]["failures"]-
              r["arms"]["bounded_random_frozen"]["failures"] for r in rows]
    mean=statistics.mean(contrast)
    margin=4.302652729911275*statistics.stdev(contrast)/len(contrast)**.5
    summary=dict(study="DEV-041",status="AUDITED_DEVELOPMENT_NOT_CONFIRMATORY",
                 per_seed=rows,primary_cost_difference_mean=mean,
                 exploratory_three_seed_95pct_interval=[mean-margin,mean+margin],
                 per_seed_failure_differences=failures,
                 descriptive_success=(mean<0 and statistics.mean(failures)<=0),
                 independent_training_replication=False,
                 full_system_attribution=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"analysis.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("directories",nargs=3)
    ap.add_argument("--output",required=True)
    x=ap.parse_args();main(x.directories,x.output)
