"""DEV-042 frozen-checkpoint evaluation on paired order-sensitive public histories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from tac_osm.development.order_causal_cases import make_order_pair, last_observations
from tac_osm.development.schema_maintenance import execute
from run_frozen_writer_dev041 import ARMS, build, sha, cases_for, last_event_fit
from run_learned_schema_dev import public_stream_tensors, public_queries
from diagnose_head_cost_dev040a import last_keys


def conditions():
    for delay in (32,128,512):
        for deadline in ("before","at","after"):
            for rep in range(16):
                yield delay,deadline,rep


def score(case,p,threshold):
    services=tuple(e for e in range(8) if p[e][0]-p[e][1]>threshold)
    out=execute(case.world,case.horizon,services,0)
    return dict(failures=out["failures"],unnecessary_services=out["unnecessary_services"],
                services=len(services),total_cost=out["total_cost"])


def rows_from_model(model, threshold):
    model.eval();results=[]
    with torch.no_grad():
        for delay,deadline,rep in conditions():
            cases=make_order_pair(420000+rep,512,delay,deadline)
            x=public_stream_tensors([c.observations for c in cases])
            q=public_queries([c.tick for c in cases],[c.horizon for c in cases])
            state=model.encode(*x)
            probs=model.decode(state,q).softmax(-1)[...,1].tolist()
            for b,c in enumerate(cases):
                results.append(dict(fingerprint=c.fingerprint,delay=delay,deadline=deadline,
                                    repetition=rep,partner=b,target_entity=c.target_entity,
                                    labels=[[c.wait_targets[e],c.service_targets[e],c.current_targets[e]] for e in range(8)],
                                    probabilities=probs[b],physical=score(c,probs[b],threshold)))
    return results


def reference_rows(table,threshold):
    result=[]
    for delay,deadline,rep in conditions():
        cases=make_order_pair(420000+rep,512,delay,deadline)
        for b,c in enumerate(cases):
            p=[table.get(k,[0.,0.,0.]) for k in last_keys(c)]
            result.append(dict(fingerprint=c.fingerprint,delay=delay,deadline=deadline,
                               repetition=rep,partner=b,target_entity=c.target_entity,
                               labels=[[c.wait_targets[e],c.service_targets[e],c.current_targets[e]] for e in range(8)],
                               probabilities=p,physical=score(c,p,threshold)))
    return result


def summarize(rows):
    mature=[r for r in rows if r["deadline"]!="before"]
    pending=[r for r in rows if r["deadline"]=="before"]
    a=lambda rs,q: sum(int((r["probabilities"][r["target_entity"]][q]>.5)==bool(
        r["labels"][r["target_entity"]][q])) for r in rs)/len(rs)
    by_delay={str(d):a([r for r in mature if r["delay"]==d],0) for d in (32,128,512)}
    total={k:sum(r["physical"][k] for r in rows) for k in ("failures","unnecessary_services","services","total_cost")}
    return dict(worlds=len(rows),mature_worlds=len(mature),
                mature_wait=a(mature,0),pending_wait=a(pending,0),
                mature_service=a(mature,1),by_delay_mature_wait=by_delay,
                cost_per_entity=total["total_cost"]/(len(rows)*8),physical=total)


def run(seed,checkpoint_dir,output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/"research/dev042/protocol.json").read_text())
    if seed not in protocol["source_checkpoints"]["seeds"]:raise ValueError("unregistered seed")
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip():
        raise ValueError("freeze source before scientific evaluation")
    output=Path(output)
    if output.exists():raise ValueError("do not overwrite an evaluated result")
    report=json.loads((Path(checkpoint_dir)/f"DEV041-RAW-seed{seed}-development"/"report.json").read_text())
    if report["source"]!=protocol["source_checkpoints"]["frozen_sha"] or report["seed"]!=seed:
        raise AssertionError("wrong archived training source or seed")
    result=dict(study="DEV-042",status="ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY",
                seed=seed,source=source,protocol_sha256=sha(ROOT/"research/dev042/protocol.json"),
                model_source=report["source"],arms={})
    for arm in ARMS:
        checkpoint=Path(checkpoint_dir)/f"DEV041-RAW-seed{seed}-development"/(arm+".pt")
        assert sha(checkpoint)==report["arms"][arm]["checkpoint_sha256"]
        ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
        model=build(arm,seed)
        model.load_state_dict(ck["model"],strict=True)
        if arm=="bounded_random_frozen":
            assert all(torch.equal(ck["model"]["memory."+k],v) for k,v in ck["initial_writer"].items())
        threshold=report["arms"][arm]["results"]["threshold"]
        rows=rows_from_model(model,threshold)
        result["arms"][arm]=dict(checkpoint_sha256=sha(checkpoint),threshold=threshold,
                                  summary=summarize(rows),rows=rows)
        print(json.dumps({ "seed":seed, "arm":arm,
             "mature_wait":result["arms"][arm]["summary"]["mature_wait"],
             "cost_per_entity":result["arms"][arm]["summary"]["cost_per_entity"] }),flush=True)
    calrows=[]
    for cs, _ in cases_for("calibration",4):
        for c in cs:
            calrows.append(dict(fingerprint=c.fingerprint,
                                labels=[[c.wait_targets[e],c.service_targets[e],c.current_targets[e]] for e in range(8)]))
    table=last_event_fit(calrows,"calibration",4)
    refthreshold=report["references"]["last_event_threshold"]
    last=reference_rows(table,refthreshold)
    result["last_event"]=dict(threshold=refthreshold,summary=summarize(last),rows=last)
    # Public exact-history reference: predicted wait/service/current are the evaluator future bits.
    oracle=[]
    for delay,deadline,rep in conditions():
        cs=make_order_pair(420000+rep,512,delay,deadline)
        assert last_observations(cs[0].observations)==last_observations(cs[1].observations)
        for c in cs:
            labels=[[c.wait_targets[e],c.service_targets[e],c.current_targets[e]] for e in range(8)]
            oracle.append(dict(fingerprint=c.fingerprint,delay=delay,deadline=deadline,
                               repetition=rep,partner=0,target_entity=c.target_entity,
                               labels=labels,probabilities=labels,physical=score(c,labels,.25)))
    result["exact_public_history"]=dict(summary=summarize(oracle))
    output.mkdir(parents=True)
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({ "seed":seed,"last_event":result["last_event"]["summary"],
                    "exact_reference":result["exact_public_history"]["summary"] }),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--checkpoints",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();run(a.seed,a.checkpoints,a.output)
