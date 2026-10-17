"""DEV-051 development experiment: public-state BC and student-state aggregation.

Fit and calibrate on disjoint synthetic worlds; run all action candidates through
the pinned native Veritas runtime. This is NOT a TAC-OSM world-model superiority test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import torch
from tac_osm.development.learned_veritas_policy import (
    PublicActionPolicy,rollout,supervised_samples,public_cohort,fit,digest,aggregate)
from tac_osm.development.veritas_public_challenge import run_episode

PROTOCOL=ROOT/"research/dev051/study-protocol.json"
THRESHOLDS=(0.,.4,.6,.75,.85,.95,1.01)


def verify_environment(veritas_root):
    import investigation_world
    base=Path(veritas_root).resolve()
    assert Path(investigation_world.__file__).resolve().is_relative_to(base)
    pin=subprocess.check_output(["git","rev-parse","HEAD"],cwd=base,text=True).strip()
    assert pin==json.loads(PROTOCOL.read_text())["environment"]["commit"]
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=base,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],
                                       cwd=ROOT,text=True).strip()


def public_ids(worlds):
    return [ep.public_payload()["episode_id"] for _,_,ep in worlds]


def digest_dataset(x,y):
    packed=json.dumps(dict(x=x.tolist(),y=y.tolist()),sort_keys=True,separators=(",",":"))
    return hashlib.sha256(packed.encode()).hexdigest()


def train_pair(seed,teacher_worlds,onpolicy_worlds,output,smoke=False):
    x,y,_=supervised_samples(teacher_worlds)
    if len(x)!=len(y) or not len(x):raise AssertionError("empty teacher supervision")
    teacher_hash=digest_dataset(x,y)
    bc=PublicActionPolicy(seed)
    dagger=PublicActionPolicy(seed)
    assert all(torch.equal(p,q) for p,q in zip(bc.parameters(),dagger.parameters()))
    started=time.monotonic()
    _,bc_loss=fit(bc,x,y,2 if smoke else 160)
    optimizer,warmup=fit(dagger,x,y,2 if smoke else 80)
    dx,dy,_=supervised_samples(onpolicy_worlds,model=dagger)
    fit_x=torch.cat((x,dx))
    fit_y=torch.cat((y,dy))
    _,dagger_loss=fit(dagger,fit_x,fit_y,2 if smoke else 80,optimizer=optimizer)
    for name,model in (("bc",bc),("dagger",dagger)):
        torch.save(model.state_dict(),output/(name+".pt"))
    assert all(torch.isfinite(p).all() for m in (bc,dagger) for p in m.parameters())
    return dict(bc=bc,dagger=dagger,stats=dict(
        teacher_examples=len(x),student_encountered_examples=len(dx),
        teacher_dataset_sha256=teacher_hash,
        onpolicy_dataset_sha256=digest_dataset(dx,dy),
        optimizer_updates_per_arm=4 if smoke else 160,
        behavior_cloning_curve=bc_loss,
        aggregation_warmup=warmup,aggregation_curve=dagger_loss,
        training_seconds=time.monotonic()-started,
        parameter_count=sum(p.numel() for p in bc.parameters())),
        checkpoint_sha256={n:digest(output/(n+".pt")) for n in ("bc","dagger")})


def evaluate(worlds,name,model=None,gate=None):
    result=[]
    for seed,index,episode in worlds:
        if name in ("inspect_verify_retry","inspect_once","stale_record"):
            output=run_episode(episode,name)
            r=dict(world=episode.public_payload()["episode_id"],failures=output["failures"],
                   complete=int(output["verification"]["state"]==1),
                   total_cost=output["total_cost"],
                   blocked=sum(int(t["blocked"]) for t in output["trace"]),
                   tool_calls=output["verification"]["tool_calls"],
                   sensing_calls=output["sensing_calls"],
                   fallbacks=0,decisions=0,
                   actions=output["observations"],verification=output["verification"])
        else:
            r,_=rollout(episode,name,model,gate)
        result.append(dict(seed=seed,index=index,**r))
    return result


def select_gate(dagger,worlds):
    dev={}
    for t in THRESHOLDS:
        rows=evaluate(worlds,"gated",dagger,t)
        dev[str(t)]=dict(threshold=t,**aggregate(rows))
    eligible=[v for v in dev.values()
              if v["residual_faults"]==0 and v["blocked_actions"]==0]
    if not eligible:raise AssertionError("registered teacher fallback must be viable")
    best=min(eligible,key=lambda v:(v["cost_per_entity"],v["fallback_fraction"],v["threshold"]))
    return best["threshold"],dev


def run(seed,output,veritas_root,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads(PROTOCOL.read_text())
    if seed not in protocol["data"]["model_seeds"]:raise ValueError("unregistered seed")
    verify_environment(veritas_root)
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    groups=protocol["data"]
    def worlds(key):
        v=groups[key]
        n=2 if smoke else v["worlds_per_seed"]
        seeds=v["seeds"][:1] if smoke else v["seeds"]
        return list(public_cohort(seeds,n))
    train=worlds("teacher_worlds")
    onpolicy=worlds("onpolicy_worlds")
    calibration=worlds("calibration_worlds")
    evaluation=worlds("evaluation_worlds")
    ids={k:public_ids(w) for k,w in (
        ("teacher",train),("onpolicy",onpolicy),
        ("calibration",calibration),("evaluation",evaluation))}
    sets=[set(z) for z in ids.values()]
    assert all(len(z)==len(set(z)) for z in ids.values())
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    models=train_pair(seed,train,onpolicy,output,smoke)
    threshold,calibration_rows=select_gate(models["dagger"],calibration)
    policies=[
        ("inspect_verify_retry",None,None),("inspect_once",None,None),
        ("stale_record",None,None),("public_belief_rule",None,None),
        ("behavior_cloning",models["bc"],None),
        ("on_policy_aggregation",models["dagger"],None),
        ("confidence_gated_aggregation",models["dagger"],threshold)]
    scores={}
    for name,model,gate in policies:
        rows=evaluate(evaluation,name,model,gate)
        scores[name]=dict(summary=aggregate(rows),rows=rows)
        print(json.dumps(dict(seed=seed,policy=name,**scores[name]["summary"])),flush=True)
    # Strong identity reference: the public-belief rule must reproduce all
    # physical outcomes of the older native inspect/verify/retry implementation.
    original=scores["inspect_verify_retry"]["rows"]
    belief=scores["public_belief_rule"]["rows"]
    assert [(z["failures"],z["total_cost"],z["complete"]) for z in original]==[
            (z["failures"],z["total_cost"],z["complete"]) for z in belief]
    report=dict(study="DEV-051A",status="PUBLIC_LEARNED_POLICY_DEVELOPMENT_NOT_CONFIRMATORY",
                seed=seed,smoke=smoke,tac_source=source,
                veritas_source=protocol["environment"]["commit"],
                protocol_sha256=digest(PROTOCOL),
                split_ids=ids,
                model_fit=models["stats"],checkpoint_sha256=models["checkpoint_sha256"],
                confidence_threshold=threshold,
                calibration_candidates=calibration_rows,
                policies=scores,
                limitations=["Trained action head, not trained TAC-OSM PLM/OSM",
                             "No sensor-error or second-repair-failure generalization",
                             "Rule-gated policy may inherit a hand-coded teacher",
                             "Independent native evaluator replay not yet run"])
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("DEV-051 REPORT SAVED",seed,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--veritas-root",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.seed,args.output,args.veritas_root,args.smoke)
