"""DEV-044A: matched-data reservoir-scaling pilot; three preset profiles each.

No test-based hyperparameter selection. Training and threshold/model selection use
separate namespaces; evaluation starts only after all candidates are selected.
The unoptimized baselines are diagnostic, NOT convergence-matched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]

import torch
import torch.nn.functional as F
from tac_osm.development.bounded_mtsk import build_bounded
from tac_osm.development.memory_baselines import MemoryBaseline
from tac_osm.development.finite_clipping import clip_finite_cpu_gradients_
from tac_osm.development.schema_maintenance import make_pair,ROLES,DEADLINES,exact_public_targets
from tac_osm.development.order_causal_cases import make_order_pair
from tac_osm.development.delayed_maintenance import Action
from run_learned_schema_dev import public_stream_tensors,public_queries,labels
from run_frozen_writer_dev041 import THRESHOLDS
from diagnose_head_cost_dev040a import last_keys

PROFILES={
    "base":(0.,1.,1.),
    "sticky":(2.,1.,1.),
    "scaled":(2.,.5,2.)
}
ARM_NAMES=tuple(f"{kind}_{profile}" for kind in ("trainable","frozen") for profile in PROFILES) + (
    "schema41","gru768","delta768")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(arm,seed):
    if arm in ("schema41","gru768","delta768"):
        return MemoryBaseline(arm,seed+40000)
    kind,profile=arm.split("_",1)
    if kind not in ("trainable","frozen") or profile not in PROFILES:
        raise ValueError("unknown registered arm")
    model=build_bounded(seed+40000)
    shift,keygain,valuegain=PROFILES[profile]
    with torch.no_grad():
        model.memory.decay_logits.add_(shift)
        model.memory.input_decay.bias.add_(shift)
        model.memory.key.weight.mul_(keygain)
        model.memory.value.weight.mul_(valuegain)
    if kind=="frozen":
        for p in model.memory.parameters():p.requires_grad_(False)
    return model


def paired_profiles(seed):
    for p in PROFILES:
        trainable=build("trainable_"+p,seed)
        frozen=build("frozen_"+p,seed)
        assert trainable.state_scalars==frozen.state_scalars==768
        assert trainable.state_dict().keys()==frozen.state_dict().keys()
        assert all(torch.equal(v,frozen.state_dict()[k]) for k,v in trainable.state_dict().items())
        assert any(x.requires_grad for x in trainable.memory.parameters())
        assert all(not x.requires_grad for x in frozen.memory.parameters())


def train(model,arm,seed,updates,output,smoke):
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                          lr=.002,weight_decay=.0001)
    curve=[];fingerprints=[];started=time.monotonic()
    before_writer={k:v.detach().clone() for k,v in model.memory.state_dict().items()}
    try:
        for step in range(updates):
            cases=make_pair(seed*100000+step,32 if smoke else 512,
                     (32,128,512)[(step//4)%3],ROLES[step%4],
                     DEADLINES[(step//12)%3],domain="dev044a-train-v1")
            fingerprints.extend(c.fingerprint for c in cases)
            ev=public_stream_tensors([c.observations for c in cases])
            queries=public_queries([c.tick for c in cases],[c.horizon for c in cases])
            target=labels(cases)
            receipt=[(c.world.act(Action("inspect",random.Random(
                  f"dev044a:{seed}:{step}:{i}").randrange(8))),) for i,c in enumerate(cases)]
            re=public_stream_tensors(receipt)
            opt.zero_grad(set_to_none=True)
            state=model.encode(*ev)
            pre=model.decode(state,queries)
            post=model.decode(model.update(state,*(v[:,0] for v in re)),queries)
            loss=(F.cross_entropy(pre.reshape(-1,2),target.reshape(-1))+
                  F.cross_entropy(post.reshape(-1,2),target.reshape(-1)))/2
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite training loss")
            loss.backward()
            norm=clip_finite_cpu_gradients_(model.parameters(),1.0)
            opt.step()
            if step in (0,updates-1) or (step+1)%36==0:
                curve.append(dict(update=step+1,loss=float(loss.detach()),gradnorm=float(norm)))
                print(json.dumps(dict(seed=seed,arm=arm,**curve[-1])),flush=True)
    except Exception as exc:
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
                  rng=torch.get_rng_state(),step=step),output/(arm+"-failed.pt"))
        (output/(arm+"-failed.json")).write_text(json.dumps(
           dict(seed=seed,arm=arm,error=str(exc),step=step))+"\n")
        raise
    if arm.startswith("frozen_") and any(not torch.equal(v,before_writer[k])
                                          for k,v in model.memory.state_dict().items()):
        raise AssertionError("frozen reservoir parameters mutated")
    return dict(steps=updates,curve=curve,fingerprints=fingerprints,
                initial_writer=before_writer,
                training_seconds=time.monotonic()-started,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def cases_for(split,repetitions,smoke=False):
    history=32 if smoke else 512
    for delay in ((32,) if smoke else (32,128,512)):
        for role in (("reissued_cancel",) if smoke else ROLES):
            for deadline in (("at",) if smoke else DEADLINES):
                for rep in range(repetitions):
                    pair=make_pair(440000+rep,history,delay,role,deadline,
                                   domain="dev044a-"+split+"-v1")
                    yield pair,[dict(fingerprint=c.fingerprint,delay=delay,role=role,
                                      deadline=deadline,repetition=rep,partner=i,
                                      target_entity=c.target_entity)
                                for i,c in enumerate(pair)]


@torch.no_grad()
def predict(model,split,reps,smoke=False):
    model.eval();results=[]
    for cases,conds in cases_for(split,reps,smoke):
        ev=public_stream_tensors([c.observations for c in cases])
        query=public_queries([c.tick for c in cases],[c.horizon for c in cases])
        p=model.decode(model.encode(*ev),query).softmax(-1)[...,1].tolist()
        for i,c in enumerate(cases):
            assert exact_public_targets(c.observations,c.tick,c.horizon)==c.wait_targets
            results.append(dict(**conds[i],probabilities=p[i],labels=labels([c])[0].tolist()))
    return results


def physical(rows,threshold):
    costs=dict(failures=0,unnecessary_services=0,services=0,inspections=0,total_cost=0)
    for row in rows:
        pp=row["probabilities"];y=row["labels"]
        chosen=tuple(i for i in range(8) if pp[i][0]-pp[i][1]>threshold)
        f=sum(y[i][int(i in chosen)] for i in range(8))
        costs["failures"]+=f
        costs["unnecessary_services"]+=sum(i in chosen and y[i][0]==0 for i in range(8))
        costs["services"]+=len(chosen)
        costs["total_cost"]+=4*f+len(chosen)
    costs["cost_per_entity"]=costs["total_cost"]/(8*len(rows))
    return costs


def select_threshold(rows):
    return min(THRESHOLDS,key=lambda t:(
        physical(rows,t)["total_cost"],abs(t-.25),t))


def summary(rows,threshold):
    allcost=physical(rows,threshold)
    mature=[r for r in rows if r["deadline"]!="before"]
    cancellation=[r for r in mature if r["role"] in ("pending_cancel","reissued_cancel")]
    reissued=[r for r in mature if r["role"]=="reissued_cancel"]
    pending=[r for r in rows if r["deadline"]=="before"]
    def accuracy(cs,q=0):
        if not cs:return None
        return sum(int((c["probabilities"][c["target_entity"]][q]>.5)==bool(
                    c["labels"][c["target_entity"]][q])) for c in cs)/len(cs)
    return dict(worlds=len(rows),physical=allcost,
                wait_all=accuracy(rows),mature_wait=accuracy(mature),
                mature_cancellation_wait=accuracy(cancellation),
                mature_reissued_wait=accuracy(reissued),
                pending_wait=accuracy(pending),service_all=accuracy(rows,1))


@torch.no_grad()
def order_eval(model,threshold,reps,smoke):
    rows=[]
    for delay in ((32,) if smoke else (32,128,512)):
        for deadline in (("at",) if smoke else DEADLINES):
            for rep in range(reps):
                cs=make_order_pair(440000+rep,32 if smoke else 512,delay,deadline,
                              domain="dev044a-order-eval-v1")
                ev=public_stream_tensors([c.observations for c in cs])
                qu=public_queries([c.tick for c in cs],[c.horizon for c in cs])
                p=model.decode(model.encode(*ev),qu).softmax(-1)[...,1].tolist()
                for i,c in enumerate(cs):
                    rows.append(dict(fingerprint=c.fingerprint,delay=delay,
                        deadline=deadline,partner=i,target_entity=c.target_entity,
                        probabilities=p[i],labels=labels([c])[0].tolist()))
    signed=[]
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        if a["deadline"]=="before":continue
        e=a["target_entity"]
        assert a["labels"][e][0]==1 and b["labels"][e][0]==0
        signed.append(a["probabilities"][e][0]-b["probabilities"][e][0])
    rank=(sum(x>0 for x in signed)+.5*sum(x==0 for x in signed))/len(signed)
    mature=[r for r in rows if r["deadline"]!="before"]
    hard=sum(int((r["probabilities"][r["target_entity"]][0]>.5)==bool(
           r["labels"][r["target_entity"]][0])) for r in mature)/len(mature)
    return dict(pair_ranking=rank,hard_wait=hard,
                physical=physical(rows,threshold),rows=rows)


def last_event_reference(calibration,evaluation,smoke):
    fit={}
    for (cases,_),r1,r2 in zip(cases_for("calibration",1 if smoke else 2,smoke),
                               calibration[::2],calibration[1::2]):
        for case,row in zip(cases,(r1,r2)):
            for key,target in zip(last_keys(case),row["labels"]):
                a,n=fit.get(key,([0.,0.,0.],0))
                fit[key]=([u+v for u,v in zip(a,target)],n+1)
    table={k:[v/n for v in arr] for k,(arr,n) in fit.items()}
    def revalue(rows,split,reps):
        out=[]
        for cs,_ in cases_for(split,reps,smoke):
            for c in cs:
                known=next(r for r in rows if r["fingerprint"]==c.fingerprint)
                out.append({**known,"probabilities":[table.get(k,[0.,0.,0.])
                                                     for k in last_keys(c)]})
        return out
    dev=revalue(calibration,"calibration",1 if smoke else 2)
    threshold=select_threshold(dev)
    test=revalue(evaluation,"evaluation",1 if smoke else 8)
    return dict(threshold=threshold,summary=summary(test,threshold))


def run(seed,output,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/"research/dev044/protocol.json").read_text())
    if seed not in protocol["trial_design"]["seeds"]:raise ValueError("unknown registered seed")
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip():
        raise ValueError("dirty source")
    output=Path(output)
    if output.exists():raise ValueError("do not overwrite study evidence")
    output.mkdir(parents=True)
    paired_profiles(seed)
    report=dict(study="DEV-044A",status="DEVELOPMENT_NOT_CONFIRMATORY",
                source=source,protocol_sha256=sha(ROOT/"research/dev044/protocol.json"),
                seed=seed,smoke=smoke,arms={},selected={})
    models={}
    updates=2 if smoke else protocol["trial_design"]["train_updates"]
    for arm in ARM_NAMES:
        model=build(arm,seed)
        info=train(model,arm,seed,updates,output,smoke)
        torch.save(dict(model=model.state_dict(),
                initial_writer=info.pop("initial_writer"),
                rng=torch.get_rng_state(),updates=updates),
                output/(arm+".pt"))
        cal=predict(model,"calibration",1 if smoke else 2,smoke)
        threshold=select_threshold(cal)
        report["arms"][arm]=dict(checkpoint_sha256=sha(output/(arm+".pt")),
                 train=info,threshold=threshold,
                 calibration_cost=physical(cal,threshold),
                 calibration_fingerprints=[r["fingerprint"] for r in cal],
                 state_scalars=model.state_scalars)
        models[arm]=model
    # SELECT PROFILES ON DEV CALIBRATION BEFORE GENERATING ANY TEST CASES.
    for kind in ("trainable","frozen"):
        subset=[kind+"_"+p for p in PROFILES]
        chosen=min(subset,key=lambda a:(report["arms"][a]["calibration_cost"]["cost_per_entity"],
                    report["arms"][a]["calibration_cost"]["failures"],a))
        report["selected"][kind]=chosen
    print("DEV044A CALIBRATION PROFILES LOCKED",json.dumps(report["selected"]),flush=True)
    for arm,model in models.items():
        th=report["arms"][arm]["threshold"]
        rows=predict(model,"evaluation",1 if smoke else 8,smoke)
        order=order_eval(model,th,1 if smoke else 8,smoke)
        report["arms"][arm]["results"]=dict(evaluation=summary(rows,th),
                order=dict(pair_ranking=order["pair_ranking"],hard_wait=order["hard_wait"],
                           physical=order["physical"],rows=order["rows"]),rows=rows)
        train_fps=set(report["arms"][arm]["train"]["fingerprints"])
        calibration_fps=set(report["arms"][arm]["calibration_fingerprints"])
        eval_fps={r["fingerprint"] for r in rows}
        order_fps={r["fingerprint"] for r in order["rows"]}
        assert not train_fps&calibration_fps and not train_fps&eval_fps
        assert not calibration_fps&eval_fps and not order_fps&eval_fps
        assert not order_fps&train_fps and not order_fps&calibration_fps
    first=report["arms"][ARM_NAMES[0]]
    dev=predict(models[ARM_NAMES[0]],"calibration",1 if smoke else 2,smoke)
    report["last_event"]=last_event_reference(dev,first["results"]["rows"],smoke)
    report["selected_results"]={kind:dict(arm=arm,**report["arms"][arm]["results"]["evaluation"])
                                for kind,arm in report["selected"].items()}
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("DEV044A REGISTERED PILOT RESULT",json.dumps(dict(seed=seed,selected=report["selected"],
         costs={a:round(report["arms"][a]["results"]["evaluation"]["physical"]["cost_per_entity"],6)
                 for a in ARM_NAMES},last_event=report["last_event"]["summary"]["physical"]["cost_per_entity"])),
         flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args();run(a.seed,a.output,a.smoke)
