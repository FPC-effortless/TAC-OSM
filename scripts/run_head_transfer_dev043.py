"""DEV-043 prospective calibration-only vs head refit under mixed causal-order exposure.

Frozen DEV-041 encoders/writers/read adapters. All data is public; hidden labels
are used only for supervision/scoring. No inspected DEV-042 cohort is consulted.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import torch
from torch import nn
import torch.nn.functional as F
from tac_osm.development.schema_maintenance import make_pair, exact_public_targets, execute
from tac_osm.development.order_causal_cases import make_order_pair, last_observations
from run_frozen_writer_dev041 import build, sha, THRESHOLDS
from run_learned_schema_dev import public_stream_tensors,public_queries,labels
from run_head_controls_dev040a import feature_input
from diagnose_head_cost_dev040a import last_keys

ARMS=("bounded_trainable","bounded_random_frozen")
FRACTIONS=(0,10,50)
METHODS=("identity","temperature_positive","platt_positive")


def split_cases(split):
    if split=="head-train": repetitions=20
    elif split=="calibration": repetitions=5
    elif split=="evaluation": repetitions=16
    else:raise ValueError("unknown split")
    out=[]
    for delay in (32,128,512):
        for deadline in ("before","at","after"):
            for rep in range(repetitions):
                if split=="head-train":
                    original=make_pair(430000+rep,512,delay,"reissued_cancel",deadline,
                                       domain="dev043-base-train-v1")
                    order=make_order_pair(430000+rep,512,delay,deadline,
                                         domain="dev043-order-train-v1")
                    out.append(dict(delay=delay,deadline=deadline,rep=rep,
                                    original=original,order=order))
                else:
                    pair=make_order_pair(430000+rep,512,delay,deadline,
                                         domain="dev043-order-"+("calibration" if split=="calibration" else "eval")+"-v1")
                    out.append(dict(delay=delay,deadline=deadline,rep=rep,order=pair))
    return out


def assemble(mixed,fraction):
    assert fraction in FRACTIONS+(100,)
    cases=[];conditions=[]
    for i,row in enumerate(mixed):
        mode="order" if fraction==100 or i%10<fraction//10 else "original"
        if "original" not in row: mode="order"
        pair=row[mode]
        for partner,c in enumerate(pair):
            conditions.append(dict(delay=row["delay"],deadline=row["deadline"],
                                   rep=row["rep"],partner=partner,target_entity=c.target_entity,
                                   fingerprint=c.fingerprint,mode=mode))
            cases.append(c)
    return cases,conditions


@torch.no_grad()
def frozen_features(model,cases):
    model.eval()
    features=[];logits=[]
    for i in range(0,len(cases),2):
        group=cases[i:i+2]
        events=public_stream_tensors([c.observations for c in group])
        q=public_queries([c.tick for c in group],[c.horizon for c in group])
        state=model.encode(*events)
        feat=feature_input(model,state,q)
        features.append(feat.cpu())
        logits.append(model.head(feat).cpu())
    return torch.cat(features),torch.cat(logits)


def capture_split(model,split):
    groups=split_cases(split)
    if split=="head-train":
        cases_a,conds_a=assemble(groups,0)
        cases_b,conds_b=assemble(groups,100)
        a,la=frozen_features(model,cases_a)
        b,lb=frozen_features(model,cases_b)
        return dict(original=dict(x=a,y=labels(cases_a),conditions=conds_a,logits=la),
                    order=dict(x=b,y=labels(cases_b),conditions=conds_b,logits=lb))
    cs,conds=assemble(groups,100)
    x,lo=frozen_features(model,cs)
    return dict(x=x,y=labels(cs),conditions=conds,logits=lo)


def mix_train(source,fraction):
    base,order=source["original"],source["order"]
    assert len(base["conditions"])==len(order["conditions"])==360
    flags=torch.tensor([(i//2)%10 < fraction//10 for i in range(360)])
    x=torch.where(flags[:,None,None,None],order["x"],base["x"])
    y=torch.where(flags[:,None,None],order["y"],base["y"])
    conditions=[order["conditions"][i] if flags[i] else base["conditions"][i] for i in range(360)]
    assert sum(c["mode"]=="order" for c in conditions)==int(360*fraction/100)
    return dict(x=x,y=y,conditions=conditions)


def fit_head(initial,data,seed,steps,lr):
    torch.manual_seed(seed)
    head=copy.deepcopy(initial)
    for p in head.parameters():p.requires_grad_(True)
    x=data["x"];y=data["y"]
    mask=torch.tensor([c["deadline"]!="before" for c in data["conditions"]])
    focus=torch.tensor([c["target_entity"] for c in data["conditions"]])
    idx=torch.arange(len(y))
    opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=0.0001)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits=head(x)
        uniform=F.cross_entropy(logits.reshape(-1,2),y.reshape(-1))
        special=F.cross_entropy(logits[idx[mask],focus[mask],0],y[idx[mask],focus[mask],0])
        loss=.5*uniform+.5*special
        if not torch.isfinite(loss):raise FloatingPointError("nonfinite head loss")
        loss.backward();opt.step()
    return head,float(loss.detach())


def fit_calibrators(logits,y):
    # q-specific affine binary-logit calibrators are monotone in each role.
    z=(logits[...,1]-logits[...,0]).detach()
    out={"identity":dict(scales=[1.]*3,bias=[0.]*3)}
    for method in METHODS[1:]:
        log_scale=torch.nn.Parameter(torch.zeros(3))
        bias=torch.nn.Parameter(torch.zeros(3)) if method=="platt_positive" else None
        params=[log_scale]+([bias] if bias is not None else [])
        optimizer=torch.optim.Adam(params,lr=.06)
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            pred=z*torch.exp(log_scale)[None,None,:]
            if bias is not None:pred=pred+bias[None,None,:]
            loss=F.binary_cross_entropy_with_logits(pred,y.float())
            loss.backward();optimizer.step()
        out[method]=dict(scales=torch.exp(log_scale.detach()).tolist(),
                          bias=bias.detach().tolist() if bias is not None else [0.]*3)
    return out


def calibrated(logits,params):
    z=logits[...,1]-logits[...,0]
    a=torch.tensor(params["scales"])
    b=torch.tensor(params["bias"])
    return torch.sigmoid(z*a[None,None,:]+b[None,None,:])


def action(rows,threshold):
    f=u=s=0;total=0
    for r in rows:
        p=r["probabilities"];y=r["labels"]
        services=tuple(i for i in range(8) if p[i][0]-p[i][1]>threshold)
        failures=sum(y[i][int(i in services)] for i in range(8))
        f+=failures
        u+=sum(i in services and y[i][0]==0 for i in range(8))
        s+=len(services);total+=4*failures+len(services)
    return dict(failures=f,unnecessary_services=u,services=s,total_cost=total,
                cost_per_entity=total/(8*len(rows)))


def as_rows(data,prob):
    return [dict(**c,labels=data["y"][i].tolist(),probabilities=prob[i].tolist())
            for i,c in enumerate(data["conditions"])]


def select_method_and_threshold(cal,data):
    options=fit_calibrators(cal,data["y"])
    # Select by NLL, not observed evaluation or physical test loss.
    nll={}
    for method,params in options.items():
        p=calibrated(cal,params).clamp(1e-7,1-1e-7)
        nll[method]=float(F.binary_cross_entropy(p,data["y"].float()))
    best=min(METHODS,key=lambda m:(nll[m],METHODS.index(m)))
    rows=as_rows(data,calibrated(cal,options[best]))
    threshold=min(THRESHOLDS,key=lambda t:(action(rows,t)["total_cost"],abs(t-.25),t))
    return dict(method=best,calibrators=options,nll=nll,threshold=threshold)


def metrics(rows):
    mature=[r for r in rows if r["deadline"]!="before"]
    pending=[r for r in rows if r["deadline"]=="before"]
    correct=sum(int((r["probabilities"][r["target_entity"]][0]>.5)==bool(
        r["labels"][r["target_entity"]][0])) for r in mature)
    wait_pending=sum(int((r["probabilities"][r["target_entity"]][0]>.5)==bool(
        r["labels"][r["target_entity"]][0])) for r in pending)
    signed=[]
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        assert a["partner"]==0 and b["partner"]==1
        assert a["deadline"]==b["deadline"] and a["target_entity"]==b["target_entity"]
        if a["deadline"]=="before":continue
        e=a["target_entity"]
        assert a["labels"][e][0]==1 and b["labels"][e][0]==0
        signed.append(a["probabilities"][e][0]-b["probabilities"][e][0])
    assert len(signed)==96 and len(mature)==192
    return dict(pairwise_ranking=(sum(x>0 for x in signed)+.5*sum(x==0 for x in signed))/96,
                signed_margin_mean=sum(signed)/96,hard_mature_wait=correct/192,
                pending_wait=wait_pending/len(pending),
                all_wait_brier=sum((r["probabilities"][e][0]-r["labels"][e][0])**2
                  for r in rows for e in range(8))/(8*len(rows)),
                all_service_brier=sum((r["probabilities"][e][1]-r["labels"][e][1])**2
                  for r in rows for e in range(8))/(8*len(rows)))


def reference(data):
    # Public-history evaluator reference; not a neural architecture.
    ref=[]
    # Materialize exactly one fresh evaluation cohort, never reuse inspected DEV-042 cases.
    source={c.fingerprint:c for item in split_cases("evaluation") for c in item["order"]}
    for r in data["conditions"]:
        found=source[r["fingerprint"]]
        assert exact_public_targets(found.observations,found.tick,found.horizon)==found.wait_targets
        yy=[[found.wait_targets[e],found.service_targets[e],found.current_targets[e]] for e in range(8)]
        ref.append(dict(**r,labels=yy,probabilities=yy))
    return dict(metrics=metrics(ref),action=action(ref,.25))


def retrospective_pairwise_oracle(rows):
    # Pure hindsight upper bound: chooses a different threshold per PAIR using labels.
    # Invalid for model selection, not a deployable policy.
    total=0
    for i in range(0,len(rows),2):
        pair=rows[i:i+2]
        scores=[p[0]-p[1] for row in pair for p in row["probabilities"]]
        grid=[-2.0,2.0]+[(a+b)/2 for a,b in zip(sorted(set(scores))[:-1],sorted(set(scores))[1:])]
        best=min((sum(action([r],t)["total_cost"] for r in pair),t) for t in grid)
        total+=best[0]
    return dict(hindsight_pairwise_oracle_cost_per_entity=total/(8*len(rows)),
                note="Chosen using test labels per pair; cannot deploy or tune on this result.")


def disjoint(groups):
    keys=[]
    for data in groups:
        keys.append({c["fingerprint"] for c in data["conditions"]})
    assert all(not keys[i]&keys[j] for i in range(len(keys)) for j in range(i))
    assert all(len(v)==len(data["conditions"]) for v,data in zip(keys,groups))


def run(seed,checkpoint_root,output,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/"research/dev043/protocol.json").read_text())
    assert seed in protocol["seeds"]
    output=Path(output)
    if output.exists():raise ValueError("refuse overwrite")
    sha_source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip():
        raise ValueError("frozen git source required")
    output.mkdir(parents=True)
    saved=json.loads((Path(checkpoint_root)/f"DEV041-RAW-seed{seed}-development"/"report.json").read_text())
    assert saved["source"]==protocol["sources"]["checkpoint_training_sha"]
    result=dict(study="DEV-043",seed=seed,source=sha_source,
                protocol_sha256=sha(ROOT/"research/dev043/protocol.json"),
                source_checkpoint_sha=saved["source"],smoke=smoke,arms={},
                status="ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY")
    for arm in ARMS:
        path=Path(checkpoint_root)/f"DEV041-RAW-seed{seed}-development"/(arm+".pt")
        assert sha(path)==saved["arms"][arm]["checkpoint_sha256"]
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        model=build(arm,seed);model.load_state_dict(checkpoint["model"])
        initial={k:v.clone() for k,v in model.state_dict().items()}
        train=capture_split(model,"head-train")
        calibration=capture_split(model,"calibration")
        evaluation=capture_split(model,"evaluation")
        disjoint([train["original"],train["order"],calibration,evaluation]) # original/order distinct public streams
        interventions={}
        steps=2 if smoke else protocol["head_optimizer"]["updates"]
        # Existing original head (no fit), calibration alone on fresh order domain
        original_logits=calibration["logits"]
        selected=select_method_and_threshold(original_logits,calibration)
        variants={"original_head":dict(head=None,params=dict(scales=[1.]*3,bias=[0.]*3),
                                        method="identity",threshold=saved["arms"][arm]["results"]["threshold"])}
        variants["calibration_only"]=dict(head=None,params=selected["calibrators"][selected["method"]],
                                           method=selected["method"],threshold=selected["threshold"])
        for fraction in FRACTIONS:
            data=mix_train(train,fraction)
            head,loss=fit_head(model.head,data,seed+fraction,steps,
                               protocol["head_optimizer"]["lr"])
            fitcal=select_method_and_threshold(head(calibration["x"]).detach(),calibration)
            variants["refit_"+str(fraction)]=dict(head=head,
                params=fitcal["calibrators"][fitcal["method"]],
                method=fitcal["method"],threshold=fitcal["threshold"],train_loss=loss,
                head_state={k:v.detach().cpu() for k,v in head.state_dict().items()})
        for key,v in variants.items():
            with torch.no_grad():
                logits=evaluation["logits"] if v["head"] is None else v["head"](evaluation["x"])
                probabilities=calibrated(logits,v["params"])
            rows=as_rows(evaluation,probabilities)
            interventions[key]=dict(metrics=metrics(rows),physical=action(rows,v["threshold"]),
                      retrospective_oracle=retrospective_pairwise_oracle(rows),
                      method=v["method"],threshold=v["threshold"],
                      probabilities=[r["probabilities"] for r in rows],
                      calibration_params=v["params"])
            if "train_loss" in v:interventions[key]["train_loss"]=v["train_loss"]
        if any(not torch.equal(initial[k],v) for k,v in model.state_dict().items()):
            raise AssertionError("frozen encoder+writer+read adapter/head changed")
        torch.save(dict(source_checkpoint_sha=sha(path),source_checkpoint=checkpoint["model"],
            head_refits={k:v["head_state"] for k,v in variants.items() if "head_state" in v},
            calibration={k:dict(params=v["params"],threshold=v["threshold"]) for k,v in variants.items()}),
            output/(arm+"-heads.pt"))
        result["arms"][arm]=dict(checkpoint_sha256=sha(path),head_evidence_sha256=sha(output/(arm+"-heads.pt")),
                                variants=interventions)
        print(json.dumps(dict(seed=seed,arm=arm,
          ranking={k:round(v["metrics"]["pairwise_ranking"],4) for k,v in interventions.items()},
          cost={k:round(v["physical"]["cost_per_entity"],4) for k,v in interventions.items()})),flush=True)
    result["reference"]=reference(evaluation)
    result["evaluation_fingerprints"]=[v["fingerprint"] for v in evaluation["conditions"]]
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--checkpoints",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--smoke",action="store_true")
    v=parser.parse_args();run(v.seed,v.checkpoints,v.output,v.smoke)
