"""LATFUSE009 frozen authentic-event at-write versus at-read loss gradient.

SHA-valid original LATFUSE008 two-seed paired trained 99,390-param models.
True May2014 original high/ordinary observed source event labels queried
after L=16/64/128/512 other observed source steps; only AFTER original
historic event prediction use recorded PM2.5 event binary label for BCE.
No model optimizer or new learned weights. Input state at event is a
detached DIFFERENTIABLE leaf, so gradient to original event-time MATRIX
tests suffix credit assignment/transport, NOT to actual writer parameters.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from scripts import latfuse002_real_fit as source
from scripts import latfuse008_paired_real_training as fit
from scripts.validate_research_parity import check_manifest

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse009/REGISTERED_READONLY_GRADIENT.json"
OUT=ROOT/"research/mathematical-engineering/latfuse009/RESULTS.json"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def detached_state_leaf(s):
    return s._replace(fast=s.fast.detach().requires_grad_(True),
       slow=s.slow.detach().requires_grad_(True),
       associative=s.associative.detach().requires_grad_(True),
       entities=s.entities.detach())

def batch_probe(m,d,pairs,lag):
    b=len(pairs)
    source_id=torch.tensor([r[0] for r in pairs],dtype=torch.long)
    anchor=torch.tensor([r[1] for r in pairs],dtype=torch.long)
    with torch.no_grad():
        st=m.initial(b)
        for off in range(-23,1):
            t=anchor+off
            st=m.step(st,d["values"][source_id,t],d["valid"][source_id,t],
                  d["age"][source_id,t],d["last"][source_id,t],source_id)
    event=detached_state_leaf(st)
    now=event
    for off in range(1,lag+1):
        t=anchor+off
        now=m.step(now,d["values"][source_id,t],d["valid"][source_id,t],
                  d["age"][source_id,t],d["last"][source_id,t],source_id)
    past_time=source.calendar([int(t) for t in anchor])[:,None,:]
    logits=m.historic_event_logits(now,source_id,past_time)
    # Source-truth PM2.5 target read ONLY after final query prediction.
    real=d["raw"][source_id,anchor,0]
    if not all(torch.isfinite(real[i]) and
        (float(real[i])>=150 if row[2]==1 else float(real[i])<=35)
            for i,row in enumerate(pairs)):
        raise RuntimeError("physically recorded original event truth corrupted")
    label=(real>=150).float()
    loss=F.binary_cross_entropy_with_logits(logits[:,0],label)
    fields=(event.associative,event.fast,event.slow,
            now.associative,now.fast,now.slow)
    derivatives=torch.autograd.grad(loss,fields,allow_unused=True)
    names=("event_associative","event_fast","event_slow",
           "query_associative","query_fast","query_slow")
    metrics={}
    for name,g in zip(names,derivatives):
        if g is None:
            metrics[name]=dict(connected=False,mean_l2=0.0,
                               source_cases_with_nonzero_derivative=0)
        else:
            if not bool(torch.isfinite(g).all()):
                raise RuntimeError("original source-gradient nonfinite "+name)
            norms=g.detach().reshape(b,-1).norm(dim=1)
            metrics[name]=dict(connected=True,mean_l2=float(norms.mean()),
                               max_l2=float(norms.max()),
                               source_cases_with_nonzero_derivative=int((norms>1e-12).sum()))
    for root in ("associative","fast","slow"):
        e=metrics["event_"+root]["mean_l2"]
        q=metrics["query_"+root]["mean_l2"]
        metrics[root+"_event_over_query_gradient_ratio"]=(e/q if q>1e-30 else None)
    return dict(delay=lag,original_real_physical_source_cases=b,
                true_event_BCE_at_frozen_checkpoint=float(loss.detach()),
                measured_gradient_norms=metrics,
                sampled_true_source_labels=[int(v) for v in label.tolist()])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise RuntimeError("never overwrite original read-only gradient report")
    torch.set_num_threads(1)
    reg=json.loads(REG.read_text())
    check_manifest(reg,"preflight")
    if reg["study_id"]!="LATFUSE009-REAL-EVENT-WRITE-GRADIENT-READONLY":
        raise RuntimeError("original source study type mismatch")
    if sha(args.source)!=source.SOURCE_SHA:
        raise RuntimeError("original licensed physical source mismatch")
    d=fit.source_observations(args.source)
    all_pairs=fit.select_pairs(d,fit.EVAL_START,fit.EVAL_STOP,2)
    selected=all_pairs[:8]
    report=json.loads((fit.OUT/"RESULTS.json").read_text())
    import hashlib
    cohort_sha=hashlib.sha256(json.dumps(all_pairs,separators=(",",":")).encode()).hexdigest()
    if cohort_sha!=report["source_eval_cohort_sha256"] or len(selected)!=8:
        raise RuntimeError("original genuine time cohort changed")
    out=[]
    for seed in fit.SEEDS:
        for arm,cls in fit.ARMS.items():
            rec=next((x for x in report["model_results"] if
                     x["seed"]==seed and x["arm"]==arm),None)
            if rec is None:raise RuntimeError("original fitted model missing")
            path=fit.OUT/f"seed-{seed}"/arm/"final.pt"
            digest=sha(path)
            if digest!=rec["original_checkpoint_sha256"]:
                raise RuntimeError("trained original checkpoint SHA changed")
            cp=torch.load(path,map_location="cpu",weights_only=False)
            m=fit.create_model(seed,arm)
            m.load_state_dict(cp["original_model"],strict=True)
            m.eval()
            originals=[p.detach().clone() for p in m.parameters()]
            # True source gradients requested exclusively wrt event-time
            # state; model weights are intentionally frozen (no optimizer).
            for p in m.parameters():
                p.requires_grad_(False)
            scores={}
            for h in fit.LAGS:
                scores[str(h)]=batch_probe(m,d,selected,h)
            if sha(path)!=digest or any(not torch.equal(t,v) for t,v in
                    zip(originals,m.parameters())):
                raise RuntimeError("trained checkpoint changed during read-only test")
            row=dict(seed=seed,arm=arm,original_checkpoint_sha256=digest,
                     real_source_gradient_at_original_event_delays=scores)
            out.append(row)
            print("LATFUSE009_REAL_EVENT_GRADIENT="+json.dumps({
                 "seed":seed,"arm":arm,
                 "assoc_grad_ratios":{h:s["measured_gradient_norms"][
                   "associative_event_over_query_gradient_ratio"]
                       for h,s in scores.items()}},sort_keys=True),flush=True)
    result=dict(study_id="LATFUSE009-REAL-EVENT-WRITE-GRADIENT-READONLY",
       original_true_source_archive_sha256=source.SOURCE_SHA,
       original_frozen_models=4,original_phy_source_matched_pairs=len(selected)//2,
       original_internal_source_cohort_sha256=cohort_sha,
       original_horizon_hours=list(fit.LAGS),
       no_original_model_parameter_updates_or_checkpoint_mutation=True,
       interpretation="Event-time STATE gradient from frozen original model historical event BCE versus query-time state gradient. It is not gradient into trained writer input at event; requires more mechanistic work.",
       original_source_July_2015_or_protected_numeric_unopened=True,
       no_architecture_performance_superiority_claim=True,models=out)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE009_ORIGINAL_SOURCE_GRADIENT_PASS="+json.dumps({
        "original_models":len(out),"original_event_pairs":len(selected)//2,
        "new_optimizer_steps":0},sort_keys=True))
if __name__=="__main__":main()
