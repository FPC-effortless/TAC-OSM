#!/usr/bin/env python3
"""TAOTUNE004: source-authenticated optimizer-specific LR search, no blind claim.

The FIT phase never loads DEV documents. Only a separate checkout after
original model/optimizer/RNG checkpoint preservation may select LR using the
two preregistered, previously exposed tuning DEV document owners.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),str(ROOT/"src"),
    str(ROOT/"research/mathematical-engineering/langmem086"),
    str(ROOT/"research/mathematical-engineering/langbase024"),
]
import numpy as np
import torch
from torch.nn import functional as F
from compact_tac import CompactDualClippedTAC086
from scripts import langmem086_run as original
from scripts.tao003_pilot import independent_numeric_replay
from scripts.tao002_prefit import ARMS, ROUTE, SOURCE_SHA, MODEL_SHA, SIZE
from scripts.validate_research_parity import check_manifest
from tac_osm.optimizer_tao import build_tao
from byte_next_reference import read_sha_source

REG = ROOT/"research/mathematical-engineering/taotune004"
MANIFEST=REG/"FAIR_COMPARISON.json"
SEED=1701
STEPS=32
BATCH=4
LENGTH=256
TARGETS=STEPS*BATCH*LENGTH
WD=.01
ADAM_BACKBONE_LR=.001

def sha(bytes_): return hashlib.sha256(bytes_).hexdigest()
def weight_hash(model):return original.weight_hash(model)

def registered():
    cfg=json.loads(MANIFEST.read_text())
    result=check_manifest(cfg,"preflight")
    if result["status"]!="PASS_PREFIT_SINGLE_MODEL" or cfg["study_id"]!="TAOTUNE004":
        raise ValueError("original preregistration failed")
    expected={(mode, i) for mode in ARMS for i in range(3)}
    actual={(name.rsplit("_lr",1)[0],int(name.rsplit("_lr",1)[1]))
            for name in (m["name"] for m in cfg["models"])}
    if expected!=actual or len(cfg["models"])!=15:
        raise ValueError("frozen 15-trial treatment missing")
    if cfg["budget"]["trial_steps"]!=STEPS or cfg["budget"]["targets_per_trial"]!=TARGETS:
        raise ValueError("frozen tokens budget changed")
    if tuple(cfg["tuning_cohort"])!=("Doc/tutorial/controlflow.rst","Doc/tutorial/index.rst"):
        raise ValueError("tuning cohort changed")
    if tuple(cfg["reporting_cohort"])!=("Doc/tutorial/introduction.rst","Doc/tutorial/venv.rst"):
        raise ValueError("reporting cohort changed")
    if sha((ROOT/"research/mathematical-engineering/langmem086/compact_tac.py").read_bytes())!=MODEL_SHA:
        raise ValueError("original model graph hash changed")
    if cfg["real_world_source"]["sha256"]!=SOURCE_SHA:
        raise ValueError("original FIT registration changed")
    for mode in ARMS:
        values=cfg["learning_rate_grid"][mode]
        if len(values)!=3 or len(set(values))!=3 or any(x<=0 for x in values):
            raise ValueError("all arms require three preregistered positive LR candidates")
    return cfg

def optimizer_for(mode,model,lr):
    if mode=="adamw":
        return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=WD,foreach=False)
    opt=build_tao(model,orthogonal_names=ROUTE,mode=mode,lr=ADAM_BACKBONE_LR,weight_decay=WD)
    og=[g for g in opt.param_groups if g["kind"]=="orthogonal"]
    ag=[g for g in opt.param_groups if g["kind"]=="adamw"]
    if len(og)!=1 or len(ag)!=1 or len(og[0]["params"])!=1 or og[0]["params"][0] is not model.ffn_up.weight:
        raise ValueError("unsafe orthogonal group assignment")
    og[0]["lr"]=float(lr)
    if ag[0]["lr"]!=ADAM_BACKBONE_LR:
        raise ValueError("AdamW fallback LR drift")
    return opt

def fit_original_batches():
    docs,source=original.fit_sources086()
    if source!=SOURCE_SHA or len(docs)!=14:
        raise ValueError("original licensed publisher FIT source differs")
    receipt=json.loads((ROOT/"research/mathematical-engineering/tao002/ORIGINAL_PREFIT_RECEIPT.json").read_text())
    expected=next(x for x in receipt["samples"]["frozen_sample_schedules"] if x["seed"]==SEED)
    if original.schedule(docs,SEED)!=expected:
        raise ValueError("full original source sample plan differs")
    rows=list(itertools.islice(original.batches(docs,SEED),STEPS))
    order=hashlib.sha256();data=hashlib.sha256()
    for _,row in rows:
        for owner,start,raw in row:
            order.update(f"{owner}:{start};".encode())
            data.update(raw)
    plan={"seed":SEED,"steps":STEPS,"batch":BATCH,"context":LENGTH,"targets":TARGETS,
          "sample_order_sha256":order.hexdigest(),"sample_bytes_sha256":data.hexdigest()}
    return docs,rows,plan

def fit(out:Path):
    if out.exists():raise FileExistsError("refuse to overwrite original trained optimizer trials")
    if torch.cuda.is_available():raise ValueError("registered CPU-only trial expected no GPU")
    torch.set_num_threads(2)
    cfg=registered()
    docs,rows,schedule=fit_original_batches()
    initial=set();summary={}
    fixture=torch.tensor([list(next(iter(docs.values()))[:16])],dtype=torch.long)
    for mode in ARMS:
        summary[mode]={}
        for i,lr in enumerate(cfg["learning_rate_grid"][mode]):
            torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
            model=CompactDualClippedTAC086()
            if model.counts()["trainable"]!=SIZE or model.counts()["deployed"]!=SIZE:
                raise ValueError("actual parameter count mismatch")
            initial_hash=weight_hash(model);initial.add(initial_hash)
            opt=optimizer_for(mode,model,lr)
            altered=fixture.clone();altered[:,8:]=(altered[:,8:]+113)%256
            with torch.no_grad():
                clean,_=model(fixture,owners=("source-FIT-poison",))
                wrong,_=model(altered,owners=("source-FIT-poison",))
            if not torch.equal(clean[:,:8],wrong[:,:8]):
                raise ValueError("original model future suffix poison failed")
            losses=[];t0=time.perf_counter();opt_seconds=0.
            for step,row in rows:
                x=torch.tensor([list(raw[:-1]) for _,_,raw in row],dtype=torch.long)
                y=torch.tensor([list(raw[1:]) for _,_,raw in row],dtype=torch.long)
                opt.zero_grad(set_to_none=True)
                logits,_=model(x,owners=tuple(f"FIT-owner-{j}" for j in range(BATCH)))
                loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
                if not bool(torch.isfinite(loss)):raise ValueError("nonfinite publisher FIT loss")
                loss.backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError("missing/nonfinite genuine-data gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                a=time.perf_counter();opt.step();opt_seconds+=time.perf_counter()-a
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise ValueError("nonfinite genuine-data fitted weights")
                losses.append(float(loss.detach()))
            wall=time.perf_counter()-t0
            if weight_hash(model)==initial_hash or len(losses)!=STEPS:
                raise ValueError("untrained original model not acceptable")
            if any(int(s["step"])!=STEPS for s in opt.state.values()):
                raise ValueError("full original optimizer moments not stepped")
            with torch.no_grad():
                cold_logits,_=model(fixture,owners=("cold-check",))
            folder=out/mode/f"lr{i}"
            folder.mkdir(parents=True,exist_ok=False)
            ns=np.random.get_state()
            cp={"study":"TAOTUNE004","mode":mode,"lr_index":i,"lr":lr,"seed":SEED,
                "steps":STEPS,"model":model.state_dict(),"optimizer":opt.state_dict(),
                "torch_rng":torch.get_rng_state(),"python_rng":random.getstate(),
                "numpy_rng":(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
                "initial_weight_hash":initial_hash,"trained_weight_hash":weight_hash(model),
                "model_source_sha256":MODEL_SHA,"fit_source_sha256":SOURCE_SHA,
                "sample_receipt":schedule,"losses":losses,
                "cold_fixture":fixture,"cold_logits":cold_logits}
            path=folder/"MODEL.pt";torch.save(cp,path)
            cp_sha=sha(path.read_bytes())
            restored=torch.load(path,map_location="cpu",weights_only=True)
            reloaded=CompactDualClippedTAC086().eval();reloaded.load_state_dict(restored["model"],strict=True)
            cold_opt=optimizer_for(mode,reloaded,lr);cold_opt.load_state_dict(restored["optimizer"])
            with torch.no_grad():
                check,_=reloaded(fixture,owners=("cold-check",))
            if weight_hash(reloaded)!=cp["trained_weight_hash"] or not torch.equal(check,restored["cold_logits"]):
                raise ValueError("same-host full trained checkpoint reload differs")
            if any(int(s["step"])!=STEPS for s in cold_opt.state.values()):
                raise ValueError("cold optimizer moment replay failed")
            info={"mode":mode,"lr":lr,"lr_index":i,"seed":SEED,"trained_steps":STEPS,"targets_seen":TARGETS,
                  "source_sha256":SOURCE_SHA,"training_schedule":schedule,
                  "checkpoint_sha256":cp_sha,"trainable_parameters":SIZE,"deployed_parameters":SIZE,
                  "initial_weight_hash":initial_hash,"trained_weight_hash":cp["trained_weight_hash"],
                  "cold_model_exact":True,"cold_optimizer_steps":STEPS,
                  "full_loop_wall_seconds":wall,"optimizer_wall_seconds":opt_seconds,
                  "rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  "first_fit_nll":losses[0],"last_fit_nll":losses[-1],"fit_loss_trace":losses,
                  "real_FIT_poison":"PASS","dev_access_before_original_git_seal":False}
            (folder/"FIT_RECEIPT.json").write_text(json.dumps(info,indent=2,sort_keys=True)+"\n")
            summary[mode][str(i)]={"lr":lr,"checkpoint_sha256":cp_sha,"last_fit_nll":losses[-1],
                                  "optimizer_wall_seconds":opt_seconds}
            print(json.dumps({"phase":"SEALED_FIT_ORIGINAL","mode":mode,"lr_index":i,
                              "lr":lr,"fit_nll":losses[-1],"checkpoint_sha256":cp_sha}),flush=True)
    if len(initial)!=1 or len(summary)!=5 or sum(map(len,summary.values()))!=15:
        raise ValueError("original fifteen trial or same-seed initialization matrix incomplete")
    (out/"FIT_MATRIX.json").write_text(json.dumps({
        "study":"TAOTUNE004","tier":"INTERNAL_REAL_FIT_NO_DEV",
        "source_sha256":SOURCE_SHA,"same_seed_initial_weights":next(iter(initial)),
        "sample_receipt":schedule,"matrix":summary,
        "original_checkpoints_complete":True,"dev_access":False,
        "protected_TEST":"NOT_ACCESSED","gpu_minutes":0},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"phase":"ALL_15_ORIGINALS_SEALED","trials":15}),flush=True)

@torch.no_grad()
def tune(out:Path):
    torch.set_num_threads(2)
    cfg=registered()
    m=json.loads((out/"FIT_MATRIX.json").read_text())
    if not m.get("original_checkpoints_complete") or m["dev_access"] is not False:
        raise ValueError("all original models must be in Git before tune-DEV read")
    models={};replays={};final_cfg=json.loads(MANIFEST.read_text())
    for mode in ARMS:
        for i,lr in enumerate(cfg["learning_rate_grid"][mode]):
            folder=out/mode/f"lr{i}"
            if not (folder/"MODEL.pt").exists():
                raise FileNotFoundError("missing original Git-backed trained checkpoint")
            receipt=json.loads((folder/"FIT_RECEIPT.json").read_text())
            path=folder/"MODEL.pt"
            if sha(path.read_bytes())!=receipt["checkpoint_sha256"]:
                raise ValueError("original Git-backed original model SHA differs")
            cp=torch.load(path,map_location="cpu",weights_only=True)
            if (cp["seed"],cp["steps"],cp["mode"],cp["lr_index"],cp["lr"])!=(SEED,STEPS,mode,i,lr):
                raise ValueError("wrong trained optimizer mode/LR")
            if cp["sample_receipt"]!=m["sample_receipt"] or cp["fit_source_sha256"]!=SOURCE_SHA:
                raise ValueError("different training source/order exposed to optimizer")
            model=CompactDualClippedTAC086().eval();model.load_state_dict(cp["model"],strict=True)
            if weight_hash(model)!=cp["trained_weight_hash"]:
                raise ValueError("independently reloaded original weight SHA mismatch")
            opt=optimizer_for(mode,model,lr);opt.load_state_dict(cp["optimizer"])
            if any(int(s["step"])!=STEPS for s in opt.state.values()):
                raise ValueError("independently loaded original optimizer step mismatch")
            logits,_=model(cp["cold_fixture"],owners=("cold-check",))
            replay=independent_numeric_replay(logits,cp["cold_logits"])
            replays[f"{mode}_lr{i}"]=replay
            models[(mode,i)]=model
            row=next(row for row in final_cfg["models"] if row["name"]==f"{mode}_lr{i}")
            row["measured"]={"trainable_parameters":SIZE,"deployed_parameters":SIZE,
                              "checkpoint_sha256":receipt["checkpoint_sha256"]}
    if check_manifest(final_cfg,"final")["status"]!="PASS_SINGLE_MODEL":
        raise ValueError("final original checkpoint/100k gate failed")
    # Do not load even a byte from the within-study REPORTING documents here.
    registry=json.loads((ROOT/"research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json").read_text())
    tune_files=[r for r in registry["files"] if r["path"] in cfg["tuning_cohort"]]
    if len(tune_files)!=2:
        raise ValueError("registered tuning publisher documents not found")
    docs={e["path"]:read_sha_source(ROOT,e["path"],e) for e in tune_files}
    scores={};target_h=hashlib.sha256()
    for owner,raw in sorted(docs.items()):
        target_h.update(owner.encode()+b"\x00"+raw[1:257])
    for mode in ARMS:
        scores[mode]=[]
        for i,lr in enumerate(cfg["learning_rate_grid"][mode]):
            model=models[(mode,i)]
            sums={"nll":0.0,"brier":0.0,"correct":0,"targets":0}
            for owner,raw in sorted(docs.items()):
                x=torch.tensor([list(raw[:256])],dtype=torch.long)
                y=torch.tensor([list(raw[1:257])],dtype=torch.long)
                logits,_=model(x,owners=(owner,))
                lp=logits.log_softmax(-1);probs=lp.exp()
                py=probs.gather(-1,y.unsqueeze(-1)).squeeze(-1)
                sums["nll"]+=float((-lp.gather(-1,y.unsqueeze(-1)).squeeze(-1)).sum())
                sums["brier"]+=float((1+probs.square().sum(-1)-2*py).sum())
                sums["correct"]+=int((logits.argmax(-1)==y).sum())
                sums["targets"]+=256
            if sums["targets"]!=512:raise ValueError("wrong genuine tuning target count")
            scores[mode].append({"lr":lr,"index":i,"nll":sums["nll"]/512,
                                 "accuracy":sums["correct"]/512,"brier":sums["brier"]/512,
                                 "checkpoint_sha256":m["matrix"][mode][str(i)]["checkpoint_sha256"]})
    chosen={mode:min(scores[mode],key=lambda row:(row["nll"],row["lr"],row["index"]))
            for mode in ARMS}
    report={"study":"TAOTUNE004","phase":"EARLY_REAL_SOURCE_LR_SELECTION",
            "tier":"INTERNAL_ALREADY_EXPOSED_TUNE_DEV_NOT_BLIND",
            "selection_criterion":"minimum tuning DEV NLL per mode, lower LR tie break",
            "source_sha256":SOURCE_SHA,"sample_receipt":m["sample_receipt"],
            "tuning_target_sha256":target_h.hexdigest(),"tuning_owners":sorted(docs),
            "num_tuning_targets_per_trial":512,
            "reporting_document_owners_unread_by_tuner":cfg["reporting_cohort"],
            "trial_scores":scores,"selected":chosen,
            "original_15_git_checkpoints_exact_hash":True,
            "independent_original_optimizer_state_replay":True,
            "independent_numeric_logits_replay":replays,
            "all_15_bitwise_replay":all(v["bitwise"] for v in replays.values()),
            "no_architecture_or_independent_superiority_claim":True,
            "first_source_DEV_already_exposed_historically":True,
            "protected_TEST":"NOT_ACCESSED","gpu_minutes":0}
    path=out/"TUNING_RESULTS.json"
    if path.exists():raise FileExistsError("first tune-DEV selection results are immutable")
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"FINAL_COMPARISON.json").write_text(json.dumps(final_cfg,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"phase":"TUNING_SELECTION_COMPLETE","selection":chosen}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("fit","tune"))
    p.add_argument("--directory",type=Path,required=True)
    a=p.parse_args()
    if a.phase=="fit":fit(a.directory)
    else:tune(a.directory)

if __name__=="__main__":main()
