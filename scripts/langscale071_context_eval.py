#!/usr/bin/env python3
"""LANGSCALE071: no-fit original-checkpoint context-length scaling, 3 size tiers x 3 seeds.

Fixed authentic exposed DEV target BYTE POSITIONS for all lengths.
Checks a frozen 160-update LANGSCALE070 checkpoint, publisher source,
same owner and target; scores input lengths 16/32/64/128/256 with each
prefix reset. This DOES NOT verify persistent streaming or >128 training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),
             str(ROOT/"research/mathematical-engineering/langbase024"),
             str(ROOT/"research/mathematical-engineering/langtrain025")]
from byte_next_reference import original_source
from train_byte_native import windows as original_dev_windows,score_model
from scripts.langscale070_cpu_fit import model_for,model_source_preflight,COUNTS,STEPS
from scripts.langlayer035_sampling import fit_sources

LENGTHS=(16,32,64,128,256)
TARGETS_PER_DOCUMENT=64
SCALE_ORDER=("p250k","p500k","p1000k")
SEEDS=(1701,1702,1703)
PREVIOUS_STUDY="LANGSCALE070"


def target_cohort(dev:dict[str,bytes]):
    if len(dev)!=4:raise ValueError("wrong original DEV owner count")
    cohort=[]
    for owner,raw in sorted(dev.items()):
        if len(raw)<512:raise ValueError("original DEV source too short")
        indices=np.linspace(256,len(raw)-1,TARGETS_PER_DOCUMENT,dtype=np.int64)
        if len(set(map(int,indices)))!=TARGETS_PER_DOCUMENT:
            raise ValueError("duplicate source target coordinate")
        for t in indices:
            t=int(t)
            if not 256<=t<len(raw):raise ValueError("bad original source index")
            cohort.append((owner,t,raw[t]))
    if len(cohort)!=256:raise ValueError("target cohort not exactly 256")
    h=hashlib.sha256()
    for owner,t,y in cohort:
        h.update(owner.encode()+b":" + str(t).encode()+b":" + bytes([y])+b";")
    return cohort,h.hexdigest()


def load_original_model(artifact_dir:Path,aggregate_file:Path,tier:str,seed:int):
    if tier not in SCALE_ORDER or seed not in SEEDS:
        raise ValueError("only nine preregistered original checkpoint arms")
    summary=json.loads(aggregate_file.read_text())
    if summary.get("study")!=PREVIOUS_STUDY or summary.get("fitted_checkpoints")!=9 or summary.get("all_source_schedules_paired") is not True:
        raise ValueError("original full scaling source aggregate unavailable/invalid")
    expected=summary["metrics"][tier]["raw_by_seed"][str(seed)]
    files=list(artifact_dir.rglob("*.pt"))
    if len(files)!=1:raise ValueError("expected exactly one trained original checkpoint")
    cp=files[0]
    raw=cp.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected["checkpoint_sha256"]:
        raise ValueError("original trained checkpoint SHA mismatch")
    ck=torch.load(cp,map_location="cpu",weights_only=True)
    if (ck.get("study"),ck.get("arm"),ck.get("seed"),ck.get("updates"))!=(PREVIOUS_STUDY,tier,seed,STEPS):
        raise ValueError("wrong original fitted model/arm/steps")
    if ck.get("parameters")!={"trainable":COUNTS[tier],"deployed":COUNTS[tier]}:
        raise ValueError("original active model geometry mismatch")
    if ck.get("sample_order_sha256")!=expected["sample_order_sha256"]:
        raise ValueError("checkpoint original source schedule mismatch")
    if any(key not in ck for key in ("optimizer","rng_torch","rng_numpy","rng_python","model")):
        raise ValueError("original replay evidence missing")
    model_source_preflight()
    m=model_for(tier)
    assert m.parameter_count()==COUNTS[tier]
    m.load_state_dict(ck["model"],strict=True)
    m.eval()
    return m,{"checkpoint_sha256":expected["checkpoint_sha256"],
              "sample_order_sha256":expected["sample_order_sha256"],
              "original_DEV_NLL":expected["nll"]}


@torch.inference_mode()
def score_length(model,dev,cohort,length):
    owner_rows={}
    total={"n":0,"nll":0.,"brier":0.,"correct":0}
    input_hash=hashlib.sha256()
    cpu_start=time.process_time()
    wall_start=time.perf_counter()
    for owner in sorted(dev):
        items=[item for item in cohort if item[0]==owner]
        if len(items)!=TARGETS_PER_DOCUMENT:
            raise ValueError("lost owner in target cohort")
        aggregate={"n":0,"nll":0.,"brier":0.,"correct":0}
        raw=dev[owner]
        for start in range(0,len(items),8):
            block=items[start:start+8]
            inputs=[]
            labels=[]
            for _,t,y in block:
                prefix=raw[t-length:t]
                if len(prefix)!=length or y!=raw[t]:
                    raise ValueError("source prefix/label/mask crossover")
                inputs.append(list(prefix))
                labels.append(y)
                input_hash.update(owner.encode()+b":" + str(t).encode()+b":"+
                                  prefix+bytes([y])+b";")
            x=torch.tensor(inputs,dtype=torch.long)
            target=torch.tensor(labels,dtype=torch.long)
            logits=model(x)
            if logits.shape!=(len(block),length,256) or not torch.isfinite(logits).all():
                raise ValueError("model output shape, causal source, or finite score mismatch")
            z=logits[:,-1,:].double()
            lp=F.log_softmax(z,dim=-1)
            p=lp.exp()
            nll=-lp.gather(-1,target[:,None]).squeeze(-1)
            brier=1-2*p.gather(-1,target[:,None]).squeeze(-1)+p.square().sum(-1)
            aggregate["n"]+=len(block)
            aggregate["nll"]+=float(nll.sum())
            aggregate["brier"]+=float(brier.sum())
            aggregate["correct"]+=int((z.argmax(-1)==target).sum())
        owner_rows[owner]={"n":aggregate["n"],
                           "nll":aggregate["nll"]/aggregate["n"],
                           "accuracy":aggregate["correct"]/aggregate["n"]}
        for key in total:total[key]+=aggregate[key]
    if total["n"]!=256:
        raise ValueError("not all exact same original target positions evaluated")
    return {"context":length,"n":total["n"],"nll":total["nll"]/total["n"],
            "top1":total["correct"]/total["n"],"brier":total["brier"]/total["n"],
            "cpu_process_seconds":time.process_time()-cpu_start,
            "wall_seconds":time.perf_counter()-wall_start,
            "inputs_and_targets_sha256":input_hash.hexdigest(),
            "per_document":owner_rows}


def run(tier,seed,original_dir,aggregate):
    if torch.cuda.is_initialized():raise ValueError("registered CPU only")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fit=fit_sources()  # SHA audit of original publisher FIT+supplement and owners
    _,dev,_=original_source(ROOT)
    if set(fit)&set(dev):raise ValueError("fit/dev overlap")
    cohort,target_sha=target_cohort(dev)
    model,receipt=load_original_model(original_dir,aggregate,tier,seed)
    original_replay=score_model(model,original_dev_windows(dev))["aggregate"]
    if original_replay["n"]!=4096 or abs(original_replay["nll"]-receipt["original_DEV_NLL"])>1e-6:
        raise ValueError("original trained 070 exposed DEV score cold replay failed")
    observations=[score_length(model,dev,cohort,length) for length in LENGTHS]
    if {row["context"] for row in observations}!=set(LENGTHS) or any(row["n"]!=256 for row in observations):
        raise ValueError("invalid same-target length comparison")
    return {"study":"LANGSCALE071","status":"NO_FIT_ORIGINAL_CKPT_EXPOSED_DEV_CONTEXT_RESET_ONLY",
            "tier":tier,"seed":seed,"parameters":COUNTS[tier],
            "trained_checkpoint":receipt,"original_full_DEV_replay_nll":original_replay["nll"],
            "target_cohort_sha256":target_sha,
            "context_target_count":len(cohort),
            "source_owners":tuple(sorted(dev)),
            "lengths":observations,
            "gpu_minutes":0,"training_updates":0,
            "protected_TEST":"NOT_ACCESSED",
            "independently_qualified":False,
            "limits":["all input contexts reset, not persistent decoding or source history carryover",
                      "256-context arm exceeds 128-byte trained input horizon",
                      "sample of 256 already exposed DEV targets, NOT independent TEST",
                      "same original fixed target indices across all lengths/scales",
                      "no new model training, no parameter or training-FLOP matched architecture comparison",
                      "full mathematically source-reconciled streaming/AD/IEEE remains OPEN"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tier",choices=SCALE_ORDER,required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--original-aggregate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("preserve original audit evidence")
    result=run(a.tier,a.seed,a.checkpoint_dir,a.original_aggregate)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("ORIGINAL_CONTEXT_EVAL "+json.dumps({"tier":a.tier,"seed":a.seed,
        "nll_by_context":{x["context"]:x["nll"] for x in result["lengths"]},
        "target_cohort_sha256":result["target_cohort_sha256"]},sort_keys=True),flush=True)


if __name__=="__main__":main()
