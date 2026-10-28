#!/usr/bin/env python3
"""LANGLAYER036: non-training, source-verified checkpoint intervention probes.

Never change stored weights/checkpoints or make independent capability claims.
Only original FIT-exposed and previously exposed DEV source bytes are scored.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
for rel in ("research/mathematical-engineering/context026",
            "research/mathematical-engineering/langlayer034",
            "research/mathematical-engineering/langtrain025",
            "research/mathematical-engineering/langbase024"):
    sys.path.insert(0,str(ROOT/rel))
from context_transition import ContextTransition026
from layered_transition import LayeredContextTAC034
from train_byte_native import windows as dev_windows
from byte_next_reference import original_source
from scripts.langlayer035_sampling import fit_sources,iter_batches

ARMS={"tac_context026":500122,"tac_layer2":499725,"tac_layer3":499934}
SEEDS=(1701,1702,1703,1704,1705)
LESIONS=("block_bypass","no_recurrence","no_concept_projection",
         "no_concept_history","uniform_concept_graph")

def model_for(arm):
    if arm=="tac_context026":return ContextTransition026()
    return LayeredContextTAC034(depth=2 if arm=="tac_layer2" else 3)

def nll(model,batches):
    total=0.; count=0
    with torch.no_grad():
        for batch in batches:
            logits=model(batch[:,:-1]).float()
            targets=batch[:,1:]
            if logits.shape!=(8,128,256):raise ValueError("scoring shape changed")
            total+=float(F.cross_entropy(logits.reshape(-1,256),
                        targets.reshape(-1),reduction="sum"))
            count+=targets.numel()
    if count!=4096:raise ValueError("expected 4096 original targets")
    return total/count

def original_batches(seed):
    fit=fit_sources()
    train=[]
    for _,obs in iter_batches(fit,seed,steps=4):
        train.append(torch.tensor([list(raw) for _,_,raw in obs],dtype=torch.long))
    _,dev,_=original_source(ROOT)
    exposed=[torch.tensor([list(raw) for _,raw in blocks],dtype=torch.long)
              for _,blocks in sorted(dev_windows(dev).items())]
    if len(train)!=4 or len(exposed)!=4 or any(x.shape!=(8,129) for x in train+exposed):
        raise ValueError("source/window geometry changed")
    return train,exposed

def rms(x):
    return float(x.detach().float().square().mean().sqrt())

def representations(model,batches):
    results={}
    hooks=[]
    for i in range(1,len(model.layernorm)):
        results[str(i)]=[]
        def observe(module,args,out,idx=i):
            x=args[0].detach()
            y=out.detach()
            delta=y-x
            results[str(idx)].append({
                "ratio_residual_to_input_rms":rms(delta)/(rms(x)+1e-12),
                "cosine_input_to_output":float(F.cosine_similarity(
                    x.float(),y.float(),dim=-1).mean()),
                "gate":float(module.residual_gain.detach().sigmoid())})
        hooks.append(model.layernorm[i].register_forward_hook(observe))
    try:nll(model,batches)
    finally:
        for h in hooks:h.remove()
    return {i:{k:sum(row[k] for row in arr)/len(arr)
               for k in ("ratio_residual_to_input_rms","cosine_input_to_output","gate")}
            for i,arr in results.items()}

def ablate(model,blocks,index,kind):
    layer=model.layernorm[index]
    with ExitStack() as stack:
        if kind=="block_bypass":
            h=layer.register_forward_hook(lambda mod,args,out:args[0])
        elif kind=="no_recurrence":
            h=layer.read_gate.register_forward_hook(
                lambda mod,args,out:torch.full_like(out,-80.0))
        elif kind=="no_concept_projection":
            h=layer.concept_to_state.register_forward_hook(
                lambda mod,args,out:torch.zeros_like(out))
        elif kind=="no_concept_history":
            h=layer.concept_gate.register_forward_hook(
                lambda mod,args,out:torch.full_like(out,-80.0))
        elif kind=="uniform_concept_graph":
            h=layer.concept_q.register_forward_hook(
                lambda mod,args,out:torch.zeros_like(out))
        else:raise ValueError("unknown preregistered perturbation")
        stack.callback(h.remove)
        return nll(model,blocks)

def gradients(model,training_batch):
    saved={}
    hooks=[]
    for idx in range(1,len(model.layernorm)):
        def observe(mod,args,out,index=idx):
            out.retain_grad()
            saved[index]=out
        hooks.append(model.layernorm[idx].register_forward_hook(observe))
    model.zero_grad(set_to_none=True)
    try:
        logits=model(training_batch[:,:-1])
        F.cross_entropy(logits.float().reshape(-1,256),
                        training_batch[:,1:].reshape(-1)).backward()
        summary={}
        for idx in range(1,len(model.layernorm)):
            if saved[idx].grad is None:raise ValueError("absent layer gradient")
            metrics={"layer_output_gradient_rms":rms(saved[idx].grad)}
            for name,parameter in model.layernorm[idx].named_parameters():
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise ValueError(f"missing/nonfinite {idx}.{name} gradient")
                metrics[name+"_gradient_rms"]=rms(parameter.grad)
            summary[str(idx)]=metrics
        return summary
    finally:
        model.zero_grad(set_to_none=True)
        for h in hooks:h.remove()

def diagnose(arm,seed,checkpoint,receipt):
    torch.set_num_threads(2)
    r=json.loads(Path(receipt).read_text())
    if r.get("arm")!=arm or r.get("seed")!=seed or r.get("updates")!=320:
        raise ValueError("wrong original receipt")
    raw=Path(checkpoint).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=r["checkpoint_sha256"]:raise ValueError("checkpoint SHA changed")
    c=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    if (c["study"],c["arm"],c["seed"],c["updates"])!=("LANGLAYER035",arm,seed,320):
        raise ValueError("trained identity changed")
    if c["sample_order_sha256"]!=r["source_sample_order_sha256"]:
        raise ValueError("sample identity changed")
    model=model_for(arm).eval()
    if sum(p.numel() for p in model.parameters())!=ARMS[arm]:
        raise ValueError("parameter count changed")
    model.load_state_dict(c["model"],strict=True)
    seen,dev=original_batches(seed)
    score=nll(model,dev)
    if abs(score-r["DEV_EXPOSED_4096_ONLY"]["nll"])>1e-5:
        raise ValueError("frozen DEV NLL could not be independently replayed")
    fit_score=nll(model,seen)
    out={"study":"LANGLAYER036","arm":arm,"seed":seed,
         "original_checkpoint_sha256":digest,
         "sample_order_sha256":c["sample_order_sha256"],
         "trained_updates_original":320,"new_optimizer_updates":0,
         "fit_seen_nll":fit_score,"exposed_dev_nll":score,
         "dev_minus_seen_fit_nll":score-fit_score,
         "fit_seen_targets":4096,"exposed_dev_targets":4096,
         "protected_TEST":"NOT_ACCESSED",
         "train_flop_parity":"NOT_VERIFIED",
         "lesion_scope":"frozen checkpoint functional intervention",
         "layers":{},"gradient":{}}
    if arm!="tac_context026":
        summary=representations(model,dev)
        for i in range(1,len(model.layernorm)):
            obs=dict(summary[str(i)])
            obs["lesion_delta_dev_nll"]={
                name:ablate(model,dev,i,name)-score for name in LESIONS}
            out["layers"][str(i)]=obs
        out["gradient"]=gradients(model,seen[0])
    if any(not math.isfinite(v) for v in (fit_score,score)):raise ValueError("nonfinite")
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=tuple(ARMS),required=True)
    p.add_argument("--seed",choices=SEEDS,type=int,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=diagnose(a.arm,a.seed,a.checkpoint,a.receipt)
    if a.output.exists():raise FileExistsError("refuse overwrite")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result,sort_keys=True,allow_nan=False))
if __name__=="__main__":main()
