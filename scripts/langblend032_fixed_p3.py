#!/usr/bin/env python3
"""LANGBLEND032: no-fit five-seed fixed original-TAC / FIT-only P3 mixture.

Uses only the original five LANGCOMPARE031 TAC trained checkpoints.
No gradients, optimizer, GPU, protected TEST or λ tuning on exposed DEV.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/"src", ROOT/"research/mathematical-engineering/langbase024",
          ROOT/"research/mathematical-engineering/langtrain025"):
    sys.path.insert(0,str(p))
import torch
from torch.nn import functional as F
from byte_next_reference import original_source,train_count_model,law
from train_byte_native import windows,score_reference
from scripts.langcompare031_cpu_fit import cold_reload

SEEDS=(1701,1702,1703,1704,1705)
LAMBDAS=(0.0,0.10,0.25,0.50,0.75,0.90,0.95,0.99,1.0)
CK={
 1701:"897435d2db494a5de10ee46a159f0050a651c916f67718165e08f894f6c84e5e",
 1702:"2d95b6d9005138b2bca474bda959318f929a24a08bc6f541dc5e48eaeaf4125f",
 1703:"fe8e002a3a0d064b6ce21755593c6ddbad90fb2380fdbf1ae351d795d1b41a20",
 1704:"b7321f781fd9e5e47bda8357e32df5a3c3522bed4bf2da20b061c248f8b4ea5c",
 1705:"fc1496a813e7758dc5ec1bdc06a8b3e4883ab707ac0e14a5f06626d3352f2bf6"}
FROZEN_TAC_NLL={1701:2.1232291969328756,1702:2.0707503656392134,
                1703:2.095373876438341,1704:2.10806429239733,
                1705:2.101159023278905}

def find_checkpoints(folder:Path):
    paths={}
    for seed in SEEDS:
        found=list(folder.rglob(f"tac_context026_seed{seed}_fit320.pt"))
        if len(found)!=1:raise ValueError(f"missing/duplicate original TAC trained checkpoint seed {seed}")
        path=found[0]
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=CK[seed]:
            raise ValueError(f"original TAC checkpoint SHA256 mismatch seed {seed}")
        paths[seed]=path
    return paths

def p3_rows(model, blocks):
    # FIT-only counts and t-exclusive next target must be checked independently.
    vecs=[]
    for start,raw in blocks:
        assert len(raw)==129
        ctx=[]
        for t in range(128):
            p=law(model,raw[max(0,t-2):t+1],3)
            ctx.append(p)
        vecs.append(ctx)
    law_tensor=torch.tensor(vecs,dtype=torch.float64)
    if law_tensor.shape!=(8,128,256):
        raise ValueError("frozen P3 next-byte geometry")
    if not torch.allclose(law_tensor.sum(-1),torch.ones(8,128,dtype=torch.float64),
                          rtol=0.,atol=1e-10):
        raise ValueError("P3 not normalized")
    return law_tensor

def scores(gold:torch.Tensor,probs:torch.Tensor):
    assert probs.shape==(8,128,256) and gold.shape==(8,128)
    if not torch.isfinite(probs).all() or not (probs.min()>0):
        raise ValueError("invalid blend probability")
    if not torch.allclose(probs.sum(-1),torch.ones_like(gold,dtype=torch.float64),
                          rtol=0.,atol=1e-8):
        raise ValueError("probability mixing lost unit normalization")
    observed=probs.gather(-1,gold[...,None]).squeeze(-1)
    n=gold.numel()
    return {"n":n,
            "nll_sum":float(-observed.log().sum()),
            "brier_sum":float((1.-2.*observed+probs.square().sum(-1)).sum()),
            "top1_sum":int((probs.argmax(-1)==gold).sum()),
            "top5_sum":int((probs.topk(5,-1).indices==gold[...,None]).any(-1).sum())}

def run(input_dir:Path,output:Path):
    if output.exists():raise FileExistsError("refuse to overwrite score evidence")
    torch.set_num_threads(2)
    paths=find_checkpoints(input_dir) # before any DEV access
    fit0,dev,supp=original_source(ROOT)
    fit={**fit0,supp[0]:supp[1]}
    if len(fit)!=14 or len(dev)!=4:
        raise ValueError("source owner contract altered")
    dw=windows(dev)
    baseline=train_count_model(fit)
    independent_p3=score_reference(fit,dw,3)["aggregate"]["nll"]
    fixed_p3={name:p3_rows(baseline,blocks) for name,blocks in dw.items()}
    for seed in SEEDS:
        if seed not in paths:raise ValueError("partial cohort")
    rows={}
    for seed in SEEDS:
        model=cold_reload(paths[seed].read_bytes(),"tac_context026",seed,
                          None,source_sample_hash(seed,fit)).eval()
        seed_rows={}
        total={str(x):{"n":0,"nll_sum":0.,"brier_sum":0.,"top1_sum":0,"top5_sum":0}
               for x in LAMBDAS}
        with torch.no_grad():
            for name,blocks in dw.items():
                xb=torch.tensor([list(raw) for _,raw in blocks],dtype=torch.long)
                gold=xb[:,1:]
                logits=model(xb[:,:-1]).double()
                if logits.shape!=(8,128,256):
                    raise ValueError("native TAC geometry not original 256-byte causal")
                pt=F.softmax(logits,dim=-1)
                p3=fixed_p3[name]
                for lam in LAMBDAS:
                    prob=(1.0-lam)*pt+lam*p3
                    r=scores(gold,prob)
                    for k in total[str(lam)]:total[str(lam)][k]+=r[k]
                    seed_rows.setdefault(str(lam),{})[name]=r["nll_sum"]/r["n"]
        for lam in LAMBDAS:
            row=total[str(lam)]
            if row["n"]!=4096:raise ValueError("not 4096 original exposed DEV targets")
            n=row["n"]
            row.update({"nll":row["nll_sum"]/n,"bits_per_original_byte":row["nll_sum"]/n/math.log(2),
                        "brier":row["brier_sum"]/n,"top1":row["top1_sum"]/n,
                        "top5":row["top5_sum"]/n,"per_document_nll":seed_rows[str(lam)]})
        if abs(total["0.0"]["nll"]-FROZEN_TAC_NLL[seed])>1e-5:
            raise ValueError("pure TAC endpoint not identical to original trained benchmark")
        if abs(total["1.0"]["nll"]-independent_p3)>1e-8:
            raise ValueError("P3 endpoint diverges from archived unchanged original P3")
        rows[str(seed)]={"checkpoint_sha256":CK[seed],"lambda_results":total}
    means={str(lam):{
        "mean_nll":sum(rows[str(seed)]["lambda_results"][str(lam)]["nll"] for seed in SEEDS)/5,
        "mean_top1":sum(rows[str(seed)]["lambda_results"][str(lam)]["top1"] for seed in SEEDS)/5}
        for lam in LAMBDAS}
    doc={"study":"LANGBLEND032","status":"NO_FIT_INTERNAL_EXPOSED_DEV_FIXED_GRID",
         "source":"Original PSF CPython 3.13.0, 14 original FIT sources / 4 already exposed DEV",
         "model":"unaltered original TAC CONTEXT026 500122p",
         "predictive_interpretation":"hybrid probability mixture, not native TAC improvement",
         "seed_count":5,"original_training_updates_per_seed":320,
         "exposed_dev_targets_per_seed":4096,
         "lambda_grid":LAMBDAS,"P3_fit_only_reference_nll":independent_p3,
         "results_per_seed":rows,"means_by_lambda":means,
         "DEV_lambda_selection":"NONE_PREDEFINED_GRID_ALL_RESULTS_REPORTED",
         "training_updates_during_study":0,"GPU_minutes":0,
         "protected_TEST":"NOT_ACCESSED",
         "scientific_limit":"nonindependent exposed development comparison"}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":doc["study"],"status":doc["status"],
        "P3_reference":independent_p3,"all_fixed_lambda_means":means,
        "five_checkpoint_SHA_verifications":True},sort_keys=True,indent=2))

def source_sample_hash(seed,fit):
    from scripts.langcompare029_sampling import schedule_receipt
    return schedule_receipt(fit,seed,steps=320)["sample_order_sha256"]

def self_test():
    gold=torch.tensor([[i%256 for i in range(128)] for _ in range(8)],dtype=torch.long)
    p=torch.full((8,128,256),1./256,dtype=torch.float64)
    assert abs(scores(gold,p)["nll_sum"]/1024-math.log(256))<1e-10
    q=p.clone()
    for w in LAMBDAS:
        r=(1-w)*p+w*q
        assert scores(gold,r)["n"]==1024
    print("LANGBLEND032 no-fit probability geometry tests PASS")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:self_test()
    else:
        if a.input_dir is None or a.output is None:
            p.error("input-dir and output required for original checkpoint scoring")
        run(a.input_dir,a.output)
