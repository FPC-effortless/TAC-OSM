"""LANGTRAIN025 registered authentic-source 256-byte WaveCED TAC training.

Newly initialized model only, never original user-owned 128-symbol checkpoints.
Uses frozen SOURCE023 CPython publisher FIT, fixed 320 AdamW updates, five seeds,
no protected TEST. DEV is previously EXPOSED. Code does not merge documents,
tune from DEV, or silently select successful seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from byte_waveced_500k import ByteWaveCED500K

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langbase024"))
from byte_next_reference import original_source,train_count_model,law

SEEDS=(1701,1702,1703,1704,1705)
UPDATES=320
BATCH=8
WINDOW=129
TOKENS=128
PARAMS=499978

def sha(raw):return hashlib.sha256(raw).hexdigest()

def windows(dev):
    result={}
    for name,raw in sorted(dev.items()):
        if len(raw)<WINDOW:raise ValueError("original DEV too short for frozen 129-byte window")
        starts=np.linspace(0,len(raw)-WINDOW,8,dtype=np.int64)
        result[name]=tuple((int(s),raw[int(s):int(s)+WINDOW]) for s in starts)
    if len(result)!=4 or sum(len(a) for a in result.values())!=32:
        raise ValueError("original registered 4x8 DEV window count differs")
    return result

def numerical_metrics(logits:torch.Tensor,y:torch.Tensor):
    if logits.shape[-1]!=256 or logits.shape[:-1]!=y.shape:
        raise ValueError("prediction shape/byte target mismatch")
    lp=F.log_softmax(logits.double(),-1)
    probs=lp.exp()
    gold=lp.gather(-1,y[...,None]).squeeze(-1)
    p_gold=probs.gather(-1,y[...,None]).squeeze(-1)
    n=y.numel()
    rare=(y>=128)
    return {"n":int(n),"nll_sum":float((-gold).sum()),
            "brier_sum":float((1-2*p_gold+probs.square().sum(-1)).sum()),
            "top1_sum":int((probs.argmax(-1)==y).sum()),
            "top5_sum":int((probs.topk(5,-1).indices==y[...,None]).any(-1).sum()),
            "rare_n":int(rare.sum()),"rare_nll_sum":float((-gold)[rare].sum())}

def normalize(metrics):
    n=metrics["n"]; nll=metrics["nll_sum"]/n
    return {**metrics,"nll":nll,"bits_per_original_byte":nll/math.log(2),
            "brier":metrics["brier_sum"]/n,
            "top1":metrics["top1_sum"]/n,"top5":metrics["top5_sum"]/n,
            "rare_nll":metrics["rare_nll_sum"]/metrics["rare_n"] if metrics["rare_n"] else None}

def aggregate(per_doc):
    keys=("n","nll_sum","brier_sum","top1_sum","top5_sum","rare_n","rare_nll_sum")
    return normalize({key:sum(row[key] for row in per_doc.values()) for key in keys})

@torch.no_grad()
def score_model(model,dev_windows):
    model.eval()
    per_doc={}
    for name,blocks in dev_windows.items():
        xb=torch.tensor([[b for b in window] for _,window in blocks],dtype=torch.long)
        logits=model(xb[:,:-1])
        scores=numerical_metrics(logits,xb[:,1:])
        if scores["n"]!=8*128:raise ValueError("DEV target count altered")
        per_doc[name]=scores
    return {"aggregate":aggregate(per_doc),
            "per_doc":{name:normalize(row) for name,row in per_doc.items()}}

def score_reference(fit,dev_windows,order):
    model=train_count_model(fit)
    per_doc={}
    for name,blocks in dev_windows.items():
        totals={"n":0,"nll_sum":0.,"brier_sum":0.,"top1_sum":0,"top5_sum":0,
                "rare_n":0,"rare_nll_sum":0.}
        for _,window in blocks:
            for t in range(128):
                prefix=window[max(0,t-2):t+1]
                p=law(model,prefix,order)
                target=window[t+1]
                loss=-math.log(p[target])
                totals["n"]+=1
                totals["nll_sum"]+=loss
                totals["brier_sum"]+=1-2*p[target]+math.fsum(q*q for q in p)
                rank=sorted(range(256),key=lambda idx:(-p[idx],idx))[:5]
                totals["top1_sum"]+=int(rank[0]==target)
                totals["top5_sum"]+=int(target in rank)
                if target>=128:
                    totals["rare_n"]+=1
                    totals["rare_nll_sum"]+=loss
        per_doc[name]=totals
    return {"aggregate":aggregate(per_doc),"per_doc":{name:normalize(row) for name,row in per_doc.items()},
            "trained_on_whole_fit_byte_targets":model["targets"],"trained_on_fit_rare_targets":model["nonascii_fit_targets"]}

def source_sample(rng,fit,names,sample_digest):
    rows=[]
    for _ in range(BATCH):
        name=names[int(rng.integers(len(names)))]
        doc=fit[name]
        start=int(rng.integers(len(doc)-WINDOW+1))
        sample_digest.update(f"{name}:{start};".encode())
        rows.append(torch.tensor(list(doc[start:start+WINDOW]),dtype=torch.long))
    return torch.stack(rows)

def safe_load(blob):
    with torch.serialization.safe_globals([
        np._core.multiarray._reconstruct,np._core.multiarray.scalar,
        np.ndarray,np.dtype,np.dtypes.UInt32DType,np.dtypes.Float64DType]):
        return torch.load(io.BytesIO(blob),map_location="cpu",weights_only=True)

def run(out,seeds=SEEDS):
    torch.set_num_threads(2)
    if not seeds or any(s not in SEEDS for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError("registered unique seeds required")
    fit0,dev,supp=original_source(ROOT)
    suppname,suppbytes=supp
    fit={**fit0,suppname:suppbytes}
    if len(fit)!=14 or len(dev)!=4:raise ValueError("FIT+supplement/DEV original membership broken")
    if any(name in dev for name in fit):raise ValueError("source owner leaked FIT/DEV")
    dw=windows(dev)
    expected=4096
    baseline={f"P{o}":score_reference(fit,dw,o) for o in (0,3)}
    for model in baseline.values():
        if model["aggregate"]["n"]!=expected:raise ValueError("benchmark target position parity broken")
    out.mkdir(parents=True,exist_ok=True)
    files={}
    for p in [Path(__file__),Path(__file__).with_name("byte_waveced_500k.py")]:
        files[p.name]=sha(p.read_bytes())
    sample_names=tuple(sorted(fit))
    records=[]
    for seed in seeds:
        started=time.perf_counter()
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        sample_hash=hashlib.sha256()
        model=ByteWaveCED500K()
        if model.parameter_count()!=PARAMS:raise ValueError("active model parameter budget changed")
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        train_losses=[]
        for step in range(UPDATES):
            batch=source_sample(rng,fit,sample_names,sample_hash)
            logits=model(batch[:,:-1])
            loss=F.cross_entropy(logits.reshape(-1,256),batch[:,1:].reshape(-1))
            if not torch.isfinite(loss):raise ValueError("nonfinite train loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step==0:
                missing=[name for name,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or not torch.any(p.grad!=0)]
                if missing:raise ValueError("inactive/nonfinite parameter gradient: "+str(missing))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step in (0,39,79,159,239,319):train_losses.append([step+1,float(loss)])
        score=score_model(model,dw)
        ck={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "rng_torch":torch.get_rng_state(),"rng_numpy":np.random.get_state(),
            "rng_python":random.getstate(),"rng_sampler":rng.bit_generator.state,
            "seed":seed,"updates":UPDATES,"sample_sha256":sample_hash.hexdigest(),
            "model_source_sha256":files["byte_waveced_500k.py"],
            "dev":score}
        ckpath=out/f"byte_waveced_seed_{seed}.pt"
        torch.save(ck,ckpath)
        replica=ByteWaveCED500K()
        record=safe_load(ckpath.read_bytes())
        replica.load_state_dict(record["model"],strict=True)
        replay=score_model(replica,dw)
        if abs(replay["aggregate"]["nll"]-score["aggregate"]["nll"])>1e-6:
            raise ValueError("frozen checkpoint local reload NLL mismatch")
        records.append({"seed":seed,"checkpoint":ckpath.name,
                        "checkpoint_sha256":sha(ckpath.read_bytes()),
                        "model_learned_parameters":model.parameter_count(),
                        "updates":UPDATES,"batch":BATCH,"context":TOKENS,
                        "training_samples_sha256":sample_hash.hexdigest(),
                        "loss_log":train_losses,"scoring":score,
                        "local_checkpoint_reload":"PASS",
                        "seconds":time.perf_counter()-started})
        print(f"seed={seed} train_end={train_losses[-1][1]:.6f} native_DEV_NLL={score['aggregate']['nll']:.6f} baseline_P3={baseline['P3']['aggregate']['nll']:.6f} REPLAY=PASS",flush=True)
    mean=sum(s["scoring"]["aggregate"]["nll"] for s in records)/len(records)
    p3=baseline["P3"]["aggregate"]
    best={"neural_mean_nll":mean,"baseline_P3_nll":p3["nll"],
          "delta_neural_minus_P3":mean-p3["nll"],
          "neural_win_seeds":sum(s["scoring"]["aggregate"]["nll"]<p3["nll"] for s in records),
          "preregistered_gain_pass":mean<p3["nll"] and sum(s["scoring"]["aggregate"]["nll"]<p3["nll"] for s in records)>=4 and
             sum(s["scoring"]["aggregate"]["brier"] for s in records)/len(records)<=p3["brier"]}
    receipt={"study":"LANGTRAIN025_NATIVE_256_BYTE_500K",
       "scientific_status":"AUTHENTIC_PREVIOUSLY_EXPOSED_DEV_NOT_INDEPENDENT",
       "original_fit_docs":13,"unicode_fit_supplement_docs":1,"exposed_dev_docs":4,
       "source_publisher_commit":"60403a5409ff2c3f3b07dd2ca91a7a3e096839c7",
       "context":TOKENS,"predicted_targets_per_seed":4096,
       "optimizer":"AdamW lr .001 default weight decay; norm clip 1",
       "source_lossless_byte_codec":True,"hyperparameter_tuning_on_dev":False,
       "training_updates":UPDATES,"active_learned_parameters":PARAMS,
       "code_sha256":files,"baselines":baseline,"model_runs":records,
       "registered_primary":best,"matched_FIT_exposure_to_n_gram":False,
       "matched_neural_SSM_G_R_U_train_compute":False,
       "original_Vim_checkpoint_Git_replay":"BLOCKED_SEPARATE_ORIGINAL_RIGHTS",
       "protected_TEST":"NOT_RUN","native_intelligence_qualification":"NOT_ESTABLISHED"}
    target=out/"LANGTRAIN025_AUTHENTIC_EXPOSED_DEV_RECEIPT.json"
    target.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS))
    args=parser.parse_args()
    run(args.output,tuple(args.seeds))
if __name__=="__main__":main()
