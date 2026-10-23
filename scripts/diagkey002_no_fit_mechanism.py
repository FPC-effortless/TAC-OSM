"""DIAGKEY002: no-fit retrospective keyed-recall ladder + exact 8x8 delta oracle.

This is a software-only synthetic mechanism probe and a diagnostic of the
QUARANTINED, optimizer-unguarded original DIAGKEY001 pilot checkpoints.
It CANNOT qualify real-world skill, memory superiority or a new model fit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.diagkey001_associative_recall import ARMS, SEEDS, CHANCE, NVAL, model_for, model_counts

ROOT=Path("research/replay-bundles/DIAGKEY002")
SOURCE=Path("research/replay-bundles/DIAGKEY001")
DELAYS=(0,1,2,4,8,16,64,256)
TRIALS_PER_DELAY=512
ORACLE_TRIALS=1000
ORACLE_SEED=75201

def original_key_query(seed,step,delay,batch_size=32,split="LADDER"):
    # Same FOUR initial key/value stores as the original DIAGKEY001 pilot.
    # The first four original writes are never re-exposed to the query.
    string=f"DIAGKEY002:{split}:{seed}:{step}:{delay}:four_pairs:32_values"
    rng=np.random.default_rng(int(hashlib.sha256(string.encode()).hexdigest()[:16],16))
    n=delay+5
    kind=np.ones((batch_size,n),np.int64)
    keys=np.zeros((batch_size,n),np.int64)
    values=np.zeros((batch_size,n),np.int64)
    answer=np.zeros((batch_size,),np.int64)
    kind[:,:4]=0
    kind[:,-1]=2
    for i in range(batch_size):
        targets=rng.choice(16,4,replace=False)
        vals=rng.choice(NVAL,4,replace=False)
        selector=int(rng.integers(4))
        keys[i,:4]=targets+1
        values[i,:4]=vals+1
        keys[i,4:-1]=rng.integers(17,65,size=delay)
        values[i,4:-1]=rng.integers(1,33,size=delay)
        keys[i,-1]=targets[selector]+1
        answer[i]=vals[selector]
        assert values[i,-1]==0
        assert not (keys[i,4:-1]==keys[i,-1]).any()
    return tuple(torch.from_numpy(z) for z in (kind,keys,values,answer))

def wilson(correct,n):
    z=1.959963984540054
    p=correct/n
    mid=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/(1+z*z/n)
    return [float(mid-half),float(mid+half)]

def git_owned_checkpoint(seed,arm):
    p=SOURCE/f"seed-{seed}"/f"{arm}.pt"
    rec=SOURCE/f"seed-{seed}"/f"{arm}.json"
    if not p.is_file() or not rec.is_file():
        raise FileNotFoundError(f"previously original Git-owned diagnostic model missing: {seed}/{arm}")
    original=subprocess.check_output(["git","show","HEAD:"+p.as_posix()])
    if original!=p.read_bytes():
        raise ValueError("source model has changed from original Git bytes")
    row=json.loads(rec.read_text())
    if row.get("live_budget_receipt_sha256"):
        raise ValueError("THIS study examines unguarded first pilot only, do not conflate with guarded fits")
    if hashlib.sha256(p.read_bytes()).hexdigest()!=row["original_checkpoint_sha256"]:
        raise ValueError("original unqualified pilot model SHA mismatch")
    learned=torch.load(p,weights_only=False,map_location="cpu")
    model=model_for(arm)
    model.load_state_dict(learned["weights"],strict=True)
    model.eval()
    return model,row

def original_ladder():
    torch.set_num_threads(1)
    counts=model_counts()
    all_rows={}
    for seed in SEEDS:
        all_rows[str(seed)]={}
        for arm in ARMS:
            model,row=git_owned_checkpoint(seed,arm)
            output={}
            with torch.no_grad():
                for delay in DELAYS:
                    correct=0
                    bits=[]
                    nll=0.
                    for iteration in range(TRIALS_PER_DELAY//32):
                        t,k,v,y=original_key_query(seed,iteration,delay)
                        logits=model(t,k,v)
                        c=(logits.argmax(-1)==y).to(torch.int64).tolist()
                        bits.extend(c)
                        correct+=sum(c)
                        nll+=float(F.cross_entropy(logits,y,reduction="sum"))
                    output[str(delay)]={
                        "correct":correct,"trials":TRIALS_PER_DELAY,
                        "accuracy":correct/TRIALS_PER_DELAY,
                        "Wilson_95_interval":wilson(correct,TRIALS_PER_DELAY),
                        "mean_NLL":nll/TRIALS_PER_DELAY,
                        "correct_bits":bits,
                    }
            all_rows[str(seed)][arm]=output
            print("ORIGINAL_UNQUALIFIED_PILOT_LADDER="+json.dumps({
                "seed":seed,"arm":arm,
                "accuracy":{key:z["accuracy"] for key,z in output.items()},
                "trained_new_models":0}),flush=True)
    summary={}
    for delay in DELAYS:
        summary[str(delay)]={}
        for arm in ARMS:
            means=[all_rows[str(s)][arm][str(delay)]["accuracy"] for s in SEEDS]
            summary[str(delay)][arm]={"five_seed_accuracy":means,
                                       "mean_accuracy":float(np.mean(means))}
    return {"original_seed_scores":all_rows,"five_seed_summary":summary,
            "original_trained_models_reused":len(SEEDS)*len(ARMS),
            "training_updates":0,
            "original_pilot_optimizer_guard_pass":False,
            "different_query_length_from_original_FIT":True,
            "chance_exact_value_accuracy":CHANCE}

def oracle_update(mem,key,value,*,normalize=False):
    old=mem
    new=old+np.outer(value-old@key,key)
    if normalize:new=new/max(1.,np.linalg.norm(new))
    return new

def delta_oracle_one(delay,scenario,rng):
    basis=np.eye(8,dtype=np.float64)
    indices=rng.choice(8,size=4,replace=False)
    values=rng.choice(8,size=4,replace=False)
    mem=np.zeros((8,8),np.float64)
    for k,v in zip(indices,values):
        mem=oracle_update(mem,basis[k],basis[v],normalize=(scenario=="global_norm_all"))
    selected=int(rng.integers(4))
    target_key=int(indices[selected])
    target_value=int(values[selected])
    for _ in range(delay):
        key=int(rng.integers(8))
        val=int(rng.integers(8))
        if scenario in ("masked_write","query_overwrites"):
            if key in indices:continue  # explicit oracle, NOT learned admission
        mem=oracle_update(mem,basis[key],basis[val],normalize=(scenario=="global_norm_all"))
    if scenario=="query_overwrites":
        mem=oracle_update(mem,basis[target_key],np.zeros(8))
    read=mem@basis[target_key]
    return int(read.argmax()==target_value),float(read[target_value]),float(np.linalg.norm(read-basis[target_value]))

def oracle_audit():
    out={}
    for scenario in ("masked_write","write_all","global_norm_all","query_overwrites"):
        per={}
        for delay in DELAYS:
            rng=np.random.default_rng(ORACLE_SEED+delay)
            z=np.asarray([delta_oracle_one(delay,scenario,rng) for _ in range(ORACLE_TRIALS)])
            count=int(z[:,0].sum())
            per[str(delay)]={
                "n":ORACLE_TRIALS,
                "accuracy":count/ORACLE_TRIALS,
                "Wilson_95_interval":wilson(count,ORACLE_TRIALS),
                "mean_true_read_amplitude":float(z[:,1].mean()),
                "mean_value_L2_error":float(z[:,2].mean())
            }
        out[scenario]=per
    return out

def gradient_path_audit():
    torch.set_num_threads(1)
    seed=SEEDS[0]
    model,_=git_owned_checkpoint(seed,"tac")
    results={}
    for delay in (0,1,4,16,64,256):
        grad_tensor=[]
        def watch(_module,inputs,output):
            output.retain_grad()
            grad_tensor.append(output)
        handle=model.encoder.register_forward_hook(watch)
        model.zero_grad(set_to_none=True)
        t,k,v,y=original_key_query(seed,904,delay,batch_size=4,split="GRAD")
        logits=model(t,k,v)
        loss=F.cross_entropy(logits,y)
        loss.backward()
        handle.remove()
        early=float(grad_tensor[0].grad[:,:4].norm())
        query=float(grad_tensor[0].grad[:,-1].norm())
        write=float(model.write_map.weight.grad.norm())
        read=float(model.query.weight.grad.norm())
        fast=float(model.fast_map.weight.grad.norm())
        results[str(delay)]={
            "loss":float(loss),
            "first_four_write_encoder_gradient_L2":early,
            "query_encoder_gradient_L2":query,
            "early_to_query_gradient_ratio":early/max(query,1e-16),
            "write_map_parameter_gradient_L2":write,
            "read_query_parameter_gradient_L2":read,
            "fast_map_parameter_gradient_L2":fast,
            "finite":all(math.isfinite(val) for val in (early,query,write,read,fast)),
            "mechanistic_interpretation":"Nonzero gradients show differentiability only; not successful learned associative binding."
        }
    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("oracle","ladder","all"),default="oracle")
    args=parser.parse_args()
    result={
        "study":"DIAGKEY002_SOFTWARE_ONLY_ORACLE_AND_FROZEN_LADDER",
        "status":"UNTRAINED_ORACLE_AND_PREVIOUSLY_EXPOSED_SYNTHETIC_PILOT",
        "real_world_evidence":False,
        "new_training_updates":0,
        "original_DIAGKEY001_scientific_training_guard":"FAILED",
        "note":"Genuine original 100k recurrent pilot was trained only at 16/64/256; shorter ladder is OUTSIDE ITS TRAINING DISTRIBUTION. A failure at delay0 does not prove learning is impossible with an actual zero-delay curriculum.",
        "oracle":oracle_audit()
    }
    if args.mode in ("ladder","all"):
        result["original_frozen_ladder"]=original_ladder()
        result["original_frozen_gradient_paths"]=gradient_path_audit()
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/("ORACLE_AND_FROZEN_LADDER.json" if args.mode in ("ladder","all") else "ORACLE_ONLY.json")
    if path.exists():raise FileExistsError("Original diagnostic evidence is immutable")
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("DIAGKEY002_SOFTWARE_ONLY="+json.dumps({
        "oracle":{k:{d:v[str(d)]["accuracy"] for d in (0,16,64,256)}
                  for k,v in result["oracle"].items()},
        "new_model_fits":0,
        "frozen_ladder_measured":"original_frozen_ladder" in result},sort_keys=True),flush=True)

if __name__=="__main__":main()
