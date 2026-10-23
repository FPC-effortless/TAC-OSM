"""Original frozen TAC associative WRITE/RETAIN/READ pathway trace.

Synthetic-only, no model fits, no change to originally trained weights.
Numerically reproduces original Native.forward logits BEFORE reporting gates.
"""
from __future__ import annotations
import json,hashlib,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.diagkey002_no_fit_mechanism import original_key_query,git_owned_checkpoint

OUT=Path("research/replay-bundles/DIAGKEY002/ACTUAL_LEARNED_WRITE_READ_GATES.json")
SEEDS=(75001,75002,75003,75004,75005)
DELAYS=(0,4,16,64,256)
N=64

@torch.no_grad()
def trace(model,t,keys,values):
    x=torch.tanh(model.encoder(model.kind_embed(t)+model.key(keys)+model.value(values)))
    f=x.new_zeros(len(x),model.width);s=f.clone()
    m=x.new_zeros(len(x),8,8)
    traces={"initial_write":[],"distractor_write":[],"final_query":[]}
    original_k=[]
    at_four=None;prior_to_query=None
    query_gate_values=None
    for j in range(x.shape[1]):
        current=x[:,j]
        query=F.normalize(f[:,:8]+s[:,:8],dim=-1,eps=1e-6)
        recall=torch.bmm(m,query.unsqueeze(-1)).squeeze(-1)
        val,gate=model.fast_map(torch.cat((current,f,recall),-1)).chunk(2,-1)
        f=torch.lerp(f,val.tanh(),gate.sigmoid())
        delta,gate=model.slow_map(torch.cat((f,s,recall),-1)).chunk(2,-1)
        s=torch.lerp(s,(s+delta.tanh()).tanh(),.25*gate.sigmoid())
        kk,vv,beta,relevance=model.write_map(torch.cat((current,f,s),-1)).split((8,8,1,1),-1)
        kk=F.normalize(kk,dim=-1,eps=1e-6)
        strength=beta.sigmoid()*relevance.sigmoid()
        err=vv.tanh()-torch.bmm(m,kk.unsqueeze(-1)).squeeze(-1)
        prev=m
        next_mem=m+strength.unsqueeze(-1)*err.unsqueeze(-1)*kk.unsqueeze(1)
        norm=next_mem.square().sum((1,2),keepdim=True).sqrt().clamp_min(1.)
        m=next_mem/norm
        label="initial_write" if j<4 else ("final_query" if j==x.shape[1]-1 else "distractor_write")
        traces[label].append({
            "gate":strength.squeeze(-1).cpu(),
            "write_vector_norm":(m-prev).flatten(1).norm(dim=1).cpu(),
            "memory_norm":m.flatten(1).norm(dim=1).cpu()
        })
        if j<4:original_k.append(kk)
        if j==3:at_four=m.clone()
        if j==x.shape[1]-2:prior_to_query=m.clone()
        if j==x.shape[1]-1:query_gate_values=strength.squeeze(-1)
    query=F.normalize(model.query(x[:,-1]),dim=-1,eps=1e-6)
    lookup=torch.bmm(m,query.unsqueeze(-1)).squeeze(-1)
    neural=model.readout(torch.cat((f,s,lookup,x[:,-1]),-1))
    selected=(keys[:,:4]==keys[:,-1,None]).to(torch.int64).argmax(-1)
    stored_keys=torch.stack(original_k,dim=1)[torch.arange(len(t)),selected]
    target_alignment=(stored_keys*query).sum(-1)
    written_target=torch.bmm(at_four,stored_keys.unsqueeze(-1)).squeeze(-1)
    pre_target=torch.bmm(prior_to_query,stored_keys.unsqueeze(-1)).squeeze(-1) if prior_to_query is not None else written_target
    after_target=torch.bmm(m,stored_keys.unsqueeze(-1)).squeeze(-1)
    return neural,{
        "gate_by_event":traces,
        "original_key_query_cosine":target_alignment.cpu(),
        "target_read_norm_after_first_4_writes":written_target.norm(dim=-1).cpu(),
        "target_read_norm_before_final_query":pre_target.norm(dim=-1).cpu(),
        "target_read_norm_after_query_write":after_target.norm(dim=-1).cpu(),
        "final_actual_read_norm":lookup.norm(dim=-1).cpu(),
        "read_query_key_alignment":target_alignment.cpu(),
        "query_gate":query_gate_values.cpu()
    }

def summarize_vectors(vectors):
    x=np.concatenate([v.numpy().reshape(-1) for v in vectors])
    return {"count":int(x.size),"mean":float(x.mean()),
            "median":float(np.median(x)),
            "p10":float(np.quantile(x,.1)),
            "p90":float(np.quantile(x,.9))}
def main():
    torch.set_num_threads(1)
    result={"study":"DIAGKEY002_REAL_FROZEN_MODEL_WRITE_READ_SIGNAL_TRACE_SYNTHETIC_ONLY",
            "original_models":5,"training_updates":0,"source":"quarantined first DIAGKEY001 pilot",
            "all_original_forward_logits_match_manual":True,
            "real_world_evidence":False,
            "limits":["Read/write gate levels on the failed-training pilot cannot establish mechanism superiority",
                      "The keys are learned 8D projections, not orthogonal onehot oracle keys",
                      "These are synthetic model traces, not actual business process memory or real robots"],
            "per_seed":{}, "summary":{}}
    for seed in SEEDS:
        model,_=git_owned_checkpoint(seed,"tac")
        result["per_seed"][str(seed)]={}
        for delay in DELAYS:
            samples={"initial_write":[],"distractor_write":[],"final_query":[]}
            misc={k:[] for k in ("original_key_query_cosine","target_read_norm_after_first_4_writes",
                                  "target_read_norm_before_final_query","target_read_norm_after_query_write",
                                  "final_actual_read_norm")}
            for batch in range(0,N,16):
                kind,key,value,y=original_key_query(seed,batch//16,delay,batch_size=16,split="TRACE")
                with torch.no_grad():
                    estimated=model(kind,key,value)
                    recreated,data=trace(model,kind,key,value)
                    torch.testing.assert_close(recreated,estimated,atol=2e-6,rtol=2e-6)
                for label in samples:
                    for row in data["gate_by_event"][label]:
                        samples[label].append(row)
                for k in misc:misc[k].append(data[k])
            report={}
            for label,values in samples.items():
                report[label+"_write_gate"]=summarize_vectors([q["gate"] for q in values]) if values else None
                report[label+"_state_delta_L2"]=summarize_vectors([q["write_vector_norm"] for q in values]) if values else None
            for k,z in misc.items():report[k]=summarize_vectors(z)
            result["per_seed"][str(seed)][str(delay)]=report
        print(f"DIAGKEY002_ACTUAL_MODEL_TRACED_{seed}_NO_TRAINING",flush=True)
    for delay in DELAYS:
        k=str(delay)
        result["summary"][k]={}
        keys=list(result["per_seed"][str(SEEDS[0])][k])
        for metric in keys:
            if result["per_seed"][str(SEEDS[0])][k][metric] is None:
                result["summary"][k][metric]=None
                continue
            result["summary"][k][metric]={
                "mean_over_5_original_seeds_of_per_seed_means":float(np.mean(
                    [result["per_seed"][str(s)][k][metric]["mean"] for s in SEEDS])),
                "five_original_seed_means":[result["per_seed"][str(s)][k][metric]["mean"] for s in SEEDS]
            }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if OUT.exists():raise FileExistsError("Original learned gate trace already exists")
    OUT.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("DIAGKEY002_LEARNED_GATES_ORIGINAL_FIVE_SEEDS="+json.dumps(result["summary"],sort_keys=True),flush=True)
if __name__=="__main__":main()
