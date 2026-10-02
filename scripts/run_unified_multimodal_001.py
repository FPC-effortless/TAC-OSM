#!/usr/bin/env python3
"""Registered unified native-model benchmark for language, image and audio.

The benchmark is synthetic by design. It tests whether a single causal
substrate can learn held-out predictive structure across three observation
modalities. It is not a claim of natural-world competence.
"""
from __future__ import annotations
import argparse, json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TRAIN_WORLD_SEED=17001
VAL_WORLD_SEED=17002
TEST_WORLD_SEED=17003
MODALITIES=("language","image","audio")
VOCAB=48
SEQ=5
N_OPS=4
LATENT=32
SLOTS=4
SLOT_DIM=8
TRAIN_STEPS=420
BATCH_SIZE=18
SEEDS=(0,1,2,3,4)

@dataclass(frozen=True)
class Example:
    modality:str
    state_id:int
    x:object
    action:int
    y:object
    semantic:Tuple[int,int,int]

class World:
    def __init__(self, seed:int, n:int, split:str):
        self.rng=np.random.default_rng(seed); self.n=n; self.split=split
        self.residue={"train":2,"val":1,"test":0}[split]
    def semantic(self, i:int)->Tuple[int,int,int]:
        a=int(self.rng.integers(0,8)); b=int(self.rng.integers(0,8)); c=int(self.rng.integers(0,4))
        for _ in range(256):
            if (a*3+b+c)%5==self.residue: return a,b,c
            b=int(self.rng.integers(0,8))
        raise RuntimeError("semantic generator failed to hit registered residue")
    @staticmethod
    def transition(s, op):
        a,b,c=s
        if op==0: a=(a+1)%8
        elif op==1: b=(b+1)%8
        elif op==2: c=(c+1)%4
        elif op==3: a=(a+1)%8; b=(b+2)%8
        else: raise ValueError(op)
        return a,b,c
    def render(self, m, s):
        a,b,c=s
        if m=="language":
            return np.array([1+a,12+b,24+c,32+((a+b)%8),40+((b+c)%8)],dtype=np.int64)
        if m=="image":
            img=np.zeros((8,8),dtype=np.float32); x=a%6; y=b%6
            img[y:y+2,x:x+2]=1.0; img[(c+3)%8,x]=0.5; img[y,(a+b+c)%8]=0.25
            return img.reshape(-1)
        t=np.arange(64,dtype=np.float32); freq=2+a+(b%3); phase=0.15*c
        return (0.5*np.sin(2*np.pi*freq*t/64+phase)+0.1*np.sin(2*np.pi*(freq+1)*t/64)).astype(np.float32)
    def make(self):
        rows=[]
        for i in range(self.n):
            s=self.semantic(i); op=int(self.rng.integers(N_OPS)); ns=self.transition(s,op)
            for m in MODALITIES:
                rows.append(Example(m,i,self.render(m,s),op,self.render(m,ns),s))
        return rows

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class Encoder(nn.Module):
    def __init__(self,m):
        super().__init__(); self.m=m
        if m=="language":
            self.emb=nn.Embedding(VOCAB,LATENT)
            self.net=nn.Sequential(nn.Linear(LATENT,LATENT),nn.GELU(),nn.Linear(LATENT,LATENT))
        else:
            self.net=nn.Sequential(nn.Linear(64,96),nn.GELU(),nn.Linear(96,LATENT))
    def forward(self,x):
        if self.m=="language": return self.net(self.emb(x).mean(1))
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self,m):
        super().__init__(); self.m=m
        self.head=nn.Linear(LATENT,SEQ*VOCAB) if m=="language" else nn.Sequential(nn.Linear(LATENT,96),nn.GELU(),nn.Linear(96,64))
    def forward(self,z): return self.head(z)

class UnifiedBrain(nn.Module):
    def __init__(self,use_structure=True,use_persistence=True,use_prediction=True):
        super().__init__()
        self.use_structure=use_structure; self.use_persistence=use_persistence; self.use_prediction=use_prediction
        self.enc=nn.ModuleDict({m:Encoder(m) for m in MODALITIES})
        self.dec=nn.ModuleDict({m:Decoder(m) for m in MODALITIES})
        self.mod_gate=nn.Linear(LATENT+3,LATENT)
        self.slot_proj=nn.Linear(LATENT,SLOTS*SLOT_DIM)
        self.slot_score=nn.Linear(LATENT,SLOTS)
        self.struct_out=nn.Linear(SLOT_DIM,LATENT)
        self.action_emb=nn.Embedding(N_OPS,LATENT)
        self.op_delta=nn.Parameter(torch.randn(N_OPS,LATENT)*0.02)
        self.transition=nn.Sequential(nn.Linear(2*LATENT,LATENT),nn.GELU(),nn.Linear(LATENT,LATENT))
        self.conf_head=nn.Sequential(nn.Linear(2*LATENT,LATENT),nn.GELU(),nn.Linear(LATENT,1))
    def encode(self,m,x):
        z=self.enc[m](x)
        idx=MODALITIES.index(m)
        one=F.one_hot(torch.full((z.shape[0],),idx,dtype=torch.long,device=z.device),3).float()
        return self.mod_gate(torch.cat([z,one],-1))
    def discover(self,z):
        if not self.use_structure:
            return z, torch.full((z.shape[0],SLOTS),1.0/SLOTS,device=z.device)
        slots=self.slot_proj(z).view(-1,SLOTS,SLOT_DIM)
        w=F.softmax(self.slot_score(z),-1)
        return self.struct_out((slots*w.unsqueeze(-1)).sum(1)),w
    def retrieve(self,z,memory,k=2):
        if not self.use_persistence or not memory:
            return torch.zeros_like(z)
        M=torch.stack(memory).to(z.device)
        idx=(F.normalize(z,-1)@F.normalize(M,-1).T).topk(min(k,len(memory)),-1).indices
        return M[idx].mean(1)
    def predict(self,z,a,memory,proposal_bias=None):
        base,_=self.discover(z)
        if self.use_persistence and memory:
            base=base+0.15*self.retrieve(z,memory)
        ae=self.action_emb(a)+self.op_delta[a]
        if proposal_bias is not None: ae=ae+proposal_bias
        if not self.use_prediction: return base
        return self.transition(torch.cat([base,ae],-1))
    def confidence(self,zp,zt):
        return torch.sigmoid(self.conf_head(torch.cat([zp,zt],-1))).squeeze(-1)

class DurableMemory:
    def __init__(self,max_items=256):
        self.items=[]; self.operator_delta={}; self.max_items=max_items
    def add(self,z,a,zn,confidence):
        if confidence<0.60: return False
        self.items.append(zn.detach().cpu()); self.items=self.items[-self.max_items:]
        d=(zn-z).detach().cpu()
        self.operator_delta[a]=d if a not in self.operator_delta else 0.9*self.operator_delta[a]+0.1*d
        return True
    def tensors(self): return list(self.items)

def batch_tensors(rows):
    out={}
    for m in MODALITIES:
        rr=[r for r in rows if r.modality==m]
        if not rr: continue
        xs=[r.x for r in rr]; ys=[r.y for r in rr]; acts=[r.action for r in rr]
        if m=="language":
            out[m]=(torch.tensor(np.stack(xs),dtype=torch.long),torch.tensor(np.stack(ys),dtype=torch.long),torch.tensor(acts,dtype=torch.long))
        else:
            out[m]=(torch.tensor(np.stack(xs),dtype=torch.float32),torch.tensor(np.stack(ys),dtype=torch.float32),torch.tensor(acts,dtype=torch.long))
    return out

@torch.no_grad()
def evaluate(model, rows, memory, include_memory=True):
    model.eval(); data=batch_tensors(rows); metrics={}
    all_z=[]; by_mod={}
    for m,(x,y,a) in data.items():
        z=model.encode(m,x); zy=model.encode(m,y)
        mem=memory.tensors() if include_memory else []
        zp=model.predict(z,a,mem)
        out=model.dec[m](zp)
        if m=="language":
            p=out.view(-1,SEQ,VOCAB).argmax(-1)
            acc=float((p==y).float().mean())
            loss=float(F.cross_entropy(out.view(-1,VOCAB),y.view(-1)))
            chance=1.0/VOCAB
            score=(acc-chance)/(1.0-chance)
            metrics[m]={"token_accuracy":acc,"normalized_score":score,"loss":loss}
        else:
            mse=float(F.mse_loss(out,y))
            score=1.0/(1.0+mse)
            metrics[m]={"mse":mse,"normalized_score":score}
        preds=torch.stack([model.predict(z,torch.full_like(a,k),mem) for k in range(N_OPS)],1)
        route_dist=((preds-zy.unsqueeze(1))**2).mean(-1)
        choice=route_dist.argmin(1)
        metrics[m]["route_accuracy"]=float((choice==a).float().mean())
        all_z.append(F.normalize(z,-1)); by_mod[m]=z
    if len(all_z)==3:
        pos=((all_z[0]-all_z[1]).pow(2).mean()+(all_z[0]-all_z[2]).pow(2).mean())/2
        shuffled=all_z[1][torch.randperm(all_z[1].shape[0])]
        neg=((all_z[0]-shuffled).pow(2).mean())
        metrics["cross_modal"]={"paired_mse":float(pos),"shuffled_mse":float(neg),"separation":float(neg-pos)}
    return metrics

def train_one(seed, disable, steps, batch_size):
    seed_all(seed)
    train_rows=World(TRAIN_WORLD_SEED,96,"train").make()
    val_rows=World(VAL_WORLD_SEED,48,"val").make()
    test_rows=World(TEST_WORLD_SEED,48,"test").make()
    model=UnifiedBrain(use_structure=("structure" not in disable),use_persistence=("persistence" not in disable),use_prediction=("prediction" not in disable))
    opt=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.0001)
    memory=DurableMemory(); rng=np.random.default_rng(seed+500)
    state_count=96
    for step in range(steps):
        ids=rng.integers(0,state_count,batch_size)
        batch=[train_rows[int(i)*3+j] for i in ids for j in range(3)]
        data=batch_tensors(batch); opt.zero_grad(); total=torch.tensor(0.0)
        for m,(x,y,a) in data.items():
            z=model.encode(m,x); zy=model.encode(m,y)
            zp=model.predict(z,a,memory.tensors(),include_memory if False else None)
            out=model.dec[m](zp)
            recon=F.cross_entropy(out.view(-1,VOCAB),y.view(-1)) if m=="language" else 8.0*F.mse_loss(out,y)
            pred=F.mse_loss(zp,zy)
            total=total+recon+0.35*pred
            for i in range(z.shape[0]):
                conf=1.0/(1.0+float(((zp[i].detach()-zy[i].detach())**2).mean()))
                memory.add(z[i].detach(),int(a[i]),zy[i].detach(),conf)
        if "alignment" not in disable:
            zs=[F.normalize(model.encode(m,data[m][0]),-1) for m in MODALITIES if m in data]
            if len(zs)==3: total=total+0.08*((zs[0]-zs[1]).pow(2).mean()+(zs[0]-zs[2]).pow(2).mean())
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    val=evaluate(model,val_rows,memory); test=evaluate(model,test_rows,memory)
    # Two-stage proposal then expensive route; proposal uses only cheap action prototypes and goal delta.
    route={m:{"proposal_recall":0.0,"conditional_route":0.0} for m in MODALITIES}
    for m in MODALITIES:
        rr=[r for r in test_rows if r.modality==m]
        x,y,a=batch_tensors(rr)[m]; z=model.encode(m,x); zy=model.encode(m,y)
        delta=zy-z
        proto=model.action_emb.weight+model.op_delta
        cheap=F.normalize(delta,-1)@F.normalize(proto,-1).T
        proposed=cheap.topk(min(2,N_OPS),-1).indices
        prop_hit=(proposed==a.unsqueeze(1)).any(1)
        route[m]["proposal_recall"]=float(prop_hit.float().mean())
        good=0; den=int(prop_hit.sum())
        mem=memory.tensors()
        for i in torch.where(prop_hit)[0].tolist():
            cand=proposed[i]
            preds=torch.stack([model.predict(z[i:i+1],torch.tensor([int(k)]),mem) for k in cand],1).squeeze(0)
            d=((preds-zy[i])**2).mean(-1); good+=int(int(cand[int(d.argmin())])==int(a[i]))
        route[m]["conditional_route"]=good/den if den else 0.0
    for m in MODALITIES: test[m].update(route[m])
    return {"seed":seed,"ablation":disable or ["none"],"validation":val,"test":test,"memory_items":len(memory.items),"operator_library":len(memory.operator_delta)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--confirmatory",action="store_true"); ap.add_argument("--steps",type=int,default=TRAIN_STEPS); ap.add_argument("--seeds",default="0,1,2,3,4"); args=ap.parse_args()
    if not args.confirmatory: raise SystemExit("This runner only emits research results with --confirmatory; pre-run smoke belongs in CI tests.")
    seeds=tuple(int(x) for x in args.seeds.split(","))
    arms=[(),("persistence",),("prediction",),("alignment",),("structure",)]
    # Contract fixes these arms before any confirmatory run.
    results=[train_one(s,a,args.steps,BATCH_SIZE) for a in arms for s in seeds]
    payload={"protocol":{"experiment_id":"TACOSM-UNIFIED-MULTIMODAL-001","train_world_seed":TRAIN_WORLD_SEED,"validation_world_seed":VAL_WORLD_SEED,"test_world_seed":TEST_WORLD_SEED,"semantic_split":"residue classes 2/1/0 over 5","seeds":list(seeds),"steps":args.steps,"batch_size":BATCH_SIZE,"modalities":list(MODALITIES),"confirmatory":args.confirmatory},"results":results}
    out=Path("artifacts/TACOSM-UNIFIED-MULTIMODAL-001.json"); out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__": main()
