#!/usr/bin/env python3
"""ME-001B: authentic SCANIA TRAIN-internal exploratory native TAC recurrence.
NOT model promotion, semantic entity discovery, action evaluation, or SSM conversion.
Reuses frozen SEQ010 official TRAIN-only acquisition and chronological window contract.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts import seq010_real_recurrent as seq

SEEDS=(119001,119002,119003)
SPLIT_SALT="me001b-vehicle-disjoint-train-internal-20261019-v1"
STEPS=250
BATCH=64
SLOTS=4
WIDTH=24
FEATURES=seq.N_FEATURES
HISTORY=seq.HISTORY
# These are *sensor-feature* embeddings, NOT learned machine identities.
# Each sample contains only one vehicle's actual chronological history.

class NativeTACAssembly(nn.Module):
    def __init__(self, slots=SLOTS, width=WIDTH):
        super().__init__()
        self.slots,self.width=slots,width
        self.sensor_embedding=nn.Parameter(torch.randn(FEATURES,width)*0.025)
        self.anchors=nn.Parameter(torch.randn(slots,width)*0.025)
        self.token_key=nn.Linear(width+2,width,bias=False)
        self.slot_query=nn.Linear(width*2,width,bias=False)
        self.null_key=nn.Linear(width+2,1)
        self.drive=nn.Linear(width+1,width)
        self.recurrent=nn.Linear(width,width,bias=False)
        self.write=nn.Linear(width*2,1)
        self.raw_decay=nn.Parameter(torch.full((width,),-3.0))
        self.output_query=nn.Linear(width,width,bias=False)
        self.output=nn.Linear(width*2,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.bias=nn.Parameter(torch.zeros(FEATURES))

    def forward(self,x,reset_at=None,return_states=False):
        if x.ndim!=3 or x.shape[-1]!=FEATURES*2:
            raise ValueError("prepared shape must be [B,T,48]")
        batch,steps,_=x.shape
        h=x.new_zeros(batch,self.slots,self.width)
        all_states=[]
        emb=self.sensor_embedding
        for t in range(steps):
            if reset_at is not None and t==reset_at:
                h=torch.zeros_like(h)
            observed=x[:,t,FEATURES:].clamp(0,1)
            values=x[:,t,:FEATURES]*observed
            token=torch.cat((emb.unsqueeze(0).expand(batch,-1,-1),
                            values.unsqueeze(-1),observed.unsqueeze(-1)),dim=-1)
            q=self.slot_query(torch.cat((h,self.anchors.unsqueeze(0).expand(batch,-1,-1)),dim=-1))
            key=self.token_key(token)
            similarity=torch.einsum("bnh,bfh->bnf",q,key)/(self.width**0.5)
            logits=torch.cat((similarity,self.null_key(token).transpose(1,2)),dim=1)
            allocation=F.softmax(logits,dim=1)[:,:self.slots]*observed[:,None,:]
            mass=allocation.sum(-1,keepdim=True)
            value=torch.einsum("bnf,bf->bn",allocation,values).unsqueeze(-1)/mass.clamp_min(1e-7)
            source=torch.einsum("bnf,fh->bnh",allocation,emb)/mass.clamp_min(1e-7)
            incoming=torch.tanh(self.drive(torch.cat((source,value),dim=-1)))
            retained=h*torch.exp(-F.softplus(self.raw_decay).view(1,1,-1))
            proposal=torch.tanh(self.recurrent(retained)+incoming)
            gate=torch.sigmoid(self.write(torch.cat((retained,incoming),dim=-1)))
            gate=gate*(mass>1e-8)
            # convex bounded native recurrent TAC state; no implicit SSM backbone.
            h=(1-gate)*retained+gate*proposal
            if return_states: all_states.append(h)
        queries=self.output_query(emb)
        select=F.softmax(torch.einsum("fh,bnh->bfn",queries,h)/(self.width**0.5),dim=-1)
        context=torch.einsum("bfn,bnh->bfh",select,h)
        read=torch.cat((context,emb.unsqueeze(0).expand(batch,-1,-1)),dim=-1)
        residual=self.output(read).squeeze(-1)+self.bias
        if return_states: return residual,torch.stack(all_states,1)
        return residual

class GlobalGRU(nn.Module):
    def __init__(self,hidden=32):
        super().__init__()
        self.gru=nn.GRU(FEATURES*2,hidden,batch_first=True)
        self.head=nn.Linear(hidden,FEATURES)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self,x,**kwargs):
        return self.head(self.gru(x)[0][:,-1])

def make_model(name):
    return NativeTACAssembly() if name=="native_tac_assembly" else GlobalGRU() if name=="global_gru" else None

def evaluate(model,data,thresholds=None,reset_at=None):
    x, delta, mask=data[:3]
    model.eval()
    sums=np.zeros(FEATURES,dtype=np.float64)
    cnt=np.zeros(FEATURES,dtype=np.float64)
    changes=np.zeros(FEATURES,dtype=np.float64)
    changes_n=np.zeros(FEATURES,dtype=np.float64)
    correct=np.zeros(FEATURES,dtype=np.float64)
    with torch.inference_mode():
        for start in range(0,len(x),128):
            bx,by,bm=x[start:start+128],delta[start:start+128],mask[start:start+128]
            pred=model(bx,reset_at=reset_at)
            error=(pred-by).abs().double()
            sums+=(error*bm).sum(0).numpy()
            cnt+=bm.sum(0).numpy()
            if thresholds is not None:
                big=(by.abs()>thresholds.unsqueeze(0)) & bm
                changes+=(error*big).sum(0).numpy()
                changes_n+=big.sum(0).numpy()
                correct+=((torch.sign(pred)==torch.sign(by))&big).sum(0).numpy()
    if np.any(cnt<100):raise ValueError("too few paired heldout targets")
    result={"macro_mae":float(np.mean(sums/cnt)),
            "micro_mae":float(np.sum(sums)/np.sum(cnt)),
            "min_targets":int(np.min(cnt))}
    if thresholds is not None:
        included=changes_n>=10
        result.update({"change_macro_mae":float(np.mean((changes/np.maximum(changes_n,1))[included])) if included.any() else None,
                       "change_direction_accuracy":float(np.sum(correct)/np.sum(changes_n)) if changes_n.sum()>0 else None,
                       "change_total":int(changes_n.sum()),"eligible_change_features":int(included.sum())})
    return result

def train_thresholds(training):
    delta,mask=training[1:3]
    vals=[]
    for f in range(FEATURES):
        observed=delta[:,f][mask[:,f]].abs()
        if len(observed)==0: raise ValueError("feature has no train observations")
        vals.append(max(float(torch.quantile(observed,0.9)),1e-4))
    return torch.tensor(vals,dtype=delta.dtype)

def fit(name,seed,train,val,thresholds,out,steps):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=make_model(name)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(seed+1000)
    x,delta,mask=train[:3]
    curve=[]
    tic=time.monotonic()
    for step in range(steps):
        ix=torch.randint(len(x),(BATCH,),generator=generator)
        prediction=model(x[ix])
        err=F.smooth_l1_loss(prediction,delta[ix],beta=0.1,reduction="none")
        loss=(err*mask[ix]).sum()/mask[ix].sum().clamp_min(1)
        if not torch.isfinite(loss):raise ValueError("nonfinite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
        if not torch.isfinite(grad_norm):raise ValueError("nonfinite gradient")
        opt.step()
        if step+1 in (1,steps//2,steps):
            curve.append({"step":step+1,"loss":float(loss.detach())})
    recorded=evaluate(model,val,thresholds=thresholds)
    reset=evaluate(model,val,thresholds=thresholds,reset_at=HISTORY//2) if name=="native_tac_assembly" else None
    path=out/"checkpoints"/f"{name}-{seed}.pt"
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":name,"seed":seed,"steps":steps,"state_dict":model.state_dict()},path)
    return {"model":name,"seed":seed,"steps":steps,"params":sum(p.numel() for p in model.parameters()),
            "validation":recorded,"reset_midway_control":reset,
            "loss_curve":curve,"wall_seconds":round(time.monotonic()-tic,2),
            "checkpoint":str(path.relative_to(out)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def run(out,steps=STEPS):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    seq.SPLIT_SALT=SPLIT_SALT
    file,source=seq.acquire_training_csv(out)
    (out/"source.json").write_text(json.dumps(source,indent=2,sort_keys=True))
    train,val,cohort=seq.load_real_windows(file)
    file.unlink(missing_ok=True)
    prepared_train=seq.prepare(*train)
    prepared_val=seq.prepare(*val)
    thresholds=train_thresholds(prepared_train)
    baseline=seq.persistence_score(prepared_val)
    baseline_change=evaluate_zero_baseline(prepared_val,thresholds)
    report={"study":"ME001B","kind":"TRAIN_INTERNAL_EXPLORATORY_REAL",
            "source":source,"cohort":cohort,"split_salt":SPLIT_SALT,
            "thresholds_training_only":thresholds.tolist(),
            "baseline":{"persistence":baseline,"change":baseline_change},
            "fits":[],"official_validation_read":False,"official_test_read":False,
            "entity_discovery_tested":False,"decision_utility_tested":False,
            "promoted":False}
    for seed in SEEDS:
        for name in ("global_gru","native_tac_assembly"):
            result=fit(name,seed,prepared_train,prepared_val,thresholds,out,steps)
            print(json.dumps(result,sort_keys=True),flush=True)
            report["fits"].append(result)
    (out/"results.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

def evaluate_zero_baseline(data,thresholds):
    _,delta,mask=data[:3]
    changes=(delta.abs()>thresholds[None,:])&mask
    sum_changes=(delta.abs()*changes).sum(0)
    cnt_changes=changes.sum(0)
    eligible=cnt_changes>=10
    return {"change_macro_mae":float((sum_changes[eligible]/cnt_changes[eligible]).mean()) if eligible.any() else None,
            "change_total":int(changes.sum()),"eligible_change_features":int(eligible.sum()),
            "change_direction_accuracy":0.0}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/me001b")
    p.add_argument("--steps",type=int,default=STEPS)
    p.add_argument("--property-only",action="store_true")
    args=p.parse_args()
    if args.property_only:
        check_properties()
    else:
        run(args.output,args.steps)

def check_properties():
    torch.set_num_threads(1)
    torch.manual_seed(20261019)
    x=torch.randn(2,HISTORY,FEATURES*2)
    x[:,:,FEATURES:]=(torch.rand(2,HISTORY,FEATURES)>0.3).float()
    x[:,:,:FEATURES]*=x[:,:,FEATURES:]
    model=NativeTACAssembly()
    y,s=model(x,return_states=True)
    assert y.shape==(2,FEATURES) and torch.isfinite(y).all()
    assert torch.isfinite(s).all() and s.abs().max()<=1.000001
    corrupted=x.clone()
    corrupted[:,:,:FEATURES]+=1e4*(1-x[:,:,FEATURES:])
    assert torch.allclose(model(corrupted),y,rtol=0,atol=0)
    state_changed=(s[:,-1]-s[:,0]).abs().max().item()
    assert state_changed>0
    (y.square().mean()+s.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.token_key.weight.grad is not None
    assert model.token_key.weight.grad.abs().max().item()>0
    # Constant null observation must not inject sensory value; decay may still act.
    missing=torch.zeros_like(x)
    _,history=model(missing,return_states=True)
    assert history.abs().max()==0
    print(json.dumps({"status":"PROPERTY_ONLY","finite":True,"no_masked_leak":True,
                      "state_changed":state_changed,"trainable_router":True},sort_keys=True))

if __name__=="__main__":
    main()
