#!/usr/bin/env python3
"""ME016: native identity-coupled slow state, fixed math, 600-step strong GRU controls.

Five preregistered equal-capacity arms on authentic KuaiRec FIT/old DEV only.
NO re-opening previously exposed Aug14-Sep06 future as a held-out test.
"""
from __future__ import annotations
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.me014b_exact_action_memory import (
   read_early,build_examples,empirical_baselines,binary_scores,ActionMemory,
   parameters,batch,VOCAB,EMB,HISTORY,SEEDS,UPDATES,BATCH,FIT_CUT,DEV_CUT,
   PIN_MD5)

ARMS=("native_chrono_anchor","native_identity_anchor","gru_anchor","gru_original")
UPDATES=600
BATCH=128
FAST_UPDATE=1/8
SLOW_UPDATE=1/24
SOURCE_SHA="c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c"

class NativeIdentitySlow(ActionMemory):
    """Parameter-free identity fast->slow write. Exact same native parameters."""
    def __init__(self):
        super().__init__("native_tac")
    def forward(self,items,metadata,action,duration,ignore_history=False):
        if items.ndim!=2 or items.shape[1]!=HISTORY or metadata.shape!=(len(items),HISTORY,3):
            raise ValueError("unexpected causal history")
        if action.shape!=(len(items),) or duration.shape!=(len(items),1):
            raise ValueError("unexpected action metadata")
        x=torch.cat((self.emb(items),metadata),dim=-1)
        if ignore_history:x=torch.zeros_like(x)
        z=x.new_zeros(len(x),32);s=z.clone()
        for t in range(HISTORY):
            u=x[:,t]
            q=torch.tanh(self.enc(u)+self.transition(z))
            gf=torch.sigmoid(self.gate(torch.cat((u,z),dim=-1)))
            z=(1-gf)*z+gf*q
            context=torch.cat((z,s),dim=-1)
            gs=torch.sigmoid(self.slow_gate(context))
            s=(1-gs)*s+gs*torch.tanh(self.slow(context)+z)
        return self.head(torch.cat((z,s,self.emb(action),duration),dim=-1))

class EngineeredAgent(nn.Module):
    """Identical trainables; only gate initialization or analytic causal prior."""
    def __init__(self,kind):
        super().__init__()
        if kind not in ARMS:raise ValueError("unregistered arm")
        self.kind=kind
        recurrent="gru" if kind in ("gru_anchor","gru_original") else "native_tac"
        self.core=NativeIdentitySlow() if kind=="native_identity_anchor" else ActionMemory(recurrent)
        self.anchored=kind in ("native_chrono_anchor","native_identity_anchor","gru_anchor")
        if recurrent=="native_tac":
            with torch.no_grad():
                self.core.gate.bias.fill_(math.log(FAST_UPDATE/(1-FAST_UPDATE)))
                self.core.slow_gate.bias.fill_(math.log(SLOW_UPDATE/(1-SLOW_UPDATE)))

    def forward(self,items,metadata,action,duration,ignore_history=False):
        logits=self.core(items,metadata,action,duration,ignore_history=ignore_history)
        if self.anchored and not ignore_history:
            # The CURRENT y is never in metadata. Only prior32 past y are.
            count=metadata[:,:,1].sum(-1,keepdim=True)
            if not bool(((metadata[:,:,1]>=0)&(metadata[:,:,1]<=1)).all()):
                raise ValueError("future/invalid historical feedback entry")
            prior=(count+1)/(HISTORY+2)
            logits=logits+torch.logit(prior)
        return logits

def evaluate(model,records):
    model.eval();pred=[]
    with torch.inference_mode():
        for first in range(0,len(records["y"]),512):
            sl=slice(first,min(first+512,len(records["y"])))
            logits=model(*batch(records,sl))
            pred.append(torch.sigmoid(logits).reshape(-1))
    return binary_scores(torch.cat(pred).numpy(),records["y"].numpy())

def gate_telemetry(model,records,n=256):
    if model.core.arm!="native_tac":return None
    # Direct replay of original genuine native recurrence on real old DEV histories.
    net=model.core;net.eval()
    n=min(n,len(records["y"]))
    with torch.no_grad():
        ids=records["items"][:n];attrs=records["metadata"][:n]
        x=torch.cat((net.emb(ids),attrs),dim=-1)
        z=torch.zeros(n,32);s=z.clone()
        fast=[];slow=[]
        for t in range(HISTORY):
            u=x[:,t]
            q=torch.tanh(net.enc(u)+net.transition(z))
            fg=torch.sigmoid(net.gate(torch.cat((u,z),-1)))
            z=(1-fg)*z+fg*q
            ctx=torch.cat((z,s),dim=-1)
            sg=torch.sigmoid(net.slow_gate(ctx))
            candidate=net.slow(ctx)+z if model.kind=="native_identity_anchor" else net.slow(ctx)
            s=(1-sg)*s+sg*torch.tanh(candidate)
            fast.append(fg);slow.append(sg)
        f=torch.stack(fast,1);g=torch.stack(slow,1)
        h=net.head.weight.detach()[0]
    return {"mean_fast_gate":float(f.mean()),
      "mean_slow_gate":float(g.mean()),
      "fast_gate_q10":float(torch.quantile(f,.1)),
      "fast_gate_q90":float(torch.quantile(f,.9)),
      "slow_gate_q10":float(torch.quantile(g,.1)),
      "slow_gate_q90":float(torch.quantile(g,.9)),
      "fast_head_L2":float(h[:32].norm()),
      "slow_head_L2":float(h[32:64].norm()),
      "initial_fast_bias":float(net.gate.bias.mean()),
      "initial_slow_bias":float(net.slow_gate.bias.mean()),
      "genuine_dev_histories_probed":n}

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    real,src=read_early()
    if src["zip_md5"]!=PIN_MD5 or src["zip_sha256"]!=SOURCE_SHA:
        raise ValueError("original authentic source SHA mismatch")
    if src["future_test_labels_parsed"] or src["later_rows_skipped_before_label_parsing"]<1:
        raise ValueError("training violated original real chronological boundary")
    train=build_examples(real,0,FIT_CUT)
    dev=build_examples(real,FIT_CUT,DEV_CUT)
    if len(train["y"])!=68825 or len(dev["y"])!=106245:
        raise ValueError("original genuine cohort/train/split changed")
    baselines=empirical_baselines(train,dev)
    manifest={
      "study":"ME016_NATIVE_SLOW_IDENTITY_CREDIT_DIAGNOSIS",
      "source":src,"real_train_examples":len(train["y"]),
      "historical_dev_examples":len(dev["y"]),
      "historical_dev_is_not_independent":True,
      "future_Aug14_Sept06_reopened":False,
      "original_Me014B_future_data_not_used":True,
      "all_prior_user1_user2_and_future_user3_labels_excluded":True,
      "seed_alignment":list(SEEDS),
      "fixed_optimizer_updates":UPDATES,"fixed_batch_size":BATCH,
      "original_GRU_strict_parameter_count":135050,
      "historical_baselines":{
        name:binary_scores(baselines[key],dev["y"].numpy())
        for name,key in (("global","global_predictions"),
                         ("item","item_predictions"),
                         ("causal_prior32","history_mean_predictions"))},
      "models":[],"architecture_promoted":False
    }
    torch.set_num_threads(1)
    for seed in SEEDS:
        for kind in ARMS:
            torch.manual_seed(seed)
            model=EngineeredAgent(kind)
            num=parameters(model)
            if num!=135050:raise ValueError("compulsory exact parameter mismatch")
            before=evaluate(model,dev)
            optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
            gen=torch.Generator().manual_seed(seed+300)
            started=time.perf_counter();traces=[]
            for step in range(1,UPDATES+1):
                idx=torch.randint(len(train["y"]),(BATCH,),generator=gen)
                y=train["y"][idx]
                logits=model(*batch(train,idx))
                loss=F.binary_cross_entropy_with_logits(logits,y)
                if not bool(torch.isfinite(loss)):raise ArithmeticError("nonfinite loss")
                optim.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                if not bool(torch.isfinite(grad)):raise ArithmeticError("nonfinite gradients")
                optim.step()
                if step in (1,UPDATES//2,UPDATES):
                    traces.append({"step":step,"loss":float(loss.detach()),"grad_norm":float(grad.detach())})
            current=evaluate(model,dev)
            dynamics=gate_telemetry(model,dev)
            ck=out/"checkpoints"/f"{kind}-{seed}.pt"
            ck.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"kind":kind,"seed":seed,"steps":UPDATES,
                        "source_sha256":src["zip_sha256"],
                        "state_dict":model.state_dict()},ck)
            entry={"model":kind,"seed":seed,"trainable_params":num,
                "actual_training_seconds":round(time.perf_counter()-started,3),
                "initial_historical_score":before,
                "trained_historical_score":current,
                "gate_diagnostics_real_historical_dev":dynamics,
                "loss_trace":traces,
                "checkpoint":str(ck.relative_to(out)),
                "checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest()}
            manifest["models"].append(entry)
            print(json.dumps({"kind":kind,"seed":seed,
                  "brier":current["brier"],"logloss":current["binary_logloss"],
                  "training_s":entry["actual_training_seconds"],
                  "real_slow_gate":dynamics["mean_slow_gate"] if dynamics else None},sort_keys=True),flush=True)
    report=out/"results.json"
    report.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":"ME016_600_STEP_ORIGINAL_DIAGNOSTIC_FITS_COMPLETE",
        "trained_models":len(manifest["models"]),
        "fresh_independent_success_claim":False,
        "unopened_future_test_used":False},sort_keys=True),flush=True)

def preflight():
    torch.manual_seed(30);torch.set_num_threads(1)
    ids=torch.randint(VOCAB,(3,HISTORY))
    meta=torch.rand(3,HISTORY,3)
    meta[:,:,1]=(meta[:,:,1]>.5).float()
    action=torch.randint(VOCAB,(3,))
    duration=torch.rand(3,1)
    expected=(meta[:,:,1].sum(1,keepdim=True)+1)/(HISTORY+2)
    counts={}
    for mode in ARMS:
        m=EngineeredAgent(mode);counts[mode]=parameters(m)
        p=torch.sigmoid(m(ids,meta,action,duration))
        target=expected if m.anchored else torch.full_like(expected,.5)
        assert torch.allclose(p,target,atol=1e-6),(mode,p,target)
        assert torch.allclose(torch.sigmoid(m(ids,meta,action,duration,ignore_history=True)),
                              torch.full_like(expected,.5),atol=1e-6)
        if mode in ("native_chrono_anchor","native_identity_anchor"):
            assert abs(float(torch.sigmoid(m.core.gate.bias).mean())-FAST_UPDATE)<1e-6
            assert abs(float(torch.sigmoid(m.core.slow_gate.bias).mean())-SLOW_UPDATE)<1e-6
        loss=F.binary_cross_entropy_with_logits(m(ids,meta,action,duration),torch.rand_like(expected))
        loss.backward()
        assert all(bool(torch.isfinite(v.grad).all()) for v in m.parameters() if v.grad is not None)
    assert set(counts.values())=={135050},counts
    assert len(ARMS)==4
    print(json.dumps({"status":"ME016_IDENTITY_SLOW_CREDIT_MATH_PASS",
       "trainable_params":counts,"new_learnable_modules_added":False},sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/me016")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    preflight() if a.self_test else run(a.out)
