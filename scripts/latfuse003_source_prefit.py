"""LATFUSE003 original real observation/gradient preflight: ZERO optimizer updates.

This checks candidate source wiring, exact base inherited initialization and
measured incomplete CPU-profiler compute. DO NOT interpret prefit loss or
constructed masks as trained prediction or matching full training FLOPs.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import torch
from torch.profiler import profile,ProfilerActivity
from scripts import latfuse002_real_fit as base
from tac_osm.latent_fusion_002 import FusedLatentFixedState,state_contract
from tac_osm.latent_fusion_003 import FusedLatentFourHead,exact_state_contract

def case(model,data,source_ids,times,add_aux):
    batch=len(source_ids)
    sid=torch.tensor(source_ids,dtype=torch.long)
    origins=torch.tensor(times,dtype=torch.long)
    state=model.initial(batch)
    enc=[]
    hook=model.embed.register_forward_hook(lambda _m,_args,out:enc.append(out))
    try:
        for i in range(base.H):
            t=origins-base.H+1+i
            state=model.step(state,data["values"][sid,t],data["valid"][sid,t],
                             data["age"][sid,t],data["last"][sid,t],sid)
    finally:
        hook.remove()
    # Exogenous future-known timestamp, not physical action or outcome.
    q=base.calendar([int(t+base.HORIZON) for t in origins]).unsqueeze(1)
    forecast,scale=model.forecast(state,sid,q)
    # Open authentic FIT future numeric measurements only AFTER forecast.
    y=data["values"][sid,origins+base.HORIZON]
    mask=data["valid"][sid,origins+base.HORIZON]
    z=torch.where(mask,y,forecast[:,0,:].detach())
    residual=(z-forecast[:,0,:])/scale[:,0,:]
    nll=(.5*residual.square()+scale[:,0,:].log()+.9189385332046727)
    loss=(.8*nll+.2*(z-forecast[:,0,:]).abs())*mask
    loss=loss.sum()/mask.sum().clamp_min(1)
    additional={"orthogonality":0.,"balance":0.}
    if add_aux:
        u=torch.stack(enc,dim=1)
        window=torch.stack([data["values"][sid,origins-base.H+1+i] for i in range(base.H)],1)
        seen=torch.stack([data["valid"][sid,origins-base.H+1+i] for i in range(base.H)],1)
        ortho=model.source_key_separation_loss(u,window,seen)
        balance=model.head_utilization_loss(u,seen)
        loss=loss+.01*ortho+.005*balance
        additional={k:float(v.detach()) for k,v in (("orthogonality",ortho),("balance",balance))}
    return loss,additional

def run(source):
    torch.set_num_threads(1)
    data=base.source_frames(source,phase="fit")
    assert hashlib.sha256(source.read_bytes()).hexdigest()==base.SOURCE_SHA
    assert len(base.SITES)==12
    # FIT only at recorded hour, every original physical station once.
    origin=base.hour(dt.datetime(2014,6,2,0))
    slots=list(range(8))
    at=[origin]*8
    assert origin+base.HORIZON<base.hour(base.TRAIN_END)
    torch.manual_seed(31027)
    old=FusedLatentFixedState()
    torch.manual_seed(31027)
    new=FusedLatentFourHead()
    orig=old.state_dict();curr=new.state_dict()
    for name,p in orig.items():
        if name not in curr or not torch.equal(p,curr[name]):
            raise RuntimeError("candidate changed original parent initializer tensor: "+name)
    baseline=state_contract(old)
    candidate=exact_state_contract(new)
    if (baseline["nominal_fp32_bytes"]!=candidate["nominal_bytes"] or
        (candidate["trainable_parameters"]-baseline["trainable_parameters"])/
              candidate["trainable_parameters"]>.01):
        raise RuntimeError("source fit registered fixed-FP32 state and <=1% weights violated")
    reports=[]
    for label,m,aux in (("original_BASE",old,False),("four_head",new,True)):
        original_parameters=[p.detach().clone() for p in m.parameters()]
        m.train()
        m.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU],with_flops=True,
                     record_shapes=False) as events:
            loss,penalties=case(m,data,slots,at,aux)
            loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(label+" source prefit nonfinite")
        gradients={}
        for name,p in m.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError(label+" missing or nonfinite real FIT gradient for "+name)
            gradients[name]=float(p.grad.detach().norm())
        if any(not torch.equal(p.detach(),init)
               for p,init in zip(m.parameters(),original_parameters)):
            raise RuntimeError(label+" no-fit audit mutated learned weights")
        flop=sum(t.flops for t in events.key_averages())
        reports.append(dict(model=label,recorded_source_fit_loss_no_update=float(loss.detach()),
            external_target_from_real_source_only=True,
            gradient_tensors_with_finite_values=len(gradients),
            measured_partial_torch_profiler_flops=flop,
            flops_scope="torch CPU supported operators forward+autograd; misses optimizer and many ops",
            auxiliary_penalties=penalties,optimizer_updates=0,
            model_parameters=sum(p.numel() for p in m.parameters())))
    return dict(study="LATFUSE003-NOFIT-PREFLIGHT",
        source_sha256=base.SOURCE_SHA,source_phase="FIT_2013_2014_ONLY",
        input_source="original 8 physical stations 2014-06-02 00:00; exactly 24 real past hourly readings; observed +6 FIT future target only enters no-update loss after forecast",
        inheritance_parent_tensor_exact=True,original_state_bytes=6800,
        candidate_state_bytes=6800,candidate_head_matrix_cells=256,
        candidate_parameters=99747,parent_parameters=98775,
        parameter_gap_fraction=(99747-98775)/99747,
        no_optimizer_created=True,no_optimizer_updates=True,
        no_real_capability_or_equal_compute_superiority_claim=True,
        tested_model_graphs=reports)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    a=p.parse_args()
    print("LATFUSE003_SOURCE_PREFIT="+json.dumps(run(a.source),sort_keys=True),flush=True)
