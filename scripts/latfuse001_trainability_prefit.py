"""LATFUSE001 real FIT-only *no optimizer* 100k-scale source trainability preflight.

Future calendar is a legally future-known exogenous covariate, NOT a candidate
physical intervention. Actual future physical readings are used only after
read-only forecast to form a source-authorized training gradient; NO step,
checkpoints, TEST labels or architecture ranking. No model capability claimed.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import torch
from tac_osm.latent_fusion_001 import FusedLatentCandidate
from latfuse001_source_diagnostic import load_first_real_fit

def known_calendar(hour):
    dt=datetime.datetime(2013,3,1)+datetime.timedelta(hours=int(hour))
    angle=[2*math.pi*dt.hour/24,2*math.pi*dt.weekday()/7,
           2*math.pi*(dt.timetuple().tm_yday-1)/366]
    return torch.tensor([math.sin(v) for v in angle]+[math.cos(v) for v in angle],dtype=torch.float32)

def run(path):
    torch.manual_seed(20261027)
    torch.set_num_threads(1)
    source, digest=load_first_real_fit(path)
    assert source.shape==(12,512,13)
    transformed=source.clone()
    transformed[:,:,:6]=torch.where(
        torch.isfinite(source[:,:,:6]),torch.log1p(source[:,:,:6].clamp_min(0)),
        source[:,:,:6])
    train=transformed[:,:256]
    mask=torch.isfinite(train)
    mu=torch.where(mask,train,0.).sum((0,1))/mask.sum((0,1)).clamp_min(1)
    std=(torch.where(mask,train-mu,0.).square().sum((0,1))/mask.sum((0,1)).clamp_min(1)).clamp_min(1e-6).sqrt()
    src=((transformed-mu)/std).float()
    model=FusedLatentCandidate(13,6,hidden=80,key_dim=16,val_dim=16,entity_count=16)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed=sum(p.numel() for p in model.parameters())
    if not (95000<=params<=105000 and params==deployed):
        raise RuntimeError(f"100k source prefit INELIGIBLE: {params}/{deployed}")
    # Explicit recorded source station identity; 12 distinct original stations.
    station=torch.arange(12)
    state=model.initial(12)
    last=torch.zeros((12,13),dtype=torch.float32)
    age=torch.full((12,13),256.,dtype=torch.float32)
    saved=[p.detach().clone() for p in model.parameters()]
    # Only causal source up through t=319 reaches model.
    for t in range(256,320):
        x=src[:,t,:]
        observed=torch.isfinite(x)
        source_x=torch.where(observed,x,0.)
        elapsed=torch.where(observed,torch.zeros_like(age),torch.clamp(age+1,max=256))
        state=model.step(state,source_x,observed,elapsed,last,station)
        last=torch.where(observed,source_x,last)
        age=elapsed
    # The future *calendar* is known from t=319 (not a synthetic action).
    q=known_calendar(325).view(1,1,6).expand(12,1,6)
    mu_pred, sigma_pred=model.forecast(state,station,q)
    mu_pred=mu_pred[:,0,:];sigma_pred=sigma_pred[:,0,:]
    assert mu_pred.shape == sigma_pred.shape == (12,13)
    assert torch.isfinite(mu_pred).all() and (sigma_pred>0).all()
    before=(mu_pred.detach().clone(),sigma_pred.detach().clone())
    # True FIT-only numeric future readings are supervision only, never passed to model.
    target=src[:,325,:]
    eligible=torch.isfinite(target)
    safe_y=torch.where(eligible,target,mu_pred.detach())
    gaussian=.5*(math.log(2*math.pi)+2*torch.log(sigma_pred)+((safe_y-mu_pred)/sigma_pred).square())
    loss=(gaussian*eligible).sum()/eligible.sum()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite real FIT source loss")
    grads=torch.autograd.grad(loss,list(model.parameters()),allow_unused=True)
    nonzero=sum(int(g is not None and bool((g.abs()>0).any())) for g in grads)
    tensor_count=len(saved)
    if nonzero!=tensor_count:
        raise RuntimeError(f"actual source gradient dead paths: {nonzero}/{tensor_count}")
    if not all(torch.equal(p.detach(),s) for p,s in zip(model.parameters(),saved)):
        raise RuntimeError("optimizer-free source analysis altered learned weights")
    again=model.forecast(state,station,q)
    if not (torch.equal(before[0],again[0][:,0,:]) and torch.equal(before[1],again[1][:,0,:])):
        raise RuntimeError("predictions depend on future source training labels")
    return dict(status="PREFIT_ONLY_NO_MODEL_FIT",source_sha256=digest,
                source="UCI501 original 12 stations, first512 2013 hours only",
                training_history="2013 source hours256..319, cutoff319",
                future_calendar="deterministically known +6 hours at 325, exogenous not an action",
                FIT_target="2013 source hour325 observed channels only, accessed after forecast",
                learned_parameters=params,deployed_learned_parameters=deployed,
                eligible_parameter_band=True,optimization_steps=0,loss_kind="masked FIT-only Gaussian forward/backward",
                training_score_claim=False,eligible_future_target_channels=int(eligible.sum()),
                active_gradient_tensors=nonzero,total_parameter_tensors=tensor_count,
                weights_exactly_unchanged=True,source_prediction_purity=True,
                model_prediction_qualification=False,independent_source_anticheat=False,
                admission={"research_parity_manifest":"diagnostic only; NOT a preregistered fit manifest",
                           "matched_neural_control":False,"training_allowed":False})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True)
    args=ap.parse_args()
    print("LATFUSE001_100K_PREFIT_JSON="+json.dumps(run(args.source),sort_keys=True))
if __name__=="__main__":
    main()
