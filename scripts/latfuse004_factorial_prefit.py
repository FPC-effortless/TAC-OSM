"""LATFUSE004 four-arm genuine-FIT input parity preflight: NO optimizer, NO scores.

Profile identical 8 authentic station-hour source histories with each model.
Partial torch-supported forward+autograd FLOPs are NOT full train FLOP parity.
Report actual parameter pairwise gap, source-only gradient and matching input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from torch.profiler import profile,ProfilerActivity

from scripts import latfuse002_real_fit as original
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_004_factorial import ARMS,FactorialFusion,contract

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse004/FAIR_COMPARISON.json"


def source_batch(data,model,examples):
    sid=torch.tensor([r[0] for r in examples])
    ti=torch.tensor([r[1] for r in examples])
    memory=model.initial(len(examples))
    enc=[]
    handle=model.embed.register_forward_hook(lambda mod,inp,out:enc.append(out))
    try:
        for j in range(24):
            t=ti-23+j
            memory=model.step(memory,data["values"][sid,t],data["valid"][sid,t],
                              data["age"][sid,t],data["last"][sid,t],sid)
    finally:
        handle.remove()
    clock=original.calendar([int(t+6) for t in ti]).unsqueeze(1)
    mean,sd=model.forecast(memory,sid,clock)
    # Exact legal boundary: future outcomes only read after predict.
    target=data["values"][sid,ti+6]
    valid=data["valid"][sid,ti+6]
    mu=mean[:,0];scale=sd[:,0]
    y=torch.where(valid,target,mu.detach())
    loss_gauss=.5*(1.8378770664093453+2*scale.log()+((y-mu)/scale).square())
    main=((.8*loss_gauss+.2*(y-mu).abs())*valid).sum()/valid.sum().clamp_min(1)
    z=torch.stack(enc,dim=1)
    historic=torch.stack([ti-23+j for j in range(24)],dim=1)
    values=data["values"][sid[:,None],historic]
    mask=data["valid"][sid[:,None],historic]
    key=model.common_key_loss(z,values,mask)
    balance=model.common_head_balance(z,mask)
    return main+.25*key+.03*balance, (main,key,balance)


def run(src):
    if hashlib.sha256(src.read_bytes()).hexdigest()!=original.SOURCE_SHA:
        raise RuntimeError("real source SHA mismatch")
    check_manifest(json.loads(REG.read_text()),"preflight")
    d=original.source_frames(src,phase="fit")
    # fixed source IDs/sample order all arms; no tuning or model-dependent filtering
    r=random.Random(404)
    examples=[(r.randrange(12),r.randrange(24,original.hour(original.TRAIN_END)-7))
               for _ in range(8)]
    assert len(examples)==8
    results=[]
    for arm in ARMS:
        torch.manual_seed(40411)
        model=FactorialFusion(arm)
        n=sum(p.numel() for p in model.parameters())
        before=[p.detach().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU],with_flops=True) as prof:
            loss,components=source_batch(d,model,examples)
            loss.backward()
        grads={name:p.grad for name,p in model.named_parameters()}
        if not torch.isfinite(loss) or any(g is None or not torch.isfinite(g).all()
                                          for g in grads.values()):
            raise RuntimeError(arm+" original source AD disconnected/nonfinite")
        if any(not torch.equal(p,q) for p,q in zip(model.parameters(),before)):
            raise RuntimeError("no-fit trained weights changed")
        flops=sum(evt.flops for evt in prof.key_averages())
        results.append(dict(arm=arm,source_gradient_tensors=len(grads),
                            partial_profiled_forward_backward_flops=flops,
                            fp32_state_bytes=contract(arm)["per_stream_nominal_bytes"],
                            trainable_parameters=n,
                            source_masked_no_optimizer_loss=float(loss.detach()),
                            component_source_losses=[float(x.detach()) for x in components],
                            optimizer_updates=0))
        print("LATFUSE004_PREFIT_ARM="+json.dumps(results[-1],sort_keys=True),flush=True)
    flops=[r["partial_profiled_forward_backward_flops"] for r in results]
    weights=[r["trainable_parameters"] for r in results]
    gap=(max(flops)-min(flops))/max(flops)
    weightgap=(max(weights)-min(weights))/max(weights)
    return dict(study_id="LATFUSE004-FACTORIAL-READONLY-PREFLIGHT",
        source_sha256=original.SOURCE_SHA,scope="original real FIT 2013-2014 genuine station-hour histories",
        real_source_selection_sha256=hashlib.sha256(json.dumps(examples,separators=(",",":")).encode()).hexdigest(),
        verified_6800_state_in_all_arms=True,exact_same_station_hour_cases=True,
        weight_parameter_gap=weightgap,partial_profiled_compute_gap=gap,
        preliminary_partial_compute_parity_1pct=(gap<=.01),
        full_forward_backward_optimizer_flops_1pct_verified=False,
        matched_fair_training_authorized=False,optimizers_created=0,
        exact_real_source_output_qualification=False,
        no_protected_or_2015_numeric_scored=True,models=results)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    print("LATFUSE004_PREFIT_RESULT="+json.dumps(run(args.source),sort_keys=True),flush=True)
if __name__=="__main__":
    main()
