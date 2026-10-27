"""LATFUSE004 read-only trained factorial head routing/collision diagnosis.

Two source-authed original fitted seeds x four source-frozen arms, genuine
June 2014 physical hourly measurements, no future targets/DEV/TEST, no
optimization. Compare actual writer probabilities, sparse writes, read weights
and within-head source-distinct key aliasing (NOT factual retrieval accuracy).
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts import latfuse002_real_fit as original
from scripts import latfuse004_factorial_training as train
from tac_osm.latent_fusion_004_factorial import FactorialFusion,ARMS,contract,head_weights

START=original.hour(dt.datetime(2014,6,1))
END=START+256


def probe_one(m,d):
    slots=torch.arange(12)
    s=m.initial(12)
    keys=[];encodings=[];routing=[];eligible=[]
    with torch.no_grad():
        for t in range(START,END):
            x=d["values"][:,t];vis=d["valid"][:,t]
            u=m.encode(x,vis,d["age"][:,t],d["last"][:,t])
            k=F.normalize(m.key_map(u).reshape(12,m.head_count,m.key_width),dim=-1,eps=1e-8)
            weights,selected=head_weights(m.write_router(u),m.log_head_temperature,
                                          head_count=m.head_count,soft=m.soft_routing)
            weights=weights*vis.any(-1)[:,None].to(weights.dtype)
            keys.append(k);encodings.append(u);routing.append(weights);eligible.append(
                selected & vis.any(-1)[:,None])
            s=m.step(s,x,vis,d["age"][:,t],d["last"][:,t],slots)
    k=torch.stack(keys,dim=1)[:,-64:]
    u=torch.stack(encodings,dim=1)[:,-64:]
    weight=torch.stack(routing,dim=1)
    selection=torch.stack(eligible,dim=1)
    past=d["values"][:,END-64:END]
    observed=d["valid"][:,END-64:END]
    joint=observed[:,:,None,:]&observed[:,None,:,:]
    overlap=joint.sum(-1)
    dist=(torch.where(joint,(past[:,:,None,:]-past[:,None,:,:]).square(),0)
          .sum(-1)/overlap.clamp_min(1)).sqrt()
    upper=torch.triu_indices(64,64,offset=1)
    distinct=((overlap>=3)&(dist>=1))[:,upper[0],upper[1]]
    n=int(distinct.sum())
    if n<1000:raise RuntimeError("missing genuine physical source-distinct pairs")
    all_valid=torch.stack([d["valid"][:,t].any(-1) for t in range(START,END)],dim=1)
    uses=all_valid.sum().clamp_min(1)
    p=weight[all_valid]
    entropy=-(p.clamp_min(1e-9).log()*p).sum(-1)
    normalized_entropy=float(entropy.mean()/torch.log(torch.tensor(float(m.head_count))))
    concentration=float((p.square().sum(-1)).mean())
    hdiagnostics=[]
    for h in range(m.head_count):
        cos=(k[:,:,h]@k[:,:,h].transpose(1,2))[:,upper[0],upper[1]].abs()
        past_weight=weight[:,-64:,h]
        pairw=(past_weight[:,:,None]*past_weight[:,None,:])[:,upper[0],upper[1]]
        mass=pairw*distinct.to(pairw.dtype)
        denominator=float(mass.sum())
        collision=float(((cos>.9)*mass).sum()/mass.sum()) if denominator>1e-7 else None
        co_selected=selection[:,-64:,h]
        both=(co_selected[:,:,None]&co_selected[:,None,:])[:,upper[0],upper[1]]
        selected_distinct=distinct&both
        nselected=int(selected_distinct.sum())
        hdiagnostics.append(dict(
           head=h,
           mean_actual_write_mass=float(weight[:,:,h].sum()/uses),
           selected_source_step_fraction=float(selection[:,:,h].sum()/uses),
           original_distinct_co_selected_pairs=nselected,
           co_selected_abs_cos_gt_0_9=float((cos[selected_distinct]>.9).float().mean()) if nselected else None,
           source_weighted_abs_cos_gt_0_9=collision,
           weighted_source_distinct_pair_mass=denominator,
           source_key_width=m.key_width))
    read_prob,_=head_weights(m.read_router(torch.cat((s.fast,s.slow),-1)),
                             m.log_head_temperature,head_count=m.head_count,soft=True)
    cand=original.calendar([END+6]*12).unsqueeze(1)
    mean,sd=m.forecast(s,slots,cand)
    ablations=[]
    for h in range(m.head_count):
        matrix=s.associative.clone().reshape(12,16,m.head_count,m.key_width)
        matrix[:,:,h,:]=0
        tampered=s._replace(associative=matrix.reshape(12,16,16))
        pm,ps=m.forecast(tampered,slots,cand)
        ablations.append(dict(head=h,
          mean_abs_forecast_location_change=float((mean-pm).abs().mean()),
          mean_abs_forecast_scale_change=float((sd-ps).abs().mean())))
    return dict(
       input_original_256_hourly_steps=256,
       original_12_source_stations=12,
       source_distinct_recent_pairs=n,
       normalized_write_entropy_mean=normalized_entropy,
       write_probability_concentration_mean=concentration,
       soft_writer_all_receives_nonzero_gradient_if_source_observed=m.soft_routing,
       per_head=hdiagnostics,
       mean_read_head_weights=[float(v) for v in read_prob.mean(0)],
       trained_head_log_temperatures=[float(z) for z in m.log_head_temperature],
       local_head_lesions=ablations,
       source_outcomes_not_scored=True,
       all_original_weights_unchanged=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise RuntimeError("do not overwrite original FIT trained-head diagnostic")
    torch.set_num_threads(1)
    if hashlib.sha256(a.source.read_bytes()).hexdigest()!=original.SOURCE_SHA:
        raise RuntimeError("historical original dataset hash mismatch")
    data=original.source_frames(a.source,phase="fit")
    report=json.loads((train.OUT/"RESULTS.json").read_text())
    if report["total_original_optimizer_updates"]!=1024:
        raise RuntimeError("original experiment not completed")
    results=[]
    with torch.no_grad():
        for seed in train.SEEDS:
            for arm in ARMS:
                record=next((x for x in report["models"]
                    if x["seed"]==seed and x["arm"]==arm),None)
                if record is None:raise RuntimeError("eight-arm source receipt missing")
                path=train.OUT/f"seed-{seed}"/arm/"final.pt"
                sha=hashlib.sha256(path.read_bytes()).hexdigest()
                if sha!=record["original_checkpoint_sha256"]:
                    raise RuntimeError("frozen original FIT checkpoint digest invalid")
                cp=torch.load(path,map_location="cpu",weights_only=False)
                m=FactorialFusion(arm)
                m.load_state_dict(cp["model"],strict=True)
                m.eval()
                assert contract(arm)["per_stream_nominal_bytes"]==6800
                before=[p.detach().clone() for p in m.parameters()]
                outcome=probe_one(m,data)
                if sha!=hashlib.sha256(path.read_bytes()).hexdigest() or any(
                    not torch.equal(x,y) for x,y in zip(before,m.parameters())):
                    raise RuntimeError("no-fit probe changed trained original model")
                row=dict(arm=arm,seed=seed,checkpoint_sha256=sha,diagnosis=outcome)
                results.append(row)
                print("LATFUSE004_FROZEN_ARM="+json.dumps(
                    dict(arm=arm,seed=seed,entropy=outcome["normalized_write_entropy_mean"],
                    write=[r["mean_actual_write_mass"] for r in outcome["per_head"]],
                    weighted_collision=[r["source_weighted_abs_cos_gt_0_9"]
                                        for r in outcome["per_head"]]),sort_keys=True),flush=True)
    r=dict(study="LATFUSE004-FROZEN-HEAD-ROUTING",
       source_sha256=original.SOURCE_SHA,
       scope="June2014 original FIT 12 source-authenticated physical station histories 256 prior hours, no label, protected TEST or exposed DEV targets",
       original_models=8,
       source_pair_difference="at least three jointly observed standardized channels, original sensor normalized RMS >=1",
       head_collision_threshold="absolute cosine >0.9 in actual per-head normalized subspace",
       interpretations="Soft head probability balance and physically distinct cosine angles are proxies only, NOT direct historical event retrieval accuracy",
       original_models_modified=False,optimizer_steps=0,no_protected_numeric_read=True,models=results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LATFUSE004_TRAINED_HEAD_DIAGNOSTIC_PASS",flush=True)
if __name__=="__main__":main()
