"""LATFUSE003 original-trained source-authentic head forensics, no refit/outcomes.

Model original SHA from source-frozen two-seed result, genuine June2014 FIT real
past observed sensor pairs. Diagnose key/head collapse, route specialization,
per-head forecast contribution and selected-head collisions. Does NOT infer
retrieval accuracy/causality/calibration or score later exposed/protected years.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from scripts import latfuse002_real_fit as data
from scripts import latfuse003_real_training as fit
from tac_osm.latent_fusion_003 import FusedLatentFourHead,exact_state_contract,_router

START=data.hour(dt.datetime(2014,6,1))
END=START+256

def score_model(model,d):
    b=len(data.SITES)
    ids=torch.arange(b)
    st=model.initial(b)
    keys=[];encoder=[];write_probs=[];write_which=[];raw=[]
    for t in range(START,END):
        values=d["values"][:,t];mask=d["valid"][:,t]
        u=model.encode(values,mask,d["age"][:,t],d["last"][:,t])
        encoder.append(u)
        k=F.normalize(model.key_map(u).reshape(b,4,4),dim=-1)
        keys.append(k)
        distribution,selected=_router(model.write_router(u),model.log_head_temperature,top_k=2)
        write_probs.append(distribution)
        write_which.append(selected & mask.any(-1)[:,None])
        st=model.step(st,values,mask,d["age"][:,t],d["last"][:,t],ids)
    keys=torch.stack(keys,1)
    embeddings=torch.stack(encoder,1)
    route=torch.stack(write_which,1)
    probs=torch.stack(write_probs,1)
    observed=d["valid"][:,START:END]
    source=d["values"][:,START:END]
    valid=observed.any(-1)
    # All source distinct pair predicates grounded on 13 actual measured
    # standardized physical channels from the same real monitoring station.
    recent=64
    s=source[:,-recent:]
    m=observed[:,-recent:]
    k=keys[:,-recent:]
    e=embeddings[:,-recent:]
    h=route[:,-recent:]
    joint=m[:,:,None,:]&m[:,None,:,:]
    count=joint.sum(-1)
    dist=(torch.where(joint,(s[:,:,None,:]-s[:,None,:,:]).square(),0)
          .sum(-1)/count.clamp_min(1)).sqrt()
    upper=torch.triu_indices(recent,recent,offset=1)
    distinct=((count>=3)&(dist>=1))[:,upper[0],upper[1]]
    total=int(distinct.sum())
    if total<1000:raise RuntimeError("real source distinct pairs insufficient")
    # Global16 key is NOT the actual per-head read key; report distinctly.
    flat=F.normalize(k.flatten(-2),dim=-1)
    fullcos=(flat@flat.transpose(1,2))[:,upper[0],upper[1]].abs()
    enc=F.normalize(e,dim=-1)
    encos=(enc@enc.transpose(1,2))[:,upper[0],upper[1]].abs()
    perhead=[]
    for i in range(4):
        cos=(k[:,:,i]@k[:,:,i].transpose(1,2))[:,upper[0],upper[1]].abs()
        both=h[:,upper[0],i]&h[:,upper[1],i]
        eligible=distinct&both
        n=int(eligible.sum())
        perhead.append(dict(head=i,source_distinct_shared_routed_pairs=n,
            distinct_co_routed_abs_cos_gt_0_9=float((cos[eligible]>.9).float().mean()) if n else None,
            mean_cos_source_distinct_co_routed=float(cos[eligible].mean()) if n else None,
            source_step_selected_fraction=float((route[:,:,i]&valid).sum()/valid.sum().clamp_min(1)),
            source_step_write_probability_mean=float(torch.where(valid,probs[:,:,i],0).sum()/valid.sum().clamp_min(1))))
    future=data.calendar([END+6]*b).unsqueeze(1)
    mean,scale=model.forecast(st,ids,future)
    delta=[]
    for i in range(4):
        matrix=st.associative.clone().reshape(b,16,4,4)
        matrix[:,:,i,:]=0.
        prior=st._replace(associative=matrix.reshape(b,16,16))
        m0,sd0=model.forecast(prior,ids,future)
        delta.append(dict(head=i,mean_abs_forecast_location_change=float((mean-m0).abs().mean()),
                          mean_abs_forecast_scale_change=float((scale-sd0).abs().mean())))
    read_router_weights,_=_router(model.read_router(torch.cat((st.fast,st.slow),dim=-1)),
                 model.log_head_temperature,top_k=4)
    return dict(
        physical_source_distinct_pair_count=total,
        encoder80_abs_cos_gt_0_9_for_distinct=float((encos[distinct]>.9).float().mean()),
        global16_abs_cos_gt_0_9_for_distinct=float((fullcos[distinct]>.9).float().mean()),
        global16_mean_abs_cosine_source_distinct=float(fullcos[distinct].mean()),
        mean_write_route_weights_per_head=[
            float((route[:,:,i]&valid).sum()/valid.sum().clamp_min(1)) for i in range(4)],
        active_head_count_per_actual_observation_exact_two=True,
        per_head=perhead,
        mean_read_head_weight=[float(x) for x in read_router_weights.mean(0)],
        forecast_ablations=delta,
        trained_log_head_temperature=[float(x) for x in model.log_head_temperature],
        all_original_weights_frozen=True,
        source_outcomes_scored=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    arg=p.parse_args()
    if arg.output.exists():raise RuntimeError("read-only source original report already exists")
    torch.set_num_threads(1)
    if hashlib.sha256(arg.source.read_bytes()).hexdigest()!=data.SOURCE_SHA:
        raise RuntimeError("source SHA mismatch")
    d=data.source_frames(arg.source,phase="fit")
    if END>=data.hour(data.TRAIN_END):
        raise RuntimeError("original FIT-only limits exceeded")
    original=json.loads((fit.OUT/"RESULTS.json").read_text())
    if original["trained_seeds"]!=list(fit.SEEDS):raise RuntimeError("source original fit list mismatch")
    evidence=[]
    with torch.no_grad():
        for r in original["results"]:
            seed=r["seed"]
            filename=fit.OUT/f"seed-{seed}"/"final.pt"
            sha=hashlib.sha256(filename.read_bytes()).hexdigest()
            if sha!=r["original_checkpoint_sha256"]:raise RuntimeError("original checkpoint changed")
            cp=torch.load(filename,map_location="cpu",weights_only=False)
            m=FusedLatentFourHead()
            m.load_state_dict(cp["model"],strict=True)
            m.eval()
            assert exact_state_contract(m)["nominal_bytes"]==6800
            before=[v.detach().clone() for v in m.parameters()]
            metrics=score_model(m,d)
            if sha!=hashlib.sha256(filename.read_bytes()).hexdigest() or any(
               not torch.equal(v,z) for v,z in zip(m.parameters(),before)):
                raise RuntimeError("read-only source experiment mutated learned model")
            evidence.append(dict(seed=seed,original_checkpoint_sha256=sha,diagnosis=metrics))
            print("LATFUSE003_TRAINED_HEAD_JSON="+json.dumps(evidence[-1],sort_keys=True),flush=True)
    out=dict(study="LATFUSE003-TRAINED-HEAD-READONLY",
       source_sha256=data.SOURCE_SHA,
       scope="original UCI501 FIT original June2014 12 registered stations; 256 source inputs only",
       no_fit=True,no_optimizer=True,no_protected_or_exposed_DEV_numeric_read=True,
       no_retrieval_ground_truth_or_heldout_capability_claim=True,
       associative_capacity_cells=256,fp32_recurrent_state_bytes=6800,
       models=evidence)
    arg.output.parent.mkdir(parents=True,exist_ok=True)
    arg.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE003_TRAINED_HEAD_READONLY_PASS",flush=True)

if __name__=="__main__":
    main()
