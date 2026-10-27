"""NATIVELATENT002: locate encoder-vs-address aliasing and memory influence timescale.

Runs only on SHA-restored five original native ME111 learned checkpoints and
authentic source FIT observations. READ ONLY. No scores/targets/fits/TEST.
A state-memory zero intervention is a local OOD mechanism probe, not causality.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F

from scripts import me111_conditional_training as run
from scripts.native_latent_001_diagnosis import checked_index,read_frame,model_for,SEEDS,ORIGIN

def original_sensor_conditional_geometry(model, inputs):
    retained={}
    def capture_enc(mod,ins,out):
        if out.ndim==3:
            retained["embedding"]=out.detach().clone()
    def capture_key(mod,ins,out):
        if out.ndim==3:
            retained["key"]=F.normalize(out.detach().clone(),dim=-1,eps=1e-6)
    henc=model.encoder.register_forward_hook(capture_enc)
    hkey=model.address.register_forward_hook(capture_key)
    try:
        model(*inputs)
    finally:
        henc.remove();hkey.remove()
    if set(retained)!={"embedding","key"}:
        raise RuntimeError("source native encoder or key schedule was not captured")
    e=retained["embedding"][:,-64:]
    k=retained["key"][:,-64:]
    sensor=inputs[0][:,-64:]
    mask=inputs[1][:,-64:]
    upper=torch.triu_indices(64,64,offset=1)
    joint=mask[:,:,None,:]&mask[:,None,:,:]
    overlap=joint.sum(-1)
    ds=(sensor[:,:,None,:]-sensor[:,None,:,:]).square()
    source_rms=(torch.where(joint,ds,0).sum(-1)/overlap.clamp_min(1)).sqrt()
    distinct=(source_rms[:,upper[0],upper[1]]>=1)&(overlap[:,upper[0],upper[1]]>=3)
    if int(distinct.sum())<200:raise RuntimeError("not enough distinct original sensor source pairs")
    pe=F.normalize(e,dim=-1,eps=1e-6)
    ec=(pe@pe.transpose(1,2))[:,upper[0],upper[1]].abs()
    kc=(k@k.transpose(1,2))[:,upper[0],upper[1]].abs()
    # Report both all-pair and source-different similarities: all-pair
    # overlap may legitimately reflect slowly varying atmospheric readings.
    result={
      "candidate_pairs_with_observed_different_values":int(distinct.sum()),
      "mean_abs_encoder_cosine_all_pairs":float(ec.mean()),
      "mean_abs_address_cosine_all_pairs":float(kc.mean()),
      "mean_abs_encoder_cosine_distinct_source_pairs":float(ec[distinct].mean()),
      "mean_abs_address_cosine_distinct_source_pairs":float(kc[distinct].mean()),
      "encoder_sim_gt_0_9_for_distinct_source":float((ec[distinct]>.9).float().mean()),
      "address_sim_gt_0_9_for_distinct_source":float((kc[distinct]>.9).float().mean()),
      "address_minus_encoder_mean_abs_similarity_distinct_source":float(
        kc[distinct].mean()-ec[distinct].mean()),
      "causal_embedding_claim":False,
    }
    return result

def tail_memory_lesions(model, inputs):
    T=inputs[0].shape[1]
    out={}
    for L in (1,4,16,64,128):
        cut=T-L
        pref=[x[:,:cut] if i!=4 and x.ndim>=2 else x for i,x in enumerate(inputs)]
        tail=[x[:,cut:] if i!=4 and x.ndim>=2 else x for i,x in enumerate(inputs)]
        _,s=model(*pref)
        normal,_=model(*tail,state=s)
        lesion=replace(s,memory=torch.zeros_like(s.memory))
        removed,_=model(*tail,state=lesion)
        out[str(L)]={
          "log_location_mean_absolute_change":float(
            (normal.log_location-removed.log_location).abs().mean()),
          "left_width_mean_absolute_change":float(
            (normal.left_scale-removed.left_scale).abs().mean()),
          "right_width_mean_absolute_change":float(
            (normal.right_scale-removed.right_scale).abs().mean()),
          "original_pre_tail_memory_frobenius_mean":float(
            s.memory.norm(dim=(-2,-1)).mean()),
          "perturbation":"prefix memory set to exact zero at same checkpoint; continuation real chronological FIT inputs unchanged",
        }
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise RuntimeError("diagnostic output already committed: no overwrite")
    torch.set_num_threads(1)
    d=run.fit_data()
    if d["phase"]!="fit" or hashlib.sha256(run.data.ARCHIVE.read_bytes()).hexdigest()!=run.data.SHA:
        raise RuntimeError("original source authority failed")
    idx=checked_index(run)
    inputs=read_frame(d,[(i,ORIGIN) for i in range(12)],512)
    rows=[]
    with torch.no_grad():
        for seed in SEEDS:
            m,path,sha=model_for(run,d,seed,idx[seed])
            before=m.versions()
            geometry=original_sensor_conditional_geometry(m,inputs)
            memory=tail_memory_lesions(m,inputs)
            if before!=m.versions() or sha!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise RuntimeError("original checkpoint modified during read-only diagnosis")
            result={"seed":seed,"original_checkpoint_sha256":sha,
                    "source_geometry":geometry,"tail_memory_influence":memory}
            rows.append(result)
            print("NATIVE_ADDRESS_SEED_JSON="+json.dumps(result,sort_keys=True),flush=True)
    out={"study":"NATIVELATENT002","scope":"READ_ONLY_POST_HOC_FIT_2013_2014",
         "source_sha256":run.data.SHA,"original_fit_origin_hour":ORIGIN,
         "station_count":12,"lookback":512,"no_fit":True,
         "protected_2016_2017_untouched":True,
         "source_exposed_2015_report_untouched":True,
         "future_labels_not_read":True,"archival_checkpoint_mutation":False,
         "identification_limit":"encoder-vs-key geometry and OOD memory lesions do not prove usable content-specific long-term recall",
         "models":rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("NATIVE_ADDRESS_RESULT_JSON="+json.dumps({"status":"PASS","models":len(rows),"no_fit":True},
       sort_keys=True),flush=True)

if __name__=="__main__":
    main()
