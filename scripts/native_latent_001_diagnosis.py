"""NATIVELATENT001: checkpoint-restored native ME111/ME110 memory forensics.

NO OPTIMIZER. NO NEW FIT. NO TARGET LABELS. NO PROTECTED/2015 REPORT.
Source: original FIT 2013-2014 UCI501; 12 physical stations, authentic chronology.
All local memory ablations are OOD sensitivity, NOT causal effects or retrieval accuracy.
The native 152-FP32-scalar carried state is source/evidence statistics plus M8x8,
not the earlier prototype's supposedly carried fast/slow 32D vectors.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts import me111_conditional_training as new
from scripts import me110_typed_training as old

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(111001,111002,111003,111004,111005)
ORIGIN=12000
HISTORIES=(16,64,256,512)
INPUTS=("values","observed","ages","clock","owners","hours")

def checked_index(study):
    file=study.OUT/"FIT_INDEX.json"
    index=json.loads(file.read_text())
    assert len(index["models"])==5
    return {r["seed"]:r["checkpoint_sha256"] for r in index["models"]}

def read_frame(d,origins,length):
    # Same six authenticated prediction-time inputs as original runner, but
    # no future-target extraction, label masks, augmentation or changed ages.
    ids=np.array([p[0] for p in origins],dtype=np.int64)
    at=np.array([p[1] for p in origins],dtype=np.int64)
    ix=at[:,None]-np.arange(length-1,-1,-1)[None,:]
    if (ix.min()<0 or at.max()+6>=new.data.FIT_STOP or
        np.any(ids<0) or np.any(ids>=12)):
        raise ValueError("requested frame outside FIT/source boundaries")
    names=("values","observed","ages","clock","hours")
    vals=[torch.from_numpy(d[n][ids[:,None],ix].copy()) for n in names]
    return vals[0],vals[1],vals[2],vals[3],torch.from_numpy(ids),vals[4]

def model_for(run,d,seed,checked_sha):
    path=run.OUT/run.ARCH/f"seed-{seed}"/"final.pt"
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=checked_sha:raise RuntimeError("original native model checkpoint hash invalid")
    cp=torch.load(path,map_location="cpu",weights_only=False)
    assert cp["seed"]==seed and cp["steps"]==run.STEPS
    m=run.initialize(d,seed)
    m.load_state_dict(cp["model"],strict=True)
    count=sum(p.numel() for p in m.parameters())
    if count!=104663 or any(p.dtype!=torch.float32 for p in m.parameters()):
        raise RuntimeError("original model 104663 FP32 trained params not preserved")
    m.eval()
    return m,path,actual

def observed_norm(x):
    v=x.detach().float()
    return dict(mean=float(v.mean()),min=float(v.min()),max=float(v.max()))

def value_delta(a,b):
    return float((a.log_location-b.log_location).detach().abs().mean())

def inspect_native(m, d, inputs):
    B=inputs[0].shape[0]
    capture={"key":[],"embedding":[]}
    def address_hook(mod,ins,out):
        capture["key"].append(F.normalize(out.detach().clone(),dim=-1,eps=1e-6))
    def encoder_hook(mod,ins,out):
        capture["embedding"].append(out.detach().clone())
    h1=m.address.register_forward_hook(address_hook)
    h2=m.encoder.register_forward_hook(encoder_hook)
    try:
        with torch.no_grad():
            base,state,trace=m(*inputs,return_trace=True)
    finally:
        h1.remove();h2.remove()
    assert capture["key"] and capture["embedding"]
    first_key=next(t for t in capture["key"] if t.ndim==3)
    first_embedding=next(t for t in capture["embedding"] if t.ndim==3)
    assert first_key.shape==(B,inputs[0].shape[1],8)
    assert first_embedding.shape==(B,inputs[0].shape[1],48)
    assert state.memory.shape==(B,8,8)
    assert state.source_sum.shape==(B,13) and state.source_mass.shape==(B,13)
    assert state.evidence_sum.shape==(B,62)
    assert 13+13+62+8*8==152

    k=first_key[:,-64:];e=first_embedding[:,-64:]
    pair=k@k.transpose(1,2)
    upper=torch.triu_indices(pair.shape[1],pair.shape[2],offset=1)
    off=pair[:,upper[0],upper[1]].abs()
    mask=input_mask=inputs[1][:,-64:]
    sensor=inputs[0][:,-64:]
    # Grounded collision indicator: keys near-identical even when actual
    # OBSERVED normalized sensor coordinates differ by >=1.0 L2.
    ss=(sensor[:,:,None,:]-sensor[:,None,:,:]).square()
    jointly_visible=mask[:,:,None,:]&mask[:,None,:,:]
    overlap=jointly_visible.sum(-1)
    l2=(torch.where(jointly_visible,ss,0.).sum(-1)/overlap.clamp_min(1)).sqrt()
    distant=(l2[:,upper[0],upper[1]]>=1.)&(overlap[:,upper[0],upper[1]]>=3)
    collision=((off>.9)&distant)
    # Factual recent write keys not independent, intra-station time overlap is
    # treated as correlated observations, NOT independent retrieved items.
    S=torch.linalg.svdvals(state.memory)
    rank=(S>S.max(-1,keepdim=True).values.clamp_min(1e-8)*.001).sum(-1)
    mem=(state.memory.square().sum((-1,-2))).sqrt()
    mass=state.source_mass[:, :6]
    widths=torch.cat((base.left_scale,base.right_scale),-1)
    assert torch.isfinite(widths).all()
    return dict(
      carried_state_float_values=152,carried_state_fp32_bytes=608,
      final_memory_norm=observed_norm(mem),
      rank_at_1e_3=observed_norm(rank.float()),
      address_64step_mean_abs_offdiag_cosine=float(off.mean()),
      key_collisions_with_distinct_source_features=int(collision.sum()),
      eligible_distinct_source_feature_pairs=int(distant.sum()),
      collision_fraction_distinct_features=float(collision.sum()/distant.sum()) if distant.any() else None,
      encoder_48d_feature_std=float(e.std(dim=(0,1)).mean()),
      observed_mean_channels_per_hour=float(mask.float().sum(-1).mean()),
      source_evidence_mass=observed_norm(mass),
      forecast_scale_mean=float(widths.mean()),
      final_forecast=base,final_state=state
    )

def memory_block_probe(m,inputs):
    T=inputs[0].shape[1]
    cut=T-64
    prev=[z[:,:cut] if z.ndim>=2 and i!=4 else z for i,z in enumerate(inputs)]
    nxt=[z[:,cut:] if z.ndim>=2 and i!=4 else z for i,z in enumerate(inputs)]
    with torch.no_grad():
        _,s=m(*prev,return_trace=False)
        baseline,_=m(*nxt,state=s,return_trace=False)
        # Registered original model provenance / state token retained:
        # only legal tensor field changed in intervention.
        candidates={
          "associative":replace(s,memory=torch.zeros_like(s.memory)),
          "source_observation_statistics":replace(s,source_sum=torch.zeros_like(s.source_sum),
             source_mass=torch.zeros_like(s.source_mass)),
          "evidence_trace":replace(s,evidence_sum=torch.zeros_like(s.evidence_sum))
        }
        result={}
        for label,modified in candidates.items():
            f,_=m(*nxt,state=modified)
            result[label]=value_delta(f,baseline)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",default="fit",choices=("fit",))
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    torch.set_num_threads(1)
    d=new.fit_data()
    assert d["phase"]=="fit" and new.data.SHA=="d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8"
    source=new.data.ARCHIVE
    assert hashlib.sha256(source.read_bytes()).hexdigest()==new.data.SHA
    origins=[(sid,ORIGIN) for sid in range(12)]
    ix=checked_index(new)
    evidence=[]
    # Native state ownership relies on parameter._version; inference_mode
    # disables version counters. no_grad preserves source guards without AD/fit.
    with torch.no_grad():
        for seed in SEEDS:
            m,cp,sha=model_for(new,d,seed,ix[seed])
            original_versions=m.versions()
            cases=[]
            for length in HISTORIES:
                x=read_frame(d,origins,length)
                meta=inspect_native(m,d,x)
                meta.pop("final_forecast");meta.pop("final_state")
                cases.append(dict(H=length,**meta))
            sensitivity=memory_block_probe(m,read_frame(d,origins,512))
            if original_versions!=m.versions():
                raise RuntimeError("frozen model parameter version changed")
            evidence.append(dict(seed=seed,checkpoint_sha256=sha,horizons=cases,
                                 prefix_memory_group_lesion_H512_to_H64=sensitivity))
            print("NATIVE_LATENT_SEED="+json.dumps(evidence[-1],sort_keys=True),flush=True)
    out={
      "study":"NATIVELATENT001",
      "checkpoint_family":"ME111 final source-trained native TAC only",
      "source_sha256":new.data.SHA,
      "source_time":"FIT 2013-2014 original source station-hour12000,12 independently source-ID-matched stations; no targets accessed",
      "cutoff":"2015-01-01; protected 2016-17 unread; exposed 2015 REPORT unread",
      "input_boundary":list(INPUTS),
      "state_semantics":"source_sum13+source_mass13+evidence_sum62+associative8x8=152 FP32; fast/slow are transient within-run features",
      "status":"POSTHOC_READ_ONLY_ENGINEERING_DIAG",
      "no_training":True,"no_model_mutation":True,"future_target_scoring":False,
      "memory_lesions_are_OOD_sensitivities_only":True,
      "causal_claim":False,"qualified_architecture_superiority":False,
      "anti_cheat_full_independent_proof":"OPEN",
      "models":evidence}
    if args.output is not None:
        if args.output.exists():
            raise RuntimeError("original diagnostic report exists; do not overwrite")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("NATIVE_LATENT_RESULT_JSON="+json.dumps(out,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
