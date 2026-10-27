"""LATFUSE002 key interference and source-authentic memory transport diagnosis.

READ-ONLY original trained checkpoints. Genuine UCI501 FIT observations 2014-06;
2015 DEVELOPMENT, 2016-2017 protected, future-target labels NEVER read.
No optimizer/fit/accuracy claims. Out-of-distribution state lesion = local
mechanistic sensitivity, not causal forecasting or world-model correctness.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts import latfuse002_real_fit as original
from tac_osm.latent_fusion_002 import FusedLatentFixedState,state_contract

START=original.hour(dt.datetime(2014,6,1))
FRAMES=256
TAILS=(1,4,16,64,128)
SEEDS=(202601,202602)


def read_original_model(seed,original_receipts):
    receipt=next((r for r in original_receipts["results"] if r["seed"]==seed),None)
    if receipt is None:raise RuntimeError("missing original fitted seed receipt")
    path=original.OUT/f"seed-{seed}"/"final.pt"
    identity=hashlib.sha256(path.read_bytes()).hexdigest()
    if identity!=receipt["original_checkpoint_sha256"]:
        raise RuntimeError("original checkpoint SHA changed")
    cp=torch.load(path,map_location="cpu",weights_only=False)
    if cp["seed"]!=seed or cp["steps"]!=128 or cp["original_source_sha256"]!=original.SOURCE_SHA:
        raise RuntimeError("original trained checkpoint metadata mismatch")
    model=FusedLatentFixedState()
    model.load_state_dict(cp["model"],strict=True)
    model.eval()
    if state_contract(model)!={"float_values":1696,"occupancy_flags":16,
            "nominal_fp32_bytes":6800,"dtype":"torch.float32",
            "trainable_parameters":98775,"deployed_learned_parameters":98775}:
        raise RuntimeError("FP32 state or model count diverged")
    return model,path,identity


def actual_stream(d,t):
    observed=d["valid"][:,t]
    return (d["values"][:,t], observed,d["age"][:,t],d["last"][:,t])


def step(model,state,d,t,slots):
    x,observed,age,last=actual_stream(d,t)
    return model.step(state,x,observed,age,last,slots)


def key_geometry(model,d,slots):
    keys=[];encodings=[]
    with torch.no_grad():
        for t in range(START+FRAMES-64,START+FRAMES):
            x,obs,age,last=actual_stream(d,t)
            u=model.encode(x,obs,age,last)
            encodings.append(u)
            keys.append(F.normalize(model.key_map(u),dim=-1,eps=1e-8))
    e=torch.stack(encodings,dim=1)
    k=torch.stack(keys,dim=1)
    sensor=d["values"][:,START+FRAMES-64:START+FRAMES]
    observed=d["valid"][:,START+FRAMES-64:START+FRAMES]
    upper=torch.triu_indices(64,64,offset=1)
    joint=observed[:,:,None,:]&observed[:,None,:,:]
    count=joint.sum(-1)
    dist=(torch.where(joint,(sensor[:,:,None,:]-sensor[:,None,:,:]).square(),0)
          .sum(-1)/count.clamp_min(1)).sqrt()
    different=(count[:,upper[0],upper[1]]>=3)&(dist[:,upper[0],upper[1]]>=1)
    if int(different.sum())<200:
        raise RuntimeError("insufficient actually distinct FIT source pairs")
    ec=(F.normalize(e,dim=-1,eps=1e-8)@F.normalize(e,dim=-1,eps=1e-8).transpose(1,2)
        )[:,upper[0],upper[1]].abs()
    kc=(k@k.transpose(1,2))[:,upper[0],upper[1]].abs()
    dcount=int(different.sum())
    return dict(
        compared_source_distinct_pairs=dcount,
        observed_sensor_rms_difference_at_least_1_and_joint_features_at_least_3=True,
        mean_abs_encoder_cosine_distinct=float(ec[different].mean()),
        mean_abs_key_cosine_distinct=float(kc[different].mean()),
        encoder_abs_cos_gt_0_9_distinct_fraction=float((ec[different]>.9).float().mean()),
        key_abs_cos_gt_0_9_distinct_fraction=float((kc[different]>.9).float().mean()),
        mean_abs_key_cosine_all_pairs=float(kc.mean()),
        key_dim=k.shape[-1],encoder_dim=e.shape[-1],
        source_pair_outcomes_NOT_ground_truth_retrieval=True,
    )


def transport_and_forecast_lesion(model,d,slots,length):
    cutoff=START+FRAMES-length
    # Always same final source hour, identical real 2014 input sequence. No future labels.
    state=model.initial(len(slots))
    for t in range(START,cutoff):
        state=step(model,state,d,t,slots)
    retained_prefix=state.associative.detach().clone()
    # Two continuations differ ONLY by prefix associative content.
    erased=state._replace(associative=torch.zeros_like(state.associative))
    transfer=torch.eye(model.key_dim,dtype=torch.float32).unsqueeze(0).expand(
        len(slots),-1,-1).clone()
    alphas=[];betas=[];projections=[]
    for t in range(cutoff,START+FRAMES):
        x,obs,age,last=actual_stream(d,t)
        u=model.encode(x,obs,age,last)
        key=F.normalize(model.key_map(u),dim=-1,eps=1e-8)
        alpha=torch.sigmoid(model.retention_map(u))
        beta=torch.sigmoid(model.write_map(u))
        eye=torch.eye(model.key_dim,device=key.device,dtype=key.dtype).unsqueeze(0)
        P=alpha.unsqueeze(-1)*(eye-beta.unsqueeze(-1)*key.unsqueeze(-1)*key.unsqueeze(-2))
        transfer=transfer@P
        alphas.append(alpha);betas.append(beta)
        state=model.step(state,x,obs,age,last,slots)
        erased=model.step(erased,x,obs,age,last,slots)
    actual=state.associative-erased.associative
    expected=retained_prefix@transfer
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
    # The following comparison measures effect on forecast only, not labeled skill.
    q=original.calendar([START+FRAMES+5]*len(slots)).unsqueeze(1)
    mu,scale=model.forecast(state,slots,q)
    mu0,scale0=model.forecast(erased,slots,q)
    ratio=actual.flatten(1).norm(dim=1)/retained_prefix.flatten(1).norm(dim=1).clamp_min(1e-9)
    spectral=torch.linalg.matrix_norm(transfer,ord=2)
    return dict(
        mean_effective_retention=float(torch.cat(alphas).mean()),
        mean_write_strength=float(torch.cat(betas).mean()),
        source_prefix_memory_norm_mean=float(retained_prefix.norm(dim=(-2,-1)).mean()),
        memory_difference_norm_ratio_mean=float(ratio.mean()),
        right_transport_spectral_norm_mean=float(spectral.mean()),
        native_actual_vs_independent_affine_max_absolute_difference=float((actual-expected).abs().max()),
        memory_lesion_forecast_mean_absolute_delta=float((mu-mu0).abs().mean()),
        memory_lesion_scale_mean_absolute_delta=float((scale-scale0).abs().mean()),
        outcomes_scored=False,
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise RuntimeError("FIT-only diagnostic already exists; no overwrite")
    torch.set_num_threads(1)
    source=hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source!=original.SOURCE_SHA:raise RuntimeError("source SHA mismatch")
    data=original.source_frames(args.source,phase="fit")
    if data["source_sha"]!=source or START+FRAMES+6>=original.hour(original.TRAIN_END):
        raise RuntimeError("diagnostic outside original source-authentic train period")
    receipts=json.loads((original.OUT/"RESULTS.json").read_text())
    if tuple(r["seed"] for r in receipts["results"])!=SEEDS:
        raise RuntimeError("original fitted seed list mismatch")
    slots=torch.arange(len(original.SITES))
    models=[]
    with torch.no_grad():
        for seed in SEEDS:
            model,path,sha=read_original_model(seed,receipts)
            orig_parameters=[p.detach().clone() for p in model.parameters()]
            geometry=key_geometry(model,data,slots)
            transport={str(h):transport_and_forecast_lesion(model,data,slots,h) for h in TAILS}
            if (hashlib.sha256(path.read_bytes()).hexdigest()!=sha or
                any(not torch.equal(p,initial) for p,initial in zip(model.parameters(),orig_parameters))):
                raise RuntimeError("original parameters mutated during diagnosis")
            finding=dict(seed=seed,checkpoint_sha256=sha,key_geometry=geometry,
                         associative_memory_transport=transport)
            models.append(finding)
            print("LATFUSE002_FORENSIC_SEED="+json.dumps(finding,sort_keys=True),flush=True)
    result=dict(study="LATFUSE002-FORENSICS-003",
        evidence_tier="ORIGINAL_TRAINED_CHECKPOINT + GENUINE_FIT_INPUTS_ONLY",
        source_archive_sha256=source,real_source_scope="2014-06 authentic 12 original Beijing stations,256 observed input hours; no future numeric targets",
        source_authorized_covariates="original available measurements, masks, elapsed age, chronological last seen, physical station ids, known future calendar",
        no_training=True,no_new_model_fits=True,all_original_weights_unchanged=True,
        no_exposed_DEV_or_protected_TEST_read=True,
        causal_decision_or_content_retrieval_skill_proven=False,
        diagnosed_state_bytes=6800,models=models)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LATFUSE002_FORENSICS_STATUS=PASS",flush=True)

if __name__=="__main__":main()
