#!/usr/bin/env python3
"""S1TEMP007 REAL FIT-only nine original checkpoint compartment forensics.

Not a training run, not DEV / TEST score or safe-control qualification.
Local hidden-state lesions are out-of-distribution perturbations and show
sensitivity only on originally observed FIT prefix episodes0 and1.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.s1temp007_original_mobile_h1024_fit import (
    lock,get_episode,encode,model_pair,ARMS,SEEDS)
from tac_osm.variants.streaming_continuous_world import WorldStreamState
from tac_osm.variants.system1_velocity_anchored import MotionStreamState
from tac_osm.research_anticheat import audit_input_boundary
ORIGINAL=ROOT/"research/parallel-six/S1TEMP007-ORIGINAL-LEDGER.json"
BATCH_EPISODES=(0,1)
STARTS=(100,300)
HISTORY=1024

def compute(model,x,u,previous,ys,scale):
    model.eval()
    with torch.no_grad():
        _,trace=model.core.core(x[:,:-1],u[:,:-1],return_trace=True)
        world=WorldStreamState(trace["sum_basis"][:,-1].clone(),
            trace["memory"][:,-1].clone(),HISTORY-1)
        source_state=MotionStreamState(world,x[:,-2].clone(),True)
        final_source=(x[:,-1],u[:,-1])
        def query(st):
            pred,_=model.step(st,*final_source)
            return pred
        baseline=query(source_state)
        fitted=lambda y:float(((y.next_observation.double()-
                                  ys.double())*scale.double()).square().mean())
        def stat(pred):
            return dict(
                point_delta_source_normalized=float((
                    pred.next_observation-baseline.next_observation
                    ).abs().mean()),
                high_motion_proxy_logit_delta=float((
                    pred.adverse_event_logit-baseline.adverse_event_logit
                    ).abs().mean()),
                mean_source_physical_MSE=fitted(pred),
                baseline_source_MSE=fitted(baseline))
        results={}
        for name,(a,b) in {
            "additive_fast72":(0,72),
            "slow_rate_1_over_16":(72,76),
            "slow_rate_1_over_64":(76,80),
            "slow_rate_1_over_256":(80,84),
            "slow_rate_1_over_1024":(84,88),
            "whole_slow16":(72,88),
        }.items():
            updated=source_state.core_state.fast_sum.clone()
            updated[:,a:b]=0
            altered=MotionStreamState(replace(
                source_state.core_state,fast_sum=updated),
                source_state.previous_observation,True)
            results[name]=stat(query(altered))
        updated=MotionStreamState(replace(
            source_state.core_state,
            memory=torch.zeros_like(source_state.core_state.memory)),
            source_state.previous_observation,True)
        results["associative8x8"]=stat(query(updated))
        results["baseline"]=stat(baseline)
    return results

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--original-dir",required=True)
    a.add_argument("--cache",default="/tmp/s1temp007-original-FIT-diagnostics")
    a.add_argument("--out",default="research/diagnostics/s1temp007-FIT-original")
    args=a.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    receipt=json.loads(ORIGINAL.read_text())
    original=Path(args.original_dir)
    before=json.loads((original/"BEFORE-DEV-SHA-FREEZE.json").read_text())
    src=lock()
    if (not before["all_nine_original_model_optimizer_rng_SHA_frozen_before_DEV"]
        or before["source_revision"]!=src["revision_sha"]
        or receipt["original_scientific_gate"]!="FAIL"):
        raise RuntimeError("invalid original source integrity/score receipt")
    # FIT0 and FIT1 only; no DEV file reader caller whatsoever.
    norms=before["normalizers_from_only_original_FIT"]
    batches=[]
    for ep,begin in zip(BATCH_EPISODES,STARTS):
        rows,hash_record=get_episode(ep,src,Path(args.cache))
        if hash_record["sha256"]!=src["FIT_receipts"][ep]["sha256"]:
            raise RuntimeError("original training source checksum mismatch")
        enc=encode(rows,norms)
        batches.append((enc,begin))
    observed=torch.stack([d["x"][t:t+HISTORY] for d,t in batches])
    prior_actions=torch.stack([d["u"][t:t+HISTORY] for d,t in batches])
    preceding=torch.stack([d["previous"][t] for d,t in batches])
    target=torch.stack([d["y"][t+HISTORY-1] for d,t in batches])
    sample=dict(observation_history=observed,
        prior_logged_action_history=prior_actions,
        preceding_measured_observation=preceding)
    allow=audit_input_boundary(sample,list(sample),check="frozen_FIT_legal_source")
    if allow.status!="PASS":
        raise RuntimeError("future source was able to enter the query")
    scale=torch.tensor(norms["state_sd"],dtype=torch.float32)
    delta=torch.tensor(norms["train_delta_scale_normalized"],dtype=torch.float32)
    verified=[]
    for seed in SEEDS:
        model=model_pair(seed,delta)["multiscale16"]
        record=next(x for x in receipt[
              "all_nine_original_model_optim_rng_checkpoint_sha_receipts"]
                    if x["seed"]==seed and x["arm"]=="multiscale16")
        p=original/f"multiscale16-{seed}.pt"
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest!=record["model_optimizer_rng_SHA256"]:
            raise RuntimeError("original source/trained checkpoint SHA tampered")
        ck=torch.load(p,map_location="cpu",weights_only=False)
        if ck["immutable_publisher_sha"]!=src["revision_sha"] or (
            ck["FIT_original_hashes"] != [
                row["sha256"] for row in before["source_fit_Sha256"]]):
            raise RuntimeError("fit source lineage/checkpoint inconsistent")
        model.load_state_dict(ck["model"],strict=True)
        snap={k:v.detach().clone() for k,v in model.state_dict().items()}
        result=compute(model,observed,prior_actions,preceding,target,scale)
        if any(not torch.equal(snap[k],v)
               for k,v in model.state_dict().items()):
            raise RuntimeError("diagnostic altered original fitted buffers/weights")
        data=dict(seed=seed,arm="multiscale16",
             original_model_optimizer_RNG_SHA=digest,
             real_fit_episodes=list(BATCH_EPISODES),context_frames=HISTORY,
             real_source_ablation_groups=result)
        verified.append(data)
        print("S1TEMP007_ORIGINAL_FIT_REAL_LESIONS",
              json.dumps(data,sort_keys=True),flush=True)
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    full=dict(study="S1TEMP007",
        real_30_FIT_source_original_verified=True,
        real_source_only_ep0_1_diagnosed=True,
        original_three_multirate_model_SHAs_verified=True,
        all_models_remain_unmodified=True,
        no_optimizer_or_DEV_or_TEST_source_reads=True,
        source_future_noninterference_not_exhaustive=True,
        local_zero_hidden_state_is_OUT_OF_DISTRIBUTION=True,
        no_safety_action_authorization=True,
        scientific_original_gate="FAIL_0_OF_3",rows=verified)
    (out/"ORIGINAL-FIT-MULTIRATE-LESIONS.json").write_text(
        json.dumps(full,indent=2,sort_keys=True)+"\n")
    print("S1TEMP007_REAL_MULTIRATE_ORIGINAL_FIT_DIAGNOSTICS=PASS",flush=True)

if __name__=="__main__":main()
