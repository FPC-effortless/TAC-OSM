"""PAR-TIME-002 pretraining REAL-source gate; no model fits, no test peeking.

Profiles one full CPU FP32 real-source forward, automatic differentiation,
clip and AdamW with lr=0 on two 96k-parameter architecture arms. The
optimizer does arithmetic but applies no learned updates. This is NOT an
equal-compute scientific proof: torch.profiler reports only supported
matmul FLOPs, and bounded unsupported operator accounting is advisory.
Training is FORBIDDEN until a separately audited complete budget closes.

No synthetic performance claim; source case is authentic and hash-locked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch

from scripts.partime002_real_source import (
    SHA256, read_source, chronology, batch_from_cases, score,
)
from tac_osm.variants.parallel_residual_world import ParallelResidualWorld
from tac_osm.variants.parallel_time_age_world import TimeAwareParallelResidualWorld

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"research/parallel-six/PAR-TIME-002-PROTOCOL.json"
SEED=271028
BATCH=4
HISTORY=12

def sha_bytes(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,
                         separators=(",",":")).encode()).hexdigest()

def model_pair():
    torch.manual_seed(SEED)
    a=ParallelResidualWorld()
    torch.manual_seed(SEED)
    b=TimeAwareParallelResidualWorld()
    # Parent and treatment must have identical shared learned weights at
    # first prediction. TimeAware's subclass construction additionally
    # initializes an expanded core, so repeating a seed alone is NOT enough.
    state=b.core.load_state_dict(a.core.state_dict(),strict=False)
    if state.unexpected_keys or state.missing_keys!=["time_projection.weight"]:
        raise RuntimeError("parent-to-treatment common learned weight map changed")
    b.residual_ffn.load_state_dict(a.residual_ffn.state_dict(),strict=True)
    with torch.no_grad():
        b.residual_logit.copy_(a.residual_logit)
    for model in (a,b):
        model.train()
        if any(not p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()):
            raise RuntimeError("unaccounted learned parameter in reference architecture")
    params=[sum(p.numel() for p in model.parameters()) for model in (a,b)]
    if params != [96303,96639] or abs(params[1]-params[0])/params[1]>=0.01:
        raise RuntimeError("pre-registered active parameter equality broken")
    return a,b,params

def real_batch(source_path):
    cfg=json.loads(PROTOCOL.read_text())
    if (cfg["source_original_zip_SHA256"]!=SHA256 or
        cfg["train_period"]!="2014-03-01/2014-04-18" or
        cfg["exposed_development_period"]!="2014-05-01/2014-06-10" or
        cfg["steps_proposed"]!=64 or
        cfg["batch"]!=BATCH or cfg["history"]!=HISTORY or
        cfg["planned_seeds"]!=[271028,271029,271030,271031,271032]):
        raise RuntimeError("frozen prefit study declaration not respected")
    d=read_source(source_path)
    train,exposed=chronology(d)
    rng=random.Random(SEED)
    # Source-time-only stratification for DIAGNOSTIC profiling. We do not
    # select on future outcomes, score or any model prediction. The
    # original random sample accidentally consisted solely of 1h gaps.
    def largest_observed_gap(case):
        site,ix,_=case
        prior=d["events"][site][ix-HISTORY+1:ix+1]
        return max(b-a for a,b in zip(prior,prior[1:]))
    irregular=[case for case in train if largest_observed_gap(case)>1]
    if len(irregular)<2:
        raise RuntimeError("real irregular clock cases unavailable: age study must stop")
    regular=[case for case in train if largest_observed_gap(case)==1]
    if len(regular)<2:
        raise RuntimeError("genuine regular source contrast unavailable")
    selected=rng.sample(irregular,2)+rng.sample(regular,2)
    batch=batch_from_cases(d,selected)
    # 2014 exposed development cases are counted through authenticated
    # known-at-time timestamps, but outcomes are NEVER scored here.
    result={"authentic_source_verified":True,"source_sha":SHA256,
       "source_training_case_count":len(train),
       "historically_exposed_development_case_count":len(exposed),
       "train_case_digest":sha_bytes(train),
       "exposed_case_digest":sha_bytes(exposed),
       "source_selected_case_digest":sha_bytes(selected),
       "source_diag_irregular_case_count":len(irregular),
       "source_diag_regular_case_count":len(regular),
       "source_diag_selected_irregular_histories":2,
       "source_physical_elapsed_delta_min":float(batch[2][:,1:].min()),
       "source_physical_elapsed_delta_max":float(batch[2][:,1:].max()),
       "source_actions_are_only_availability_and_known_clock":True,
       "training_target_not_used_as_model_input":True,
       "source_years_2015_2017_numerics_never_parsed":True}
    return batch,result

def step(model,inputs,*,time_aware,profile=False):
    obs,actions,elapsed,writes,truth,valid=inputs
    opt=torch.optim.AdamW(model.parameters(),lr=0.0,
                           weight_decay=0.001,foreach=False,fused=False)
    def execute():
        opt.zero_grad(set_to_none=True)
        if time_aware:
            prediction=model(obs,actions,elapsed_dt=elapsed,write_mask=writes)
        else:
            prediction=model(obs,actions)
        loss=score(prediction[:,-1],truth,valid)
        if not torch.isfinite(loss).item():
            raise RuntimeError("authentic nonfinite real-source loss")
        loss.backward()
        grads=list(model.named_parameters())
        if any(p.grad is None or not p.grad.isfinite().all().item()
               for _,p in grads):
            raise RuntimeError("disconnected actual-source model gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()   # lr zero: optimizer math computed, weights preserved
        return float(loss.detach()),grads
    if not profile:
        return execute(),None
    from torch.profiler import profile as profiler, ProfilerActivity
    with profiler(activities=[ProfilerActivity.CPU],
                  record_shapes=True,with_flops=True) as p:
        result=execute()
    events=[]
    for event in p.key_averages(group_by_input_shape=True):
        if not event.key.startswith("aten::"):
            continue
        events.append({"op":event.key,"shape":event.input_shapes,
                       "count":event.count,"flops":int(event.flops or 0)})
    return result,events

def summary(events):
    return {"supported_matmul_flops":sum(e["flops"] for e in events),
            "all_ATen_op_invocations":sum(e["count"] for e in events),
            "unsupported_mathematical_operator_invocations":
              sum(e["count"] for e in events if e["flops"]==0),
            "operator_shape_kinds":len(events)}

def key(e):
    return json.dumps((e["op"],e["shape"]),sort_keys=True)

def compare(a,b):
    x={key(e):e for e in a}
    y={key(e):e for e in b}
    base=sum(e["flops"] for e in a)
    test=sum(e["flops"] for e in b)
    if min(base,test)<=0:
        raise RuntimeError("no profiler matmul FLOP support: cannot evaluate gate")
    supported_gap=abs(base-test)/max(base,test)
    missing=[]
    for k in set(x)|set(y):
        left=x.get(k);right=y.get(k)
        l=left["count"] if left else 0
        r=right["count"] if right else 0
        if l!=r:
            missing.append({
                "key":k[:190],"baseline_count":l,
                "temporal_count":r,
                "supported_flops_delta":(right["flops"] if right else 0)-
                                        (left["flops"] if left else 0)})
    missing.sort(key=lambda z:abs(z["temporal_count"]-z["baseline_count"]),reverse=True)
    return {
        "supported_ops_matmul_relative_flops_gap":supported_gap,
        "supported_ops_gap_below_1pct":supported_gap<0.01,
        "unsupported_op_shape_differences":len(
            [q for q in missing if q["supported_flops_delta"]==0]),
        "differing_op_shape_signatures":len(missing),
        "shape_difference_sample_first_20":missing[:20],
        "FULL_training_math_FLOP_parity_CERTIFIED":False,
        "candidate_training_authorized":False,
        "reason":"Torch profiler supports subset of true nonlinear/AD/optimizer FLOPs; full operator closure not yet certified",
    }

def run(path:Path):
    torch.set_num_threads(1)
    inputs,source=real_batch(path)
    parent,candidate,params=model_pair()
    pa_before=[p.detach().clone() for p in parent.parameters()]
    ca_before=[p.detach().clone() for p in candidate.parameters()]
    # With all-source-write=True and zero time projection, source-forward
    # should agree with the earlier time-blind model before optimization.
    with torch.no_grad():
        yp=parent(inputs[0],inputs[1])
        yc=candidate(inputs[0],inputs[1],elapsed_dt=inputs[2],write_mask=inputs[3])
        max_abs=float((yp-yc).abs().max())
        if max_abs>0.00002:
            raise RuntimeError("parent-preserving new temporal projection failed")
    (pl,_),pevents=step(parent,inputs,time_aware=False,profile=True)
    (cl,_),cevents=step(candidate,inputs,time_aware=True,profile=True)
    for model,initial in ((parent,pa_before),(candidate,ca_before)):
        if not all(torch.equal(old,now) for old,now in zip(initial,model.parameters())):
            raise RuntimeError("preflight optimizer was not truly zero-update")
    out={"study":"PAR-TIME-002", "phase":"REAL_SOURCE_FULL_STEP_PREFLIGHT_NO_FIT",
         "status":"SOURCE_OBSERVATIONS_VERIFIED; TRAINING_BLOCKED_FULL_FLOPS_OPEN",
         "source":source,"params":params,
         "parameter_difference_relative":(params[1]-params[0])/params[1],
         "same_seed_parent_to_time_initial_prediction_max_abs":max_abs,
         "real_source_loss_with_zero_optimizer_update":
            {"time_blind":pl,"time_aware":cl},
         "parent_operator_counts":summary(pevents),
         "temporal_operator_counts":summary(cevents),
         "operator_comparison":compare(pevents,cevents),
         "source_real_not_independently_qualified":True,
         "has_run_actual_training":False,
         "new_checkpoint_count":0,
         "protected_source_years_opened":False}
    print("PAR_TIME_002_REAL_SOURCE_NOFIT="+json.dumps(out,sort_keys=True,allow_nan=False),
          flush=True)
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    out=run(args.source)
    if args.output:
        args.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
