"""LATFUSE008 nofit source-true two-arm historic read FLOP parity audit.

No optimizer/weights updated. Model comparison NOT qualified until full
forward/backward + optimizer operation parity independently validated.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import profile,ProfilerActivity

from scripts import latfuse002_real_fit as source
from scripts.latfuse006_event_decoder import event_cohort,TRAIN_START,TRAIN_STOP
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_007_historic_query import SourceTimedEventQuery
from tac_osm.latent_fusion_008_nonassoc_control import SourceTimedNonassociativeQuery

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse008/NOFIT_REGISTRATION.json"

def one_case(m,d,cohort):
    sid=torch.tensor([r[0] for r in cohort],dtype=torch.long)
    anchor=torch.tensor([r[1] for r in cohort],dtype=torch.long)
    s=m.initial(len(cohort))
    for offset in range(-23,65):
        t=anchor+offset
        s=m.step(s,d["values"][sid,t],d["valid"][sid,t],
                 d["age"][sid,t],d["last"][sid,t],sid)
    known_past=source.calendar([int(i) for i in anchor])[:,None,:]
    logits=m.historic_event_logits(s,sid,known_past)
    future_clock=source.calendar([int(i+70) for i in anchor])[:,None,:]
    forecast,scale=m.forecast(s,sid,future_clock)
    # Only now open target values; never an input to either query or encoder
    historical=d["values"][sid,anchor,0]
    event=((historical>=1.25)).float()
    if not all(bool(d["valid"][sid[i],anchor[i],0]) and (
            float(historical[i])>=1.25 if cohort[i][2] else
            abs(float(historical[i]))<=.25) for i in range(len(cohort))):
        raise RuntimeError("historical true event/control labels invalid")
    observed=d["valid"][sid,anchor+70]
    truth=d["values"][sid,anchor+70]
    mean=forecast[:,0];sigma=scale[:,0]
    clean=torch.where(observed,truth,mean.detach())
    nll=.5*(math.log(2*math.pi)+2*sigma.log()+((clean-mean)/sigma).square())
    forecast_loss=((.8*nll+.2*(clean-mean).abs())*observed).sum()/observed.sum().clamp_min(1)
    label_loss=F.binary_cross_entropy_with_logits(logits[:,0],event)
    return label_loss+forecast_loss,(label_loss.detach(),forecast_loss.detach())

def audit(source_path):
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=source.SOURCE_SHA:
        raise RuntimeError("nonmatching original physical-source ZIP")
    reg=json.loads(REG.read_text())
    check_manifest(reg,"preflight")
    if reg["study_id"]!="LATFUSE008-QUERY-CONTROL-SOURCE-PREFIT":
        raise RuntimeError("historical query nofit registration mismatch")
    d=source.source_frames(source_path,phase="fit")
    cohort=event_cohort(d,TRAIN_START,TRAIN_STOP,8)[:8]
    recorded_sha=hashlib.sha256(json.dumps(cohort,separators=(",",":")).encode()).hexdigest()
    results=[]
    for name,model_cls in (("assoc_historic_query",SourceTimedEventQuery),
                           ("fast_slow_historic_query",SourceTimedNonassociativeQuery)):
        torch.manual_seed(8008)
        m=model_cls()
        orig=m.original_state_and_weight_contract()
        if not (orig["trainable_parameters"]==orig["deployed_parameters"]==99390 and
                orig["original_nominal_memory_bytes"]==6800):
            raise RuntimeError("registered model size disagreement")
        before=[v.detach().clone() for v in m.parameters()]
        with profile(activities=[ProfilerActivity.CPU],with_flops=True) as p:
            loss,components=one_case(m,d,cohort)
            loss.backward()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite original source objective")
        missing=[name for name,p in m.named_parameters() if
                 p.grad is None or not bool(torch.isfinite(p.grad).all())]
        if missing:
            raise RuntimeError("training graph disconnected original tensor "+repr(missing))
        if any(not torch.equal(x,y) for x,y in zip(before,m.parameters())):
            raise RuntimeError("no-fit read path modified original model parameters")
        n=sum(e.flops for e in p.key_averages())
        results.append(dict(arm=name,live_params=99390,carried_fp32_state_bytes=6800,
                genuine_original_source_training_case_order_sha256=recorded_sha,
                real_source_gradient_tensors=sum(1 for _ in m.parameters()),
                torch_CPU_supported_operator_forward_AD_flops=n,
                original_real_event_BCE_nofit=float(components[0]),
                original_real_forecast_loss_nofit=float(components[1]),
                optimizer_updates=0))
        print("LATFUSE008_ONE_REAL_SOURCE_ARM="+json.dumps(results[-1],sort_keys=True),flush=True)
    maxima=max(x["torch_CPU_supported_operator_forward_AD_flops"] for x in results)
    minima=min(x["torch_CPU_supported_operator_forward_AD_flops"] for x in results)
    diff=(maxima-minima)/maxima
    report=dict(study="LATFUSE008-QUERY-CONTROL-SOURCE-PREFIT",
        source_sha256=source.SOURCE_SHA,source_cohort_sha256=recorded_sha,
        original_genuine_event_control_pairs=len(cohort)//2,
        original_parent_model_and_optimizers_untouched=True,
        two_models_identical_live_parameters_and_recurrent_state=True,
        equal_identical_true_source_inputs_and_masks_and_labels=True,
        supported_ops_relative_flops_gap=diff,
        partial_profiled_compute_within_one_pct=diff<=.01,
        optimizer_and_unsupported_ops_excluded=True,
        full_training_compute_parity_verified=False,
        comparative_original_training_permitted=False,
        models=results)
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise RuntimeError("source-first prereg result already exists")
    torch.set_num_threads(1)
    result=audit(a.source)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE008_REAL_SOURCE_PREFIT_COMPLETE="+json.dumps(result,sort_keys=True),flush=True)
if __name__=="__main__":main()
