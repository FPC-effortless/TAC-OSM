"""LATFUSE007 no-fit physical-source +6 forecast and source-historic query gradients.

No optimizer, no pretrained original checkpoint modifications, no source
event label query leak. Does NOT demonstrate trained historic event memory.
"""
from __future__ import annotations
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

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse007/REGISTERED_NOGAIN_NOMODELFIT.json"

def run(src):
    if hashlib.sha256(src.read_bytes()).hexdigest()!=source.SOURCE_SHA:
        raise RuntimeError("source original sha mismatch")
    reg=json.loads(REG.read_text())
    check_manifest(reg,"preflight")
    if reg["study_id"]!="LATFUSE007-HISTORIC-QUERY-NOFIT-001":
        raise RuntimeError("source event query study prereg mismatch")
    torch.set_num_threads(1)
    d=source.source_frames(src,phase="fit")
    # Four original physical event/control matched pairs, no model/outcome
    # based sample selection. All anchored inside original FIT training.
    cohort=event_cohort(d,TRAIN_START,TRAIN_STOP,8)[:8]
    site=torch.tensor([z[0] for z in cohort],dtype=torch.long)
    anchor=torch.tensor([z[1] for z in cohort],dtype=torch.long)
    assert all(anchor+64+6<source.hour(source.TRAIN_END))
    torch.manual_seed(7007)
    model=SourceTimedEventQuery()
    contract=model.original_state_and_weight_contract()
    if (contract["original_nominal_memory_bytes"]!=6800 or
        contract["trainable_parameters"]!=99390):
        raise RuntimeError("original memory/parameter count mismatch")
    before=[p.detach().clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as recorder:
        s=model.initial(len(cohort))
        for offset in range(-23,65):
            tt=anchor+offset
            s=model.step(s,d["values"][site,tt],d["valid"][site,tt],
                         d["age"][site,tt],d["last"][site,tt],site)
        # Historic query depends ONLY on source known timestamp (anchor),
        # source identifier currently contained in legal model state.
        past_clock=source.calendar([int(i) for i in anchor])[:,None,:]
        logits=model.historic_event_logits(s,site,past_clock)
        # Forecast future known calendar alone, no actual future conditions.
        future=source.calendar([int(i+64+6) for i in anchor])[:,None,:]
        mu,scale=model.forecast(s,site,future)
        # Ground truth first consulted AFTER BOTH original outputs existed.
        recorded=d["values"][site,anchor,0]
        eligible=d["valid"][site,anchor,0]
        label=torch.where(recorded>=1.25,torch.ones_like(recorded),
                          torch.zeros_like(recorded))
        # Original known cohort positive/ordinary selection integrity.
        if not bool(eligible.all()) or not all(
            (float(v)>=1.25 if c[2] else abs(float(v))<=.25)
            for v,c in zip(recorded,cohort)):
            raise RuntimeError("invalid observed authentic historical event target")
        true_future=d["values"][site,anchor+64+6]
        true_mask=d["valid"][site,anchor+64+6]
        forecast=mu[:,0,:];sd=scale[:,0,:]
        clean=torch.where(true_mask,true_future,forecast.detach())
        nll=.5*(math.log(2*math.pi)+2*sd.log()+((clean-forecast)/sd).square())
        loss_forecast=((.8*nll+.2*(clean-forecast).abs())*true_mask).sum()/true_mask.sum().clamp_min(1)
        loss_event=F.binary_cross_entropy_with_logits(logits[:,0],label)
        loss=loss_forecast+loss_event
        loss.backward()
    grads={name:p.grad for name,p in model.named_parameters()}
    if not bool(torch.isfinite(loss).all()) or any(g is None or
          not bool(torch.isfinite(g).all()) for g in grads.values()):
        raise RuntimeError("unconnected true original-source gradients")
    if any(not torch.equal(p,z) for p,z in zip(model.parameters(),before)):
        raise RuntimeError("original nofit source model weights changed")
    measured_flops=sum(x.flops for x in recorder.key_averages())
    return dict(study_id="LATFUSE007-HISTORIC-QUERY-NOFIT-001",
        source_sha256=source.SOURCE_SHA,
        original_source_FIT_only=True,observed_source_paired_facts=len(cohort)//2,
        original_recurrent_fp32_state_bytes=6800,trainable_live_parameters=99390,
        total_live_parameter_tensors=len(grads),all_real_source_grads_finite=True,
        no_optimizer_or_model_fit=True,no_history_token_KV_cache=True,
        factual_historic_query_only_source_known_clock=True,
        future_outcomes_read_only_after_both_predictions=True,
        original_gaussian_training_loss=float(loss_forecast.detach()),
        original_real_historic_binary_BCE_before_any_training=float(loss_event.detach()),
        incomplete_cpu_supported_torch_flops_forward_backward=measured_flops,
        FLOPs_do_not_establish_training_budget_parity=True,
        no_accuracy_or_architecture_superiority_claim=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError("do not overwrite authentic registered nofit preflight")
    result=run(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE007_ORIGINAL_EVENT_NOFIT="+json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":main()
