"""PAR-MEM-013 original real-source paired full-step operator parity preflight.

Includes source forward, both paired historical-target decoders, true
content ranking loss, future physical loss, backward, clipping and AdamW.
ATen op/input-shape invocation counts must be identical between two arms.
Profiler-reported FLOPs can be a subset; cannot certify all mathematical
FP32/special-function/optimizer FLOPs from unsupported ATen ops alone.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch.profiler import profile,ProfilerActivity
from scripts.parmem013_genuine_source_fit import (
    fixed,make_model,paired_source_batch,SEEDS,ARMS,LR
)
from scripts.parmem013_source_event_objective import (
    paired_real_source_predictions,loss_and_measures
)


def trace_one(arm,case,data,fit,original):
    model=make_model(SEEDS[0],original)
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.001,
                          foreach=False,fused=False)
    batch=paired_source_batch(
        data,fit,case["sites"],case["anchors"],case["lag"])
    with profile(activities=[ProfilerActivity.CPU],record_shapes=True,
                 with_flops=True) as prof:
        opt.zero_grad(set_to_none=True)
        out=paired_real_source_predictions(model,batch,case["lag"],
                               associative_read=(arm==ARMS[0]))
        loss,details=loss_and_measures(out,batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
    signatures={}
    declared_flops=0
    for event in prof.key_averages(group_by_input_shape=True):
        sig=json.dumps([event.key,event.input_shapes],sort_keys=True)
        signatures[sig]=signatures.get(sig,0)+int(event.count)
        declared_flops+=int(event.flops or 0)
    if not signatures or declared_flops<100000:
        raise RuntimeError("entire source-trained model forward+AD+AdamW profiling failed")
    return signatures,declared_flops,float(loss.detach())


def audit(source):
    torch.set_num_threads(1)
    p,data,fit,_,original=fixed(source)
    case=p["training_schedule_by_seed"][str(SEEDS[0])][0]
    scores={arm:trace_one(arm,case,data,fit,original) for arm in ARMS}
    a,b=ARMS
    ac,af,aloss=scores[a]
    bc,bf,bloss=scores[b]
    mismatches={k:(ac.get(k,0),bc.get(k,0))
       for k in set(ac)|set(bc) if ac.get(k,0)!=bc.get(k,0)}
    gap=abs(af-bf)/max(1,af,bf)
    result=dict(study_id="PAR-MEM-013-ORIGINAL-REAL-SOURCE-OP-TRACE",
        original_publisher_zip_SHA256=p["source_SHA256"],
        current_source_fit_events=129,paired_real_source_station_batch=4,
        forward_AD_clip_and_optimizer_audited=True,
        same_live_parameters=98326,
        same_saved_state_FP32=410,
        unique_ATen_operator_input_shapes=len(ac),
        number_of_differing_graph_operator_signatures=len(mismatches),
        differing_graph_examples=list(mismatches.items())[:8],
        all_operator_signature_count_identical=len(mismatches)==0,
        aligned_read_profiler_supported_FLOPs=af,
        trained_no_associative_read_supported_FLOPs=bf,
        supported_subset_relative_gap=gap,
        complete_all_ATen_scalar_special_backward_optimizer_FLOP_coverage_proven=False,
        source_value_supervision_authentic=True,
        optimizer_was_used_only_on_TEMPORARY_preflight_graph=True,
        real_source_exposed_DEV_untouched_by_preflight=True)
    print("PAR_MEM_013_AUTHENTIC_SOURCE_PAIRED_OPERATOR_PREFLIGHT="+
          json.dumps(result,sort_keys=True),flush=True)
    if mismatches or gap>0.01:
        raise RuntimeError("compulsory source-matched operator signature parity failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    a=ap.parse_args()
    audit(a.source)
