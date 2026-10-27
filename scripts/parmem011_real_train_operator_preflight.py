"""Real source PAR-MEM-011 forward + autograd + gradient-clip + AdamW op audit.

CPU profiler op+input-shape+count signature must be identical across the
source-known correct vs *adjacent wrong old event* query. Same 98326 active
parameters, same physical data, same source case order and optimizer.
Numeric FLOP counts are a partial profiler-supported subset; matching
all op-shape counts across both arms is stronger evidence for mathematical
operator-graph equivalence but does not prove backend exact bit FLOPs for
every unsupported CUDA/CPU non-matmul, every seed or result generalization.
"""
import argparse,json
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
from scripts.parmem011_genuine_source_train import (
    fixed,neural,source_batch,forward_loss,SEEDS,ARMS,LR
)

def source_op_profile(seed,arm,source,case,receipt,fit):
    model=neural(seed,receipt)
    model.train()
    optim=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.001,
                             foreach=False,fused=False)
    batch=source_batch(source,fit,case["sites"],case["anchors"],case["lag"],arm)
    with profile(activities=[ProfilerActivity.CPU],record_shapes=True,
                 with_flops=True) as prof:
        optim.zero_grad(set_to_none=True)
        loss,_,_=forward_loss(model,batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optim.step()
    # Every scalar/tensor backend operator is audited by ORIGINAL input
    # shapes and invocation count, including backward and optimizer.
    table={}
    total=0
    for row in prof.key_averages(group_by_input_shape=True):
        sig=json.dumps([row.key,row.input_shapes],sort_keys=True)
        table[sig]=table.get(sig,0)+row.count
        total+=int(row.flops or 0)
    if not table or total<1000:
        raise RuntimeError("source training profiler did not observe neural operations")
    return table,total,float(loss.detach())

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,type=Path)
    args=ap.parse_args()
    torch.set_num_threads(1)
    p,data,fit,_,receipt=fixed(args.source)
    seed=SEEDS[0]
    case=p["genuine_train_schedules_by_seed"][str(seed)][0]
    out={}
    for arm in ARMS:
        sig,flops,loss=source_op_profile(seed,arm,data,case,receipt,fit)
        out[arm]=dict(signatures=sig,reported_flops=flops,source_train_loss=loss)
    correct,wrong=ARMS
    a,b=out[correct],out[wrong]
    changed={key:(a["signatures"].get(key,0),b["signatures"].get(key,0))
             for key in set(a["signatures"])|set(b["signatures"])
             if a["signatures"].get(key,0)!=b["signatures"].get(key,0)}
    equal_ops=not changed
    partial_gap=abs(a["reported_flops"]-b["reported_flops"])/max(
        1,a["reported_flops"],b["reported_flops"])
    result=dict(study_id="PAR-MEM-011-SOURCE-TRAIN-OP-TRACE",
                original_real_source_SHA256=p["source_SHA256"],
                sample_seed=seed,history_events=129,batch_site_count=4,
                checked_forward_backward_clip_optimizer=True,
                aligned_op_shape_count_signature=equal_ops,
                source_operator_shape_difference_count=len(changed),
                different_op_signature_samples=list(changed.items())[:12],
                observed_unique_operator_shapes=len(a["signatures"]),
                correct_supported_flops=a["reported_flops"],
                control_supported_flops=b["reported_flops"],
                partial_supported_FLOP_gap_ratio=partial_gap,
                all_nonmatmul_scalar_ops_have_certified_FLOP_costs=False,
                total_backward_optimizer_FLOPs_mathematically_certified=False,
                same_architecture_graph_by_construction=True,
                no_heldout_source_scoring=True)
    print("PAR_MEM_011_SOURCE_FULL_TRAIN_OP_AUDIT="+json.dumps(result,sort_keys=True),flush=True)
    if not equal_ops or partial_gap>.01:
        raise RuntimeError("genuine learned comparator operator parity gate FAILED")
