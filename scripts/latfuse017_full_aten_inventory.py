"""LATFUSE017 REAL-source full ATen operator SHAPE/PATH inventory.

This is NOT full hardware FLOP parity proof. Torch op semantics, exponentials,
optimizer and memory operations differ; preserve residual uncertainty, never
mislabel operator inventory as exact full training compute. Runs optimizer
with lr=0 (Adam states computed; neural weights BITWISE unchanged).
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from scripts import latfuse002_real_fit as source
from scripts import latfuse008_paired_real_training as original
from scripts.latfuse016_authentic_content_nofit import (
    original_source_forward_loss,actual_rows,SEEDS,ARMS)
from scripts.validate_research_parity import check_manifest

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse017/REGISTERED_OP_INVENTORY.json"
SIMPLE={"aten::add","aten::sub","aten::mul","aten::div","aten::neg",
        "aten::abs","aten::reciprocal","aten::maximum","aten::minimum",
        "aten::clamp","aten::clamp_min","aten::clamp_max","aten::floor_divide",
        "aten::lerp","aten::where","aten::addcmul","aten::addcdiv",
        "aten::sum","aten::mean"}
TRANSCEND={"aten::exp","aten::log","aten::log1p","aten::tanh","aten::sigmoid",
       "aten::sqrt","aten::rsqrt","aten::pow","aten::softplus",
       "aten::_softmax","aten::_softmax_backward_data",
       "aten::tanh_backward","aten::sigmoid_backward","aten::threshold_backward",
       "aten::smooth_l1_loss","aten::smooth_l1_loss_backward","aten::softplus_backward"}
MATMUL={"aten::mm","aten::bmm","aten::addmm","aten::mv","aten::dot"}
# Pure metadata/view/assignment operations counted as zero ARITHMETIC FLOPs
# but included in the full operator signature for memory/performance audit.
NOFLOP={"aten::view","aten::_unsafe_view","aten::reshape","aten::unsqueeze",
"aten::squeeze","aten::squeeze_dim","aten::transpose","aten::permute",
"aten::slice","aten::select","aten::expand","aten::as_strided",
"aten::detach","aten::alias","aten::clone","aten::copy_","aten::empty",
"aten::empty_strided","aten::zeros","aten::ones","aten::zeros_like",
"aten::ones_like","aten::empty_like","aten::lift_fresh","aten::new_empty",
"aten::fill_","aten::zero_","aten::scatter_","aten::index",
"aten::index_put","aten::_to_copy","aten::to","aten::cat",
"aten::split","aten::split_with_sizes","aten::unbind",
"aten::gather","aten::select_scatter","aten::slice_scatter",
"aten::index_select","aten::embedding","aten::contiguous"}
def tensors(x):
    return [v for v in tree_leaves(x) if isinstance(v,torch.Tensor)]
def numel(shape):
    n=1
    for k in shape:n*=k
    return n
def mm_flops(name,inputs):
    try:
        if name=="aten::bmm":
            a,b=inputs[:2]
            return int(2*a.shape[0]*a.shape[-2]*a.shape[-1]*b.shape[-1])
        if name in ("aten::mm","aten::addmm"):
            a,b=inputs[-2:]
            return int(2*a.shape[-2]*a.shape[-1]*b.shape[-1]+
                (a.shape[-2]*b.shape[-1] if name=="aten::addmm" else 0))
        if name=="aten::mv":
            a,b=inputs[:2]
            return int(2*a.shape[0]*a.shape[1])
        if name=="aten::dot":
            return int(2*inputs[0].numel())
    except (IndexError,ValueError):
        return 0
    return 0

class OriginalGraphInventory(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls=collections.Counter()
        self.tokens=collections.Counter()
        self.result_elems=collections.Counter()
        self.categories=collections.Counter()
        self.unknown=collections.Counter()
        self.unknown_dtypes=collections.Counter()
        self.stage="unknown"
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        inputs=tensors((args,kwargs))
        out=func(*args,**kwargs)
        outputs=tensors(out)
        name=func._schema.name
        token=name+"."+func._schema.overload_name
        # Auditable precise per-operation tensor shape identity.
        signature=(token,
          tuple((str(t.dtype),tuple(int(z) for z in t.shape)) for t in inputs[:5]),
          tuple((str(t.dtype),tuple(int(z) for z in t.shape)) for t in outputs[:4]))
        k=json.dumps(signature,separators=(",",":"))
        phase=self.stage
        self.calls[(phase,name)]+=1
        self.tokens[(phase,k)]+=1
        output_elements=sum(int(z.numel()) for z in outputs
                            if z.is_floating_point() or z.is_complex())
        self.result_elems[(phase,name)]+=output_elements
        if name in MATMUL:
            self.categories[(phase,"exact_matmul_multiply_add_2x")]+=mm_flops(name,inputs)
        elif name in SIMPLE:
            self.categories[(phase,"elementary_float_result_elements_NOT_strict_flops")]+=output_elements
        elif name in TRANSCEND:
            self.categories[(phase,"transcendental_result_elements_UNKNOWN_implementation_flops")]+=output_elements
        elif name in NOFLOP:
            self.categories[(phase,"view_or_storage_or_index_ops")]+=1
        else:
            self.unknown[(phase,name)]+=1
            self.unknown_dtypes[(phase,name)]+=output_elements
        return out
    def receipt(self):
        return dict(
            operations=[{"stage":phase,"op":name,"calls":int(n),
                         "float_output_elements":int(self.result_elems[(phase,name)])}
                        for (phase,name),n in sorted(self.calls.items())],
            operator_shapes=[{"stage":stage,"signature":signature,"calls":int(n)}
                        for (stage,signature),n in sorted(self.tokens.items())],
            operation_category_sums=[{"stage":stage,"category":category,
                                      "proxy_or_exact":int(n)}
                        for (stage,category),n in sorted(self.categories.items())],
            unclassified_operator_names=[{"stage":stage,"name":name,"calls":int(n),
                           "floating_result_elements":int(self.unknown_dtypes[(stage,name)])}
                           for (stage,name),n in sorted(self.unknown.items())],
            operation_shape_count=len(self.tokens))

def run(args):
    if hashlib.sha256(args.source.read_bytes()).hexdigest()!=source.SOURCE_SHA:
        raise RuntimeError("original real-source archive SHA mismatch")
    m=json.loads(REG.read_text())
    check_manifest(m,"preflight")
    if m["study_id"]!="LATFUSE017-AUTHENTIC-WHOLE-TRAIN-OP-INVENTORY":
        raise RuntimeError("nonfit whole operator audit manifest changed")
    torch.set_num_threads(1)
    d=original.source_observations(args.source)
    rows=actual_rows(d)
    row_sha=hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest()
    models=[]
    for seed in SEEDS:
        for arm in ARMS:
            model=original.create_model(seed,arm)
            p_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
            if p_count!=99390:
                raise RuntimeError("real model learned budget changed")
            before=[p.detach().clone() for p in model.parameters()]
            # Actual optimizer operations are executed but learning rate ZERO.
            # No trained model weights, original checkpoint or fit produced.
            opt=torch.optim.AdamW(model.parameters(),lr=0.0,weight_decay=.001,
                                   foreach=False,fused=False)
            audit=OriginalGraphInventory()
            with audit:
                audit.stage="forward"
                model.train()
                opt.zero_grad(set_to_none=True)
                total,content,fore,target=original_source_forward_loss(model,d,rows)
                audit.stage="backward"
                total.backward()
                audit.stage="grad_clip"
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                audit.stage="adamw_lr0"
                opt.step()
            if not bool(torch.isfinite(total)):
                raise RuntimeError("source original loss not finite")
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
                raise RuntimeError("real-source model gradient invalid")
            if any(not torch.equal(x,y) for x,y in zip(before,model.parameters())):
                raise RuntimeError("ZERO-LR operator auditing modified any model parameter")
            tape=audit.receipt()
            # Summaries are mathematically unambiguous as ATen *shape* signatures.
            # Scalar exp/log/tanh FLOP costs are hardware- and kernel-dependent.
            summary=dict(seed=seed,arm=arm,
                actual_source_time_case_SHA=row_sha,
                real_source_examples=len(rows),historical_hour_steps=88,
                live_params=p_count,original_recurrent_nominal_bytes=6800,
                original_same_true_source_recorded_objective=float(total.detach()),
                optimizer_lr=0,optimizer_foreach=False,optimizer_fused=False,
                optimizer_state_tensors=len(opt.state_dict()["state"]),
                parameters_bitwise_unchanged=True,
                operations=tape["operations"],
                operator_shape_count=tape["operation_shape_count"],
                categorized_operator_tallies=tape["operation_category_sums"],
                unclassified_operator_names=tape["unclassified_operator_names"],
                full_operator_shape_receipt_SHA256=hashlib.sha256(
                    json.dumps(tape["operator_shapes"],sort_keys=True).encode()).hexdigest())
            models.append((summary,tape))
            print("LATFUSE017_ORIGINAL_SOURCE_STEP="+json.dumps({
              k:summary[k] for k in ("seed","arm","live_params","optimizer_state_tensors",
               "operator_shape_count","categorized_operator_tallies")},sort_keys=True),flush=True)
    by_seed=[]
    for seed in SEEDS:
        x,y=[z for z in models if z[0]["seed"]==seed]
        if {x[0]["arm"],y[0]["arm"]}!=set(ARMS):
            raise RuntimeError("registered training arm mismatch")
        a,b=x[1],y[1]
        full=collections.Counter({(j["stage"],j["signature"]):j["calls"]
                                  for j in a["operator_shapes"]})
        comp=collections.Counter({(j["stage"],j["signature"]):j["calls"]
                                  for j in b["operator_shapes"]})
        delta=full-comp
        removed=comp-full
        # Validate BOTH FLOP proxies with SAME original true-source workload.
        sums=lambda tape:collections.Counter({
          (j["stage"],j["category"]):j["proxy_or_exact"]
          for j in tape["operation_category_sums"]})
        a_stat=sums(a);b_stat=sums(b)
        keys=sorted(set(a_stat)|set(b_stat))
        comparisons={f"{q[0]}:{q[1]}":
          {"assoc":int(a_stat[q]),"fastslow":int(b_stat[q]),
           "relative_gap":abs(a_stat[q]-b_stat[q])/max(a_stat[q],b_stat[q],1)}
          for q in keys}
        imbal=[{"stage":phase,"signature":sig,"assoc_only_calls":int(v)}
              for (phase,sig),v in delta.most_common(30)]
        imbal+=[{"stage":phase,"signature":sig,"fastslow_only_calls":int(v)}
              for (phase,sig),v in removed.most_common(30)]
        # Find if all UNSUPPORTED OP shapes match; mismatch alone invalidates
        # a claim of exhaustive scalar training FLOPs, even when FLOP proxy~1%.
        unknown_a={(r["stage"],r["name"]):(r["calls"],r["floating_result_elements"])
            for r in x[0]["unclassified_operator_names"]}
        unknown_b={(r["stage"],r["name"]):(r["calls"],r["floating_result_elements"])
            for r in y[0]["unclassified_operator_names"]}
        unknown_changed=[{"stage":p[0],"op":p[1],"assoc":unknown_a.get(p),
                   "fastslow":unknown_b.get(p)}
                   for p in sorted(set(unknown_a)|set(unknown_b))
                   if unknown_a.get(p)!=unknown_b.get(p)]
        by_seed.append(dict(seed=seed,
          stage_exact_category_proxies=comparisons,
          full_operator_shape_signature_excess_in_each_arm_preview=imbal,
          unmatched_operator_signature_counts={"assoc_excess":sum(delta.values()),
                 "fastslow_excess":sum(removed.values())},
          unsupported_operator_counts_or_elements_differ=unknown_changed,
          full_training_FLOPs_PARITY_VERIFIED=False,
          reason="Operator inventory covers forward/back/gradient clip/AdamW, but exact FLOPs of transcendental, custom kernels and backend optimizer operations cannot be derived from count/shape signatures alone."))
    out=dict(study_id=m["study_id"],real_source_zip_sha256=source.SOURCE_SHA,
      original_source_case_sha256=row_sha,original_source_cases=len(rows),
      original_model_weights_bitwise_unchanged=True,optimizer_update_lr0_only=True,
      trained_models_created=0,original_checkpoint_modified=False,
      compare_trained_original_full_FLOPs_qualifies=False,
      original_source_March_only_no_exposed_2015_or_protected_2016_17_numeric=True,
      method="TorchDispatchMode captures ALL invoked ATen tensor calls by stage and full input/output shape signature, explicit exact matmul FLOP inventory and remaining elementary/transcendental/unknown cost tallies. Zero-learning-rate optimizer states updated but original weights not changed.",
      models=[z[0] for z in models],original_seed_pair_audit=by_seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise RuntimeError("would overwrite precommitted empirical source audit")
    args.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE017_REAL_SOURCE_FULL_ATEN_INVENTORY="+json.dumps({
      "real_source_cases":len(rows),"four_modeled_graphs":len(models),
      "optimizer_zero_lr":True,"full_flop_parity":False,
      "shape_deltas":[u["unmatched_operator_signature_counts"] for u in by_seed]
    },sort_keys=True))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    run(args)
if __name__=="__main__":main()
