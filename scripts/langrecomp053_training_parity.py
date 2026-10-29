#!/usr/bin/env python3
"""LANGRECOMP053: real-trained TAC CPU autograd checkpoint gradient parity.

Whole-original-window non-reentrant checkpoint wraps the *entire* actual TAC
PyTorch graph; source, labels, weights and stochastic state identical.
No optimizer step, GPU, original GRAPH011 implementation or VRAM claim.
"""
from __future__ import annotations
import argparse,hashlib,json,math,platform,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.langrecomp052_replay import load_original
from scripts.langlayer035_sampling import fit_sources,iter_batches

FWD_ATOL=3e-5
FWD_RTOL=1e-4
GRAD_ATOL=3e-5
GRAD_RTOL=1e-4
GEOMETRY=((1,32),(2,64))
RNG_SEED=53153

def source_batch(fit,batch_size,length):
    """First frozen seed1701 authentic FIT owner windows, labels strictly t+1."""
    _,records=next(iter_batches(fit,1701,steps=1))
    if not 1<=batch_size<=len(records) or not 4<=length<=128:
        raise ValueError("unregistered source shape")
    rows=[];targets=[];owners=[];hashes=[]
    for name,start,original in records[:batch_size]:
        if len(original)!=129 or original!=fit[name][start:start+129]:
            raise ValueError("original owner window bytes changed")
        rows.append(list(original[:length]))
        targets.append(list(original[1:length+1]))
        owners.append({"owner":name,"start":start,
                       "window_sha256":hashlib.sha256(original).hexdigest()})
        hashes.append(original)
    x=torch.tensor(rows,dtype=torch.long)
    y=torch.tensor(targets,dtype=torch.long)
    if x.shape!=y.shape or not torch.equal(x[:,1:],y[:,:-1]):
        raise ValueError("true next-byte source labels misaligned")
    return x,y,owners,hashlib.sha256(b"".join(hashes)).hexdigest()

class SavedRequestCounter:
    """Logical autograd save requests only: NOT physical VRAM or unique storage."""
    def __init__(self):
        self.calls=0
        self.logical_bytes=0
        self.float_bytes=0
    def pack(self,t):
        self.calls+=1
        count=t.numel()*t.element_size()
        self.logical_bytes+=count
        if t.is_floating_point():self.float_bytes+=count
        return t
    @staticmethod
    def unpack(t):return t

def one_pass(model,x,y,policy):
    """Exact same original trainable weights with zero initialized gradients."""
    if policy not in ("native","checkpoint_entire_graph"):
        raise ValueError("unregistered graph")
    model.zero_grad(set_to_none=True)
    counter=SavedRequestCounter()
    input_before=x.detach().clone()
    target_before=y.detach().clone()
    before_rng=torch.get_rng_state().clone()
    start=time.perf_counter_ns()
    with torch.autograd.graph.saved_tensors_hooks(counter.pack,counter.unpack):
        if policy=="native":
            logits=model(x)
        else:
            logits=checkpoint(lambda tokens:model(tokens),x,
                              use_reentrant=False,preserve_rng_state=True,
                              determinism_check="default")
        if logits.shape!=(x.size(0),x.size(1),256):
            raise ValueError("original neural logits dimensions changed")
        loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
    fwd_time=(time.perf_counter_ns()-start)/1e6
    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
        raise ValueError("model loss or logits not finite")
    if not torch.equal(x,input_before) or not torch.equal(y,target_before):
        raise ValueError("source input/label mutated")
    fwd_rng_unchanged=torch.equal(before_rng,torch.get_rng_state())
    tick=time.perf_counter_ns()
    loss.backward()
    bwd_time=(time.perf_counter_ns()-tick)/1e6
    grads={}
    missing=[]
    for name,param in model.named_parameters():
        if param.grad is None:
            missing.append(name)
        else:
            if not torch.isfinite(param.grad).all():
                raise ValueError("nonfinite original model gradient "+name)
            grads[name]=param.grad.detach().clone()
    return {"logits":logits.detach().clone(),
        "loss":loss.detach().clone(),
        "grads":grads,"missing":sorted(missing),
        "forward_ms":fwd_time,"backward_ms":bwd_time,
        "fwd_rng_unchanged":fwd_rng_unchanged,
        "saved_logical_tensor_requests":counter.calls,
        "saved_logical_tensor_bytes":counter.logical_bytes,
        "saved_logical_floating_bytes":counter.float_bytes,
        "time_wall_ms":fwd_time+bwd_time}

def compare_outputs(a,b):
    if a["missing"]!=b["missing"]:
        raise ValueError("recompute changed missing gradient paths")
    if set(a["grads"])!=set(b["grads"]):
        raise ValueError("recompute changed trainable gradient tensor schema")
    if not torch.allclose(a["logits"],b["logits"],atol=FWD_ATOL,rtol=FWD_RTOL):
        raise ValueError("forward neural logits changed under checkpoint")
    if not torch.allclose(a["loss"],b["loss"],atol=FWD_ATOL,rtol=FWD_RTOL):
        raise ValueError("original next-byte NLL changed under checkpoint")
    max_abs=0.;max_rel=0.;bad=[];total=0
    for name in a["grads"]:
        x=a["grads"][name];y=b["grads"][name]
        if x.shape!=y.shape:
            raise ValueError("original parameter gradient tensor changed: "+name)
        diff=(x-y).abs()
        d=float(diff.max()) if diff.numel() else 0.
        denom=torch.maximum(torch.maximum(x.abs(),y.abs()),torch.full_like(x,1e-6))
        rel=float((diff/denom).max()) if diff.numel() else 0.
        max_abs=max(max_abs,d);max_rel=max(max_rel,rel)
        total+=x.numel()
        if not torch.allclose(x,y,atol=GRAD_ATOL,rtol=GRAD_RTOL):bad.append(name)
    if bad:
        raise ValueError("original trained graph gradients changed: "+",".join(bad[:12]))
    return {"max_logits_abs":float((a["logits"]-b["logits"]).abs().max()),
        "max_nll_abs":float((a["loss"]-b["loss"]).abs()),
        "max_gradient_abs":max_abs,
        "max_relative_gradient_including_near_zero":max_rel,
        "gradient_tensor_count":len(a["grads"]),
        "gradient_element_count":total,
        "matching_missing_gradient_parameters":a["missing"],
        "original_qualifications":"conditional PyTorch FP32 allclose, not IEEE mathematical proof"}

def run(original_dir):
    if torch.cuda.is_initialized():
        raise ValueError("GPU unregistered; CPU phase only")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    fit=fit_sources()
    model,receipt=load_original(original_dir,fit)
    model.train()
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params==499623
    # Same model instance: no optimizer update ever, zero fresh grads each pass.
    state_before={n:p.detach().clone() for n,p in model.named_parameters()}
    rows=[]
    for b,length in GEOMETRY:
        x,y,owners,source_hash=source_batch(fit,b,length)
        common_rng=torch.manual_seed(RNG_SEED)
        rng=torch.get_rng_state().clone()
        torch.set_rng_state(rng.clone())
        ref=one_pass(model,x,y,"native")
        state_after_ref=torch.get_rng_state().clone()
        torch.set_rng_state(rng.clone())
        alt=one_pass(model,x,y,"checkpoint_entire_graph")
        state_after_alt=torch.get_rng_state().clone()
        comparison=compare_outputs(ref,alt)
        if not torch.equal(state_after_ref,state_after_alt):
            raise ValueError("forward/backward RNG difference")
        for name,param in model.named_parameters():
            if not torch.equal(param.detach(),state_before[name]):
                raise ValueError("original model weights changed during diagnostic "+name)
        rows.append({"batch_size":b,"window_tokens":length,
            "actual_original_next_byte_targets":b*length,
            "original_fit_owners":owners,
            "source_windows_sha256":source_hash,
            "parity":comparison,
            "native":{"loss_nll":float(ref["loss"]),**{k:ref[k] for k in
                ("forward_ms","backward_ms","time_wall_ms","fwd_rng_unchanged",
                 "saved_logical_tensor_requests","saved_logical_tensor_bytes",
                 "saved_logical_floating_bytes")}},
            "checkpoint_entire_graph":{"loss_nll":float(alt["loss"]),**{k:alt[k] for k in
                ("forward_ms","backward_ms","time_wall_ms","fwd_rng_unchanged",
                 "saved_logical_tensor_requests","saved_logical_tensor_bytes",
                 "saved_logical_floating_bytes")}},
            "identical_cpu_rng_after_backward":True})
    return {"study":"LANGRECOMP053",
        "status":"INTERNAL_ORIGINAL_TRAINED_MODEL_SOURCE_FIT_CPU_FORWARD_BACKWARD_PARITY",
        "reference":receipt,
        "geometry":[list(x) for x in GEOMETRY],
        "trained_original_parameters":params,
        "optimizer_updates":0,
        "training_updates":0,
        "trainable_parameter_bytes_FP32":params*4,
        "original_FIT_causal_next_byte_source":True,
        "pytorch":torch.__version__,
        "platform":platform.platform(),
        "cpu_threads":2,
        "checkpoint_method":"torch.utils.checkpoint.checkpoint use_reentrant=False wrapping entire original forward training graph",
        "verification":{"abs_tol":GRAD_ATOL,"rel_tol":GRAD_RTOL,
            "original_unchanged_weights":True,"new_fits":False,
            "whole_model_peak_cuda_vram":"NOT_MEASURED_NO_GPU",
            "optimizer_step":"NOT_RUN"},
        "rows":rows,
        "independently_qualified":False,
        "graph011_original_method":"NOT_LOCATED_OR_IMPORTED",
        "gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "limits":["No 5.51MB→0.0014MB or 99% GPU VRAM verification",
            "Logical saved tensor requests count includes aliases and excludes runtime allocator; not peak memory or unique device residency",
            "Entire-graph nonreentrant checkpoint recomputation is not a tuned segment checkpointing implementation",
            "Identical trained original model on already exposed original FIT data; not external generalization",
            "Backward parity is not optimizer equivalence or new trained model capability",
            "No PNDS whole-model Lean/IEEE/AD proof, validation independent of PyTorch, or GPU metrics"],
        "promotion":False}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("do not overwrite original evidence")
    result=run(a.original_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":result["study"],"status":result["status"],
        "rows":[{"batch":r["batch_size"],"length":r["window_tokens"],
        "loss":r["native"]["loss_nll"],
        "logits_error":r["parity"]["max_logits_abs"],
        "gradient_error":r["parity"]["max_gradient_abs"],
        "gradient_count":r["parity"]["gradient_tensor_count"],
        "normal_wall_ms":r["native"]["time_wall_ms"],
        "checkpoint_wall_ms":r["checkpoint_entire_graph"]["time_wall_ms"],
        "normal_saved_logical_bytes":r["native"]["saved_logical_tensor_bytes"],
        "checkpoint_saved_logical_bytes":r["checkpoint_entire_graph"]["saved_logical_tensor_bytes"]}
        for r in result["rows"]]},sort_keys=True))
if __name__=="__main__":main()
