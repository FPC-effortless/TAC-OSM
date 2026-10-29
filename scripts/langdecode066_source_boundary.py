#!/usr/bin/env python3
"""LANGDECODE066: constructed finite-precision, high-clock cache/source audit.

No trained checkpoint, no language scores, and no claimed universal IEEE
proof. Validates the actual original Python source imported from TAC-OSM.
Reports an independent integrity gap in the experimental 064 engine:
the cache is mutable but not included in the original signature check.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),
              str(ROOT/"research/mathematical-engineering/langstream043")]
from incremental import MAX_CLOCK, SHAPES, phi_position
from candidate import transform
from factorial_depth import FactorialDepthControl040
from scripts.langdecode057_validation import BatchedValidationTAC057
from scripts.langdecode064_cached_cap import CachedCapPositionTAC064

# Includes clocks beyond FP32's exact integer representation, beyond FP64's
# integer-exact boundary, and near the registered int64 upper limit.
HIGH_CLOCKS=(127,128,129,255,512,8192,65536,
             2**24-1,2**24,2**24+1,
             2**31-1,2**31,2**31+1,
             2**53-1,2**53,2**53+1,
             MAX_CLOCK-2)


def check(dtype:torch.dtype)->dict:
    torch.manual_seed(66066)
    model=FactorialDepthControl040(state_width=48,upper_graph=False).to(dtype).eval()
    orig=BatchedValidationTAC057(model,"cap_all")
    candidate=CachedCapPositionTAC064(model,"cap_all")
    assert candidate._cap_all_ref is not None
    reference=candidate._cap_all_ref
    checks=0
    max_abs=0.0
    with torch.inference_mode():
        for start in range(0,65536,4096):
            tt=torch.arange(start,min(65536,start+4096),dtype=torch.long)
            original=transform(phi_position(tt,dtype),"cap_all")
            cached=torch.where(
                (tt>=128)[:,None],
                reference.expand(tt.shape[0],-1), original)
            checks+=int(tt.numel())
            if not torch.equal(original,cached):
                raise ValueError("cached positional bitwise mismatch in dense clock range")
            max_abs=max(max_abs,(original-cached).abs().max().item())
        for t in HIGH_CLOCKS:
            tt=torch.tensor([t],dtype=torch.long)
            original=transform(phi_position(tt,dtype),"cap_all")
            cached=reference if t>=128 else original
            if not torch.equal(original,cached):
                raise ValueError("cached positional bitwise mismatch at high clock "+str(t))
            checks+=1
            max_abs=max(max_abs,(original-cached).abs().max().item())
        # High-clock exact recurrent/output comparison without pretending to
        # reconstruct any historical source prefix at large t.
        tokens=torch.tensor([12,211],dtype=torch.long)
        mask=torch.tensor([True,False])
        for t in HIGH_CLOCKS:
            if t<128:continue
            s=replace(orig.start(("owner-a","owner-b")),clock=t)
            y,st=orig.step(tokens,s,owners=s.owners,write_mask=mask)
            z,sc=candidate.step(tokens,s,owners=s.owners,write_mask=mask)
            if not torch.equal(y,z) or any(
                    not torch.equal(st.tensors[k],sc.tensors[k]) for k in SHAPES):
                raise ValueError("clock/state/logits original-candidate mismatch "+str(t))
            if sc.tensor_bytes()!=2*976*torch.tensor([],dtype=dtype).element_size():
                raise ValueError("persistent per-owner state changed")
    # Mathematical single-position projection gradient check, not
    # neural full-graph training/AD verification.
    weight=model.embedding.weight.detach()[:2].clone()
    features=transform(phi_position(torch.tensor([256,4096]),dtype),"cap_all")
    cache=reference.expand(2,-1)
    if not torch.equal(features,cache):raise ValueError("gradient probes differ")
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        (weight+model.pos(features)).tanh().sum().backward()
        g1=model.pos.weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        (weight+model.pos(cache)).tanh().sum().backward()
        g2=model.pos.weight.grad.detach().clone()
        if not torch.equal(g1,g2):raise ValueError("position-layer weight gradient mismatch")
    # Negative control: demonstrate untracked mutable cache on 064, not
    # a real adversarial exploit and not a passed integrity/safety check.
    before=reference.detach().clone()
    with torch.no_grad():reference[0,0].add_(0.125)
    try:
        candidate._check_model()
        original_guard_detected_cache_mutation=False
    except ValueError:
        original_guard_detected_cache_mutation=True
    if torch.equal(before,reference):
        raise ValueError("negative-control mutation did not change tensor")
    # No mutation persists beyond this disposable constructed model instance.
    return {"dtype":str(dtype),"dense_clock_tests":65536,
            "large_clock_tests":len(HIGH_CLOCKS),"checked_positions":checks,
            "large_clock_state_transitions":len([t for t in HIGH_CLOCKS if t>=128]),
            "bitwise_original_position_and_state":True,
            "position_linear_weight_gradient_bitwise":True,
            "max_abs_original_position_error":max_abs,
            "original_guard_detected_cache_mutation":original_guard_detected_cache_mutation,
            "integrity_status":("GUARDED" if original_guard_detected_cache_mutation
                                else "OPEN_MUTABLE_ENGINE_CACHE_UNTRACKED")}


def run()->dict:
    torch.set_num_threads(2)
    rows=[check(torch.float32),check(torch.float64)]
    if not all(row["bitwise_original_position_and_state"] and
               row["position_linear_weight_gradient_bitwise"] for row in rows):
        raise ValueError("audit numerical equality failed")
    return {"study":"LANGDECODE066","purpose":"constructed_source_bridge_only",
            "status":"NUMERICAL_RANGE_PASS_ENGINE_CACHE_INTEGRITY_OPEN",
            "rows":rows,"source":"original TAC-OSM context026 + langmemory042 imported",
            "no_new_neural_training":True,"gpu_minutes":0,
            "protected_TEST":"NOT_ACCESSED","original_checkpoint":"NOT_LOADED",
            "full_source_ieee_ad_proof":"OPEN",
            "independent_mobile_latency":"NOT_MEASURED",
            "limits":["source-scoped empirical finite clocks, not all int64 clocks",
                      "probe positional linear AD only, not whole source backwards",
                      "mutated cache negative control is a constructed failure of original cache integrity monitoring",
                      "high-clock synthetic state does not imply genuine long-context accuracy"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse to overwrite audit evidence")
    report=run()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(report,sort_keys=True))


if __name__=="__main__":main()
