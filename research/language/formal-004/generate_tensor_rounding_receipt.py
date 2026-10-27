#!/usr/bin/env python3
"""Replayable CPU per-operation floating-point diagnostic receipt.

No original experiment/checkpoint is refit or re-evaluated; this report is
ONLY synthetic numerical verification of local PyTorch dot reductions.
The calculation is independent of float64 reference reduction because the
input/product/reference exact sums are Python binary rationals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import torch
from audit_tensor_rounding import audit_float32_dot
from parallel_tac_language_gate_original import ParallelTACLanguage

ORIGINAL_SOURCE_SHA256="835212d74cc0d2d4d22ff7e75beb12771586cce58f2e63943690e5d859417457"

def produce_receipt():
    source=Path(__file__).with_name("parallel_tac_language_gate_original.py")
    found=hashlib.sha256(source.read_bytes()).hexdigest()
    if found != ORIGINAL_SOURCE_SHA256:
        raise ValueError("source hash changed: do not replay a mutated original model")
    torch.manual_seed(22051)
    torch.set_num_threads(1)
    records=[]
    for n in (8,16,48,88,128,248):
        for scale in (1e-3,1.,1e3):
            gen=torch.Generator().manual_seed(501+n+int(scale*1000))
            a=torch.randn(n,generator=gen,dtype=torch.float32)*scale
            b=torch.randn(n,generator=gen,dtype=torch.float32)/scale
            for backend in ("matmul","einsum","rounded_product_sum"):
                rec=audit_float32_dot(a,b,backend)
                rec["source"]="constructed_operand_pair"
                rec["scale"]=scale
                records.append(rec)
    model=ParallelTACLanguage(decoder=75,post=10,position_mode="multiscale",
                              gate_mode="silu").float().eval()
    with torch.no_grad():
        from source_independent_reference import phi
        addr=phi(17,torch.float32,torch.device("cpu"),1)[0]
        k=torch.nn.functional.normalize(addr @ model.rotation[0],dim=0,eps=1e-6)
        mem=torch.randn(16,8)
        for row in mem:
            rec=audit_float32_dot(row,k,"einsum")
            rec["source"]="original_source_8d_aligned_address"
            records.append(rec)
        input248=torch.randn(248)
        for i in (0,1,37,74):
            rec=audit_float32_dot(model.read_in.weight[i],input248,"matmul")
            rec["source"]="original_source_248d_decoder_row"
            records.append(rec)
    if not all(r["within_candidate_budget"] for r in records):
        bad=[r for r in records if not r["within_candidate_budget"]]
        raise AssertionError(f"sampled CPU backend exceeds candidate envelope: {bad[:3]}")
    return {
        "scope":"constructed CPU FP32 kernel diagnostics, NOT universal IEEE or language performance",
        "original_source_sha256":found,
        "backend":"cpu",
        "torch_version":str(torch.__version__),
        "python_version":platform.python_version(),
        "reference":"exact binary rationals of FP32 input operands, not FP64",
        "candidate_envelope":"gamma_{2n} sum_i |a_i*b_i| + 8*n*2^-149; verified only on recorded cases",
        "total_cases":len(records),
        "all_sampled_cases_within_candidate_envelope":True,
        "cases":records
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("tensor_rounding_receipt.json"))
    a=parser.parse_args()
    receipt=produce_receipt()
    a.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(f"Verified {receipt['total_cases']} exact-rational local cases; receipt: {a.output}")

if __name__=="__main__":
    main()
