#!/usr/bin/env python3
"""Exact symbolic parameter ledger for the PINNED official Mamba-3 SISO LM.

No importing of mamba_ssm, GPU allocation or training. Check this accounting
against the real deployed object on a BF16-capable accelerator before any fit.
Only applies to LANGCOMPARE029's SISO config, never to MIMO or other widths.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

D=80
LAYERS=6
FF=128
VOCAB=256
EXPAND=2
INNER=160
HEAD_DIM=32
HEADS=5
STATE=32
GROUPS=1
MIMO_RANK=1
ROPE_FRACTION=0.5
ROPE_ANGLES=8

def ledger():
    assert INNER == D*EXPAND and INNER%HEAD_DIM==0 and HEADS==INNER//HEAD_DIM
    assert ROPE_ANGLES==int(STATE*ROPE_FRACTION)//2
    # Shapes are counted from pinned state-spaces/mamba source:
    # mamba3.py Mamba3.__init__ SISO; mlp.py GatedMLP; mixer_seq_simple.py
    input_width = 2*INNER+2*STATE*GROUPS*MIMO_RANK+3*HEADS+ROPE_ANGLES
    mixer={
        "in_proj.weight [407,80]":input_width*D,
        "dt_bias [5]":HEADS,
        "B_bias [5,1,32]":HEADS*MIMO_RANK*STATE,
        "C_bias [5,1,32]":HEADS*MIMO_RANK*STATE,
        "B_norm.weight [32]":STATE,
        "C_norm.weight [32]":STATE,
        "D [5]":HEADS,
        "out_proj.weight [80,160]":D*INNER,
    }
    assert input_width==407
    # GatedMLP uses two bias-free projections: 80 -> 2*128 -> 80.
    # At width=128, "multiple_of=128" leaves it unchanged.
    mlp={"fc1.weight [256,80]":2*FF*D, "fc2.weight [80,128]":D*FF}
    norms={"block.norm [80w,80b]":2*D,"block.norm2 [80w,80b]":2*D}
    per_block=sum(mixer.values())+sum(mlp.values())+sum(norms.values())
    model={
        "embedding [256,80]":VOCAB*D,
        "mamba3_siso_6_blocks":per_block*LAYERS,
        "final_layernorm [80w,80b]":2*D,
        "untied_lm_head [256,80]":VOCAB*D,
    }
    result={"status":"ANALYTICAL_ONLY_NOT_INSTANTIATED",
            "config":{"d_model":D,"layers":LAYERS,"d_intermediate":FF,
                      "vocab":VOCAB,"expand":EXPAND,"d_inner":INNER,
                      "head_dim":HEAD_DIM,"heads":HEADS,"d_state":STATE,
                      "ngroups":GROUPS,"mimo_rank":MIMO_RANK,
                      "siso":True,"tied_embedding":False,"rope_fraction":ROPE_FRACTION},
            "individual_siso_mixer_parameters":mixer,
            "individual_mlp_parameters":mlp,
            "individual_block_norm_parameters":norms,
            "per_layer_parameters":per_block,
            "model_ledger":model,
            "analytical_total":sum(model.values()),
            "actual_cuda_model_count":"PENDING",
            "training_updates":0,
            "GPU_quota_consumed":0}
    if sum(mixer.values())!=45754 or per_block!=76794 or result["analytical_total"]!=501884:
        raise ValueError("Mamba-3 SISO source-derived geometry/count mismatch")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=ledger()
    s=json.dumps(result,sort_keys=True,indent=2)+"\n"
    if args.output:
        if args.output.exists():raise ValueError("refuse to overwrite receipt")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(s)
    print(s,end="")

if __name__=="__main__":
    main()
