#!/usr/bin/env python3
"""LANGSCALE073 no-fit positional-extrapolation differential on frozen 070 models.

Explicitly follows 072 strict cross-host replay failures, using frozen 1e-6\nNLL/Brier replay tolerances and exact source SHA/top1. Original native
position vs source-registered cap_all transform at L256, with L128
unchanged-position negative control. Previously EXPOSED DEV, post-071
hypothesis; does not establish causal root cause or generalization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),
             str(ROOT/"research/mathematical-engineering/langmemory042")]
from candidate import controlled_position
from scripts.langscale071_context_eval import (
    SEEDS,SCALE_ORDER,load_original_model,target_cohort,score_length)
from byte_next_reference import original_source


def model_sha(model):
    h=hashlib.sha256()
    for name,x in model.state_dict().items():
        h.update(name.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(tier,seed,checkpoint_dir,aggregate,old_context):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    _,dev,_=original_source(ROOT)
    cohort,targetsha=target_cohort(dev)
    previous=json.loads(old_context.read_text())
    if (previous.get("study")!="LANGSCALE071" or
        previous.get("tier")!=tier or previous.get("seed")!=seed or
        previous.get("target_cohort_sha256")!=targetsha or
        previous.get("training_updates")!=0):
        raise ValueError("not original same-target 071 source cohort")
    model,receipt=load_original_model(checkpoint_dir,aggregate,tier,seed)
    if receipt["checkpoint_sha256"]!=previous["trained_checkpoint"]["checkpoint_sha256"]:
        raise ValueError("original frozen model identity differs")
    preweight=model_sha(model)
    if bool(model.pos._forward_pre_hooks) or bool(model.pos._forward_hooks):
        raise ValueError("original checkpoint positional hook not clean")
    original128=score_length(model,dev,cohort,128)
    original256=score_length(model,dev,cohort,256)
    old_rows={x["context"]:x for x in previous["lengths"]}
    # LANGSCALE073 separately preregistered after LANGSCALE072's 4/9 strict
    # cross-host numerical replay failures. The 1e-6 tolerance is frozen
    # BEFORE any new 073 run, consistent with original 070 cold-replay rule.
    # Original source owner/input/target SHA and top1 remain EXACT.
    replay_errors={
        str(context):{
          "abs_nll":abs(result["nll"]-old_rows[context]["nll"]),
          "abs_brier":abs(result["brier"]-old_rows[context]["brier"]),
          "top1_identical":result["top1"]==old_rows[context]["top1"],
          "source_input_sha_identical":result["inputs_and_targets_sha256"]==
                                       old_rows[context]["inputs_and_targets_sha256"],
        }
        for context,result in ((128,original128),(256,original256))
    }
    for key,record in replay_errors.items():
        if not record["source_input_sha_identical"] or not record["top1_identical"]:
            raise ValueError("original target source/accuracy mismatch at L"+key+
                             ": "+str(record))
        if record["abs_nll"]>1e-6 or record["abs_brier"]>1e-6:
            raise ValueError("original cross-host numerical replay beyond predeclared 1e-6 at L"+key+
                             ": "+str(record))
    with controlled_position(model,"cap_all"):
        cap128=score_length(model,dev,cohort,128)
        cap256=score_length(model,dev,cohort,256)
    if bool(model.pos._forward_pre_hooks) or bool(model.pos._forward_hooks):
        raise ValueError("temporary positional hook leaked outside scoring")
    if preweight!=model_sha(model):
        raise ValueError("source pretrained model checkpoint changed")
    # At clocks 0..127 the original source cap_all must act as a no-op.
    for key in ("n","nll","top1","brier","inputs_and_targets_sha256"):
        if cap128[key]!=original128[key]:
            raise ValueError("source cap_all unexpectedly changed original trained-range positional law at128: "+key)
    for x in (cap128,cap256,original128,original256):
        if x["n"]!=256 or x["inputs_and_targets_sha256"] not in {
            original128["inputs_and_targets_sha256"],
            original256["inputs_and_targets_sha256"]}:
            raise ValueError("different source owner/target cohort in positional comparison")
    if cap256["inputs_and_targets_sha256"]!=original256["inputs_and_targets_sha256"]:
        raise ValueError("position change confounded with different original source bytes")
    return {
        "study":"LANGSCALE073","status":"SOURCE_POSITION_POST_FAILED_EXACT_CROSS_HOST_REPLAY_NO_FIT_DIAGNOSTIC",
        "tier":tier,"seed":seed,"checkpoint_sha256":receipt["checkpoint_sha256"],
        "target_cohort_sha256":targetsha,"same_targets":256,
        "old_071_native_256_nll":old_rows[256]["nll"],
        "cross_host_replay_abs_tolerance":1e-6,
        "cross_host_replay_errors":replay_errors,
        "weights_unchanged_sha256":preweight,
        "hook_removed_and_model_weights_original":True,
        "original_position_128":original128,
        "cap_position_128_negative_control":cap128,
        "original_position_256":original256,
        "cap_position_256":cap256,
        "cap256_minus_native256_nll":cap256["nll"]-original256["nll"],
        "training_updates":0,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "independently_qualified":False,
        "prior_LANGSCALE072_exact_replay_status":"FAILED_4_OF_9",
        "limits":["selected after observing 071 L256 degradation on already EXPOSED DEV",
                  "inference-time transform not trained/revalidated on unseen data",
                  "position-only cap does not isolate recurrent state mathematical degradation",
                  "all windows reset and are not persistent online streaming",
                  "no model weights changed or trained, no mobile device evidence",
                  "complete PNDS actual source IEEE/AD/full graph proof OPEN"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tier",choices=SCALE_ORDER,required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--original-aggregate",type=Path,required=True)
    p.add_argument("--previous-context",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse overwrite evidence")
    r=evaluate(a.tier,a.seed,a.checkpoint_dir,a.original_aggregate,a.previous_context)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("POSITION_DIAGNOSTIC "+json.dumps({
        "tier":a.tier,"seed":a.seed,
        "native128":r["original_position_128"]["nll"],
        "native256":r["original_position_256"]["nll"],
        "cap256":r["cap_position_256"]["nll"],
        "cap_minus_native":r["cap256_minus_native256_nll"]},sort_keys=True))


if __name__=="__main__":main()
