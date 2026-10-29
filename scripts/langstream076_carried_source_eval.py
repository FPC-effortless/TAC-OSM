#!/usr/bin/env python3
"""LANGSTREAM076 original trained source carried-state and native-memory lesion.

This is a genuine PSF publisher real DEV next-byte *diagnostic*, not a new
training run and not independently held-out validation (DEV already exposed).
All source owners are carried continuously from byte 0 through 319; native
memory early-clear and one-step native lesion are causal counterfactual
controls, NOT an exact causal attribution of old facts or total state value.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
for name in ("langbase024","langscale075","langscale070","langstream076"):
    path=str(ROOT/"research/mathematical-engineering"/name)
    if path not in sys.path:sys.path.insert(0,path)
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from byte_next_reference import original_source
from native_matrix import NativeMatrixTAC075,COUNTS,WIDTHS
from native_streaming import NativeStreamingTAC076
from scripts.langscale075_cpu_fit import model_source_preflight
from scripts.langlayer035_sampling import fit_sources

SEEDS=(1701,1702,1703)
TIERS=("m8","m12","m16")
TARGET_CLOCKS=tuple(range(128,320,3))
EXPECTED_STATE_BYTES={"m8":3904,"m12":4752,"m16":5728}
ORIGINAL_SOURCES={"m8":("LANGSCALE070","p500k"),"m12":("LANGSCALE075","m12"),
                  "m16":("LANGSCALE075","m16")}


def load_original(tier,seed,root,summary_file):
    if tier not in TIERS or seed not in SEEDS:
        raise ValueError("registered original native-memory model only")
    summary=json.loads(summary_file.read_text())
    if summary.get("study")!="LANGSCALE075" or summary.get("new_checkpoints_verified")!=6:
        raise ValueError("original 075 training+read-only recovery unverified")
    if not summary.get("source_paired") or not summary.get("parameter_parity_within_one_percent"):
        raise ValueError("source model study not paired")
    entries=summary["metrics"][tier]["original_seed_receipts"]
    receipt=next((x for x in entries if x["seed"]==seed),None)
    if receipt is None:raise ValueError("missing source original model seed receipt")
    files=list(root.rglob("*.pt"))
    if len(files)!=1:raise ValueError("must supply exactly one original trained source checkpoint")
    raw=files[0].read_bytes()
    if hashlib.sha256(raw).hexdigest()!=receipt["checkpoint_sha256"]:
        raise ValueError("model checkpoint SHA does not match original verified summary")
    ck=torch.load(files[0],map_location="cpu",weights_only=True)
    actual=ORIGINAL_SOURCES[tier]
    if (ck.get("study"),ck.get("arm"),ck.get("seed"),ck.get("updates"))!=(
            actual[0],actual[1],seed,160):
        raise ValueError("original trained source checkpoint metadata mismatch")
    if ck.get("sample_order_sha256")!=receipt["source_sample_order_sha256"]:
        raise ValueError("original fitted source sample order corrupted")
    if ck.get("source_commit")!="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7":
        raise ValueError("publisher source SHA commit changed")
    if ck.get("parameters")!={"trainable":COUNTS[tier],"deployed":COUNTS[tier]}:
        raise ValueError("original trained active source geometry mismatch")
    if not all(n in ck for n in ("optimizer","rng_python","rng_numpy","rng_torch","model")):
        raise ValueError("trained source optimizer/RNG checkpoint missing")
    model_source_preflight()
    m=NativeMatrixTAC075(tier).eval()
    m.load_state_dict(ck["model"],strict=True)
    if m.parameter_count()!=COUNTS[tier]:
        raise ValueError("native model weights changed")
    return m,receipt


def score(logits,observed_byte):
    logits=logits.squeeze(0)
    lp=F.log_softmax(logits.double(),dim=-1)
    p=lp.exp()
    y=int(observed_byte)
    return {"nll":float(-lp[y]),"brier":float(1-2*p[y]+p.square().sum()),
            "top1":int(logits.argmax().item()==y)}


def evaluate(tier,seed,checkpoint_dir,summary_file):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fit=fit_sources()
    _,dev,_=original_source(ROOT)
    if len(fit)!=14 or len(dev)!=4 or set(fit)&set(dev):
        raise ValueError("source publisher owner partition compromised")
    model,ck=load_original(tier,seed,checkpoint_dir,summary_file)
    engine=NativeStreamingTAC076(model,"cap_all")
    owners=tuple(sorted(dev))
    all_rows=[]
    sig=hashlib.sha256()
    started=time.process_time()
    for owner in owners:
        raw=dev[owner]
        if len(raw)<=321:raise ValueError("original DEV owner too short")
        # The carried original source from byte 0 to 127 is complete.
        prefix=torch.tensor(list(raw[:128]),dtype=torch.long).view(1,128)
        with torch.inference_mode():
            original=model(prefix)[:,-1]
            pref_logits,state=engine.prefill(prefix,owners=(owner,))
            if (original-pref_logits).abs().max().item()>3e-5:
                raise ValueError("original parallel prefill differs from actual carried source")
            assert state.clock==128 and state.tensor_bytes()==EXPECTED_STATE_BYTES[tier]
            early_wiped=replace(state,tensors={**state.tensors,
                                             "native":torch.zeros_like(state.tensors["native"])})
            if early_wiped.clock!=state.clock:raise ValueError("source time altered")
            for t in range(128,320):
                token=torch.tensor([raw[t]],dtype=torch.long)
                if state.clock!=t or early_wiped.clock!=t:
                    raise ValueError("source owner state clock not continuous")
                # The model has not read raw[t+1] before predicting it.
                full,newstate=engine.step(token,state,owners=(owner,))
                wiped,nextwipe=engine.step(token,early_wiped,owners=(owner,))
                if t in TARGET_CLOCKS:
                    pre_lesion=replace(state,tensors={**state.tensors,
                              "native":torch.zeros_like(state.tensors["native"])})
                    one_step,_=engine.step(token,pre_lesion,owners=(owner,))
                    y=raw[t+1]
                    sig.update(owner.encode()+b":"+str(t).encode()+b":"+bytes([y])+b";")
                    all_rows.append({"owner":owner,"t":t,"source_target_byte":y,
                        "full":score(full,y),"early_native_clear_at_128":score(wiped,y),
                        "one_step_native_zero":score(one_step,y),
                        "early_logit_abs_difference":float((full-wiped).abs().max()),
                        "lesion_logit_abs_difference":float((full-one_step).abs().max())})
                state,early_wiped=newstate,nextwipe
            if state.clock!=320 or early_wiped.clock!=320:
                raise ValueError("original source carry ended at wrong clock")
            if state.tensor_bytes()!=EXPECTED_STATE_BYTES[tier]:
                raise ValueError("native state size not fixed across source history")
    if len(all_rows)!=4*len(TARGET_CLOCKS):
        raise ValueError("source outcome positions do not match original preregistration")
    summary={}
    for arm in ("full","early_native_clear_at_128","one_step_native_zero"):
        summary[arm]={key:sum(r[arm][key] for r in all_rows)/len(all_rows)
                      for key in ("nll","brier","top1")}
    return {"study":"LANGSTREAM076","tier":tier,"seed":seed,
      "status":"SOURCE_CARRIED_NATIVE_MATRIX_DIAGNOSTIC_ONLY",
      "trained_checkpoint_sha256":ck["checkpoint_sha256"],
      "trained_source_schedule_sha256":ck["source_sample_order_sha256"],
      "parameters":COUNTS[tier],"native_matrix_dim":WIDTHS[tier],
      "position_mode":"cap_all_AFTER127_UNTRAINED_INFERENCE_POLICY",
      "state_bytes_FP32_per_stream":EXPECTED_STATE_BYTES[tier],
      "carried_history_last_clock":320,"warm_prefix_original_bytes":128,
      "original_source_owners":owners,"source_target_count":len(all_rows),
      "source_target_fingerprint":sig.hexdigest(),
      "target_clock_positions":TARGET_CLOCKS,"source_rows":all_rows,
      "mean_metrics":summary,
      "mean_abs_logit_early_diff":sum(r["early_logit_abs_difference"] for r in all_rows)/len(all_rows),
      "mean_abs_logit_lesion_diff":sum(r["lesion_logit_abs_difference"] for r in all_rows)/len(all_rows),
      "cpu_process_seconds":time.process_time()-started,
      "no_new_training":True,"gpu_minutes":0,
      "protected_TEST":"NOT_ACCESSED","independently_qualified":False,
      "limits":["same previously exposed publisher DEV documents as earlier 070/075 studies",
                "true carried state but source cap_all inference policy selected after 073 exposed DEV results",
                "native-only early wipe does not erase history in aligned/upper/wave/concepts",
                "one-step native zero suppresses old native before current write, not all memory dependence",
                "no direct long-range fact recovery or independent generalization demonstrated",
                "different native k changes aligned address+readout and training FLOPs unmatched",
                "full source IEEE/AD/Lean mathematical equivalence and mobile runtime OPEN"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tier",choices=TIERS,required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--original-aggregate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("preserve original read-only source results")
    result=evaluate(a.tier,a.seed,a.checkpoint_dir,a.original_aggregate)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("STREAM076_SOURCE_RESULT "+json.dumps({
      "tier":a.tier,"seed":a.seed,"source_target_count":result["source_target_count"],
      "state_bytes":result["state_bytes_FP32_per_stream"],
      "metrics":result["mean_metrics"],
      "early_max_logit_diff":result["mean_abs_logit_early_diff"],
      "lesion_max_logit_diff":result["mean_abs_logit_lesion_diff"]
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
