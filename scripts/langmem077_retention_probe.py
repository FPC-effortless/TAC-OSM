#!/usr/bin/env python3
"""LANGMEM077: original trained native memory retention-vs-erasure no-fit diagnosis.

Unlike LANGSTREAM076's late NLL-only lesion, this traces exact SOURCE-native
state difference dynamics under identical real publisher bytes and original
frozen weights. The native transition is N' = N P + Q, where
P = .995 sigmoid(retain_logit) (I - sigmoid(erase_logit) ww^T).
Since the source input path of P/Q does not read native N, subtracting two
same-input runs with different initial N cancels Q:
    Delta' = Delta P.
For unit w and erase beta in [0,1], spectral norm(P) <= alpha; hence
||Delta_{t+d}||_F <= ||Delta_t||_F * product(alpha).
This is an analytical upper bound on one initial native-state perturbation,
NOT proof about total historical information in aligned/wave/concept/upper.
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
for name in ("langstream076","langscale075","langscale070","langbase024","langmemory042","context026"):
    p=str(ROOT/"research/mathematical-engineering"/name)
    if p not in sys.path: sys.path.insert(0,p)
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from byte_next_reference import original_source
from candidate import transform
from context_transition import phi_position
from native_streaming import NativeStreamingTAC076
from scripts.langstream076_carried_source_eval import (
    load_original, EXPECTED_STATE_BYTES, SEEDS, TIERS, score
)
from scripts.langlayer035_sampling import fit_sources

START=128
STOP=320
CLOCKS=(128,129,131,135,143,159,191,255,287) # age 1,2,4,8,16,32,64,128,160

@torch.inference_mode()
def native_transition_factors(model,state,token,position_mode):
    """Reconstruct exact trained-source P and the normalized read query.

    Input: only previously observed per-owner state, clock and current byte.
    Neither next byte nor any future information is used.
    """
    s=state.tensors;t=state.clock
    tt=torch.tensor([t],dtype=torch.long,device=token.device)
    embed=model.embedding(token)
    e=(embed+model.pos(transform(phi_position(tt,embed.dtype),position_mode))).tanh()
    lex=embed.tanh()
    local=lex
    for lag in range(3):
        local=local+model.context_lag_weights[lag].tanh()*s["lexical"][:,lag]
    query=F.normalize(model.native_key(local.tanh()),dim=-1,eps=1e-6)
    ab=model.native_gate(e).sigmoid()
    alpha=.995*ab[:,0]
    beta=ab[:,1]
    wk=s["previous_key"]
    k=model.native_dim
    ident=torch.eye(k,dtype=e.dtype,device=e.device).unsqueeze(0)
    p=alpha[:,None,None]*(ident-beta[:,None,None]*wk[:,:,None]*wk[:,None,:])
    return {"P":p,"alpha":float(alpha.item()),"erase":float(beta.item()),
            "query":query,
            "write_key_norm":float(torch.linalg.vector_norm(wk).item()),
            "query_write_cosine":float((query*wk).sum().item()),
            "old_query_write_key":wk.clone()}

@torch.inference_mode()
def evaluate(tier,seed,checkpoint_dir,verified_summary):
    torch.set_num_threads(2)
    fit=fit_sources()
    _,dev,_=original_source(ROOT)
    if len(fit)!=14 or len(dev)!=4 or set(fit)&set(dev):
        raise ValueError("original publisher FIT/DEV source identities differ")
    model,receipt=load_original(tier,seed,checkpoint_dir,verified_summary)
    engine=NativeStreamingTAC076(model,"cap_all")
    target_hasher=hashlib.sha256()
    original_rows=[]
    started=time.process_time()
    owners=tuple(sorted(dev))
    for owner in owners:
        raw=dev[owner]
        if len(raw)<STOP+1:raise ValueError("original owner too short for tested clock")
        src=torch.tensor(list(raw[:START]),dtype=torch.long).reshape(1,START)
        initial_logits,initial=engine.prefill(src,owners=(owner,))
        if initial.clock!=START or initial.tensor_bytes()!=EXPECTED_STATE_BYTES[tier]:
            raise ValueError("bad original source state/owner/time")
        # Source full-prefix check before any lesion.
        full_logits=model(src)[:,-1]
        if (initial_logits-full_logits).abs().max().item()>3e-5:
            raise ValueError("original full source prefix not faithfully reconstructed")
        erased=replace(initial,tensors={**initial.tensors,
                     "native":torch.zeros_like(initial.tensors["native"])})
        if erased.clock!=initial.clock:raise ValueError("native-only lesion changed time")
        first_norm=float(torch.linalg.matrix_norm(initial.tensors["native"].double()).item())
        if not math.isfinite(first_norm) or first_norm<=1e-10:
            raise ValueError("pre-128 native matrix has no viable original memory state")
        product_alpha=1.
        log_alpha=0.
        state=initial
        cosines=[]
        for t in range(START,STOP):
            if state.clock!=t or erased.clock!=t:
                raise ValueError("state carry reset, source label leakage or owner mismatch")
            observed=torch.tensor([raw[t]],dtype=torch.long)
            factor=native_transition_factors(model,state,observed,"cap_all")
            # Subtract only the two pre-existing native matrices, with all
            # source inputs, clock and write keys identical.
            delta_before=(state.tensors["native"]-erased.tensors["native"]).double()
            predicted=(delta_before@factor["P"].double()).squeeze(0)
            clean_logits,next_clean=engine.step(observed,state,owners=(owner,))
            erased_logits,next_erased=engine.step(observed,erased,owners=(owner,))
            actual=(next_clean.tensors["native"]-next_erased.tensors["native"]).double().squeeze(0)
            transition_err=float((actual-predicted).abs().max().item())
            if transition_err>1e-4:
                raise ValueError(f"native Delta' = Delta P source deviation at clock{t}: {transition_err}")
            if not (0<factor["alpha"]<1 and 0<factor["erase"]<1):
                raise ValueError("source-native gate impossible")
            product_alpha*=factor["alpha"]
            log_alpha+=math.log(factor["alpha"])
            ratio=float(torch.linalg.matrix_norm(actual).item())/first_norm
            # Scalar retention is a tight upper bound on the spectral norm of
            # P_t when write-key norm is <=1, the erase gate is in (0,1).
            # Float32 cancellation produces a measurable machine-precision floor.
            if ratio>product_alpha+1e-4:
                raise ValueError(f"native retention bound violated at clock{t}")
            read_key=factor["query"].double()
            native_read_delta=(model.native_read_proj(
                 (next_clean.tensors["native"]@factor["query"].unsqueeze(-1)).squeeze(-1))-
                 model.native_read_proj(
                 (next_erased.tensors["native"]@factor["query"].unsqueeze(-1)).squeeze(-1)))
            read_delta=float(torch.linalg.vector_norm(native_read_delta.double()).item())
            if not (math.isfinite(read_delta) and math.isfinite(ratio)):
                raise ValueError("nonfinite original native source diagnosis")
            if factor["write_key_norm"]>1.00001:
                raise ValueError("non-normalized trained native write address")
            if abs(factor["query_write_cosine"])>1.00001:
                raise ValueError("query/write key dot outside normalized range")
            cosines.append(factor["query_write_cosine"])
            if t not in range(START,STOP):raise ValueError("source clock mismatched")
            y=raw[t+1]  # accessed ONLY after the model predicted.
            target_hasher.update(owner.encode()+b":"+str(t).encode()+b":"+
                                 bytes([y])+b";")
            clean_score=score(clean_logits,y)
            erased_score=score(erased_logits,y)
            row={"owner":owner,"clock":t,"age_since_native_erase":t-START+1,
                 "source_target_byte":int(y),
                 "native_alpha":factor["alpha"],"native_erase_gate":factor["erase"],
                 "write_key_norm":factor["write_key_norm"],
                 "query_write_cosine":factor["query_write_cosine"],
                 "initial_native_norm":first_norm,
                 "observed_native_difference_norm_ratio":ratio,
                 "retention_only_upper_bound_product_alpha":product_alpha,
                 "source_native_delta_update_abs_error":transition_err,
                 "native_read_vector_l2_difference":read_delta,
                 "native_erase_logit_max_abs":float((clean_logits-erased_logits).abs().max()),
                 "full":clean_score,"early_erase":erased_score}
            original_rows.append(row)
            state,erased=next_clean,next_erased
        if state.clock!=STOP or erased.clock!=STOP:
            raise ValueError("source did not complete full stream")
    if len(original_rows)!=(STOP-START)*4:
        raise ValueError("original source owner/clock target count changed")
    summary={}
    for age in [t-START+1 for t in CLOCKS]:
        rows=[r for r in original_rows if r["age_since_native_erase"]==age]
        if len(rows)!=4:raise ValueError("lost original source clock")
        summary[str(age)]={
           "mean_alpha":sum(z["native_alpha"] for z in rows)/4,
           "mean_erase_gate":sum(z["native_erase_gate"] for z in rows)/4,
           "mean_retention_upper_bound":sum(z["retention_only_upper_bound_product_alpha"] for z in rows)/4,
           "mean_actual_native_difference_ratio":sum(z["observed_native_difference_norm_ratio"] for z in rows)/4,
           "mean_native_read_delta_l2":sum(z["native_read_vector_l2_difference"] for z in rows)/4,
           "mean_full_nll":sum(z["full"]["nll"] for z in rows)/4,
           "mean_early_erase_minus_full_nll":sum(z["early_erase"]["nll"]-z["full"]["nll"] for z in rows)/4,
           "max_source_native_delta_update_abs_error":max(z["source_native_delta_update_abs_error"] for z in rows)}
    mean_log_alpha=sum(math.log(x["native_alpha"]) for x in original_rows)/len(original_rows)
    if mean_log_alpha>=0:raise ValueError("source retention gate not a contraction")
    metrics={
        "native_retention_alpha_mean":sum(x["native_alpha"] for x in original_rows)/len(original_rows),
        "native_forget_half_life_source_steps_from_mean_log_alpha":math.log(.5)/mean_log_alpha,
        "native_erase_beta_mean":sum(x["native_erase_gate"] for x in original_rows)/len(original_rows),
        "mean_abs_query_write_cosine":sum(abs(x["query_write_cosine"]) for x in original_rows)/len(original_rows),
        "fraction_abs_query_write_cosine_ge_0_9":sum(abs(x["query_write_cosine"])>=.9 for x in original_rows)/len(original_rows),
        "mean_full_nll":sum(x["full"]["nll"] for x in original_rows)/len(original_rows),
        "mean_early_erase_minus_full_nll":sum(x["early_erase"]["nll"]-x["full"]["nll"] for x in original_rows)/len(original_rows),
        "max_source_native_delta_update_abs_error":max(x["source_native_delta_update_abs_error"] for x in original_rows)
    }
    return {"study":"LANGMEM077","tier":tier,"seed":seed,
        "status":"ORIGINAL_SOURCE_NATIVE_GATE_RETENTION_DIAGNOSTIC_EXPOSED_DEV",
        "original_model_checkpoint_sha256":receipt["checkpoint_sha256"],
        "original_fit_sample_order_sha256":receipt["source_sample_order_sha256"],
        "registered_native_dim":model.native_dim,"model_parameters":model.parameter_count(),
        "fp32_state_bytes_per_owner":EXPECTED_STATE_BYTES[tier],
        "original_source_owners":owners,"original_source_label_sha256":target_hasher.hexdigest(),
        "original_target_count":len(original_rows),"clock_first":START,"clock_end_exclusive":STOP,
        "native_erased_at_clock":START,"sample_ages":tuple(t-START+1 for t in CLOCKS),
        "source_rows":original_rows,"selected_age_summary":summary,"metrics":metrics,
        "cpu_process_seconds":time.process_time()-started,
        "no_fits":True,"no_optimizer_steps":True,"gpu_minutes":0,
        "protected_TEST":"NOT_ACCESSED","independently_qualified":False,
        "limits":[
          "original PSF genuine publisher source but DEV cohort previously exposed and all labels reused",
          "cap_all positional policy itself selected post 073 exposed DEV",
          "difference dynamics measure perturbation of native state at clock128 only; other memories remain",
          "query/write-key cosine is geometric diagnostic, NOT content hit or factual retrieval accuracy",
          "float32 cancellation floor can make relative native delta measurement uninformative for late ages",
          "training FLOPs differed in original k8/12/16 source trained model fits",
          "no new trained model, independent TEST, mobile speed or externally qualified claim"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tier",choices=TIERS,required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--original-aggregate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("source evidence overwrite denied")
    r=evaluate(args.tier,args.seed,args.checkpoint_dir,args.original_aggregate)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,allow_nan=False,sort_keys=True,indent=2)+"\n")
    print("NATIVE_RETENTION077 "+json.dumps({"tier":args.tier,"seed":args.seed,
        "metrics":r["metrics"],
        "ages":{k:{"bound":v["mean_retention_upper_bound"],
                   "actual":v["mean_actual_native_difference_ratio"]}
                for k,v in r["selected_age_summary"].items()}},sort_keys=True))


if __name__=="__main__":
    main()
