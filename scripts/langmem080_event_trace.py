#!/usr/bin/env python3
"""LANGMEM080: exact original-source trained native write/event attribution.

No neural training. Real publisher DEV (already exposed); original verified
dual 078 trained checkpoint; original 079 carry. An *event* here is one
historical, genuinely observed next-byte INPUT write Q_a into a latent 8x8
matrix, NOT a readable word/fact. The diagnostic measures tagged write
survival, current query projection and one-query-step causal logits only.

Read raw[t+1] only *after* original model sees raw[t] and both counterfactual
logits have been calculated. No future query/answer text is used to select
anchors, keys, labels or model params.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
for n in ("langmem079","langmem078","langscale075","langscale070",
          "context026","langmemory042","langbase024","langstream076"):
    path=str(ROOT/"research/mathematical-engineering"/n)
    if path not in sys.path:sys.path.insert(0,path)
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from dual_streaming import DualNativeStreamingTAC079
from scripts.langmem079_carried_eval import restore_original
from scripts.langstream076_carried_source_eval import score
from scripts.langlayer035_sampling import fit_sources
from byte_next_reference import original_source
from context_transition import phi_position
from candidate import transform

ANCHORS=(32,64,96,128,192)
DELAYS=(1,8,32,64,128,192)
LAST_CLOCK=320
PAIRS=tuple((anchor,lag) for anchor in ANCHORS for lag in DELAYS
            if anchor+lag<LAST_CLOCK)
SEEDS=(1701,1702,1703)
LANES=("fast","slow")
SOURCE_ERROR=1e-5
SCORE_POSITIONS={anchor+lag for anchor,lag in PAIRS}


@torch.no_grad()
def source_factors(model,state,token,*,mode="cap_all"):
    """Frozen actual trained original 079 source P/Q on a single observed byte.

    This implements the *same* source transforms as DualNativeStreamingTAC079
    without reading from prior native matrices, the future true byte or a
    generated model output. Original P and Q are input-only.
    """
    if type(state.clock) is not int or token.shape!=(1,) or token.dtype!=torch.long:
        raise ValueError("one original observed source byte, scalar clock required")
    t=state.clock
    m=model;s=state.tensors
    tt=torch.tensor([t],dtype=torch.long,device=token.device)
    embed=m.embedding(token)
    lex=embed.tanh()
    e=(embed+m.pos(transform(phi_position(tt,embed.dtype),mode))).tanh()
    local=lex
    for lag in range(3):
        local=local+m.context_lag_weights[lag].tanh()*s["lexical"][:,lag]
    query=F.normalize(m.native_key(local.tanh()),dim=-1,eps=1e-6)
    wk=s["previous_key"]
    value=m.native_value(e).tanh()
    ab=m.native_gate(e).sigmoid()
    eye=torch.eye(8,dtype=e.dtype,device=e.device).unsqueeze(0)
    alpha=.995*ab[:,0]
    erase=ab[:,1]
    pf=alpha[:,None,None]*(eye-erase[:,None,None]*wk[:,:,None]*wk[:,None,:])
    qf=erase[:,None,None]*value[:,:,None]*wk[:,None,:]
    rho=torch.exp(-F.softplus(m.slow_log_decay))
    sbeta=.05*torch.sigmoid(m.slow_erase_logit)
    swrite=.01*torch.sigmoid(m.slow_write_logit)
    ps=rho*(eye-sbeta*wk[:,:,None]*wk[:,None,:])
    qs=swrite*value[:,:,None]*wk[:,None,:]
    if t==0:
        pf=eye.clone();qf=torch.zeros_like(qf)
        ps=eye.clone();qs=torch.zeros_like(qs)
    return {"fast":{"p":pf,"q":qf,"retention_bound":float(alpha.item())},
            "slow":{"p":ps,"q":qs,"retention_bound":float(rho.item())},
            "query":query,"write_key":wk,"value":value,
            "alpha_fast":float(alpha.item()),"rho_slow":float(rho.item()),
            "beta_fast":float(erase.item()),"beta_slow":float(sbeta.item()),
            "write_slow":float(swrite.item()),
            "read_mix_slow":float(torch.sigmoid(m.slow_read_logit).item())}


def norm(x):
    return float(torch.linalg.matrix_norm(x.double()).item())


@torch.no_grad()
def evaluate(seed,ckpt_dir,verified):
    torch.set_num_threads(2)
    fit=fit_sources()
    _,dev,_=original_source(ROOT)
    if len(fit)!=14 or len(dev)!=4 or set(fit)&set(dev):
        raise ValueError("original genuine publisher FIT/DEV owner manifest differs")
    model,receipt=restore_original(seed,ckpt_dir,verified)
    engine=DualNativeStreamingTAC079(model,"cap_all")
    if model.parameter_count()!=499627:
        raise ValueError("original source-trained model geometry differs")
    started=time.process_time()
    source_target_sha=hashlib.sha256()
    source_anchor_sha=hashlib.sha256()
    rows=[]
    source_steps_verified=0
    sham_verified=0
    max_source_error=0.
    max_event_error=0.
    max_read_error=0.
    for owner in sorted(dev):
        raw=dev[owner]
        if len(raw)<LAST_CLOCK+1:
            raise ValueError("registered genuine source owner too short")
        state=engine.start((owner,))
        # key=(anchor,lane): the exact propagated contribution of *one*
        # original admitted historical write, and a bound from that write.
        active={}
        for t in range(LAST_CLOCK):
            if state.clock!=t or state.tensor_bytes()!=4160:
                raise ValueError("original stream clock/state invalid")
            token=torch.tensor([raw[t]],dtype=torch.long)
            factors=source_factors(model,state,token)
            prior=state
            full_logits,next_state=engine.step(token,prior,owners=(owner,))
            for lane,name in (("fast","native"),("slow","slow_native")):
                expected=prior.tensors[name]@factors[lane]["p"]+factors[lane]["q"]
                mismatch=float((expected-next_state.tensors[name]).abs().max().item())
                max_source_error=max(max_source_error,mismatch)
                if mismatch>SOURCE_ERROR:
                    raise ValueError(f"source trained {lane} affine P/Q mismatch at original clock {t}: {mismatch}")
            source_steps_verified+=1
            if t==127:
                # Not a retrained/extrapolated result. Source original 128-byte
                # parallel graph must agree with genuine carried original.
                prefix=torch.tensor(list(raw[:128]),dtype=torch.long).reshape(1,128)
                ref=model(prefix)[:,-1]
                err=float((full_logits-ref).abs().max())
                if err>3e-5:
                    raise ValueError("original source sequential prefill prefix mismatch")
            # Transport all previously observed event *writes* with current
            # exact original P_t. No current Q_t can enter an older anchor.
            next_contrib={}
            for (anchor,lane),(c,origin,retention_bound) in active.items():
                now=c@factors[lane]["p"]
                next_contrib[(anchor,lane)]=(now,origin,
                     retention_bound*factors[lane]["retention_bound"])
            if t in ANCHORS:
                for lane in LANES:
                    q=factors[lane]["q"].clone()
                    initial=norm(q)
                    if initial<=1e-12:raise ValueError("original event did not make a source-native write")
                    next_contrib[(t,lane)]=(q,initial,1.)
                source_anchor_sha.update(owner.encode()+b":"+str(t).encode()+b":"+
                    bytes([raw[t-1],raw[t]])+b";")
            # For all registered a<t, the event was present in prestate.
            # The current-model answer raw[t+1] is NOT read until after both
            # model predictions and no label-dependent selection is possible.
            for a,lag in PAIRS:
                if a+lag!=t:continue
                for lane,name in (("fast","native"),("slow","slow_native")):
                    if (a,lane) not in active or (a,lane) not in next_contrib:
                        raise ValueError("missing causal historical original write")
                    before,origin,_=active[a,lane]
                    after,_,bound=next_contrib[a,lane]
                    survival=norm(after)/origin
                    if survival>bound+1e-4:
                        raise ValueError("old single-write contribution norm exceeds original affine contraction bound")
                    modified=replace(prior,tensors={**prior.tensors,
                                          name:prior.tensors[name]-before})
                    candidate,next_alt=engine.step(token,modified,owners=(owner,))
                    difference=(next_state.tensors[name]-next_alt.tensors[name])
                    error=float((difference-after).abs().max().item())
                    max_event_error=max(max_event_error,error)
                    if error>SOURCE_ERROR:
                        raise ValueError(f"source one historical write transport wrong lane={lane} clock={t}: {error}")
                    # The native/slow write is read post-update at *this same*
                    # clock and original observed current-byte query.
                    read_direct=(after@factors["query"].unsqueeze(-1)).squeeze(-1)
                    if lane=="slow":read_direct=read_direct*factors["read_mix_slow"]
                    read_actual=(difference@factors["query"].unsqueeze(-1)).squeeze(-1)
                    if lane=="slow":read_actual=read_actual*factors["read_mix_slow"]
                    read_error=float((read_direct-read_actual).abs().max())
                    max_read_error=max(max_read_error,read_error)
                    if read_error>SOURCE_ERROR:
                        raise ValueError("source one-event native read projection not exact")
                    # Zero negative control at registered source query clocks:
                    # source state unchanged and result must be bitwise equal.
                    if (a,lag)==PAIRS[0]:
                        sham=replace(prior,tensors={**prior.tensors,
                                     name:prior.tensors[name]-torch.zeros_like(before)})
                        sham_logits,_=engine.step(token,sham,owners=(owner,))
                        if not torch.equal(sham_logits,full_logits):
                            raise ValueError("zero-event negative control not bitwise original")
                        sham_verified+=1
                    # Both causal source model predictions are complete now.
                    y=int(raw[t+1])
                    event_byte=int(raw[a]);event_prev_byte=int(raw[a-1])
                    source_target_sha.update(owner.encode()+b":"+str(a).encode()+b":"+
                         str(t).encode()+b":"+bytes([y])+b":"+lane.encode()+b";")
                    full=score(full_logits,y)
                    alt=score(candidate,y)
                    rows.append({"owner":owner,"anchor_clock":a,"query_clock":t,"lag":lag,
                        "lane":lane,"historical_observed_source_byte":event_byte,
                        "historical_predecessor_source_byte":event_prev_byte,
                        "source_next_byte_target":y,
                        "origin_write_frobenius":origin,
                        "surviving_write_frobenius_ratio":survival,
                        "survival_source_alpha_bound":bound,
                        "direct_old_event_read_l2":float(torch.linalg.vector_norm(read_direct.double()).item()),
                        "event_matrix_source_diff_max_abs_error":error,
                        "event_read_source_diff_max_abs_error":read_error,
                        "full_nll":full["nll"],"without_event_nll":alt["nll"],
                        "without_event_minus_full_nll":alt["nll"]-full["nll"],
                        "full_top1":full["top1"],"without_event_top1":alt["top1"],
                        "original_current_token_observed_before_query":int(raw[t]),
                        "old_event_matrix_numerically_resolved":bool(
                            (prior.tensors[name]-before!=prior.tensors[name]).any().item())})
            active=next_contrib
            state=next_state
        if state.clock!=LAST_CLOCK:
            raise ValueError("original genuine source carry stopped early")
    if len(rows)!=len(dev)*len(PAIRS)*2:
        raise ValueError("source registered original anchor/delay matrix incomplete")
    if sham_verified!=len(dev)*2:
        raise ValueError("original source sham counterfactual controls missing")
    if source_steps_verified!=4*LAST_CLOCK or max_source_error>SOURCE_ERROR or max_event_error>SOURCE_ERROR:
        raise ValueError("some original source affine/event identity was not verified")
    return {"study":"LANGMEM080","seed":seed,
      "status":"REAL_LICENSED_ORIGINAL_SOURCE_EVENT_WRITE_TRANSPORT_AND_READ_ATTRIBUTION",
      "original_checkpoint_sha256":receipt["checkpoint_sha256"],
      "original_fit_sample_order_sha256":receipt["source_sample_order_sha256"],
      "parameters":499627,"state_fp32_bytes_per_owner":4160,
      "position_mode":"cap_all_POST_EXPOSED_DEV","original_owners":tuple(sorted(dev)),
      "source_anchors":ANCHORS,"source_delays":DELAYS,
      "registered_source_pairs_per_owner":len(PAIRS),
      "source_target_pairs_with_both_lanes":len(rows),
      "source_target_fingerprint_sha256":source_target_sha.hexdigest(),
      "source_observed_anchor_fingerprint_sha256":source_anchor_sha.hexdigest(),
      "original_stream_steps_verified":source_steps_verified,
      "zero_negative_control_bitwise_count":sham_verified,
      "max_original_source_affine_update_abs_error":max_source_error,
      "max_original_source_event_transport_abs_error":max_event_error,
      "max_original_source_read_projection_abs_error":max_read_error,
      "rows":rows,
      "cpu_process_seconds":time.process_time()-started,
      "no_new_fits":True,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
      "independently_qualified":False,
      "limitations":["original genuine PSF publisher DEV owner data PREVIOUSLY EXPOSED",
        "source query keys derived from current observed real byte and past; no source future byte access",
        "event is a source-written latent matrix Q at one time, not ground-truth source byte decoded or fact correctness",
        "one-query-step event contribution removal holds other downstream memory states fixed, not full historical replay",
        "old event disappearance below float32 arithmetic floor is not exact real zero",
        "no changed learned weights or retraining, no newly unseen corpus, no protected TEST",
        "original 078 extra slow scan training FLOPs not matched to original fast-only",
        "source cap_all positional inference postselected on earlier exposed DEV; full source IEEE AD PNDS and old factual content recovery OPEN"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--original-aggregate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("original genuine source evidence overwrite prohibited")
    r=evaluate(a.seed,a.checkpoint_dir,a.original_aggregate)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LANGMEM080_EVENT_TRACE "+json.dumps({"seed":a.seed,
      "source_paired_events":r["source_target_pairs_with_both_lanes"],
      "source_steps":r["original_stream_steps_verified"],
      "source_sham":r["zero_negative_control_bitwise_count"],
      "max_write_source_err":r["max_original_source_event_transport_abs_error"],
      "fast_mean_survival":sum(x["surviving_write_frobenius_ratio"] for x in r["rows"] if x["lane"]=="fast")/(len(r["rows"])//2),
      "slow_mean_survival":sum(x["surviving_write_frobenius_ratio"] for x in r["rows"] if x["lane"]=="slow")/(len(r["rows"])//2)
    },sort_keys=True))
if __name__=="__main__":main()
