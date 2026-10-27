"""LATFUSE011 original real FIT source pre-architecture falsifiers.

1. Source-paired below-chance sign tests and nuisance polarity controls.
2. Exact d=8 isotropic cosine null and wrong-within-physical-pair keys.
4. Full event-read lesion, event-specific source memory signal/interference,
   exact learned alpha(I-beta kkT) event-step matrix difference transport.

No optimizer, retraining, selected checkpoint, source target in model input,
future numeric source, or retroactive cohort tuning. Diagnostic only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import torch
import torch.nn.functional as F

from scripts import latfuse002_real_fit as source
from scripts import latfuse008_paired_real_training as fit
from scripts import latfuse010_real_event_writer_attribution as prior
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_004_factorial import head_weights

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse011/REGISTERED_FALSIFIERS.json"
DELAYS=(16,64,128,512)
Z95=1.959963984540054

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def wilson(wins,n):
    if n==0:return [None,None]
    p=wins/n;z=Z95
    root=math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    den=1+z*z/n
    return [max(0,(p+z*z/(2*n)-z*root)/den),
            min(1,(p+z*z/(2*n)+z*root)/den)]

def exact_two_sided_sign_p(k,n):
    # Two-sided exact binomial sign test for strict non-tied pairs;
    # cannot treat multiple seeds on the same physical pairs as independent.
    if n==0:return None
    t=min(k,n-k)
    return min(1.0,2*sum(math.comb(n,i) for i in range(t+1))/2**n)

def paired_summary(values):
    if values.ndim!=1 or values.numel()%2 or not torch.isfinite(values).all():
        raise RuntimeError("missing/nonfinite authentic paired forecast values")
    diff=values[::2]-values[1::2]
    wins=int((diff>0).sum())
    loses=int((diff<0).sum())
    ties=int((diff==0).sum())
    non_tied=wins+loses
    p=(wins+.5*ties)/len(diff)
    return dict(pairs=len(diff),rank=p,winning_pairs=wins,
        losing_pairs=loses,tied_pairs=ties,
        rank_Wilson_95_interval=wilson(wins,non_tied) if not ties else None,
        exact_two_sided_sign_p=exact_two_sided_sign_p(wins,non_tied),
        mean_positive_minus_ordinary=float(diff.mean()),
        mean_high=float(values[::2].mean()),mean_ordinary=float(values[1::2].mean()))

def read_query(m,state,anchor):
    clock=source.calendar([int(t) for t in anchor])[:,None,:]
    joint=torch.cat((state.fast,state.slow),-1)
    keys=F.normalize((torch.tanh(m.historic_clock_query(clock))+
         m.belief_query(joint)[:,None,:]).reshape(len(anchor),1,2,8),
         dim=-1,eps=1e-8)[:,0]
    route,_=head_weights(m.read_router(joint),m.log_head_temperature,
                         head_count=2,soft=True)
    return keys,route

def projected_read(mat,keys,route):
    b=len(keys)
    heads=mat.reshape(b,16,2,8).permute(0,2,1,3)
    r=torch.einsum("bhvk,bhk->bhv",heads,keys)
    return (r*route[:,:,None]).sum(1)

def prior_past_pm_raw(raw,site,t):
    vals=[]
    for s,now in zip(site.tolist(),t.tolist()):
        v=raw[s,:now+1,0]
        valid=v[torch.isfinite(v)]
        if len(valid)==0:raise RuntimeError("no previous original physically observed PM2.5")
        vals.append(valid[-1])
    return torch.stack(vals)

def anchors(m,d,cases):
    station=torch.tensor([r[0] for r in cases],dtype=torch.long)
    anchor=torch.tensor([r[1] for r in cases],dtype=torch.long)
    st=m.initial(len(cases))
    for offset in range(-23,0):
        t=anchor+offset
        st=m.step(st,d["values"][station,t],d["valid"][station,t],
                  d["age"][station,t],d["last"][station,t],station)
    initial=st.associative.clone()
    t=anchor
    u=m.encode(d["values"][station,t],d["valid"][station,t],
               d["age"][station,t],d["last"][station,t])
    write=F.normalize(m.key_map(u).reshape(len(cases),2,8),dim=-1,eps=1e-8)
    after=m.step(st,d["values"][station,t],d["valid"][station,t],
                 d["age"][station,t],d["last"][station,t],station)
    difference=after.associative-initial
    no_event=after._replace(associative=initial)
    return station,anchor,after,no_event,difference,write

def study(m,d,cases):
    b=len(cases)
    site,anchor,normal,lesioned,event_difference,write_key=anchors(m,d,cases)
    if b!=48:raise RuntimeError("unexpected 24 physical event pairs")
    labels=torch.tensor([z[2] for z in cases])
    if not torch.equal(labels[::2],torch.ones_like(labels[::2])) or not torch.equal(
           labels[1::2],torch.zeros_like(labels[1::2])):
        raise RuntimeError("authentic source labels not ordered as preregistered")
    # All source labels are used for metrics only AFTER the frozen model predicts.
    eye=torch.eye(8).reshape(1,1,8,8)
    transfer=eye.expand(b,2,-1,-1).clone()
    start=event_difference.reshape(b,16,2,8).permute(0,2,1,3).contiguous()
    actual={} 
    for age in range(1,max(DELAYS)+1):
        now=anchor+age
        x=d["values"][site,now]
        mask=d["valid"][site,now]
        u=m.encode(x,mask,d["age"][site,now],d["last"][site,now])
        key=F.normalize(m.key_map(u).reshape(b,2,8),dim=-1,eps=1e-8)
        w,selected=head_weights(m.write_router(u),m.log_head_temperature,
                                 head_count=2,soft=True)
        good=mask.any(-1)[:,None]&selected
        alpha=torch.where(good,torch.sigmoid(m.retention_map(u)).expand(-1,2),
                          torch.ones_like(w))
        beta=torch.sigmoid(m.write_map(u))*w*good.to(w.dtype)
        P=alpha[:,:,None,None]*(eye-
            beta[:,:,None,None]*key[:,:,:,None]*key[:,:,None,:])
        transfer=transfer@P
        normal=m.step(normal,x,mask,d["age"][site,now],d["last"][site,now],site)
        lesioned=m.step(lesioned,x,mask,d["age"][site,now],d["last"][site,now],site)
        if age not in DELAYS:continue
        if any(not torch.equal(a,v) for a,v in zip(
          (normal.fast,normal.slow,normal.entities,normal.active),
          (lesioned.fast,lesioned.slow,lesioned.entities,lesioned.active))):
            raise RuntimeError("real source non-associative states changed during event-only lesion")
        original_delta=normal.associative-lesioned.associative
        predicted=(start@transfer).permute(0,2,1,3).contiguous().reshape(b,16,16)
        torch.testing.assert_close(original_delta,predicted,atol=1e-5,rtol=2e-4)
        known=source.calendar([int(t) for t in anchor])[:,None,:]
        logits=m.historic_event_logits(normal,site,known)[:,0]
        zeros=normal._replace(associative=torch.zeros_like(normal.associative))
        logits_zero=m.historic_event_logits(zeros,site,known)[:,0]
        logits_eventless=m.historic_event_logits(lesioned,site,known)[:,0]
        # No source label is ever supplied to these three model outputs.
        q,route=read_query(m,normal,anchor)
        core=projected_read(normal.associative,q,route)
        signal=projected_read(original_delta,q,route)
        nuisance=projected_read(lesioned.associative,q,route)
        signal_norm=signal.norm(dim=-1)
        noise_norm=nuisance.norm(dim=-1)
        snr=signal_norm/(noise_norm+1e-8)
        allfraction=signal_norm/(core.norm(dim=-1)+1e-8)
        actual_logits_difference=logits-logits_eventless
        # Diagnostic independent exact linear native head check.
        direct=m.historic_event_head(signal).squeeze(-1)-m.historic_event_head(
            torch.zeros_like(signal)).squeeze(-1)
        if m.__class__.__name__=="SourceTimedEventQuery":
            torch.testing.assert_close(actual_logits_difference,direct,atol=3e-6,rtol=2e-4)
        else:
            if not torch.equal(logits,logits_zero) or not torch.equal(logits,logits_eventless):
                raise RuntimeError("noassoc learned historical decoder illegally reads original matrix")
        # Matched wrong-event key within same station/hour pair as empirical
        # physically grounded mismatch, not a theoretical random vector.
        wrong=write_key.reshape(b//2,2,2,8).flip(1).reshape(b,2,8)
        angular=(q*write_key).sum(-1)
        angular_wrong=(q*wrong).sum(-1)
        prob=torch.sigmoid(logits)
        prob_zero=torch.sigmoid(logits_zero)
        prob_eventless=torch.sigmoid(logits_eventless)
        current=d["raw"][site,now,0]
        current_available=torch.isfinite(current)
        last=prior_past_pm_raw(d["raw"],site,now)
        prior_true=d["raw"][site,anchor,0]
        if not bool(torch.isfinite(prior_true).all()):
            raise RuntimeError("source event truth not recorded")
        if any(not (float(v)>=150 if r[2] else float(v)<=35)
                for v,r in zip(prior_true,cases)):
            raise RuntimeError("same source paired labels no longer valid")
        # Avoid treating unavailable current sensors as zero source values.
        recent_filled=torch.where(current_available,current,last)
        month_offset=(anchor[::2]-anchor[1::2]).abs().float()/24
        stat=dict(delay_hours=age,
          event_pair_rank=paired_summary(prob),
          full_read_zero_rank=paired_summary(prob_zero),
          original_event_write_bypass_rank=paired_summary(prob_eventless),
          source_current_pm25_rank=paired_summary(recent_filled),
          source_last_observed_pm25_rank=paired_summary(last),
          source_current_pm25_observed_fraction=float(current_available.float().mean()),
          original_matched_source_date_gap_days={
             "mean":float(month_offset.mean()),"median":float(month_offset.median()),
             "max":float(month_offset.max())},
          original_matched_site_same_clock=True,
          query_write_signed_cos_mean=float((angular*route).sum(-1).mean()),
          query_write_abs_cos_mean=float((angular.abs()*route).sum(-1).mean()),
          query_wrong_anchor_abs_cos_mean=float((angular_wrong.abs()*route).sum(-1).mean()),
          per_head_cos_abs_mean=[float(angular[:,i].abs().mean()) for i in range(2)],
          entire_associative_read_lesion_prob_abs_change=float(
              (prob-prob_zero).abs().mean()),
          entire_associative_read_lesion_logit_abs_change=float(
              (logits-logits_zero).abs().mean()),
          event_write_step_lesion_prob_abs_change=float(
              (prob-prob_eventless).abs().mean()),
          event_signal_to_interference_norm_ratio_mean=float(snr.mean()),
          event_signal_norm_mean=float(signal_norm.mean()),
          other_writes_read_norm_mean=float(noise_norm.mean()),
          full_read_norm_mean=float(core.norm(dim=-1).mean()),
          relative_event_signal_to_full_read_norm_mean=float(allfraction.mean()),
          exact_event_delta_P_product_abs_max=float((predicted-original_delta).abs().max()),
          event_read_signal_logit_effect_mean=float(actual_logits_difference.abs().mean()),
          source_channel_labels_used_only_after_frozen_model_prediction=True)
        actual[str(age)]=stat
    return actual

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise RuntimeError("original real-source pre-architecture evidence exists; never overwrite")
    torch.set_num_threads(1)
    j=json.loads(REG.read_text())
    check_manifest(j,"preflight")
    if (j["study_id"]!="LATFUSE011-PRE-ARCH-FALSIFIERS" or
        j["lags"]!=list(DELAYS) or j["original_source_sha256"]!=source.SOURCE_SHA):
        raise RuntimeError("prospective real source diagnostic registration changed")
    if sha(a.source)!=source.SOURCE_SHA:raise RuntimeError("unauthorized original real source archive")
    d=fit.source_observations(a.source)
    cases=fit.select_pairs(d,fit.EVAL_START,fit.EVAL_STOP,2)
    if len(cases)!=48:raise RuntimeError("original registered source cohort changed")
    prior_report=json.loads((fit.OUT/"RESULTS.json").read_text())
    cohort_sha=hashlib.sha256(json.dumps(cases,separators=(",",":")).encode()).hexdigest()
    if prior_report["source_eval_cohort_sha256"]!=cohort_sha:
        raise RuntimeError("past genuine source labels/normal matching altered")
    analytic=math.gamma(8/2)/(math.sqrt(math.pi)*math.gamma((8+1)/2))
    approximation=math.sqrt(2/(math.pi*8))
    if not abs(analytic-.2910)<.002:raise RuntimeError("8D exact cosine null sanity")
    records=[]
    with torch.no_grad():
        for seed in fit.SEEDS:
            for arm in fit.ARMS:
                rec=next((z for z in prior_report["model_results"] if
                          z["seed"]==seed and z["arm"]==arm),None)
                if rec is None:raise RuntimeError("missing genuine original pretrained checkpoint")
                path=fit.OUT/f"seed-{seed}"/arm/"final.pt"
                dg=sha(path)
                if dg!=rec["original_checkpoint_sha256"]:
                    raise RuntimeError("original source trained checkpoint changed")
                cp=torch.load(path,map_location="cpu",weights_only=False)
                m=fit.create_model(seed,arm)
                m.load_state_dict(cp["original_model"],strict=True)
                m.eval()
                before=[x.detach().clone() for x in m.parameters()]
                obs=study(m,d,cases)
                if dg!=sha(path) or any(not torch.equal(x,y)
                     for x,y in zip(m.parameters(),before)):
                    raise RuntimeError("original source model mutated")
                item=dict(seed=seed,arm=arm,original_checkpoint_sha256=dg,delays=obs)
                records.append(item)
                print("LATFUSE011_SOURCE_FALSIFIERS="+json.dumps({
                    "seed":seed,"arm":arm,
                    "rank":{h:v["event_pair_rank"]["rank"] for h,v in obs.items()},
                    "full_read_prob_effect":{h:v["entire_associative_read_lesion_prob_abs_change"]
                                for h,v in obs.items()}},sort_keys=True),flush=True)
    out=dict(study_id=j["study_id"],source_sha256=source.SOURCE_SHA,
      original_matched_physical_source_pairs=len(cases)//2,
      original_event_cohort_sha256=cohort_sha,
      d8_exact_random_isotropic_abs_cosine=analytic,
      d8_asymptotic_approx_abs_cosine=approximation,
      baseline_warning="random isotropic cosine expectation is NOT conditional/source learned key null",
      stored_source_evaluation_is_not_virgin_heldout=True,
      repeated_pretrained_seeds_are_not_independent_source_pair_replications=True,
      no_tac_optimizer_or_model_changes=True,
      no_synthetic_task_scientific_evidence=True,
      no_2015_DEV_or_protected_2016_17_numeric_source=True,
      physical_model_conditions="All observed true 24 paired May-Jun2014 cases across original station/hour matching",
      original_trained_models=records)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE011_GENUINE_SOURCE_READONLY_PASS="+json.dumps({
      "models":len(records),"physical_pairs":len(cases)//2,"no_fit":True},sort_keys=True))
if __name__=="__main__":main()
