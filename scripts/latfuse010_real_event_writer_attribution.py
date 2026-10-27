"""LATFUSE010: Original-source event WRITE key/value gradient and READ alignment.

Read-only after true original LATFUSE008 historical event-trained checkpoints.
Actual prior source observation enters state normally, and event source x is
made differentiable only to capture genuine source writer activation gradients
under frozen original model parameters. Uses exact four precommitted high vs
ordinary source-station/hour matched May2014 physical event pairs.
Model source historic query NEVER receives the observed historic event label.

Exact event-time matrix-write lesion cancels only this one state transition's
associative matrix CHANGE (which includes decay and write); that is a local
mechanistic counterfactual, not a real alternative physical sensor world.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts import latfuse002_real_fit as source
from scripts import latfuse008_paired_real_training as fitted
from scripts.validate_research_parity import check_manifest
from tac_osm.latent_fusion_004_factorial import head_weights

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research/mathematical-engineering/latfuse010/REGISTERED_ATTRIBUTION.json"
LAGS=(16,64,128,512)

def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def capture_event_write(model,pre,d,source_id,anchor):
    """Capture actual source event write pre-normalized key and tanh-input value.

    The source x requires gradient, ORIGINAL MODEL parameters are frozen,
    but original key_map/val_map outputs remain differentiable through x.
    """
    captured={}
    def hook(name):
        def store(_module,_inputs,output):
            if name in captured:
                raise RuntimeError("duplicate source-event write in recorded hook")
            captured[name]=output
        return store
    handles=(model.key_map.register_forward_hook(hook("key")),
             model.val_map.register_forward_hook(hook("value")))
    t=anchor
    source_x=d["values"][source_id,t].detach().clone().requires_grad_(True)
    try:
        actual=model.step(pre,source_x,d["valid"][source_id,t],
                 d["age"][source_id,t],d["last"][source_id,t],source_id)
    finally:
        for h in handles:h.remove()
    if set(captured)!={"key","value"}:
        raise RuntimeError("original fitted write key/value not captured")
    for name,z in captured.items():
        if not z.requires_grad or not bool(torch.isfinite(z).all()):
            raise RuntimeError("original source actual writer not differentiable "+name)
    return actual,source_x,captured

def actual_source_event_gates(model,d,source_id,anchor):
    with torch.no_grad():
        u=model.encode(d["values"][source_id,anchor],
          d["valid"][source_id,anchor],d["age"][source_id,anchor],
          d["last"][source_id,anchor])
        key=F.normalize(model.key_map(u).reshape(len(source_id),2,8),dim=-1,eps=1e-8)
        val=torch.tanh(model.val_map(u))
        routed,eligible=head_weights(model.write_router(u),model.log_head_temperature,
                               head_count=2,soft=True)
        if not bool(eligible.all()):
            raise RuntimeError("some source write head ineligible unexpectedly")
        observed=d["valid"][source_id,anchor].any(-1)
        alpha=torch.where(observed[:,None],
                 torch.sigmoid(model.retention_map(u)).expand(-1,2),torch.ones_like(routed))
        beta=torch.sigmoid(model.write_map(u))*routed*observed[:,None].to(routed.dtype)
    return dict(keys=key,values=val,alpha=alpha,beta=beta,route=routed)

def original_historic_read_query(model,at,anchor):
    with torch.no_grad():
        b=at.fast.shape[0]
        cal=source.calendar([int(t) for t in anchor])[:,None,:]
        joint=torch.cat((at.fast,at.slow),dim=-1)
        q=F.normalize((torch.tanh(model.historic_clock_query(cal))+
            model.belief_query(joint)[:,None,:]).reshape(b,1,2,8),dim=-1,eps=1e-8)
        route,_=head_weights(model.read_router(joint),model.log_head_temperature,
                             head_count=2,soft=True)
    return q[:,0],route,cal

def l2summary(g,b):
    if g is None:
        return dict(connected=False,mean_norm=0.0,
             nonzero_cases=0,max_norm=0.0)
    if not bool(torch.isfinite(g).all()):
        raise RuntimeError("original source write activation derivative nonfinite")
    n=g.detach().reshape(b,-1).norm(dim=1)
    return dict(connected=True,mean_norm=float(n.mean()),
                max_norm=float(n.max()),nonzero_cases=int((n>1e-12).sum()))

def source_pair_rank(p):
    return float(((p[::2]>p[1::2]).float()+
                .5*(p[::2]==p[1::2]).float()).mean())

def observed_readout_loss(model,at,source_id,anchor,true_source):
    time=source.calendar([int(v) for v in anchor])[:,None,:]
    logits=model.historic_event_logits(at,source_id,time)[:,0]
    # Labels MAY ONLY be read after the frozen model historical query.
    label=(true_source>=150).to(logits.dtype)
    loss=F.binary_cross_entropy_with_logits(logits,label)
    return logits,loss,label

def run_one(model,d,pairs,lag):
    b=len(pairs)
    site=torch.tensor([r[0] for r in pairs],dtype=torch.long)
    anchor=torch.tensor([r[1] for r in pairs],dtype=torch.long)
    with torch.no_grad():
        pre=model.initial(b)
        for offset in range(-23,0):
            ts=anchor+offset
            pre=model.step(pre,d["values"][site,ts],
               d["valid"][site,ts],d["age"][site,ts],
               d["last"][site,ts],site)
    # Event write captured through the ACTUAL source-trained model step.
    after,input_leaf,captured=capture_event_write(model,pre,d,site,anchor)
    lesion=after._replace(associative=pre.associative)
    source_write=actual_source_event_gates(model,d,site,anchor)
    now=after
    with torch.no_grad():
        untouched_lesion=lesion
    for offset in range(1,lag+1):
        ts=anchor+offset
        now=model.step(now,d["values"][site,ts],
                 d["valid"][site,ts],d["age"][site,ts],
                 d["last"][site,ts],site)
        with torch.no_grad():
            untouched_lesion=model.step(untouched_lesion,
               d["values"][site,ts],d["valid"][site,ts],
               d["age"][site,ts],d["last"][site,ts],site)
    if not all(torch.equal(p.detach(),q.detach()) for p,q in
                zip((now.fast,now.slow,now.entities,now.active),
                    (untouched_lesion.fast,untouched_lesion.slow,
                     untouched_lesion.entities,untouched_lesion.active))):
        raise RuntimeError("event-matrix lesion influenced nonassociative source states")
    # Prediction first, original observed source target only AFTER.
    known_clock=source.calendar([int(t) for t in anchor])[:,None,:]
    logits=model.historic_event_logits(now,site,known_clock)[:,0]
    with torch.no_grad():
        lesion_logits=model.historic_event_logits(
                           untouched_lesion,site,known_clock)[:,0]
        original_recorded=d["raw"][site,anchor,0]
        if not all(bool(torch.isfinite(v)) and (
            float(v)>=150 if row[2] else float(v)<=35)
                for v,row in zip(original_recorded,pairs)):
            raise RuntimeError("pre-committed original source event ground truth mismatch")
        historical_truth=(original_recorded>=150).float()
    loss=F.binary_cross_entropy_with_logits(logits,historical_truth)
    derivatives=torch.autograd.grad(loss,
         (captured["key"],captured["value"],input_leaf,after.associative),
         allow_unused=True)
    with torch.no_grad():
        # The keys are the ORIGINAL training learned 2x8-D write keys,
        # not the historical oracle input to the actual model query.
        read_query,read_weight,_=original_historic_read_query(model,now,anchor)
        cosine=(read_query*source_write["keys"]).sum(-1)
        signed=(cosine*read_weight).sum(-1)
        absval=(cosine.abs()*read_weight).sum(-1)
        p=torch.sigmoid(logits.detach())
        z=torch.sigmoid(lesion_logits)
        norm_matrix=(now.associative-untouched_lesion.associative).flatten(1).norm(dim=1)
        original_update=(after.associative-pre.associative).flatten(1).norm(dim=1)
        summaries={}
        for target in (0,1):
            sel=historical_truth==target
            summaries["high" if target else "ordinary"]=dict(
                physically_observed_cases=int(sel.sum()),
                source_true_historic_pred_probability_mean=float(p[sel].mean()),
                learned_read_to_event_write_weighted_abs_cos=float(absval[sel].mean()),
                learned_read_to_event_write_weighted_signed_cos=float(signed[sel].mean()),
                event_exact_write_lesion_abs_prediction_change_mean=float((p[sel]-z[sel]).abs().mean()),
                full_event_write_delta_matrix_norm_mean=float(original_update[sel].mean()),
                after_lag_event_matrix_influence_norm_mean=float(norm_matrix[sel].mean()))
        no_assoc=(model.__class__.__name__=="SourceTimedNonassociativeQuery")
        if no_assoc and not torch.equal(p,z):
            raise RuntimeError("noassoc event-query wrongly depends on event associative write")
        result=dict(lag_hours=lag,
            true_source_event_count=int(historical_truth.sum()),
            original_source_event_write_gate_alpha_mean=float(source_write["alpha"].mean()),
            original_source_event_write_gate_beta_mean=float(source_write["beta"].mean()),
            original_event_write_router_entropy=float(
              (-(source_write["route"]*source_write["route"].clamp_min(1e-9).log()).sum(-1)/
               torch.log(torch.tensor(2.0))).mean()),
            true_source_historic_event_BCE=float(loss.detach()),
            source_pair_event_rank=source_pair_rank(p),
            original_exact_source_event_write_lesion_source_pair_rank=source_pair_rank(z),
            overall_exact_event_write_lesion_mean_absolute_probability_effect=float((p-z).abs().mean()),
            learned_historic_read_key_to_event_write_key_headwise_abs_cos_mean=[
                float(cosine[:,h].abs().mean()) for h in range(2)],
            learned_historic_read_weighted_event_key_abs_cos_mean=float(absval.mean()),
            source_event_key_gradient=l2summary(derivatives[0],b),
            source_event_value_gradient=l2summary(derivatives[1],b),
            source_event_input_gradient=l2summary(derivatives[2],b),
            source_post_event_matrix_gradient=l2summary(derivatives[3],b),
            under_model_original_memory_storage_budget_6800B=True,
            by_event_type=summaries)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise RuntimeError("read-only source-key-value attribution original evidence already exists")
    torch.set_num_threads(1)
    reg=json.loads(REG.read_text())
    check_manifest(reg,"preflight")
    if reg["study_id"]!="LATFUSE010-REAL-EVENT-WRITE-READ-ATTRIBUTION" or (
       tuple(reg["lags_hours"])!=LAGS):
        raise RuntimeError("registered original true source delayed attribution mismatched")
    if sha(args.source)!=source.SOURCE_SHA:
        raise RuntimeError("wrong genuine original UCI501 historical archive")
    d=fitted.source_observations(args.source)
    whole=fitted.select_pairs(d,fitted.EVAL_START,fitted.EVAL_STOP,2)
    chosen=whole[:8]
    prior=json.loads((fitted.OUT/"RESULTS.json").read_text())
    original_cohort_sha=hashlib.sha256(json.dumps(whole,separators=(",",":")).encode()).hexdigest()
    if (original_cohort_sha!=prior["source_eval_cohort_sha256"] or len(chosen)!=8
        or prior["total_optimizer_steps"]!=128):
        raise RuntimeError("original trained source cohort/checkpoints were changed")
    evidence=[]
    for seed in fitted.SEEDS:
        for arm in fitted.ARMS:
            receipt=next((z for z in prior["model_results"] if
                    z["seed"]==seed and z["arm"]==arm),None)
            if receipt is None:raise RuntimeError("original source-trained arm receipt missing")
            cp=fitted.OUT/f"seed-{seed}"/arm/"final.pt"
            original_hash=sha(cp)
            if original_hash!=receipt["original_checkpoint_sha256"]:
                raise RuntimeError("original trained model SHA changed")
            m=fitted.create_model(seed,arm)
            pretrained=torch.load(cp,map_location="cpu",weights_only=False)
            m.load_state_dict(pretrained["original_model"],strict=True)
            m.eval()
            weights=[p.detach().clone() for p in m.parameters()]
            m.requires_grad_(False)  # input activation still differentiated
            delays={str(h):run_one(m,d,chosen,h) for h in LAGS}
            if sha(cp)!=original_hash or any(not torch.equal(x,y) for x,y
                                            in zip(weights,m.parameters())):
                raise RuntimeError("actual source-trained original weights mutated by diagnosis")
            evidence.append(dict(seed=seed,arm=arm,original_checkpoint_sha256=original_hash,
                                 lagged_source_event_attribution=delays))
            print("LATFUSE010_REAL_SOURCE_WRITER_ARM="+json.dumps({
                "arm":arm,"seed":seed,
                "event_key_gradient":{h:v["source_event_key_gradient"]["mean_norm"]
                    for h,v in delays.items()},
                "real_write_lesion_probability_effect":{h:
                    v["overall_exact_event_write_lesion_mean_absolute_probability_effect"]
                    for h,v in delays.items()}},sort_keys=True),flush=True)
    output=dict(study_id=reg["study_id"],original_source_archive_sha256=source.SOURCE_SHA,
      original_matched_source_cohort_sha256=original_cohort_sha,
      original_observed_real_source_pairs=len(chosen)//2,
      source_lags_hours=list(LAGS),source_true_original_frozen_model_count=len(evidence),
      original_checkpoint_parameters_unchanged=True,original_optimizer_updates_this_study=0,
      original_event_write_gradients_are_activation_not_parameter_gradients=True,
      event_step_matrix_lesion_is_local_model_counterfactual_not_alternative_physical_world=True,
      no_ORACLE_event_write_key_supplied_to_model_query=True,
      original_2015_DEV_2016_2017_protected_numeric_unopened=True,
      no_qualified_model_accuracy_or_superiority=True,models=evidence)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LATFUSE010_REAL_EVENT_WRITER_ATTRIBUTION_PASS="+json.dumps({
      "models":len(evidence),"physical_source_pairs":len(chosen)//2,
      "new_optimizer_steps":0},sort_keys=True),flush=True)

if __name__=="__main__":main()
