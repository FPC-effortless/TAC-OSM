#!/usr/bin/env python3
"""C5 bridge: execution-feedback learning + product-key selective routing.

The candidate population is fixed within each M level. Target codes are held
out from the 48-item training codebook. The learner updates from post-execution
verifier labels; the product-key index is rebuilt every N learner updates.
Inference cost and training-only dense feedback cost are reported separately.
"""
from __future__ import annotations
import json, math, random, statistics, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm import Candidate, Outcome, PersistentState, Query, VerificationResult
from tac_osm.execution_product_key import ExecutionFeedbackProductKeyRouter
from tac_osm.energy_router import EnergyRouterConfig, RepresentationEnergyRouter
from tac_osm.state import PersistentStore, StateConfig
from tac_osm.noisy_state_tasks import CODEBOOK

SEEDS=(10,11,12)
M_LEVELS=(128,256,512)
TRAIN_STEPS=900
EVAL_STEPS=300
STATE_BITS=10
LATENT_DIM=16
TARGET_CODES=tuple(CODEBOOK[:16])
TRAIN_CODES=tuple(CODEBOOK[16:64])
REFRESH=32

def all_decoys():
    excluded=set(TARGET_CODES)|set(TRAIN_CODES)
    out=[]
    for value in range(1<<STATE_BITS):
        code=tuple((value>>(STATE_BITS-1-i))&1 for i in range(STATE_BITS))
        if code not in excluded: out.append(code)
    return tuple(out)

DECOYS=all_decoys()

def population(seed,m):
    vals=list(TARGET_CODES)+list(TRAIN_CODES)
    rng=random.Random(seed*100003+m*7919)
    dec=list(DECOYS); rng.shuffle(dec); vals.extend(dec[:m-64]); rng.shuffle(vals)
    return tuple(Candidate(
        key=f"state-{seed}-{m}-{i:04d}",
        descriptor=tuple(v),
        action=i,
        provenance="c5_feedback_productkey",
    ) for i,v in enumerate(vals))

def query_for(seed,m,step,target):
    flip=(seed+3*step+m)%STATE_BITS
    bits=list(target); bits[flip]^=1
    return Query(
        text=" ".join(str(int(x)) for x in bits),
        context=(1,)*STATE_BITS,
        step=step,
        provenance="c5_execution_feedback_productkey_query",
    )

def target_index(candidates,target):
    return next(i for i,c in enumerate(candidates) if tuple(c.descriptor)==tuple(target))

def make_arm(seed,m,product):
    candidates=population(seed,m)
    training_keys=[c.key for c in candidates if tuple(c.descriptor) in set(TRAIN_CODES)]
    state=PersistentStore(StateConfig(seed=seed+100000,memory_dim=STATE_BITS,n_slots=2048)) if False else PersistentStore(StateConfig(seed=seed+100000,n_slots=2048))
    if product:
        router=ExecutionFeedbackProductKeyRouter(
            seed=seed,input_dim=STATE_BITS,latent_dim=LATENT_DIM,
            factor_count=3,factor_size=16,factor_beam=7,max_shortlist=32,
            refresh_interval_updates=REFRESH,learning_rate=0.01,margin=0.1,
        )
    else:
        router=RepresentationEnergyRouter(EnergyRouterConfig(
            input_dim=STATE_BITS,latent_dim=LATENT_DIM,
            learning_rate=0.01,margin=0.1,top_k=1,seed=seed
        ))
    train=[]
    for step in range(TRAIN_STEPS):
        rng=random.Random(seed*1009+m*7919+step*104729)
        target=rng.choice(TARGET_CODES)
        q=query_for(seed,m,step,target)
        ti=target_index(candidates,target)
        if product:
            decision,diag=router.route(q,state,candidates,training_keys=training_keys)
        else:
            decision=router.route(q,state,candidates)
            diag=None
        selected=decision.selected
        ok=selected==ti
        detail=SimpleNamespace(gold_index=ti,candidates=candidates)
        outcome=Outcome(ok,float(ok),"ok" if ok else "wrong_candidate",detail)
        ver=VerificationResult(ok,"accept" if ok else "reject")
        if product:
            memory=router.representation.last_memory
            if memory is None:
                memory=router.representation.addressor.address(q,state)
            feedback_scores=router.representation.score(q,memory,candidates)
            feedback_selected=selected
            if feedback_selected < 0:
                feedback_order=sorted(range(len(feedback_scores)),
                                      key=lambda i:(-feedback_scores[i],i))
                feedback_selected=next(i for i in feedback_order if i != ti)
            router.learn_from_verifier(
                query=q,state=state,candidates=candidates,
                selected=feedback_selected,outcome=outcome,verification=ver,
                scores=feedback_scores,
            )
        else:
            router.learn_from_outcome(
                q,state,candidates,selected,success=ok,scores=decision.scores
            )
    metrics=[]
    admitted=[]
    rerank=[]
    factor_macs=[]
    pair_ops=[]
    build_macs=[]
    accepted=0
    for step in range(EVAL_STEPS):
        rng=random.Random(seed*1009+m*7919+(TRAIN_STEPS+step)*104729)
        target=rng.choice(TARGET_CODES); ti=target_index(candidates,target)
        q=query_for(seed,m,TRAIN_STEPS+step,target)
        if product:
            decision,diag=router.route(q,state,candidates,training_keys=training_keys)
            scores=decision.scores
            rank=None
            shortlist_indices=[i for i,s in enumerate(scores) if math.isfinite(s)]
            order=sorted(shortlist_indices,key=lambda i:(-scores[i],i))
            admitted.append(int(ti in shortlist_indices))
            rerank.append(diag.state_candidates_scored)
            factor_macs.append(diag.factor_score_macs)
            pair_ops.append(diag.pair_generation_ops)
            build_macs.append(diag.build_total_macs/ max(1,diag.refresh_count))
        else:
            decision=router.route(q,state,candidates)
            order=sorted(range(len(decision.scores)),key=lambda i:(-decision.scores[i],i))
            rank=order.index(ti)+1
        ok=decision.selected==ti
        accepted+=int(ok)
        metrics.append({
            "target_rank": (order.index(ti)+1) if ti in order else None,
            "selected":int(decision.selected==ti),
        })
    n=len(metrics)
    result={
        "seed":seed,"M":m,"product_key":product,
        "eval_top1_recall":accepted/n,
        "mean_target_rank":statistics.fmean([x["target_rank"] for x in metrics if x["target_rank"] is not None]),
        "router_updates":router.updates,
    }
    if product:
        result.update({
            "proposal_admission_recall":statistics.fmean(admitted),
            "conditional_selection_given_admission":(
                accepted/sum(admitted) if sum(admitted) else 0.0
            ),
            "state_candidates_scored_mean":statistics.fmean(rerank),
            "states_scored_over_M":statistics.fmean(rerank)/m,
            "factor_score_macs_mean":statistics.fmean(factor_macs),
            "pair_generation_ops_mean":statistics.fmean(pair_ops),
            "amortized_build_macs_per_query":statistics.fmean(build_macs),
            "max_shortlist":32,"factor_beam":7,
        })
    return result

def main():
    rows=[]
    for seed in SEEDS:
        for m in M_LEVELS:
            rows.append(make_arm(seed,m,False))
            rows.append(make_arm(seed,m,True))
    pooled={}
    for product in (False,True):
        for m in M_LEVELS:
            vals=[r for r in rows if r["product_key"]==product and r["M"]==m]
            pooled[("product_key" if product else "dense",m)]={
                "M":m,"arm":"product_key" if product else "dense",
                "eval_top1_recall":statistics.fmean(v["eval_top1_recall"] for v in vals),
                "mean_target_rank":statistics.fmean(v["mean_target_rank"] for v in vals),
                "router_updates":statistics.fmean(v["router_updates"] for v in vals),
                **({"proposal_admission_recall":statistics.fmean(v["proposal_admission_recall"] for v in vals),
                    "conditional_selection_given_admission":statistics.fmean(v["conditional_selection_given_admission"] for v in vals),
                    "state_candidates_scored_mean":statistics.fmean(v["state_candidates_scored_mean"] for v in vals),
                    "states_scored_over_M":statistics.fmean(v["states_scored_over_M"] for v in vals),
                    "factor_score_macs_mean":statistics.fmean(v["factor_score_macs_mean"] for v in vals),
                    "pair_generation_ops_mean":statistics.fmean(v["pair_generation_ops_mean"] for v in vals),
                    "amortized_build_macs_per_query":statistics.fmean(v["amortized_build_macs_per_query"] for v in vals)}, product_key),
            }
    return {"protocol":{"name":"TACOSM-C5-EXECUTION-FEEDBACK-PRODUCTKEY-001",
                        "seeds":list(SEEDS),"M_levels":list(M_LEVELS),
                        "train_steps":TRAIN_STEPS,"eval_steps":EVAL_STEPS,
                        "refresh_interval_updates":REFRESH,"factor_count":3,
                        "factor_size":16,"factor_beam":7,"max_shortlist":32},
            "pooled":{f"{k[0]}:{k[1]}":v for k,v in pooled.items()},"cells":rows}

if __name__=="__main__":
    result=main()
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/TACOSM-C5-EXECUTION-FEEDBACK-PRODUCTKEY-001.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
