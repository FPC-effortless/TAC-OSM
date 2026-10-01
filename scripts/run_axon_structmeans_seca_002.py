#!/usr/bin/env python3
"""Registered AXON/StructMeans/SECA phase-2 experiment."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac_osm.operator_learning import (
    ExperienceStore, FixedAXONConsolidator, PrimitiveOperator,
    PSTLearner, SECAEngine, SparseOperatorRouter, StructMeans,
    TransitionRecord, apply_operator,
)

DIM=8
KINDS=("toggle","set1","set0")
OPS=(
    PrimitiveOperator("toggle",(0,1,0,1,0,1,0,1)),
    PrimitiveOperator("set1",(1,0,1,0,1,0,1,0)),
    PrimitiveOperator("set0",(1,1,0,0,1,1,0,0)),
)

def independent_apply(state, op):
    out=list(state)
    for i,active in enumerate(op.mask):
        if not active: continue
        if op.kind=="toggle": out[i]=1-out[i]
        elif op.kind=="set1": out[i]=1
        elif op.kind=="set0": out[i]=0
        else: raise ValueError(op.kind)
    return tuple(out)

class Composite:
    def __init__(self,name,steps): self.name=name; self.steps=tuple(steps)
    def execute(self,state):
        cur=tuple(state)
        for step in self.steps: cur=apply_operator(cur,step)
        return cur

def independent_reference(state,macro):
    cur=tuple(state)
    for step in macro.steps: cur=independent_apply(cur,step)
    return cur

def make_records(seed=22,per_op=120):
    rng=random.Random(seed); rows=[]
    for op in OPS:
        for i in range(per_op):
            state=tuple(rng.randrange(2) for _ in range(DIM))
            rows.append(TransitionRecord(state,op,apply_operator(state,op),True,i,0))
    rng.shuffle(rows); return rows

def route_success(router, library, rows):
    good=0
    for state,goal in rows:
        chosen,_=router.route(state,goal,library,budget=1)
        good += int(chosen and chosen[0].execute(state)==goal)
    return good/len(rows) if rows else 0.0

def main():
    train=make_records()
    store=ExperienceStore()
    for rec in train: store.append(rec)
    bad=TransitionRecord(train[0].before,train[0].operator,
                         tuple(1-x for x in train[0].after),False,99999,0)
    rejected_committed=store.append(bad)
    pst=store.reconstruct_pst(KINDS)
    rng=random.Random(91)
    held=[]
    for i in range(180):
        op=OPS[i%len(OPS)]; state=tuple(rng.randrange(2) for _ in range(DIM))
        held.append(TransitionRecord(state,op,apply_operator(state,op),True,1000+i,0))
    transition_acc=pst.transition_accuracy(held)

    sm=StructMeans(3,KINDS,seed=4); sm.fit(train)
    sig_purity=sm.signature_purity(train)
    kind_purity=sm.purity(train)

    macros=FixedAXONConsolidator(min_support=20).consolidate(train)
    sparse=SparseOperatorRouter(pst)
    single=[]
    for _ in range(90):
        op=OPS[rng.randrange(len(OPS))]; state=tuple(rng.randrange(2) for _ in range(DIM))
        single.append((state,independent_apply(state,op)))
    single_success=route_success(sparse,macros,single)

    first=next(m for m in macros if m.steps[0].kind=="toggle")
    second=next(m for m in macros if m.steps[0].kind=="set1")
    target=Composite("target",first.steps+second.steps)

    composite=[]
    for _ in range(128):
        state=tuple(rng.randrange(2) for _ in range(DIM)); goal=target.execute(state)
        if all(m.execute(state)!=goal for m in macros): composite.append((state,goal))

    pre=route_success(sparse,macros,composite)
    seca=SECAEngine()
    proposals=seca.propose(macros,max_pairs=12)
    verify_states=[s for s,_ in composite[:16]]
    accepted=seca.verify(proposals,verify_states,independent_reference)
    target_name=f"seca:{first.name}+{second.name}"
    discovered=any(m.name==target_name for m in accepted)
    library=tuple(macros)+tuple(accepted)
    post=route_success(sparse,library,composite)

    result={
        "protocol":{"name":"TACOSM-AXON-STRUCTMEANS-SECA-002","train_records":len(train),
                    "heldout_records":len(held),"operator_count":len(OPS),
                    "single_operator_budget":1,"composite_tasks":len(composite)},
        "regm":{"verified_records":len(store.records),"rejected_committed":int(rejected_committed),
                "retained_competence_after_reconstruction":transition_acc},
        "pst":{"heldout_transition_accuracy":transition_acc},
        "structmeans":{"clusters":len(sm.centroids),"kind_purity":kind_purity,
                       "signature_purity":sig_purity,
                       "verified_records_per_centroid":len(train)/len(sm.centroids)},
        "axon":{"macros":[m.name for m in macros],"supports":{m.name:m.support for m in macros},
                "single_operator_route_success":single_success},
        "seca":{"proposed":len(proposals),"verified":len(accepted),
                "target_discovered":discovered,
                "pre_seca_success":pre,"post_seca_success":post,
                "new_operator_names":[m.name for m in accepted]},
        "closed_loop":{"experience_ingested":len(store.records)==len(train),
                       "transition_model_verified":transition_acc>=.95,
                       "structure_abstraction_verified":sig_purity>=.85,
                       "operator_library_built":len(macros)>=len(OPS),
                       "novel_operator_verified":discovered,
                       "post_creation_reuse":post}}
    return result

if __name__=="__main__":
    result=main()
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/TACOSM-AXON-STRUCTMEANS-SECA-002.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
