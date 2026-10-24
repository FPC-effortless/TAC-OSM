#!/usr/bin/env python3
"""Bounded real-identity memory admission mechanism tests, NOT real skill."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
from tac_osm.variants.bounded_entity_admission import (
    init_entity_state,entity_admit_step,entity_read,entity_protect)


def test():
    torch.set_num_threads(1)
    m=init_entity_state(2,slots=2,width=8,dtype=torch.float64)
    tested=0
    a=torch.tensor([11,21]);b=torch.tensor([12,22])
    v=torch.tensor([[.7]+[0]*7,[.1]+[0]*7],dtype=torch.float64)
    w=torch.tensor([[0,.8]+[0]*6,[0,.9]+[0]*6],dtype=torch.float64)
    yes=torch.tensor([True,True]);no=torch.tensor([False,False])
    high=torch.tensor([.7,.8],dtype=torch.float64)
    m,_,accepted,replaced=entity_admit_step(m,a,v,high,yes)
    assert accepted.all() and not replaced.any()
    tested+=1
    snapshot=m.values.clone()
    m,_,accepted,replaced=entity_admit_step(m,b,w,high,yes)
    assert accepted.all() and not replaced.any()
    assert torch.equal(m.values[:,0],snapshot[:,0])
    tested+=2
    before=m
    m,_,accepted,replaced=entity_admit_step(
       m,a,torch.zeros_like(v),torch.ones(2,dtype=torch.float64),no)
    assert torch.equal(before.values,m.values)
    assert torch.equal(before.entity_ids,m.entity_ids)
    assert not accepted.any()
    tested+=3
    m=entity_protect(m,a,yes)
    before=m.values.clone()
    # Full, protected entity a cannot be evicted regardless of desired score.
    c=torch.tensor([13,23])
    too_low=torch.tensor([.75,.79],dtype=torch.float64)
    m,_,accepted,replaced=entity_admit_step(m,c,w,too_low,yes,
                                            replacement_margin=.15)
    assert not accepted.any() and not replaced.any()
    assert torch.equal(m.values,before)
    tested+=2
    # Higher known-at-time priority may replace b, but protected a stays.
    score=torch.ones(2,dtype=torch.float64)
    m,_,accepted,replaced=entity_admit_step(m,c,w,score,yes,
                                            replacement_margin=.1)
    assert accepted.all() and replaced.all()
    assert torch.equal(m.values[:,0],before[:,0])
    values_a,found_a=entity_read(m,a)
    values_b,found_b=entity_read(m,b)
    values_c,found_c=entity_read(m,c)
    assert found_a.all() and not found_b.any() and found_c.all()
    assert torch.equal(values_c,w)
    tested+=6
    # Evicted b's numerical memory must never leak into new c on partial
    # updates; a brand-new entity allocates from zero state.
    oldscore=m.priority.clone()
    d=torch.tensor([14,24]);dval=torch.tensor([[0,.2]+[0]*6,[0,.3]+[0]*6],dtype=torch.float64)
    m,_,accepted,replaced=entity_admit_step(
      m,d,dval,score,yes,beta=.25,replacement_margin=0)
    assert not accepted.any()   # score == current: strict > required
    tested+=1
    # Explicitly clear the only unprotected slot by an authorized replace.
    lower=torch.tensor([.6,.6],dtype=torch.float64)
    state0=init_entity_state(2,1,8,dtype=torch.float64)
    state0,_,_,_=entity_admit_step(state0,a,v,lower,yes)
    state1,_,ok,rep=entity_admit_step(
         state0,c,dval,score,yes,beta=.25,replacement_margin=.1)
    assert ok.all() and rep.all()
    torch.testing.assert_close(state1.values[:,0],.25*dval)
    tested+=2
    with torch.no_grad():
        for _ in range(1024):
            state1,_,_,_=entity_admit_step(
                state1,c,dval,score,yes,beta=.2)
        assert state1.values.numel()==2*8
        assert torch.equal(state1.entity_ids,c[:,None])
    tested+=2
    print(json.dumps({"status":"PASS_BOUNDED_PROTECTED_ENTITY_ADMISSION",
       "checks":tested,"slots":2,"memory_per_entity":8,
       "observational_entity_ids_required":True,
       "future_outcome_oracle_used":False,"trainable_parameters_added":0,
       "parallel_batch_routing_proven":False,
       "real_longitudinal_qualification":"NOT_EXECUTED"},sort_keys=True))


if __name__=="__main__":test()
