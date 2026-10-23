"""Synthetic fixtures check math/contracts only; capability uses actual records."""
import copy
import math
import numpy as np
import torch
from scripts.me069_workflow_source import fit_preprocessing,encode_cases
from scripts.me072_marked_world import derive_preprocessing,pack,forward,metrics
from tac_osm.workflow_me071 import detach_belief
from tac_osm.workflow_me072 import MarkedWorldModel,record_log_probability,world_loss,time_quantiles,HORIZONS


def fixture():
    raw=[{"id":str(i),"events":[["A|START" if t%3 else "B|COMPLETE","r",i*10000.+t*t]
         for t in range(70)]} for i in range(2)]
    pre=fit_preprocessing(raw);cases=encode_cases(raw,pre)
    for c in cases:c["history_log_prior"]=np.asarray([pre["transition_log_probability"][a] for a in c["activity"]],np.float32)
    pre=derive_preprocessing(cases,pre)
    torch.manual_seed(72);model=MarkedWorldModel(pre)
    with torch.no_grad():
        for head in (model.mark_head,model.survival_head,model.clock_base,model.clock_increment):head.weight.normal_(std=.01)
    return cases,pre,model


def test_multi_horizon_labels_preserve_actual_records_and_explicit_boundary():
    cases,_,_=fixture();data=pack(cases,[0],65,5)
    for i in range(5):
        for j,h in enumerate(HORIZONS):
            exists=65+i+h<70
            assert bool(data[7][0,i,j])==exists
            if exists:
                assert data[5][0,i,j]==cases[0]['activity'][65+i+h]
                assert data[6][0,i,j]==cases[0]['timestamps'][65+i+h]-cases[0]['timestamps'][65+i]
            else:assert data[6][0,i,j]==0
    assert data[8][0,-1] and not data[7][0,-1].any()


def test_joint_mass_normalizes_and_boundary_probability_increases():
    cases,_,model=fixture();p,_=forward(model,pack(cases,[0],length=6))
    joint=record_log_probability(p).exp()
    torch.testing.assert_close(joint.sum(-1),torch.ones_like(p.log_survival),rtol=1e-6,atol=1e-6)
    assert (p.log_survival[:,1:]<=p.log_survival[:,:-1]).all()
    assert (p.log_zero[:,1:]<=p.log_zero[:,:-1]).all()


def test_clock_locations_and_quantiles_cannot_reverse_over_horizons():
    cases,_,model=fixture();p,_=forward(model,pack(cases,[0],length=6))
    assert (p.log_time_location[:,1:]>=p.log_time_location[:,:-1]).all()
    q=time_quantiles(p)
    assert q.dtype==torch.float64 and torch.isfinite(q).all()
    assert (q[:,:,1:]>=q[:,:,:-1]).all()
    assert (q[:,1:]+1e-6>=q[:,:-1]).all()


def test_quantile_inverse_uses_the_same_zero_mixture_distribution():
    cases,_,model=fixture();p,_=forward(model,pack(cases,[0],length=2))
    q=time_quantiles(p);mu=p.log_time_location.double();sigma=p.log_time_scale.double().exp()[:,None,:]
    for k,level in enumerate((.05,.5,.95)):
        positive=q[:,:,k]>0
        z=(q[:,:,k].clamp_min(1e-300).log()[...,None]-mu)/sigma
        cdf=p.log_zero.double().exp()+(1-p.log_zero.double().exp())*(p.mixture_log_weight.double().exp()[:,None,:]*.5*torch.erfc(-z/math.sqrt(2))).sum(-1)
        torch.testing.assert_close(cdf[positive],torch.full_like(cdf[positive],level),rtol=1e-6,atol=1e-6)
        assert ((p.log_zero.double().exp()>=level)|positive).all()


def test_complete_joint_training_loss_has_finite_active_gradients():
    cases,_,model=fixture();data=pack(cases,[0,1],40,32)
    p,_=forward(model,data)
    loss=world_loss(p,data[5].flatten(0,1),data[6].flatten(0,1),data[7].flatten(0,1),data[8].flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())


def test_chunk_carry_and_prediction_before_observation_are_causal():
    cases,_,model=fixture();a=pack(cases,[0],0,32);b=pack(cases,[0],32,32)
    _,first=forward(model,a);_,tail=forward(model,b,detach_belief(first))
    _,whole=forward(model,pack(cases,[0],0,64))
    for x,y in zip(tail,whole):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-6)
    original,_=forward(model,a);changed=list(a);changed[0]=a[0].clone();changed[0][:,16:]=1
    future,_=forward(model,tuple(changed))
    torch.testing.assert_close(record_log_probability(original)[:16],record_log_probability(future)[:16],rtol=0,atol=0)


def test_no_time_score_is_fabricated_after_a_record_boundary():
    cases,_,model=fixture();data=pack(cases,[0],69,1);p,_=forward(model,data,all_steps=False)
    rows,_=metrics(p,data)
    for row in rows.values():
        assert row['future_record_cases']==0 and row['record_boundary_cases']==1
        assert row['waiting_median_MAE_hours'] is None and row['conditional_time_NLL_seconds'] is None


def test_new_future_labels_do_not_change_available_prefix_features():
    cases,_,_=fixture();changed=copy.deepcopy(cases);changed[0]['activity'][6:]=1
    a,b=pack(cases,[0],length=6),pack(changed,[0],length=6)
    for index in (0,1,2,3,4,8):torch.testing.assert_close(a[index],b[index],rtol=0,atol=0)
