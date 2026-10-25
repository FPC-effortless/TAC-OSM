"""Engineering tests only: synthetic tensors do not qualify robot/controller skill."""
import pytest
import torch
from tac_osm.variants.action_query_edge_world import (
    ActionQueryEdgeWorld,instantiate_edge_research_profile,
)

torch.set_num_threads(1)


def records(dtype=torch.float64):
    torch.manual_seed(400)
    m=ActionQueryEdgeWorld().to(dtype)
    B,T=2,9
    x=torch.randn(B,T,14,dtype=dtype)*.2
    prev=torch.randn(B,T,14,dtype=dtype)*.1
    commands=torch.randn(B,3,14,dtype=dtype)*.1
    hours=torch.arange(T,dtype=torch.int64)[None].expand(B,T).clone()
    owners=torch.tensor([1,7],dtype=torch.int64)
    return m,x,prev,commands,hours,owners


def test_native_dim_budget_and_non_authority():
    m,x,u,query,t,owner=records()
    assert 95000<=m.active_parameter_count<=105000
    assert m.active_parameter_count==sum(p.numel() for p in m.parameters())
    y,st=m(x,u,query,t,owner)
    assert y.mean.shape==(2,3,14)
    assert not y.trained_action_policy
    assert not y.causal_action_effect_established
    assert not y.autonomous_actuation_authorized
    assert st.evidence.basis_sum.shape==(2,88)
    assert st.evidence.memory.shape==(2,8,8)
    assert (88+64)*4==608


def test_candidate_query_does_not_mutate_factual_state():
    m,x,u,query,t,owner=records()
    y,s=m(x,u,query,t,owner)
    alternative=query.clone()
    alternative[:,1]+=1.5
    z,s2=m(x,u,alternative,t,owner)
    torch.testing.assert_close(s.evidence.memory,s2.evidence.memory,rtol=0,atol=0)
    torch.testing.assert_close(s.evidence.basis_sum,s2.evidence.basis_sum,rtol=0,atol=0)
    torch.testing.assert_close(s.evidence.count,s2.evidence.count,rtol=0,atol=0)
    torch.testing.assert_close(y.mean[:,0],z.mean[:,0],rtol=0,atol=0)
    assert not torch.allclose(y.mean[:,1],z.mean[:,1],atol=1e-12)


def test_carried_chunks_preserve_factual_state_and_readout():
    m,x,u,query,t,owner=records()
    y,full=m(x,u,query,t,owner)
    _,partial=m(x[:,:4],u[:,:4],query,t[:,:4],owner)
    z,carried=m(x[:,4:],u[:,4:],query,t[:,4:],owner,state=partial)
    torch.testing.assert_close(full.evidence.memory,carried.evidence.memory,rtol=1e-8,atol=1e-9)
    torch.testing.assert_close(full.evidence.basis_sum,carried.evidence.basis_sum,rtol=1e-8,atol=1e-9)
    torch.testing.assert_close(y.mean,z.mean,rtol=1e-8,atol=1e-9)


def test_all_active_gradients_flow_through_candidate_forecast():
    m,x,u,query,t,owner=records()
    pred,_=m(x,u,query,t,owner)
    pred.mean.square().sum().backward()
    missing=[name for name,p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing,missing
    assert m.command_head[0].weight.grad.abs().sum()>0
    assert m.core.encoder.weight.grad.abs().sum()>0
    assert m.core.readout[2].weight.grad.abs().sum()>0


def test_source_authority_invalid_and_padding_rejected():
    m,x,u,q,t,owner=records()
    _,st=m(x,u,q,t,owner)
    with pytest.raises(ValueError):m(x,u,q,t,owner,state=st) # repeated actual observed clock
    with pytest.raises(ValueError):m(x,u,q,t+20,owner.flip(0),state=st)
    with pytest.raises(ValueError):m(x,u,q,t.roll(1,dims=1),owner)
    invalid=torch.ones(2,9,dtype=torch.bool);invalid[:,-1]=False
    with pytest.raises(ValueError):m(x,u,q,t,owner,valid=invalid)
    q2=q.clone();q2[0,0,0]=float("nan")
    with pytest.raises(ValueError):m(x,u,q2,t,owner)
    with torch.no_grad():m.core.encoder.bias.add_(.01)
    with pytest.raises(ValueError):m(x,u,q,t+20,owner,state=st)


def test_domain_profiles_remain_typed_and_not_six_trained_models():
    sensor=instantiate_edge_research_profile("temporal_sensor")
    from tac_osm.variants.belief_sensor_world import BeliefSensorWorld
    assert isinstance(sensor,BeliefSensorWorld)
    assert isinstance(instantiate_edge_research_profile("robot_physical"),ActionQueryEdgeWorld)
    with pytest.raises(ValueError):instantiate_edge_research_profile("enterprise")
