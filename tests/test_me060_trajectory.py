"""Numerical/causality checks only; fabricated inputs are not capability data."""
import torch
from scripts.me060_train_trajectory import build, objective
from tests.test_me057_training_contract import identified
from tac_osm.real_robot_me060_trajectory import (
    WorldState, physical_interval, forecast_requires_review)


def test_bounds_and_direct_action_prefix_causality():
    torch.manual_seed(60001)
    m=build("native_tac",identified());b=m.initial(2)
    for _ in range(128):
        prev=torch.randn(2,14);current=prev+torch.randn_like(prev)*.05
        b=m.observe(b,prev,torch.randn_like(prev),current,torch.zeros(2,1,dtype=torch.bool))
    assert b.fast.abs().max()<=1 and b.slow.abs().max()<=1
    assert b.memory.square().sum((-2,-1)).sqrt().max()<=1.00001
    current=torch.randn(2,14);actions=torch.randn(2,16,14)
    a=m.forecast(b,current,actions)
    changed=actions.clone();changed[:,8:]+=100
    c=m.forecast(b,current,changed)
    torch.testing.assert_close(a.mean[:,:8],c.mean[:,:8],rtol=0,atol=0)
    assert torch.isfinite(a.scale).all() and (a.scale>0).all()
    torch.testing.assert_close(a.weights.sum(-1),torch.ones(2,16))


def test_no_future_target_changes_prediction_and_all_parameters_train():
    torch.manual_seed(60001)
    states=torch.randn(8,81,14);actions=torch.randn_like(states)
    flags=torch.zeros(8,81,dtype=torch.bool)
    for arm in ("native_tac","gru_world"):
        m=build(arm,identified());b=m.initial(8)
        for t in range(64):
            b=m.observe(b,states[:,t],actions[:,t],states[:,t+1],flags[:,t,None])
        before=m.forecast(b,states[:,64],actions[:,64:80])
        future=states.clone();future[:,65:]+=100
        after=m.forecast(b,future[:,64],actions[:,64:80])
        torch.testing.assert_close(before.mean,after.mean,rtol=0,atol=0)
        objective(m,(states,actions,flags)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_query_does_not_mutate_persistent_belief():
    m=build("native_tac",identified());b=m.initial(2)
    copy=WorldState(*(x.clone() for x in b))
    m.forecast(b,torch.randn(2,14),torch.randn(2,16,14))
    for left,right in zip(b,copy):
        torch.testing.assert_close(left,right,rtol=0,atol=0)


def test_uncertainty_review_uses_physical_tolerances():
    m=build("native_tac",identified())
    forecast=m.forecast(m.initial(2),torch.randn(2,14),torch.randn(2,16,14))
    lower,upper,radius=physical_interval(forecast,torch.ones(14),2.)
    assert (upper>lower).all()
    assert forecast_requires_review(radius,torch.full((14,),1e-8)).all()
    assert not forecast_requires_review(radius,torch.full((14,),1e8)).any()
