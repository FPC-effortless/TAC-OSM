"""ME054 mathematical SOFTWARE guards, never simulated capability evidence."""
import torch
import pytest
from tac_osm.real_robot_me054_stable import (
    StableRealTAC,StableRealGRU,actual_real_physical_and_takeover_loss)


def models():
    scales=[.015]*14
    return StableRealTAC(scales),StableRealGRU(scales)


def test_me054_physical_persistence_is_exact_in_both_zero_initialized_heads():
    for model in models():
        x=torch.randn(2,14)*.3
        a=torch.randn(2,14)*.1
        human=torch.tensor([[False],[True]])
        z=model.observe(model.initial(2),x)
        before=model.predict(z,x,a[:,None],human)
        assert torch.equal(before.next_joint_state[:,0],x)
        assert before.takeover_logits.shape==(2,1,1)


def test_me054_zero_true_physical_innovation_is_exact_belief_identity():
    model,_=models()
    x=torch.randn(2,14)*.2
    a=torch.randn(2,14)*.1
    h=torch.tensor([[False],[True]])
    state=model.observe(model.initial(2),x)
    forecast=model.predict(state,x,a[:,None],h)
    mask=torch.ones(2,1,dtype=torch.bool)
    masked=torch.zeros(2,1,dtype=torch.bool)
    after=model.verify_real_feedback(
        state,a,h,forecast.next_joint_state[:,0],
        forecast.next_joint_state[:,0],mask,
        torch.full((2,1),float("nan")),
        torch.full((2,1),float("nan")),masked)
    assert all(torch.equal(before,aft) for before,aft in zip(state,after))


def test_me054_real_innovation_bounded_contract_and_differentiability():
    model,_=models()
    x=torch.randn(2,14)*.2
    action=torch.randn(2,14)*.1
    authority=torch.tensor([[False],[True]])
    belief=model.observe(model.initial(2),x)
    pred=model.predict(belief,x,action[:,None],authority)
    truly_observed=x+.012
    available=torch.ones(2,1,dtype=torch.bool)
    mask=~authority
    labels=torch.tensor([[1.],[float("nan")]])
    updated=model.verify_real_feedback(belief,action,authority,
        truly_observed,pred.next_joint_state[:,0],available,
        labels,pred.takeover_logits[:,0],mask)
    # Both convex pathways target tanh-bounded state; initial/native
    # observation itself is bounded. No repeated additive state explosion.
    assert updated.fast.abs().max()<=1.00001
    assert updated.slow.abs().max()<=1.00001
    assert updated.memory.abs().max()<=1.00001
    next_pred=model.predict(model.observe(updated,truly_observed),
                            truly_observed,action[:,None],authority)
    total,_=actual_real_physical_and_takeover_loss(
        model,next_pred,truly_observed,truly_observed+.02,
        labels,mask)
    total.backward()
    assert model.native.next_observation.weight.grad is not None
    assert torch.isfinite(model.native.next_observation.weight.grad).all()
    assert model.physical_map.weight.grad is not None


def test_me054_matched_active_parameters_unchanged_from_ME052():
    a,b=models()
    ac=sum(p.numel() for p in a.parameters() if p.requires_grad)
    bc=sum(p.numel() for p in b.parameters() if p.requires_grad)
    assert ac==37985 and bc==38295
    assert abs(ac-bc)/max(ac,bc)<.01


def test_me054_rejects_leaky_or_negative_FIT_delta_scaling():
    with pytest.raises(ValueError):
        StableRealTAC([-1]*14)
    with pytest.raises(ValueError):
        StableRealGRU([.5]*13)
