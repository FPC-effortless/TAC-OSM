"""ESWAM003 software invariants only: no model intelligence inferred from fixtures."""
import torch
import pytest

from tac_osm.variants.selective_exchange_world import SelectiveExchangeWorldModel


def make_inputs(b=2):
    return (torch.randn(b,14), torch.randn(b,14),
            torch.full((b,),1.0/15), torch.arange(1,b+1,dtype=torch.long),
            torch.ones(b,14,dtype=torch.bool))


def test_new_model_counts_every_active_weight_without_padding():
    model=SelectiveExchangeWorldModel()
    total=sum(p.numel() for p in model.parameters())
    active=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total==active==101226
    assert 95000<=active<=105000


def test_no_future_motion_or_outcome_is_required_as_a_model_input():
    torch.manual_seed(93)
    model=SelectiveExchangeWorldModel()
    x,a,dt,ids,mask=make_inputs()
    p=model(x,a,dt,ids,mask)
    assert p.mean_next.shape==x.shape
    assert p.proposed_action.shape==a.shape
    assert p.movement_logit.shape==(2,)
    assert p.movement_probability.shape==(2,)
    assert torch.isfinite(p.mean_next).all()
    assert torch.isfinite(p.scale_next).all()
    assert p.movement_probability.min()>=0 and p.movement_probability.max()<=1
    assert not p.state.world.memory.occupied[:,:,2].any()


def test_no_unobserved_velocity_and_same_known_action_anchor():
    torch.manual_seed(311)
    model=SelectiveExchangeWorldModel()
    x,a,dt,ids,mask=make_inputs()
    start=model(x,a,dt,ids,mask)
    # The very first frame has no historical measured displacement.
    # Its action proposal is anchored to an *already known* recorded action.
    assert ((start.proposed_action-a).abs()<0.10).all()
    second=model(x+0.02,a,dt,ids,mask,state=start.state)
    assert second.previous_prediction_present.tolist()==[True,True]
    assert second.state.world.memory.ids[:,0].tolist()==ids.tolist()
    assert not second.state.world.memory.occupied[:,:,2].any()


def test_all_missing_frame_never_creates_observed_fact_or_artificial_velocity():
    torch.manual_seed(500)
    model=SelectiveExchangeWorldModel()
    x,a,dt,ids,mask=make_inputs(1)
    first=model(x,a,dt,ids,mask)
    not_observed=torch.full_like(x,float("nan"))
    later=model(not_observed,a,dt,ids,torch.zeros_like(mask),state=first.state)
    assert not later.observation_written.item()
    assert torch.isfinite(later.mean_next).all()
    assert torch.isfinite(later.movement_logit).all()
    torch.testing.assert_close(later.state.last_observation,x,rtol=0,atol=0)
    assert later.state.last_observed_mask.all()


def test_action_correction_is_bounded_and_gradients_are_real():
    torch.manual_seed(99)
    model=SelectiveExchangeWorldModel()
    x,a,dt,ids,mask=make_inputs()
    prior=model(x-0.03,a,dt,ids,mask)
    pred=model(x,a,dt,ids,mask,state=prior.state)
    observed_next=x+0.08
    target_action=a+0.003
    observed_motion=torch.tensor([True,False])
    loss=((pred.mean_next-observed_next)**2).mean()
    loss+=torch.nn.functional.binary_cross_entropy_with_logits(
        pred.movement_logit,observed_motion.float())
    loss+=torch.nn.functional.smooth_l1_loss(pred.proposed_action,target_action)
    loss.backward()
    assert ((pred.proposed_action-a).abs()<=0.10).all()
    for key in ("motion_head.0.weight","shrink_log_scale","velocity_log_gain",
                "action_step_logit","memory.exchange_map.weight",
                "memory.merge_gate.weight","next_heads.weight",
                "action_head.weight"):
        parameter=dict(model.named_parameters())[key]
        assert parameter.grad is not None,key
        assert torch.isfinite(parameter.grad).all(),key
    assert model.motion_head[0].weight.grad.abs().sum().item()>0


def test_rejects_bad_previous_state_and_invalid_source_actions():
    model=SelectiveExchangeWorldModel()
    x,a,dt,ids,mask=make_inputs(1)
    with pytest.raises(ValueError):
        model(x,torch.full_like(a,float("nan")),dt,ids,mask)
    with pytest.raises(ValueError):
        model(x,a,dt,torch.tensor([-1]),mask)
