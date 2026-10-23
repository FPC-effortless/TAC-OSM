"""ESWAM004: software invariants, not model skill or task performance."""
import torch
import pytest
from tac_osm.variants.kinetic_exchange_world import KineticExchangeWorldModel


def inputs(batch=2):
    torch.manual_seed(4804)
    return (torch.randn(batch,14),torch.randn(batch,14),
            torch.full((batch,),1/15.),torch.arange(5,5+batch,dtype=torch.long),
            torch.ones(batch,14,dtype=torch.bool))


def test_parameter_budget_complete_and_nonpadded():
    model=KineticExchangeWorldModel()
    active=sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed=sum(p.numel() for p in model.parameters())
    assert (active,deployed)==(101706,101706)
    assert abs(active-101226)/101226<0.01


def test_zero_kinetic_evidence_does_not_leak_future_motion():
    model=KineticExchangeWorldModel()
    x,a,dt,ids,mask=inputs()
    first=model(x,a,dt,ids,mask)
    same=model(x,a,dt,ids,mask,state=first.state)
    # Exact same actually observed position and command has zero causal
    # kinetic evidence; prior learned prospective movement estimate remains.
    assert (same.kinetic_gate>=0).all() and (same.kinetic_gate<=1).all()
    torch.testing.assert_close(same.kinetic_gate,same.movement_probability.square(),atol=1e-6,rtol=1e-6)
    assert not same.state.prior.world.memory.occupied[:,:,2].any()


def test_observed_kinetic_change_opens_gate_without_future_labels():
    model=KineticExchangeWorldModel()
    x,a,dt,ids,mask=inputs()
    old=model(x,a,dt,ids,mask)
    new=model(x+0.1,a,dt,ids,mask,state=old.state)
    assert (new.kinetic_gate>=new.movement_probability.square()-1e-7).all()
    assert (new.state.kinetic_ema>0).all()
    assert new.state.prior.world.memory.ids[:,0].tolist()==ids.tolist()
    assert new.mean_next.shape==x.shape and new.proposed_action.shape==a.shape


def test_missing_observation_does_not_create_factual_evidence():
    model=KineticExchangeWorldModel()
    x,a,dt,ids,mask=inputs(1)
    first=model(x,a,dt,ids,mask)
    missing=model(torch.full_like(x,float("nan")),a,dt,ids,
                  torch.zeros_like(mask),state=first.state)
    assert not missing.observation_written.item()
    assert torch.isfinite(missing.mean_next).all()
    assert torch.isfinite(missing.movement_probability).all()
    assert not missing.state.prior.world.memory.occupied[:,:,2].any()


def test_joint_full_model_training_gradients_reach_kinetic_and_exchange():
    torch.manual_seed(360)
    model=KineticExchangeWorldModel()
    x,a,dt,ids,mask=inputs(2)
    old=model(x,a,dt,ids,mask)
    result=model(x+0.01,a+0.015,dt,ids,mask,state=old.state)
    loss=(result.mean_next-(x+0.03)).square().mean()
    loss+=torch.nn.functional.binary_cross_entropy_with_logits(
        result.movement_logit,torch.tensor([1.,0.]))
    loss+=result.proposed_action.square().mean()+result.scale_next.mean()
    loss.backward()
    required=("kinetic_event.0.weight","kinetic_event.2.weight",
              "log_kinetic_gain","prior_velocity_logits",
              "memory.exchange_map.weight","memory.merge_gate.weight",
              "fast_map.weight","action_head.weight")
    p=dict(model.named_parameters())
    assert all(p[k].grad is not None and torch.isfinite(p[k].grad).all() for k in required)
    assert p["kinetic_event.0.weight"].grad.abs().sum()>0
    assert p["prior_velocity_logits"].grad.abs().sum()>0


def test_untrusted_inputs_are_rejected():
    model=KineticExchangeWorldModel()
    x,a,dt,ids,mask=inputs(1)
    with pytest.raises(ValueError):
        model(x,a,dt,torch.tensor([-1]),mask)
    with pytest.raises(ValueError):
        model(x,torch.full_like(a,float("nan")),dt,ids,mask)
