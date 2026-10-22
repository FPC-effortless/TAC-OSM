"""SOFTWARE fixtures only; not artificial physical-task performance evidence."""
import pytest
import torch
from tac_osm.real_robot_gru_world_control import GRURobotWorldModel
from tac_osm.real_robot_predictive_state import RealRobotPredictiveState


def sample():
    torch.manual_seed(52117)
    observation=torch.randn(3,14)*.01
    action=torch.randn(3,14)*.02
    authority=torch.tensor([[False],[True],[False]])
    return observation,action,authority


def test_established_recurrent_control_has_equal_information_and_real_outcome_shapes():
    a,u,auth=sample()
    model=GRURobotWorldModel(40)
    z=model.observe(model.initial(3),a)
    q=model.predict(z,a,u[:,None],auth)
    assert q.next_joint_state.shape==(3,1,14)
    assert q.takeover_logits.shape==(3,1,1)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)>0
    assert torch.isfinite(q.next_joint_state).all()
    both=model.predict(z,a,torch.stack((u,u*2),1),auth)
    assert both.next_joint_state.shape==(3,2,14)
    assert not torch.equal(both.next_joint_state[:,0],both.next_joint_state[:,1])


def test_rival_queries_do_not_mutate_recurrent_belief_and_missing_feedback_is_identity():
    a,u,auth=sample()
    m=GRURobotWorldModel(40)
    z=m.observe(m.initial(3),a)
    prior=z.clone()
    m.predict(z,a,u[:,None],auth)
    assert torch.equal(z,prior)
    null14=torch.full_like(a,float("nan"))
    null1=torch.full((3,1),float("nan"))
    missing=torch.zeros(3,1,dtype=torch.bool)
    unchanged=m.verify_real_feedback(z,u,auth,null14,null14,missing,
                                      null1,null1,missing)
    assert torch.equal(z,unchanged)


def test_rival_masks_current_human_authority_and_future_only_takeover():
    a,u,auth=sample()
    m=GRURobotWorldModel(40)
    z=m.observe(m.initial(3),a)
    q=m.predict(z,a,u[:,None],auth)
    with pytest.raises(ValueError,match="undefined"):
        m.verify_real_feedback(z,u,auth,a,a,torch.ones(3,1,dtype=torch.bool),
                               torch.zeros(3,1),q.takeover_logits[:,0],
                               torch.ones(3,1,dtype=torch.bool))
    available=~auth
    recovered=m.verify_real_feedback(z,u,auth,
        a+.1,q.next_joint_state[:,0],torch.ones(3,1,dtype=torch.bool),
        torch.tensor([[1.],[float("nan")],[0.]]),
        q.takeover_logits[:,0],available)
    assert recovered.shape==z.shape
    assert torch.isfinite(recovered).all()


def test_rival_factual_losses_and_full_feedback_are_differentiable():
    a,u,auth=sample()
    m=GRURobotWorldModel(40)
    z=m.observe(m.initial(3),a)
    p=m.predict(z,a,u[:,None],auth)
    flags=torch.tensor([[1.],[float("nan")],[0.]])
    target=a+.01
    availability=~auth
    losses=m.genuine_logged_losses(p,target,flags,availability)
    assert losses["actually_supervised_takeover_count"]==2
    updated=m.verify_real_feedback(z,u,auth,target,p.next_joint_state[:,0],
       torch.ones(3,1,dtype=torch.bool),flags,p.takeover_logits[:,0],availability)
    next_state=m.observe(updated,target)
    later=m.predict(next_state,target,u[:,None],auth)
    total=(losses["observed_physical_mse"]+losses["masked_real_takeover_bce"]
        +later.next_joint_state.square().mean()+later.takeover_logits.square().mean())
    total.backward()
    assert torch.isfinite(total)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.requires_grad)


def test_prefit_resource_surveys_cannot_claim_matched_training_from_parameter_count():
    from scripts.profile_real_robot_world_models_prefit import (
        construct_models,nparameters,FRAMES,GRAD_STEPS)
    native,control,census,closest=construct_models()
    assert len(census)>10
    assert FRAMES==256 and GRAD_STEPS==16
    assert nparameters(control)==closest["active_parameters"]
    assert isinstance(closest["relative_gap"],float)
    assert nparameters(native)>0
