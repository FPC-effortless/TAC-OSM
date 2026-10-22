"""Software-only numerical behavior. Fake tensors here NEVER count as model skill."""
import pytest
import torch

from tac_osm.real_robot_predictive_state import (
    RealRobotPredictiveState,RobotForecast,DOF)


def make_inputs(n=3):
    torch.manual_seed(52070)
    obs=torch.randn(n,14)*.2
    act=torch.randn(n,14)*.1
    human=torch.tensor([[False],[True],[False]])[:n]
    return obs,act,human


def test_integrated_agent_queries_actual_action_without_mutating_native_belief():
    m=RealRobotPredictiveState()
    observation,action,known=make_inputs()
    original=m.initial(3)
    state=m.observe(original,observation)
    query=m.predict(state,observation,action[:,None,:],known)
    assert query.next_joint_state.shape==(3,1,14)
    assert query.takeover_logits.shape==(3,1,1)
    assert torch.isfinite(query.next_joint_state).all()
    state_copy=tuple(t.clone() for t in state)
    alternate=m.predict(state,observation,torch.stack((action,action*.5),dim=1),known)
    assert alternate.next_joint_state.shape==(3,2,14)
    assert all(torch.equal(v,o) for v,o in zip(state,state_copy))


def test_missing_physical_and_takeover_labels_cannot_change_any_state_even_with_NaN():
    m=RealRobotPredictiveState()
    obs,act,known=make_inputs()
    belief=m.observe(m.initial(3),obs)
    unknown=torch.full((3,14),float("nan"))
    missing=torch.full((3,1),float("nan"))
    corrected=m.verify_real_feedback(
        belief,act,known,unknown,unknown,
        torch.zeros(3,1,dtype=torch.bool),missing,missing,
        torch.zeros(3,1,dtype=torch.bool))
    assert all(torch.equal(a,b) for a,b in zip(belief,corrected))


def test_real_physical_feedback_updates_state_without_inventing_takeover():
    m=RealRobotPredictiveState()
    obs,act,known=make_inputs()
    s=m.observe(m.initial(3),obs)
    q=m.predict(s,obs,act[:,None,:],known)
    # No actual operator takeover labels: only genuinely measured physical
    # next-state may correct the dynamic belief.
    updated=m.verify_real_feedback(s,act,known,obs+.02,
        q.next_joint_state[:,0],torch.ones(3,1,dtype=torch.bool),
        torch.full((3,1),float("nan")),torch.full((3,1),float("nan")),
        torch.zeros(3,1,dtype=torch.bool))
    assert torch.isfinite(updated.memory).all()
    assert any(not torch.equal(a,b) for a,b in zip(s,updated))


def test_takeover_is_masked_under_existing_human_control_and_binary_if_available():
    m=RealRobotPredictiveState()
    obs,act,known=make_inputs()
    s=m.observe(m.initial(3),obs)
    with pytest.raises(ValueError,match="already human-controlled"):
        m.verify_real_feedback(s,act,known,obs,obs,torch.ones(3,1,dtype=torch.bool),
            torch.zeros(3,1),torch.zeros(3,1),torch.ones(3,1,dtype=torch.bool))
    avail=~known
    s2=m.verify_real_feedback(s,act,known,
        torch.full((3,14),float("nan")),torch.full((3,14),float("nan")),
        torch.zeros(3,1,dtype=torch.bool),
        torch.tensor([[1.],[float("nan")],[0.]]),torch.zeros(3,1),avail)
    assert torch.isfinite(s2.fast).all()
    assert torch.equal(s2.fast[1],s.fast[1])
    with pytest.raises(ValueError,match="binary"):
        m.verify_real_feedback(s,act,known,obs,obs,
            torch.zeros(3,1,dtype=torch.bool),
            torch.tensor([[2.],[float("nan")],[0.]]),torch.zeros(3,1),avail)


def test_factual_next_state_and_separately_masked_onset_have_finite_training_gradients():
    m=RealRobotPredictiveState()
    obs,act,known=make_inputs()
    state=m.observe(m.initial(3),obs)
    forecast=m.predict(state,obs,act[:,None,:],known)
    factual_next=obs+.01
    flags=torch.tensor([[1.],[float("nan")],[0.]])
    avail=~known
    losses=m.genuine_logged_losses(forecast,factual_next,flags,avail)
    assert losses["actually_supervised_takeover_count"]==2
    # These tensors are NOT real data; check only that software can backprop.
    total=losses["observed_physical_mse"]+losses["masked_real_takeover_bce"]
    total.backward()
    assert torch.isfinite(total)
    assert m.native.next_observation.weight.grad is not None
    assert m.native.readout[-1].weight.grad is not None
    assert m.native.observation.weight.grad is not None


def test_dual_feedback_correction_is_bounded_and_learnable_through_subsequent_real_step():
    m=RealRobotPredictiveState()
    obs,act,known=make_inputs()
    before=m.observe(m.initial(3),obs)
    q=m.predict(before,obs,act[:,None,:],known)
    updated=m.verify_real_feedback(before,act,known,obs+.1,
        q.next_joint_state[:,0],torch.ones(3,1,dtype=torch.bool),
        torch.tensor([[0.],[float("nan")],[1.]]),
        q.takeover_logits[:,0],~known)
    assert torch.max(torch.abs(updated.fast-before.fast))<=.5000001
    assert torch.max(torch.abs(updated.slow-before.slow))<=.5000001
    next_state=m.observe(updated,obs+.1)
    post=m.predict(next_state,obs+.1,act[:,None,:],known)
    # Gradient requirement for *integrated* future predictive capability,
    # not an evaluation of how well the model learned anything.
    loss=post.next_joint_state.square().mean()+post.takeover_logits.square().mean()
    loss.backward()
    for p in (m.physical_map.weight,m.physical_key.weight,
              m.physical_value.weight,m.native.verify_map.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all()
