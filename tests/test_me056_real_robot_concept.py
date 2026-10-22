"""ME056 real-robot numerical contracts, NOT robot performance evidence."""
import torch
from tac_osm.real_robot_me056_concept import (
    RealRobotConceptCED, MatchedStableRobotGRU, count_active, nearest_real_gru_width)


def case():
    torch.manual_seed(5601)
    scale=torch.ones(14)*.03
    m=RealRobotConceptCED(scale)
    x=torch.randn(2,14)
    a=torch.randn(2,3,14)
    h=torch.tensor([[False],[True]])
    return m,x,a,h


def test_physical_identity_prior_and_recurrent_concept_shape():
    m,x,a,h=case()
    z=m.observe(m.initial(2),x)
    pred=m.predict(z,x,a,h)
    assert z.concepts.shape==(2,6,15)
    assert pred.next_joint_state.shape==(2,3,14)
    assert pred.takeover_logits.shape==(2,3,1)
    torch.testing.assert_close(pred.next_joint_state,x.unsqueeze(1).expand(-1,3,-1),atol=0,rtol=0)
    trace=m.predict_trace(z,x,a,h)
    assert trace.step_logits.shape==(4,2,3,1)
    assert trace.step_deltas.shape==(4,2,3)


def test_action_reordering_and_no_unverified_memory_write():
    m,x,a,h=case()
    z=m.observe(m.initial(2),x)
    saved=[t.clone() for t in (*z.core,z.concepts)]
    pred=m.predict(z,x,a,h)
    order=[2,0,1]
    q=m.predict(z,x,a[:,order],h)
    torch.testing.assert_close(pred.takeover_logits[:,order],q.takeover_logits,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(pred.next_joint_state[:,order],q.next_joint_state,atol=1e-7,rtol=1e-6)
    assert all(torch.equal(old,new) for old,new in zip(saved,(*z.core,z.concepts)))


def test_independent_feedback_masks_and_zero_error_identity():
    m,x,a,h=case()
    z=m.observe(m.initial(2),x)
    pred=m.predict(z,x,a[:,:1],h)
    missing=torch.zeros(2,1,dtype=torch.bool)
    future=torch.full((2,14),float('nan'))
    observed=torch.full((2,1),float('nan'))
    missing_z=m.verify_real_feedback(z,a[:,0],h,future,pred.next_joint_state[:,0],
          missing,observed,pred.takeover_logits[:,0],missing)
    assert all(torch.equal(old,new) for old,new in zip((*z.core,z.concepts),
               (*missing_z.core,missing_z.concepts)))
    same=m.verify_real_feedback(z,a[:,0],h,pred.next_joint_state[:,0].clone(),
          pred.next_joint_state[:,0],torch.ones(2,1,dtype=torch.bool),
          observed,pred.takeover_logits[:,0],missing)
    assert all(torch.equal(old,new) for old,new in zip((*z.core,z.concepts),
               (*same.core,same.concepts)))


def test_real_physical_surprise_updates_only_available_batch_rows():
    m,x,a,h=case()
    z=m.observe(m.initial(2),x)
    p=m.predict(z,x,a[:,:1],h)
    future=p.next_joint_state[:,0].clone()
    future[0] += .04
    future[1] = float('nan')
    mask=torch.tensor([[True],[False]])
    miss=torch.zeros(2,1,dtype=torch.bool)
    after=m.verify_real_feedback(z,a[:,0],h,future,p.next_joint_state[:,0],
                                 mask,torch.full((2,1),float('nan')),
                                 p.takeover_logits[:,0],miss)
    assert not torch.equal(after.concepts[0],z.concepts[0])
    assert torch.equal(after.concepts[1],z.concepts[1])
    for old,new in zip(z.core,after.core):
        assert torch.equal(old[1],new[1])
        assert torch.isfinite(new).all()


def test_actual_physical_prediction_gradient_reaches_concept_decoder():
    m,x,a,h=case()
    z=m.observe(m.initial(2),x)
    y=m.predict(z,x,a[:,:1],h)
    target=x+.11
    onset=torch.zeros(2,1)
    mask=~h
    losses=m.genuine_logged_losses(y,target,onset,mask)
    normalized=((y.next_joint_state[:,0]-target)/m.factual_delta_scale).square().mean()
    loss=normalized+.2*losses["masked_real_takeover_bce"]
    loss.backward()
    for module in (m.swarm.decode_propose,m.swarm.decode_gate,
                   m.swarm.outcome_delta,m.swarm.next_delta):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and
                   p.grad.abs().sum()>0 for p in module.parameters())


def test_substantive_gru_capacity_survey_uses_all_trainable_weights():
    m,x,a,h=case()
    scale=m.factual_delta_scale.detach()
    size=nearest_real_gru_width(scale)
    assert size["gru_width"]>=40
    assert size["native_tac"]==count_active(m)
    assert size["gru_world"]==count_active(MatchedStableRobotGRU(scale,size["gru_width"]))
    assert size["relative_gap"] < .02, size
    g=MatchedStableRobotGRU(scale,size["gru_width"])
    p=g.predict(g.initial(2),x,a,h)
    torch.testing.assert_close(p.next_joint_state,x.unsqueeze(1).expand(-1,3,-1),atol=0,rtol=0)
