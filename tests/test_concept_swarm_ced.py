"""Numerical/integration checks on random tensors; NOT evaluation of intelligence."""
import torch
import pytest
from tac_osm.concept_swarm_ced import ConceptSwarmCED


def fixture():
    torch.manual_seed(311)
    m = ConceptSwarmCED(5, 3, outcome_size=1, concept_count=6,
                        concept_dim=16, refinement_steps=4)
    x = torch.randn(3,5)
    a = torch.randn(3,4,3)
    return m, x, a


def test_observation_persists_both_native_and_concept_state():
    m,x,_=fixture()
    old=m.initial(3)
    state=m.observe(old,x)
    assert old.concepts.count_nonzero()==0
    assert not torch.equal(state.concepts,old.concepts)
    assert not torch.equal(state.core.fast,old.core.fast)
    assert state.concepts.shape==(3,6,16)
    assert all(torch.isfinite(z).all() for z in (*state.core,state.concepts))
    later=m.observe(state,x+.5)
    assert not torch.equal(later.concepts,state.concepts)


def test_parallel_action_queries_are_permutation_equivariant_and_read_only():
    m,x,a=fixture(); state=m.observe(m.initial(3),x)
    before=[z.clone() for z in (*state.core,state.concepts)]
    p=m.predict(state,x,a)
    assert p.outcome_logits.shape==(3,4,1)
    assert p.next_observation.shape==(3,4,5)
    assert p.future_concepts.shape==(3,4,6,16)
    assert p.policy_scores.shape==(3,4)
    assert p.information_scores.shape==(3,4)
    assert p.step_logits.shape==(4,3,4,1)
    assert p.step_deltas.shape==(4,3,4)
    assert all(torch.equal(z,old) for z,old in zip((*state.core,state.concepts),before))
    idx=[2,0,3,1]
    pp=m.predict(state,x,a[:,idx,:])
    for v,w in [(p.outcome_logits,pp.outcome_logits),(p.next_observation,pp.next_observation),
                (p.future_concepts,pp.future_concepts),(p.policy_scores,pp.policy_scores),
                (p.information_scores,pp.information_scores)]:
        torch.testing.assert_close(v[:,idx],w,atol=0,rtol=0)


def test_recurrent_refinement_moves_latents_with_shared_weights():
    m,x,a=fixture(); state=m.observe(m.initial(3),x)
    p=m.predict(state,x,a)
    assert torch.isfinite(p.step_deltas).all()
    assert p.step_deltas.min()>0
    assert not torch.equal(p.step_logits[0],p.step_logits[-1])
    assert torch.equal(p.step_logits[-1],p.outcome_logits)


def test_verification_changes_only_observed_entities_with_bounded_updates():
    m,x,a=fixture(); state=m.observe(m.initial(3),x)
    p=m.predict(state,x,a)
    mask=torch.tensor([[True],[False],[True]])
    outcome=torch.tensor([[1.0],[float('nan')],[0.0]])
    prior=p.outcome_logits[:,0].sigmoid().detach()
    new=m.verify(state,a[:,0],outcome,prior,mask)
    assert torch.equal(new.concepts[1],state.concepts[1])
    for before,after in zip(state.core,new.core):
        assert torch.equal(before[1],after[1])
    assert (new.concepts-state.concepts).abs().max() <= .125
    assert not torch.equal(new.concepts[0],state.concepts[0])


def test_unobserved_feedback_is_exact_identity_even_for_nan_inputs():
    m,x,a=fixture(); state=m.observe(m.initial(3),x)
    missing=torch.zeros(3,1,dtype=torch.bool)
    after=m.verify(state,a[:,0],torch.full((3,1),float('nan')),
                   torch.full((3,1),float('nan')),missing)
    for before,new in zip((*state.core,state.concepts),(*after.core,after.concepts)):
        assert torch.equal(before,new)


def test_gradients_reach_native_encoder_memory_and_swarm_decoder():
    m,x,a=fixture()
    state=m.observe(m.initial(3),x)
    p=m.predict(state,x,a)
    future=m.observe(state,x+.2)
    v=m.verify(future,a[:,0],torch.ones(3,1),p.outcome_logits[:,0].sigmoid(),torch.ones(3,1,dtype=torch.bool))
    loss=(p.outcome_logits.square().mean()+p.next_observation.square().mean()
          +p.future_concepts.square().mean()+p.policy_scores.square().mean()
          +p.information_scores.square().mean()+v.concepts.square().mean()
          +sum(z.square().mean() for z in v.core))
    loss.backward()
    for name in ('anchors','rel_bias'):
        grad=getattr(m,name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
    for name in ('decode_propose','q','k','v','future_map','policy_head','information_head','verify_delta','observe_propose'):
        module=getattr(m,name)
        grad_sum=sum(q.grad.abs().sum() for q in module.parameters() if q.grad is not None)
        assert grad_sum>0,name
    for name in ('transition','write','readout','verify_map','next_observation'):
        module=getattr(m.core,name)
        grad_sum=sum(q.grad.abs().sum() for q in module.parameters() if q.grad is not None)
        assert grad_sum>0,name


def test_entity_batch_separation():
    m,x,a=fixture()
    full=m.observe(m.initial(3),x); p=m.predict(full,x,a)
    for i in range(3):
        solo=m.observe(m.initial(1),x[i:i+1]); solo_p=m.predict(solo,x[i:i+1],a[i:i+1])
        for left,right in ((full.concepts,solo.concepts),(p.outcome_logits,solo_p.outcome_logits),
                           (p.future_concepts,solo_p.future_concepts)):
            torch.testing.assert_close(left[i:i+1],right,rtol=1e-5,atol=1e-6)


def test_invalid_inputs_are_rejected():
    m,x,a=fixture(); state=m.initial(3)
    with pytest.raises(ValueError):m.observe(state,x[:,:-1])
    with pytest.raises(ValueError):m.predict(state,x,a[:,:,:-1])
    with pytest.raises(ValueError):m.verify(state,a[:,0],torch.ones(3,1),torch.zeros(3,1),torch.ones(3,1))
    with pytest.raises(ValueError):m.verify(state,a[:,0],torch.full((3,1),float('nan')),torch.zeros(3,1),torch.ones(3,1,dtype=torch.bool))
    with pytest.raises(ValueError):ConceptSwarmCED(5,3,concept_count=1)
    with pytest.raises(ValueError):ConceptSwarmCED(5,3,refinement_steps=0)


def test_long_stream_stays_finite_without_claiming_retention():
    m,x,a=fixture()
    with torch.no_grad():
        s=m.initial(3)
        for j in range(130):
            s=m.observe(s,x + (j%5)*.02)
            p=m.predict(s,x,a)
            if j%9==0:
                s=m.verify(s,a[:,0],torch.zeros(3,1),p.outcome_logits[:,0].sigmoid(),torch.ones(3,1,dtype=torch.bool))
        assert all(torch.isfinite(z).all() for z in (*s.core,s.concepts))


def test_deterministic_roundtrip_untrained_only(tmp_path):
    m,x,a=fixture(); state=m.observe(m.initial(3),x)
    target=m.predict(state,x,a)
    dest=tmp_path/'model.pt';torch.save(m.state_dict(),dest)
    other=ConceptSwarmCED(5,3)
    other.load_state_dict(torch.load(dest,weights_only=True))
    s2=other.observe(other.initial(3),x)
    actual=other.predict(s2,x,a)
    for y,z in zip(target,actual):torch.testing.assert_close(y,z,rtol=0,atol=0)
