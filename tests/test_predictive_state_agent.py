"""Synthetic numerical contracts only; no intelligence or real-data evidence."""
import torch
import pytest
from tac_osm.predictive_state_agent import PredictiveStateAgent, Belief


def case():
    torch.manual_seed(52)
    model = PredictiveStateAgent(5,3)
    context = torch.randn(2,5)
    actions = torch.randn(2,4,3)
    return model, context, actions


def test_observation_updates_native_state_without_mutation():
    m,x,a = case()
    old = m.initial(2)
    state = m.observe(old,x)
    assert all(torch.count_nonzero(t) == 0 for t in old)
    assert all(torch.isfinite(t).all() for t in state)
    assert state.fast.abs().max() <= 1 and state.slow.abs().max() <= 1
    assert state.memory.shape == (2,8,8)


def test_candidates_are_queries_and_never_state_writes():
    m,x,a = case()
    state = m.observe(m.initial(2),x)
    before = [t.clone() for t in state]
    logits, transitions = m.predict(state,x,a)
    assert logits.shape == (2,4,1) and transitions.shape == (2,4,5)
    assert all(torch.equal(p,q) for p,q in zip(before,state))
    perm = [3,1,0,2]
    other,_ = m.predict(state,x,a[:,perm])
    torch.testing.assert_close(logits[:,perm],other,rtol=0,atol=0)


def test_feedback_is_not_a_prediction_argument():
    m,x,a = case()
    state = m.observe(m.initial(2),x)
    guess,_ = m.predict(state,x,a)
    after = m.verify(state,a[:,0],torch.ones(2,1),guess[:,0].sigmoid(),torch.ones(2,1,dtype=torch.bool))
    repeated,_ = m.predict(state,x,a)
    assert torch.equal(guess,repeated)
    assert not torch.equal(after.fast,state.fast)


def test_missing_outcomes_produce_exact_identity_even_with_nan_labels():
    m,x,a = case()
    state = m.observe(m.initial(2),x)
    after = m.verify(state,a[:,0],torch.full((2,1),float('nan')),
                     torch.full((2,1),float('nan')),torch.zeros(2,1,dtype=torch.bool))
    assert all(torch.equal(p,q) for p,q in zip(state,after))


def test_per_entity_mask_and_coordinate_correction_bound():
    m,x,a = case()
    state = m.observe(m.initial(2),x)
    after = m.verify(state,a[:,0],torch.tensor([[1e6],[float('nan')]]),
                     torch.zeros(2,1),torch.tensor([[True],[False]]))
    for p,q in zip(state,after):
        assert torch.equal(p[1],q[1])
    assert (after.fast-state.fast).abs().max() <= .25
    assert (after.slow-state.slow).abs().max() <= .25


def test_full_predict_verify_future_gradient_paths_are_finite():
    m,x,a = case()
    state = m.observe(m.initial(2),x)
    guess,transition = m.predict(state,x,a)
    state = m.verify(state,a[:,0],torch.ones(2,1),guess[:,0].sigmoid(),torch.ones(2,1,dtype=torch.bool))
    future = m.observe(state,x+.1)
    logits,next_x = m.predict(future,x+.1,a)
    loss = logits.square().mean()+next_x.square().mean()+guess.square().mean()+transition.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    for name in ('enc','slow','verify_map','verify_key','verify_value','readout','next_observation'):
        assert sum(p.grad.abs().sum() for p in getattr(m,name).parameters()) > 0


def test_entity_batching_does_not_cross_contaminate_state():
    m,x,a = case()
    batch = m.observe(m.initial(2),x)
    for i in range(2):
        alone = m.observe(m.initial(1),x[i:i+1])
        for p,q in zip(batch,alone):
            torch.testing.assert_close(p[i:i+1],q,rtol=1e-5,atol=1e-7)


@pytest.mark.parametrize('failure', ['context','candidates','state','mask','observed_nan'])
def test_invalid_contracts_fail(failure):
    m,x,a = case()
    state = m.initial(2)
    with pytest.raises(ValueError):
        if failure == 'context': m.observe(state,x[:,:4])
        elif failure == 'candidates': m.predict(state,x,a[:,:,:2])
        elif failure == 'state': m.observe(Belief(state.fast[:,:1],state.slow,state.memory),x)
        elif failure == 'mask': m.verify(state,a[:,0],torch.ones(2,1),torch.zeros(2,1),torch.ones(2,1))
        else: m.verify(state,a[:,0],torch.full((2,1),float('nan')),torch.zeros(2,1),torch.ones(2,1,dtype=torch.bool))


def test_long_numerical_stream_is_finite_not_a_retention_result():
    m,x,a = case()
    with torch.no_grad():
        state = m.initial(2)
        for _ in range(300):
            state = m.observe(state,x)
            logits,_ = m.predict(state,x,a)
            state = m.verify(state,a[:,0],torch.zeros(2,1),logits[:,0].sigmoid(),torch.ones(2,1,dtype=torch.bool))
        assert all(torch.isfinite(t).all() for t in state)


def test_untrained_weight_roundtrip_replays_exactly(tmp_path):
    m,x,a = case()
    state = m.observe(m.initial(2),x)
    expected = m.predict(state,x,a)
    torch.save(m.state_dict(),tmp_path/'untrained.pt')
    restored = PredictiveStateAgent(5,3)
    restored.load_state_dict(torch.load(tmp_path/'untrained.pt',weights_only=True))
    actual = restored.predict(restored.observe(restored.initial(2),x),x,a)
    assert all(torch.equal(p,q) for p,q in zip(expected,actual))


def test_nonfinite_available_prior_prediction_is_rejected():
    m,x,a = case()
    with pytest.raises(ValueError,match='preceding prediction'):
        m.verify(m.initial(2),a[:,0],torch.ones(2,1),torch.full((2,1),float('nan')),
                 torch.ones(2,1,dtype=torch.bool))
