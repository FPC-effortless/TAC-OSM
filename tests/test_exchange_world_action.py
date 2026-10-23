"""ES-WAM engineering verification: invariants are NOT learned intelligence evidence."""
import torch
import pytest
from tac_osm.variants.exchange_world_action import ExchangeMemory, ExchangeWorldActionModel


def test_exchange_operators_obey_symmetry_and_antisymmetry():
    torch.manual_seed(112)
    memory = ExchangeMemory(16, 8)
    u, v = torch.randn(5, 16), torch.randn(5, 16)
    symmetric, antisymmetric = memory.exchange(u, v)
    flipped_s, flipped_a = memory.exchange(v, u)
    torch.testing.assert_close(symmetric, flipped_s, rtol=0, atol=1e-7)
    torch.testing.assert_close(antisymmetric, -flipped_a, rtol=0, atol=1e-7)
    torch.testing.assert_close(memory.exchange(u, u)[1], torch.zeros_like(u), rtol=0, atol=1e-7)


def test_same_identity_different_roles_preserve_observation_and_prediction():
    bank = ExchangeMemory(16, 2)
    original = bank.initial(1)
    ids = torch.tensor([27], dtype=torch.long)
    key, observed, predicted = torch.randn(1,16), torch.randn(1,16), torch.randn(1,16)
    memory, yes = bank.write(original, ids, 0, key, observed, torch.tensor([True]))
    assert yes.tolist() == [True]
    memory, yes = bank.write(memory, ids, 1, key, predicted, torch.tensor([True]))
    assert yes.tolist() == [True]
    got_obs, obs_hit = bank.retrieve(memory, key, 0, requested_ids=ids)
    got_pred, pred_hit = bank.retrieve(memory, key, 1, requested_ids=ids)
    assert obs_hit.tolist() == [True] and pred_hit.tolist() == [True]
    torch.testing.assert_close(got_obs, observed, rtol=0, atol=1e-6)
    torch.testing.assert_close(got_pred, predicted, rtol=0, atol=1e-6)
    assert not torch.equal(got_obs, got_pred)
    assert not bool(memory.occupied[0,0,2])  # never silently promote to verified


def test_collisions_fail_closed_without_eviction_or_wrong_identity_retrieval():
    bank = ExchangeMemory(16, 1)
    old = bank.initial(1)
    key = torch.randn(1,16)
    old, ok = bank.write(old, torch.tensor([5]), 0, key, torch.ones(1,16), torch.tensor([True]))
    later, ok = bank.write(old, torch.tensor([6]), 0, key, torch.zeros(1,16), torch.tensor([True]))
    assert not bool(ok.item())
    assert torch.equal(old.ids, later.ids)
    assert torch.equal(old.keys, later.keys)
    assert torch.equal(old.values, later.values)
    assert torch.equal(old.occupied, later.occupied)
    missed, hit = bank.retrieve(later, key, 0, requested_ids=torch.tensor([6]))
    assert not bool(hit.item())
    assert torch.equal(missed, torch.zeros_like(missed))


def test_readonly_missing_and_invalid_writes_are_rejected():
    bank = ExchangeMemory(16, 2)
    base = bank.initial(2)
    values = torch.randn(2,16)
    kept, admitted = bank.write(base, torch.tensor([1,2]), 0, values, values,
                                torch.tensor([False,False]))
    assert not admitted.any()
    assert torch.equal(base.values, kept.values)
    assert (kept.ids == -1).all()
    with pytest.raises(ValueError):
        bank.write(base, torch.tensor([-1,2]), 0, values, values, torch.ones(2,dtype=torch.bool))
    with pytest.raises(ValueError):
        bank.write(base, torch.tensor([1,2]), 4, values, values, torch.ones(2,dtype=torch.bool))


def test_model_is_in_required_budget_and_actions_change_futures():
    torch.manual_seed(754)
    model = ExchangeWorldActionModel()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed = sum(p.numel() for p in model.parameters())
    assert count == deployed == 100343
    assert 95000 <= count <= 105000
    x = torch.randn(2,14)
    mask = torch.ones_like(x, dtype=torch.bool)
    dt = torch.full((2,), 1/15.)
    ids = torch.tensor([25,35])
    a = torch.randn_like(x)
    result1 = model(x,a,dt,ids,mask)
    result2 = model(x,-a,dt,ids,mask)
    assert result1.mean_next.shape == x.shape
    assert result1.scale_next.shape == x.shape
    assert (result1.scale_next > 0).all()
    assert result1.proposed_action.shape == a.shape
    assert not torch.equal(result1.mean_next, result2.mean_next)
    assert not torch.equal(result1.proposed_action, result2.proposed_action)
    assert result1.observation_written.all()
    assert result1.prediction_written.all()
    assert not result1.previous_prediction_present.any()
    assert not result1.state.memory.occupied[:,:,2].any()


def test_missing_observation_never_becomes_verified_fact():
    model = ExchangeWorldActionModel()
    all_missing = torch.full((1,14), float("nan"))
    result = model(all_missing, torch.zeros(1,14), torch.zeros(1),
                   torch.tensor([10]), torch.zeros(1,14,dtype=torch.bool))
    assert not bool(result.observation_written.item())
    assert bool(result.prediction_written.item())
    assert not bool(result.state.memory.occupied[0,0,0])
    assert bool(result.state.memory.occupied[0,0,1])
    assert torch.isfinite(result.mean_next).all()
    assert torch.isfinite(result.scale_next).all()


def test_recurrent_prediction_is_separate_from_verified_state_and_batched_entities():
    torch.manual_seed(151)
    model = ExchangeWorldActionModel()
    x = torch.randn(2,14)
    mask = torch.ones_like(x,dtype=torch.bool)
    dt = torch.ones(2)
    ids = torch.tensor([111,222])
    a = torch.randn_like(x)
    first = model(x,a,dt,ids,mask)
    before = first.state.memory.values.detach().clone()
    second = model(x+0.01,a,dt,ids,mask,state=first.state)
    assert second.previous_prediction_present.tolist() == [True,True]
    assert torch.equal(first.state.memory.values.detach(), before)
    assert first.state.memory.ids.tolist() == [[111]+[-1]*7,[222]+[-1]*7]
    assert second.state.memory.ids.tolist() == [[111]+[-1]*7,[222]+[-1]*7]
    assert not second.state.memory.occupied[:,:,2].any()


def test_full_model_gradients_reach_exchange_and_policy_paths():
    torch.manual_seed(133)
    model = ExchangeWorldActionModel()
    x = torch.randn(2,14)
    dt = torch.ones(2)
    ids = torch.tensor([44,55])
    mask = torch.ones_like(x,dtype=torch.bool)
    action = torch.randn_like(x)
    one = model(x,action,dt,ids,mask)
    two = model(x+0.03,action,dt,ids,mask,state=one.state)
    loss = ((two.mean_next-torch.randn_like(x))**2).mean()
    loss = loss + two.scale_next.mean() + two.proposed_action.square().mean()
    loss.backward()
    grads = {name:p.grad for name,p in model.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values()), [
        name for name,g in grads.items() if g is None]
    assert grads["memory.exchange_map.weight"].abs().sum().item() > 0
    assert grads["memory.merge_gate.weight"].abs().sum().item() > 0
    assert grads["predicted_code.weight"].abs().sum().item() > 0
    assert grads["action_head.weight"].abs().sum().item() > 0
