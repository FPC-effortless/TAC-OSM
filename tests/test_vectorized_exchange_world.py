"""Original ES-WAM versus vectorized full-model values, slots and backward parity."""
import pytest
import torch
from tac_osm.variants.exchange_world_action import ExchangeMemory, ExchangeWorldActionModel
from tac_osm.variants.vectorized_exchange_world import (
    VectorizedExchangeMemory, VectorizedExchangeWorldActionModel,
)


def same(a, b, atol=1e-6, rtol=1e-5):
    assert a.shape == b.shape and a.dtype == b.dtype
    if a.dtype == torch.bool or not a.dtype.is_floating_point:
        assert torch.equal(a, b)
    else:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def same_memory(a, b):
    for field in ("ids", "keys", "values", "occupied"):
        same(getattr(a, field), getattr(b, field))


@pytest.mark.parametrize("batch", [1, 3, 16])
@pytest.mark.parametrize("role", [0, 1, 2])
def test_all_memory_roles_writes_retrieval_and_backward(batch, role):
    torch.manual_seed(700 + batch + role)
    old, new = ExchangeMemory(8, 5), VectorizedExchangeMemory(8, 5)
    new.load_state_dict(old.state_dict())
    state_old, state_new = old.initial(batch), new.initial(batch)
    for t in range(9):
        identity = torch.full((batch,), t if t < 6 else t % 4, dtype=torch.long)
        enabled = torch.tensor([(i + t) % 3 != 0 for i in range(batch)])
        key = torch.randn(batch, 8, requires_grad=True)
        value = torch.randn(batch, 8, requires_grad=True)
        state_old, ok1 = old.write(state_old, identity, role, key, value, enabled)
        state_new, ok2 = new.write(state_new, identity, role, key, value, enabled)
        same(ok1, ok2)
        same_memory(state_old, state_new)
        query = torch.randn(batch, 8)
        for requested in (None, identity, torch.full((batch,), 99, dtype=torch.long)):
            v1, h1 = old.retrieve(state_old, query, role, requested)
            v2, h2 = new.retrieve(state_new, query, role, requested)
            same(v1, v2)
            same(h1, h2)
    q1 = torch.randn(batch, 8, requires_grad=True)
    q2 = q1.detach().clone().requires_grad_(True)
    result_old, _ = old.retrieve(state_old, q1, role)
    result_new, _ = new.retrieve(state_new, q2, role)
    (result_old.square().sum() + state_old.values.square().sum()).backward()
    (result_new.square().sum() + state_new.values.square().sum()).backward()
    same(q1.grad, q2.grad)
    for (n1, p1), (n2, p2) in zip(old.named_parameters(), new.named_parameters()):
        assert n1 == n2
        if p1.grad is None or p2.grad is None:
            assert p1.grad is None and p2.grad is None
        else:
            same(p1.grad, p2.grad, 5e-6, 2e-5)


@pytest.mark.parametrize("batch", [1, 4, 12])
def test_complete_original_model_multistep_output_and_gradients(batch):
    torch.manual_seed(11000 + batch)
    orig = ExchangeWorldActionModel(dimension=7, action_dim=7, width=32, memory_width=8, slots=3)
    vect = VectorizedExchangeWorldActionModel(dimension=7, action_dim=7, width=32, memory_width=8, slots=3)
    vect.load_state_dict(orig.state_dict(), strict=True)
    assert sum(p.numel() for p in orig.parameters()) == sum(p.numel() for p in vect.parameters())
    state_orig, state_vect = orig.initial(batch), vect.initial(batch)
    loss_orig, loss_vect = 0.0, 0.0
    for t in range(6):
        obs = torch.randn(batch, 7)
        mask = torch.rand(batch, 7) > .25
        mask[0, :] = t % 2 == 0  # all-missing frame: no factual write
        action = torch.randn(batch, 7)
        dt = torch.full((batch,), .1 + t * .01)
        identity = torch.full((batch,), t // 2, dtype=torch.long)
        target = torch.randn(batch, 7)
        a = orig(obs, action, dt, identity, mask, state=state_orig)
        b = vect(obs, action, dt, identity, mask, state=state_vect)
        for name in ("mean_next", "scale_next", "proposed_action",
                     "previous_prediction_present", "observation_written",
                     "prediction_written"):
            same(getattr(a, name), getattr(b, name))
        same(a.state.fast, b.state.fast)
        same(a.state.slow, b.state.slow)
        same_memory(a.state.memory, b.state.memory)
        state_orig, state_vect = a.state, b.state
        loss_orig = loss_orig + (a.mean_next-target).square().mean() + .1*a.scale_next.mean() + .05*a.proposed_action.square().mean()
        loss_vect = loss_vect + (b.mean_next-target).square().mean() + .1*b.scale_next.mean() + .05*b.proposed_action.square().mean()
    loss_orig = loss_orig + .02*state_orig.memory.values.square().mean()
    loss_vect = loss_vect + .02*state_vect.memory.values.square().mean()
    loss_orig.backward()
    loss_vect.backward()
    for (n1, p1), (n2, p2) in zip(orig.named_parameters(), vect.named_parameters()):
        assert n1 == n2
        if p1.grad is None or p2.grad is None:
            assert p1.grad is None and p2.grad is None
        else:
            same(p1.grad, p2.grad, 5e-6, 2e-5)


def test_adversarial_capacity_and_identity_preservation():
    old, new = ExchangeMemory(4, 2), VectorizedExchangeMemory(4, 2)
    new.load_state_dict(old.state_dict())
    x, y = old.initial(2), new.initial(2)
    for target_id, enabled in [(11, True), (12, True), (13, True), (11, True), (14, False)]:
        ids = torch.tensor([target_id, target_id])
        allow = torch.tensor([enabled, enabled])
        k, v = torch.randn(2, 4), torch.randn(2, 4)
        x, ok_old = old.write(x, ids, 0, k, v, allow)
        y, ok_new = new.write(y, ids, 0, k, v, allow)
        same(ok_old, ok_new)
        same_memory(x, y)
    assert x.ids.tolist() == [[11, 12], [11, 12]]
