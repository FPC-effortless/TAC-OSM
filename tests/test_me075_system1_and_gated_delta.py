"""ME075/early System1 engineering tests only; no artificial model-ability scores."""
import torch
import pytest
from torch.nn import functional as F
from tac_osm.parameter_budget import check_parameter_budget
from tac_osm.variants.system1_native import System1FastModel, physical_one_step_nll
from tac_osm.variants.gated_delta_memory import (
    GatedDelta8x8, delta_read, gated_delta_update
)


def test_100k_native_system1_count_and_true_all_active_gradients():
    torch.manual_seed(75)
    m = System1FastModel(dimension=14)
    counts = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert counts == 100541
    assert check_parameter_budget(counts, counts)
    x = torch.randn(3, 14)
    a = torch.randn(3, 14)
    forecast = m(x, a, torch.tensor([0., 1., 3600.]))
    assert forecast.mean_next.shape == x.shape
    assert forecast.scale_next.shape == x.shape
    assert forecast.event_logit.shape == (3,)
    assert torch.isfinite(forecast.mean_next).all()
    assert (forecast.scale_next > 0).all()
    loss = physical_one_step_nll(forecast, torch.randn_like(x), torch.tensor([True, True, False]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.requires_grad)


def test_streaming_fast_state_is_bounded_and_does_not_mutate_original():
    m = System1FastModel(14)
    old = m.initial(2)
    saved = old.clone()
    x = torch.randn(2, 14)
    a = torch.randn(2, 14)
    first = m(x, a, torch.zeros(2), old)
    next_ = m(x, a, torch.ones(2), first.fast)
    torch.testing.assert_close(old, saved, rtol=0, atol=0)
    assert first.fast.shape == (2, 96)
    assert (first.fast.abs() <= 1).all() and (next_.fast.abs() <= 1).all()
    assert not torch.equal(first.fast, next_.fast)
    assert not torch.equal(first.mean_next, next_.mean_next)


def test_system1_has_no_slow_or_associative_state_and_no_action_authority():
    m = System1FastModel()
    f = m(torch.ones(1, 14), torch.zeros(1, 14), torch.zeros(1))
    assert len(f) == 4
    assert not hasattr(f, "action")
    assert not hasattr(f, "allow")
    assert not hasattr(f, "slow")
    assert not hasattr(f, "memory")
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 100541


@pytest.mark.parametrize("invalid_time", [-1, float("nan"), float("inf")])
def test_system1_rejects_invalid_or_future_proxies(invalid_time):
    m = System1FastModel()
    with pytest.raises(ValueError):
        m(torch.ones(1, 14), torch.zeros(1, 14), torch.tensor([invalid_time]))
    with pytest.raises(ValueError):
        m(torch.ones(1, 14), torch.zeros(1, 14), torch.tensor([0.]), torch.ones(1, 64))


def test_system1_predictions_have_no_hidden_future_input():
    torch.manual_seed(75005)
    m = System1FastModel()
    x = torch.randn(2, 14)
    a = torch.randn_like(x)
    tick = torch.ones(2)
    observed = m(x, a, tick)
    # Outcomes are not part of the model inputs; an altered target cannot change a forecast.
    possible_outcome_1 = torch.randn_like(x)
    possible_outcome_2 = torch.randn_like(x)
    assert not torch.equal(possible_outcome_1, possible_outcome_2)
    with torch.no_grad():
        same = m(x, a, tick)
    torch.testing.assert_close(observed.mean_next, same.mean_next, rtol=0, atol=0)
    torch.testing.assert_close(observed.fast, same.fast, rtol=0, atol=0)


def test_gated_delta_perfect_corrective_overwrite_in_orthogonal_key_space():
    m = torch.zeros(1, 8, 8)
    k1 = F.one_hot(torch.tensor([0]), 8).float()
    k2 = F.one_hot(torch.tensor([1]), 8).float()
    v1 = F.one_hot(torch.tensor([2]), 8).float()
    v2 = F.one_hot(torch.tensor([3]), 8).float()
    overwrite = F.one_hot(torch.tensor([4]), 8).float()
    alpha = torch.ones(1)
    beta = torch.ones(1)
    m = gated_delta_update(m, k1, v1, alpha, beta)
    m = gated_delta_update(m, k2, v2, alpha, beta)
    torch.testing.assert_close(delta_read(m, k1), v1, rtol=0, atol=1e-7)
    torch.testing.assert_close(delta_read(m, k2), v2, rtol=0, atol=1e-7)
    m = gated_delta_update(m, k1, overwrite, alpha, beta)
    torch.testing.assert_close(delta_read(m, k1), overwrite, rtol=0, atol=1e-7)
    torch.testing.assert_close(delta_read(m, k2), v2, rtol=0, atol=1e-7)


def test_gated_delta_can_erase_and_never_write_query():
    cell = GatedDelta8x8(12)
    original = torch.randn(2, 8, 8)
    query = torch.randn(2, 12)
    result = cell.write(original, query, torch.tensor([False, False]))
    torch.testing.assert_close(original, result, rtol=0, atol=0)
    key = F.one_hot(torch.tensor([0]), 8).float()
    value = F.one_hot(torch.tensor([1]), 8).float()
    erased = gated_delta_update(original[:1], key, value,
                                torch.zeros(1), torch.ones(1))
    torch.testing.assert_close(delta_read(erased, key), value, rtol=0, atol=1e-7)


def test_gated_delta_gradients_exist_but_mechanism_test_is_not_intelligence():
    torch.manual_seed(75)
    cell = GatedDelta8x8(12)
    observations = torch.randn(2, 12)
    state = cell.write(cell.initial(2), observations, torch.tensor([True, True]))
    guessed = cell.query(state, observations)
    loss = guessed.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in cell.parameters() if p.requires_grad)


def test_invalid_gated_delta_inputs_fail_closed():
    x = torch.zeros(2, 8, 8)
    k = torch.ones(2, 8)
    v = torch.ones(2, 8)
    with pytest.raises(ValueError):
        gated_delta_update(x, k, v, torch.tensor([1.1, 1.]), torch.ones(2))
    with pytest.raises(ValueError):
        gated_delta_update(x, k, v, torch.ones(2), torch.tensor([float("nan"), 1.]))
    with pytest.raises(ValueError):
        gated_delta_update(x, k[:, :7], v, torch.ones(2), torch.ones(2))
