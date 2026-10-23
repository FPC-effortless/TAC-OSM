"""ME-062 software-only tests, not real-world capability evidence."""
import torch

from tests.test_me057_training_contract import identified
from tac_osm.real_robot_me062_selective import (
    SelectiveTrajectoryModel,
    selective_gaussian_forecast_loss,
)


def build(name):
    width, decoder = (32, 64) if name == "native_tac" else (44, 82)
    return SelectiveTrajectoryModel(identified(), name, width, decoder)


def test_untrained_point_forecast_exactly_persistence_both_arms():
    torch.manual_seed(62001)
    states = torch.randn(2, 14)
    actions = torch.randn(2, 16, 14)
    for name in ("native_tac", "gru_world"):
        model = build(name)
        belief = model.initial(2)
        forecast, trust = model.forecast_with_trust(belief, states, actions)
        torch.testing.assert_close(forecast.mean, states[:, None].expand(-1, 16, -1), rtol=0, atol=0)
        torch.testing.assert_close(trust, torch.full_like(trust, .5), rtol=0, atol=0)
        assert forecast.scale.shape == (2, 16, 14)
        assert torch.isfinite(forecast.scale).all() and (forecast.scale > 0).all()


def test_no_future_action_can_change_an_earlier_horizon():
    model = build("native_tac")
    # Activate both learned heads, so this is not a trivial zero-output test.
    with torch.no_grad():
        model.output.weight.normal_(std=.02)
        model.trust_head[-1].weight.normal_(std=.01)
        model.log_scale_head[-1].weight.normal_(std=.01)
    current = torch.randn(2, 14)
    actions = torch.randn(2, 16, 14)
    old = model.forecast(model.initial(2), current, actions)
    changed = actions.clone()
    changed[:, 5:] += 10
    new = model.forecast(model.initial(2), current, changed)
    torch.testing.assert_close(old.mean[:, :5], new.mean[:, :5], rtol=0, atol=0)
    torch.testing.assert_close(old.scale[:, :5], new.scale[:, :5], rtol=0, atol=0)


def test_shared_physical_objective_has_finite_gradients():
    torch.manual_seed(62002)
    history = torch.randn(2, 4, 14)
    executed = torch.randn(2, 3, 14)
    actions = torch.randn(2, 16, 14)
    y = torch.randn(2, 16, 14)
    for name in ("native_tac", "gru_world"):
        model = build(name)
        optim = torch.optim.AdamW(model.parameters(), lr=.001)
        for _ in range(2):
            optim.zero_grad(set_to_none=True)
            belief = model.initial(2)
            for t in range(3):
                belief = model.observe(belief, history[:, t], executed[:, t],
                                       history[:, t+1], torch.zeros(2, 1, dtype=torch.bool))
            p = model.forecast(belief, history[:, -1], actions)
            loss = selective_gaussian_forecast_loss(model, p, y, torch.ones(2, 16, 1, dtype=torch.bool))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
            optim.step()
