import torch

from tac_osm.real_robot_me057_fused_anchor import FusedAnchorServoRolloutTAC


def model():
    return FusedAnchorServoRolloutTAC(
        delta_scale=[.02]*14, state_mean=[1.]*14, state_scale=[2.]*14,
        action_mean=[3.]*14, action_scale=[4.]*14, servo_gain=[.1]*14,
        servo_step_limit=[.03]*14)


def test_fused_anchor_preserves_strict_parameter_parity():
    candidate = model()
    assert sum(p.numel() for p in candidate.parameters()) == 38268
    assert candidate.anchor_projection.in_features == 81
    assert candidate.anchor_projection.out_features == 90
    assert not hasattr(candidate, "boundary_projection")
    assert not hasattr(candidate, "state_projection")
    assert not hasattr(candidate, "episode_initializer")


def test_fused_anchor_updates_bounded_slow_and_trace_state():
    candidate = model()
    x = torch.randn(3, 14)
    belief = candidate.initial_from_observation(x)
    observed = candidate._observe_training(belief, x)
    consolidated = candidate._consolidate_training(observed, x)
    assert consolidated.fast.shape == (3, 32)
    assert consolidated.slow.shape == (3, 32)
    assert consolidated.memory.shape == (3, 8)
    assert torch.isfinite(consolidated.slow).all()
    assert torch.all(consolidated.memory.abs() <= 1.)


def test_fused_anchor_receives_dense_predictive_gradients():
    candidate = model()
    count = 16
    loss = candidate.dense_transition_auxiliary(
        torch.randn(count, 32), torch.randn(count, 32),
        torch.randn(count, 8), torch.randn(count, 14),
        torch.randn(count, 14), torch.randn(count, 14),
        torch.zeros(count, 1, dtype=torch.bool),
        torch.zeros(count, 1, dtype=torch.bool))
    loss.backward()
    gradient = candidate.anchor_projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient.abs().sum(1) > 0)
    assert torch.all(gradient.abs().sum(0) > 0)
    for name in ("feedback_key.weight", "feedback_value.weight",
                 "feedback_map.weight", "anchor_gain"):
        active = dict(candidate.named_parameters())[name].grad
        assert active is not None and active.abs().sum() > 0


def test_compiled_fused_anchor_matches_eager_boundary():
    candidate = model()
    x = torch.randn(2, 14)
    belief = candidate.initial_from_observation(x)
    expected = candidate._consolidate_training(belief, x)
    compiled = torch.compile(
        candidate._consolidate_training, dynamic=False, fullgraph=True,
        mode="reduce-overhead")
    actual = compiled(belief, x)
    for left, right in zip(expected, actual):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)
