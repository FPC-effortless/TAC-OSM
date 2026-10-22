import torch

from tac_osm.real_robot_me056_boundary import BoundaryServoRolloutTAC


def model():
    return BoundaryServoRolloutTAC(
        delta_scale=[.02]*14,state_mean=[1.]*14,state_scale=[2.]*14,
        action_mean=[3.]*14,action_scale=[4.]*14,servo_gain=[.1]*14,
        servo_step_limit=[.03]*14)


def test_parameter_budget_is_unchanged_and_all_split_weights_are_active():
    candidate=model()
    assert sum(p.numel() for p in candidate.parameters())==38266
    assert candidate.fast_projection.out_features==64
    assert candidate.boundary_projection.out_features==17


def test_fast_observation_preserves_slow_trace_until_boundary():
    candidate=model()
    belief=candidate.initial_from_observation(torch.randn(2,14))
    observed=candidate._observe_training(belief,torch.randn(2,14))
    assert torch.equal(observed.slow,belief.slow)
    assert torch.equal(observed.memory,belief.memory)
    consolidated=candidate._consolidate_training(observed,torch.randn(2,14))
    assert consolidated.slow.shape==(2,32)
    assert consolidated.memory.shape==(2,8)
    assert torch.isfinite(consolidated.slow).all()


def test_prediction_queries_persistent_belief_without_mutating_it():
    candidate=model()
    belief=candidate.initial_from_observation(torch.randn(2,14))
    before=tuple(value.clone() for value in belief)
    forecast=candidate._predict_training(
        belief,torch.randn(2,14),torch.randn(2,3,14),
        torch.zeros(2,1,dtype=torch.bool))
    assert forecast.next_joint_state.shape==(2,3,14)
    assert all(torch.equal(left,right) for left,right in zip(before,belief))


def test_dense_transition_auxiliary_uses_all_native_projection_parameters():
    candidate=model()
    count=12
    loss=candidate.dense_transition_auxiliary(
        torch.randn(count,32),torch.randn(count,32),torch.randn(count,8),
        torch.randn(count,14),torch.randn(count,14),torch.randn(count,14),
        torch.zeros(count,1,dtype=torch.bool),
        torch.zeros(count,1,dtype=torch.bool))
    assert loss.ndim==0 and torch.isfinite(loss)
    loss.backward()
    for name in ("boundary_projection.weight","state_projection.weight",
                 "native.observation.weight"):
        gradient=dict(candidate.named_parameters())[name].grad
        assert gradient is not None and gradient.abs().sum()>0


def test_boundary_and_prediction_paths_have_finite_gradients():
    candidate=model()
    x=torch.randn(2,14)
    belief=candidate.initial_from_observation(x)
    belief=candidate._observe_training(belief,x)
    belief=candidate._consolidate_training(belief,x)
    forecast=candidate._predict_training(
        belief,x,torch.randn(2,1,14),torch.zeros(2,1,dtype=torch.bool))
    (forecast.next_joint_state.square().mean()+
     forecast.takeover_logits.square().mean()).backward()
    for name in ("fast_projection.weight","boundary_projection.weight",
                 "state_projection.weight"):
        parameter=dict(candidate.named_parameters())[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
