"""Synthetic numerical contracts only; no robot capability evidence."""
import torch
import pytest
from tac_osm.real_robot_me055_rollout import (
    ServoRolloutGRU, ServoRolloutTAC, factual_multistep_loss)


def models():
    args = dict(delta_scale=[.02]*14, state_mean=[1.]*14, state_scale=[2.]*14,
                action_mean=[3.]*14, action_scale=[4.]*14,
                servo_gain=[.1]*14, servo_step_limit=[.03]*14)
    return ServoRolloutTAC(**args), ServoRolloutGRU(**args)


def test_shared_servo_prior_is_exact_with_zero_initialized_residual_heads():
    x = torch.zeros(2,14)
    action = torch.ones(2,1,14)
    human = torch.zeros(2,1,dtype=torch.bool)
    for model in models():
        belief = model.observe(model.initial(2), x)
        prediction = model.predict(belief, x, action, human)
        # Raw action=7, then normalized to the state coordinates as 3; .1*3
        # is clipped by the FIT-only .03 one-step physical bound.
        torch.testing.assert_close(prediction.next_joint_state[:,0],
                                   torch.full_like(x,.03),rtol=0,atol=0)


def test_fused_native_projection_preserves_parameter_budget_without_dead_weights():
    tac,gru=models()
    assert sum(p.numel() for p in tac.parameters()) == 38266
    assert sum(p.numel() for p in gru.parameters()) == 38295
    assert tac.state_projection.out_features == 73
    assert tac.fast_projection.out_features == 81
    assert tac.feedback_key.out_features == 16
    assert tac.feedback_value.out_features == 16
    assert tac.feedback_map.in_features == 87
    assert tac.native.readout[0].out_features == 147
    assert not any(name.startswith(("native.value.","native.write.",
                                    "native.slow_gate.","native.enc.",
                                    "native.transition.","native.read_fast.",
                                    "native.gate.","native.key.",
                                    "native.query.",
                                    "native.slow.",
                                    "native.verify_map.",
                                    "physical_map.",
                                    "native.verify_key.","native.verify_value.",
                                    "physical_key.","physical_value."))
                   for name,_ in tac.named_parameters())


def test_open_loop_shapes_finiteness_and_no_input_belief_mutation():
    actions = torch.randn(2,16,14)
    x = torch.randn(2,14)*.1
    human = torch.zeros(2,1,dtype=torch.bool)
    for model in models():
        belief = model.initial(2)
        before = [part.clone() for part in belief] if hasattr(belief,"fast") else belief.clone()
        forecast = model.rollout(belief,x,actions,human)
        assert forecast.joint_state.shape == (2,16,14)
        assert forecast.takeover_logits.shape == (2,16,1)
        assert torch.isfinite(forecast.joint_state).all()
        if hasattr(belief,"fast"):
            assert all(torch.equal(a,b) for a,b in zip(before,belief))
        else:
            assert torch.equal(before,belief)


def test_native_uses_compact_relevance_trace_instead_of_dense_matrix():
    model,_=models()
    belief=model.initial(3)
    assert belief.memory.shape == (3,8)
    updated=model.observe(belief,torch.randn(3,14))
    assert updated.memory.shape == (3,8)
    assert torch.isfinite(updated.memory).all()


def test_native_episode_initializer_is_observation_conditioned_and_trainable():
    model,_=models()
    context=torch.randn(3,14)
    belief=model.initial_from_observation(context)
    assert belief.fast.shape==(3,32) and belief.memory.shape==(3,8)
    sum(part.square().mean() for part in belief).backward()
    assert model.episode_initializer.weight.grad.abs().sum()>0
    assert model.initializer_gate.grad.abs().sum()>0


def test_checked_and_training_kernels_agree_before_feedback():
    x=torch.randn(2,14)*.1
    action=torch.randn(2,1,14)*.1
    human=torch.zeros(2,1,dtype=torch.bool)
    for model in models():
        initial=model.initial(2)
        checked_state=model.observe(initial,x)
        fast_state=model._observe_training(initial,x)
        if hasattr(checked_state,"fast"):
            for left,right in zip(checked_state,fast_state):
                torch.testing.assert_close(left,right,rtol=1e-6,atol=1e-7)
        else:
            torch.testing.assert_close(checked_state,fast_state,rtol=1e-6,atol=1e-7)
        checked=model.predict(checked_state,x,action,human)
        fast=model._predict_training(fast_state,x,action,human)
        torch.testing.assert_close(checked.next_joint_state,fast.next_joint_state,
                                   rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(checked.takeover_logits,fast.takeover_logits,
                                   rtol=1e-6,atol=1e-7)


def test_fused_burnin_matches_separate_training_kernels():
    current=torch.randn(2,14)*.1
    action=torch.randn(2,14)*.1
    following=torch.randn(2,14)*.1
    authority=torch.zeros(2,1,dtype=torch.bool)
    next_authority=torch.tensor([[False],[True]])
    for model in models():
        initial=model.initial(2)
        state=model._observe_training(initial,current)
        prediction=model._predict_training(
            state,current,action.unsqueeze(1),authority)
        onset=(~authority & next_authority).to(current.dtype)
        separate=model._verify_training(
            state,action,authority,following,prediction.next_joint_state[:,0],
            onset,prediction.takeover_logits[:,0])
        fused=model._burnin_training(
            initial,current,action,authority,following,next_authority)
        if hasattr(fused,"fast"):
            for left,right in zip(separate,fused):
                torch.testing.assert_close(left,right,rtol=1e-6,atol=1e-7)
        else:
            torch.testing.assert_close(separate,fused,rtol=1e-6,atol=1e-7)


def test_sixteen_step_rollout_has_full_finite_gradient_path():
    model,_ = models()
    x = torch.randn(2,14)*.1
    actions = torch.randn(2,16,14)*.1
    forecast = model.rollout(model.initial_from_observation(x),x,actions,
                             torch.zeros(2,1,dtype=torch.bool))
    future = torch.randn(2,16,14)*.1
    onset = torch.zeros(2,16,1)
    available = torch.ones(2,16,1,dtype=torch.bool)
    loss, metrics = factual_multistep_loss(
        model,forecast,future,available,onset,available)
    loss.backward()
    assert metrics["supervised_physical_steps"] == 32
    # Open-loop planning correctly does not invoke post-outcome verification
    # modules. Every prediction/observation parameter must still participate.
    for name,p in model.named_parameters():
        if name.startswith(("native.verify_", "physical_", "feedback_")):
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.native.next_observation.weight.grad.abs().sum() > 0
    assert model.fast_projection.weight.grad.abs().sum() > 0


def test_masked_nan_targets_never_enter_loss():
    model,_ = models()
    x = torch.zeros(2,14)
    actions = torch.zeros(2,4,14)
    forecast = model.rollout(model.initial(2),x,actions,
                             torch.zeros(2,1,dtype=torch.bool))
    targets = torch.zeros(2,4,14)
    targets[1] = float("nan")
    onset = torch.zeros(2,4,1)
    onset[1] = float("nan")
    available = torch.tensor([[[True]]*4,[[False]]*4])
    loss,_ = factual_multistep_loss(
        model,forecast,targets,available,onset,available)
    assert torch.isfinite(loss)


def test_rollout_refuses_future_human_control_as_candidate_plan():
    model,_ = models()
    with pytest.raises(ValueError,match="robot control"):
        model.rollout(model.initial(1),torch.zeros(1,14),torch.zeros(1,4,14),
                      torch.ones(1,1,dtype=torch.bool))


@pytest.mark.parametrize("field", ["scale","gain","limit"])
def test_fit_only_physical_contract_fails_closed(field):
    args = dict(delta_scale=[.02]*14,state_mean=[0.]*14,state_scale=[1.]*14,
                action_mean=[0.]*14,action_scale=[1.]*14,
                servo_gain=[.1]*14,servo_step_limit=[.03]*14)
    if field == "scale": args["state_scale"] = [0.]*14
    elif field == "gain": args["servo_gain"] = [1.1]*14
    else: args["servo_step_limit"] = [-.1]*14
    with pytest.raises(ValueError):
        ServoRolloutTAC(**args)
