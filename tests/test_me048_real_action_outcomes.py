"""ME048 numerical-only checks; never simulated decision-performance evidence."""
import torch
import pytest
from tac_osm.me048_real_action_outcomes import (
    ARMS,ActionOutcomeModel,true_observed_action_loss,preflight)


def test_me048_three_action_heads_native_real_math_contracts():
    result=preflight()
    assert result["status"]=="ME048_MATH_SOFTWARE_ONLY_PASS"
    assert len(result["active_coefficients"])==3
    assert not result["authentic_action_test_results_computed"]
    assert not result["sequence_of_logged_user_states_available"]


def test_me048_outputs_treatment_conditional_probability_and_two_goals():
    torch.manual_seed(48048)
    x=torch.randn(4,12)
    for arm in ARMS:
        net=ActionOutcomeModel(arm)
        treatment=net.probability(x,torch.zeros(4,dtype=torch.long))
        conversion=net.probability(x,torch.ones(4,dtype=torch.long))
        assert treatment.shape==conversion.shape==(4,2)
        assert (treatment>=0).all() and (treatment<=1).all()
        assert not torch.equal(treatment,conversion)


def test_me048_never_uses_counterfactual_outcome_as_a_supervised_label():
    torch.manual_seed(48048)
    net=ActionOutcomeModel("action_tac")
    x=torch.randn(4,12)
    action=torch.tensor([1,0,1,0])
    observed_visit=torch.tensor([1.,0.,1.,0.])
    observed_conversion=torch.tensor([0.,0.,0.,0.])
    loss=true_observed_action_loss(net,x,action,observed_visit,observed_conversion)
    assert torch.isfinite(loss)
    loss.backward()
    assert net.state.transition.weight.grad is not None
    assert net.action_head[0].weight.grad is not None
    with pytest.raises(ValueError):
        true_observed_action_loss(net,x,torch.tensor([1,0,3,0]),
                                  observed_visit,observed_conversion)


def test_me048_training_only_normalization_rejects_broken_scales():
    net=ActionOutcomeModel("action_context")
    with pytest.raises(ValueError):
        net.set_training_normalization(torch.zeros(12),torch.zeros(12))
    net.set_training_normalization(torch.ones(12),torch.ones(12))
    assert net.trained_mean.tolist()==[1.]*12
