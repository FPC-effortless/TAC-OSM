"""ME049 software and scientific validity safeguards.

Constructed tensors check original equations and training permissions only;
they NEVER constitute real-world randomized trial performance evidence.
"""
import pytest,torch
from tac_osm.me049_rct_action_models import (
  ARMS,ACTION_NAMES,ThreeActionWorldModel,actual_logged_action_loss,test_math_preflight)
from scripts.me049_train_randomized_policy import split,offline_value


def test_me049_original_three_arm_equal_active_parameter_budget_and_native_gradient():
    m=test_math_preflight()
    assert m["status"]=="ME049_NUMERICAL_NO_SYNTHETIC_SUCCESS"
    assert m["max_min_ratio"]<1.01
    assert len(m["active_parameter_counts"])==3


def test_me049_true_original_three_actions_are_supported_and_splits_disjoint():
    assert ACTION_NAMES==("No E-Mail","Mens E-Mail","Womens E-Mail")
    assert [split(i) for i in range(20)]==["FIT"]*14+["DEV"]*3+["TEST"]*3
    assert sum(split(i)=="FIT" for i in range(64000))==44800
    assert sum(split(i)=="DEV" for i in range(64000))==9600
    assert sum(split(i)=="TEST" for i in range(64000))==9600


def test_me049_original_logged_action_is_only_outcome_used_to_train():
    net=ThreeActionWorldModel("action_tac")
    inputs=torch.randn(3,12)
    actions=torch.tensor([0,1,2])
    outcomes=torch.tensor([0.,1.,0.])
    conversion=torch.zeros(3)
    spend=torch.zeros(3)
    loss=actual_logged_action_loss(net,inputs,actions,outcomes,conversion,spend)
    assert torch.isfinite(loss)
    loss.backward()
    assert net.state.transition.weight.grad is not None
    assert net.head[-1].weight.grad is not None
    with pytest.raises(ValueError):
        actual_logged_action_loss(net,inputs,torch.tensor([0,1,3]),outcomes,conversion,spend)


def test_me049_known_rct_propensity_1_3_corrects_policy_value():
    import numpy as np
    action=np.array([0,1,2,0,1,2],dtype=np.int64)
    reward=np.array([0.,1.,1.,0.,0.,1.])
    q=np.zeros((6,3),dtype=np.float64)
    policy=np.array([0,1,2,0,1,2])
    values,points=offline_value(policy,action,reward,q)
    assert abs(values["IPS"]-reward.mean()*3)<1e-12
    assert abs(values["effective_sample_size"]-6)<1e-12
    assert len(points)==6
