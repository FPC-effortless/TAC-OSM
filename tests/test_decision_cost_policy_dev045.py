"""DEV-045 action-conditioned policy: no hidden labels at inference, explicit regret."""
import pytest
torch=pytest.importorskip("torch")
from tac_osm.development.decision_cost_policy import (
    CostConvention,Decision,select_actions,expected_action_costs,
    expected_decision_regret,contrastive_cost_margin,simulator_cost_for_wait_service,
)


def test_wait_when_waiting_cheaper():
    ans=select_actions(torch.tensor([[.1]]),torch.tensor([[.1]]))
    assert ans.action.item()==Decision.WAIT
    assert ans.expected_selected_cost.item()==pytest.approx(.4)


def test_service_when_expected_failures_justify_intervention():
    ans=select_actions(torch.tensor([[.9]]),torch.tensor([[0.]]))
    assert ans.action.item()==Decision.SERVICE
    assert ans.expected_selected_cost.item()==pytest.approx(1.0)


def test_inspection_requires_explicit_contingent_outcome_value():
    wait=torch.tensor([[.75]])
    service=torch.tensor([[.5]])
    assert select_actions(wait,service).action.item()!=Decision.INSPECT
    ans=select_actions(wait,service,inspect_followup_cost=torch.tensor([[.2]]))
    assert ans.action.item()==Decision.INSPECT
    assert ans.expected_selected_cost.item()==pytest.approx(.3)


def test_inadmissibility_does_not_silently_default_to_wait():
    valid=torch.tensor([[[False,False,False]]])
    ans=select_actions(torch.tensor([[.9]]),torch.tensor([[.0]]),admissible=valid)
    assert ans.action.item()==Decision.HUMAN_REVIEW
    assert ans.requires_human_review.item()
    assert torch.isnan(ans.expected_selected_cost).item()


def test_differentiable_decision_regret_and_oracle_is_separate():
    pred=torch.nn.Parameter(torch.tensor([[[1.5,1.0,2.0]]]))
    real=torch.tensor([[[4.0,1.0,4.1]]])
    regret=expected_decision_regret(pred,real,temperature=.5)
    assert regret.item()>0
    regret.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0


def test_hard_negative_cost_ranking_loss_penalizes_bad_preference():
    poor=torch.tensor([[[.2,1.0,1.5]]])
    optimal=torch.tensor([[[4.,1.,4.1]]])
    assert contrastive_cost_margin(poor,optimal).item()>0
    good=torch.tensor([[[3.,.1,4.]])
    assert contrastive_cost_margin(good,optimal).item()==pytest.approx(0.)


def test_physical_cost_evaluator_requires_true_action_receipt():
    wait=torch.tensor([[1,0]],dtype=torch.long)
    serviced=torch.tensor([[0,0]],dtype=torch.long)
    chosen=torch.tensor([[int(Decision.SERVICE),int(Decision.WAIT)]])
    outcome=simulator_cost_for_wait_service(wait,serviced,chosen)
    assert outcome.tolist()==[[1.,0.]]
    with pytest.raises(ValueError):
        simulator_cost_for_wait_service(wait,serviced,torch.tensor([[2,0]]))
    with pytest.raises(ValueError):
        expected_action_costs(torch.tensor([[1.4]]),torch.tensor([[0.]]))
