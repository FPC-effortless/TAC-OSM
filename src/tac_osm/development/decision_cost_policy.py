"""DEV-045 cost-aware decision core.

The model-facing policy receives only action-conditioned predictions and
explicit public feasibility constraints. Hidden simulator labels are accepted
ONLY by loss/evaluator helpers, never by select_actions.
This is an engineering primitive, not a validated full TAC-OSM policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch
from torch.nn import functional as F


class Decision(IntEnum):
    WAIT=0
    SERVICE=1
    INSPECT=2
    HUMAN_REVIEW=3


@dataclass(frozen=True)
class CostConvention:
    failure:float=4.0
    service:float=1.0
    inspection:float=.1

    def __post_init__(self):
        if (self.failure<=0 or self.service<0 or self.inspection<0 or
            not all(math.isfinite(v) for v in (self.failure,self.service,self.inspection))):
            raise ValueError("finite nonnegative physical-cost coefficients required")


def _prob(name,value):
    if not torch.is_tensor(value) or not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating tensor")
    if value.numel()==0 or not torch.isfinite(value).all() or ((value<0)|(value>1)).any():
        raise ValueError(f"{name} must be nonempty finite probabilities [0,1]")


def _check_pred(wait,service):
    _prob("p_wait",wait);_prob("p_service",service)
    if wait.shape!=service.shape:raise ValueError("matching entity-shaped forecast required")


@dataclass
class DecisionBatch:
    action:torch.Tensor
    expected_action_costs:torch.Tensor
    expected_selected_cost:torch.Tensor
    requires_human_review:torch.Tensor


def expected_action_costs(p_wait,p_service,inspect_followup_cost=None,
                          convention:CostConvention=CostConvention()):
    """Forecast cost of WAIT / SERVICE / INSPECT, before any true outcomes.

    inspect_followup_cost is the expected downstream cost of a *contingent*
    post-inspection strategy, conditioned on possible verified receipts.
    It must NOT be silently set to zero or confused with inspection price.
    """
    _check_pred(p_wait,p_service)
    wait=convention.failure*p_wait
    service=convention.service+convention.failure*p_service
    if inspect_followup_cost is None:
        inspect=torch.full_like(wait,float("inf"))
    else:
        if (not torch.is_tensor(inspect_followup_cost) or
            inspect_followup_cost.shape!=wait.shape or
            not torch.is_floating_point(inspect_followup_cost) or
            not torch.isfinite(inspect_followup_cost).all() or
            (inspect_followup_cost<0).any()):
            raise ValueError("explicit finite nonnegative inspect follow-up costs required")
        inspect=inspect_followup_cost+convention.inspection
    return torch.stack((wait,service,inspect),dim=-1)


def select_actions(p_wait,p_service,inspect_followup_cost=None,admissible=None,
                   convention:CostConvention=CostConvention()):
    """Return explicit HUMAN_REVIEW if constraints exclude all known actions."""
    costs=expected_action_costs(p_wait,p_service,inspect_followup_cost,convention)
    if admissible is None:
        feasible=torch.ones_like(costs,dtype=torch.bool)
    else:
        if admissible.shape!=costs.shape or admissible.dtype!=torch.bool:
            raise ValueError("admissible must be bool tensor [...,3]")
        feasible=admissible
    feasible=feasible & torch.isfinite(costs)
    masked=costs.masked_fill(~feasible,float("inf"))
    best=masked.argmin(-1)
    has=feasible.any(-1)
    chosen=torch.where(has,best,torch.full_like(best,int(Decision.HUMAN_REVIEW)))
    chosen_cost=masked.gather(-1,best.unsqueeze(-1)).squeeze(-1)
    # Human review has unknown external cost, not an invented free WAIT action.
    chosen_cost=torch.where(has,chosen_cost,torch.full_like(chosen_cost,float("nan")))
    return DecisionBatch(chosen,masked,chosen_cost,~has)


def expected_decision_regret(predicted_costs,oracle_costs,temperature=.1,admissible=None):
    """Differentiable regret surrogate, oracle labels restricted to TRAINING.

    Oracle future costs are not available to select_actions and no policy can
    access them at deployment. Pair-wise independent entity decisions only.
    """
    if not (torch.is_tensor(predicted_costs) and torch.is_tensor(oracle_costs) and
            predicted_costs.shape==oracle_costs.shape and predicted_costs.shape[-1]==3):
        raise ValueError("matching [...,3] costs expected")
    if (not math.isfinite(temperature)) or temperature<=0:raise ValueError("positive temperature")
    if not torch.isfinite(oracle_costs).all():
        raise ValueError("finite independent simulator oracle costs required")
    if admissible is None:admissible=torch.isfinite(predicted_costs)
    if admissible.shape!=predicted_costs.shape or admissible.dtype!=torch.bool:
        raise ValueError("admissible action mask mismatch")
    valid=admissible & torch.isfinite(predicted_costs)
    if not valid.any(-1).all():raise ValueError("unresolved action mask requires external human review")
    policy=F.softmax((-predicted_costs/temperature).masked_fill(~valid,-torch.inf),dim=-1)
    masked_oracle=oracle_costs.masked_fill(~valid,float("inf"))
    reference=masked_oracle.min(-1).values.detach()
    # Avoid inf*zero while preserving continuous gradients for valid actions.
    regrets=torch.where(valid,oracle_costs-reference.unsqueeze(-1),0.)
    return (policy*regrets).sum(-1).mean()


def contrastive_cost_margin(predicted_costs,oracle_costs,margin=.2,admissible=None):
    """Hard-negative action costs: correct (lowest-oracle-cost) vs other actions."""
    if predicted_costs.shape!=oracle_costs.shape or predicted_costs.shape[-1]!=3:
        raise ValueError("matching action costs required")
    if admissible is None:admissible=torch.isfinite(predicted_costs)
    valid=admissible & torch.isfinite(predicted_costs)
    if not valid.any(-1).all():raise ValueError("no admissible action")
    oracle_best=oracle_costs.masked_fill(~valid,float("inf")).argmin(-1)
    best=predicted_costs.gather(-1,oracle_best.unsqueeze(-1))
    other=torch.arange(3,device=oracle_best.device)!=oracle_best.unsqueeze(-1)
    chosen=valid & other
    bad=F.relu(margin+best-predicted_costs)
    return torch.where(chosen,bad,0.).sum()/chosen.sum().clamp_min(1)


def simulator_cost_for_wait_service(wait_targets,service_targets,chosen,
                                    convention:CostConvention=CostConvention()):
    """AUDIT/TRAINING ONLY. True counterfactual labels NEVER enter select_actions."""
    if wait_targets.shape!=service_targets.shape or chosen.shape!=wait_targets.shape:
        raise ValueError("matching per-entity labels/actions required")
    if not torch.all((wait_targets==0)|(wait_targets==1)) or not torch.all(
           (service_targets==0)|(service_targets==1)):
        raise ValueError("binary simulator labels required")
    if not torch.all((chosen==int(Decision.WAIT))|(chosen==int(Decision.SERVICE))):
        raise ValueError("inspect/review require verified second-stage outcome")
    outcome=torch.where(chosen==int(Decision.SERVICE),service_targets,wait_targets)
    return (convention.failure*outcome+
            convention.service*(chosen==int(Decision.SERVICE)).to(outcome.dtype))
