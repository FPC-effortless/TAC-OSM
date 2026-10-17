"""DEV-048 genuinely sparse OSM calls and explicit unresolved execution.

Only selected slots enter OSM. Unselected or inadmissible actions remain REVIEW
until a separately measured public controller or sensing policy resolves them.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .decision_cost_policy import select_actions, Decision


@dataclass
class RoutedPlan:
    actions: torch.Tensor
    selected: torch.Tensor
    probabilities: torch.Tensor
    routing_candidates: int
    predicted_entities: int
    state: object


def sparse_plan(model,state,horizons,budget,admissible=None):
    slots = model.memory.read(state)
    batch,entities,_ = slots.shape
    selected,_ = model.relevance(slots,budget)
    gathered = slots.gather(1,selected[...,None].expand(-1,-1,slots.shape[-1]))
    p_wait,p_service = model.world_predictor(gathered,horizons)
    mask = None
    if admissible is not None:
        if admissible.shape!=(batch,entities,3) or admissible.dtype!=torch.bool:
            raise ValueError("public per-entity execution constraints required")
        mask = admissible.gather(1,selected[...,None].expand(-1,-1,3))
    chosen = select_actions(p_wait,p_service,admissible=mask,convention=model.costs)
    actions = torch.full((batch,entities),int(Decision.HUMAN_REVIEW),dtype=torch.long,device=slots.device)
    actions.scatter_(1,selected,chosen.action)
    return RoutedPlan(actions,selected,torch.stack((p_wait,p_service),-1),
                      batch*entities,batch*budget,state)


def admissible_services(actions, max_services=None):
    """Validate a structured action transaction before side effects."""
    if actions.ndim!=2 or not torch.all((actions==Decision.WAIT)|(actions==Decision.SERVICE)):
        raise ValueError("unresolved inspections/review require a separate contingent policy")
    services = [tuple(torch.nonzero(row==Decision.SERVICE).flatten().tolist()) for row in actions]
    if max_services is not None and (max_services<0 or any(len(s)>max_services for s in services)):
        raise ValueError("service capacity constraint rejects the transaction")
    return services


def commit_public_receipts(model,state,events,entities,ticks,verified):
    """Receipt validity is distinct from agreement with the model's prediction."""
    row = torch.arange(entities.shape[0],device=entities.device)
    if (ticks[verified] < state.last_seen[row,entities][verified]).any():
        raise ValueError("accepted receipts cannot move local event time backwards")
    return model.verified_update(state,events,entities,ticks,verified)
