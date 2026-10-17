"""DEV-046 untrained vertical slice: public events -> local state -> future -> cost.

Interfaces only, NOT a learned causal world model, not original CDL/CASM parity.
All information available to forward() is public. Privileged simulator labels
belong only in independent loss/evaluator code, never in this inference module.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn

from .decision_cost_policy import select_actions,DecisionBatch,CostConvention


@dataclass
class WorldState:
    fast:torch.Tensor
    slow:torch.Tensor
    event:torch.Tensor
    last_seen:torch.Tensor


@dataclass
class WorldPrediction:
    p_wait:torch.Tensor
    p_service:torch.Tensor
    decisions:DecisionBatch
    selected_entities:torch.Tensor
    retrievals:int
    state:WorldState


class EntityLocalState(nn.Module):
    """Three timescales; every bank changes only on an accepted local event."""
    def __init__(self,entities=8,dim=32):
        super().__init__()
        if entities<1 or dim<1:raise ValueError("positive entity count and dimensions")
        self.entities=entities;self.dim=dim
        self.fast_cell=nn.GRUCell(dim+1,dim)
        self.slow_cell=nn.GRUCell(dim+1,dim)
        self.event_cell=nn.GRUCell(dim+1,dim)
        self.slow_gate=nn.Linear(dim+1,dim)

    def initial(self,batch,device,dtype):
        shape=(batch,self.entities,self.dim)
        base=torch.zeros(shape,device=device,dtype=dtype)
        return WorldState(base.clone(),base.clone(),base.clone(),
                          torch.zeros((batch,self.entities),device=device,dtype=dtype))

    def step(self,state,features,entity,tick,admit=None):
        if features.shape!=(state.fast.shape[0],self.dim):
            raise ValueError("event feature shape mismatch")
        b=features.shape[0]
        if entity.shape!=(b,) or entity.dtype!=torch.long or (
              (entity<0)|(entity>=self.entities)).any():
            raise ValueError("explicit public integer entity IDs required")
        if tick.shape!=(b,) or not torch.isfinite(tick).all():
            raise ValueError("finite public event timestamps required")
        if admit is None:admit=torch.ones(b,dtype=torch.bool,device=features.device)
        if admit.shape!=(b,) or admit.dtype!=torch.bool:raise ValueError("boolean verified write admission")
        if not admit.any():return state
        row=torch.arange(b,device=features.device)
        dt=(tick-state.last_seen[row,entity]).clamp_min(0)
        x=torch.cat((features,torch.log1p(dt)[:,None]),dim=-1)
        oldfast=state.fast[row,entity]
        oldslow=state.slow[row,entity]
        oldevent=state.event[row,entity]
        nextfast=self.fast_cell(x,oldfast)
        candidate=self.slow_cell(x,oldslow)
        g=self.slow_gate(x).sigmoid()
        nextslow=g*candidate+(1-g)*oldslow
        nextevent=self.event_cell(x,oldevent)
        def put(bank,new):
            # clone+index-copy keeps rejected and unrelated slots exactly unchanged.
            updated=bank.clone()
            updated[row,entity]=torch.where(admit[:,None],new,bank[row,entity])
            return updated
        seen=state.last_seen.clone()
        seen[row,entity]=torch.where(admit,tick,state.last_seen[row,entity])
        return WorldState(put(state.fast,nextfast),put(state.slow,nextslow),
                          put(state.event,nextevent),seen)

    def read(self,state):
        return torch.cat((state.fast,state.slow,state.event),dim=-1)


class BudgetedRelevance(nn.Module):
    def __init__(self,dim,entities=8):
        super().__init__()
        self.entities=entities;self.score=nn.Linear(dim,1)
    def forward(self,slots,k,allowed=None):
        if slots.ndim!=3 or slots.shape[1]!=self.entities:raise ValueError("entity slot mismatch")
        if not 1<=k<=self.entities:raise ValueError("retrieval budget out of range")
        scores=self.score(slots).squeeze(-1)
        if allowed is not None:
            if allowed.shape!=scores.shape or allowed.dtype!=torch.bool:
                raise ValueError("allowed retrieval mask")
            if (allowed.sum(-1)<k).any():
                raise ValueError("insufficient public valid candidates for retrieval budget")
            scores=scores.masked_fill(~allowed,float("-inf"))
        idx=scores.topk(k,dim=-1).indices
        # Hard top-k retrieval is explicitly measured, not claimed differentiable.
        return idx,k


class ActionConditionedOSM(nn.Module):
    def __init__(self,dim,slots=96):
        super().__init__()
        self.query=nn.Linear(1,dim)
        self.action=nn.Embedding(2,dim)
        self.head=nn.Sequential(nn.Linear(slots+dim,dim),nn.Tanh(),
                                nn.Linear(dim,1))

    def forward(self,slot_features,horizon):
        if slot_features.ndim!=3 or horizon.shape!=(slot_features.shape[0],):
            raise ValueError("expected slots [B,E,D] and positive horizons [B]")
        if (horizon<=0).any() or not torch.isfinite(horizon).all():
            raise ValueError("finite positive horizons required")
        b,e,_=slot_features.shape
        query=self.query(torch.log1p(horizon.float())[:,None])
        probs=[]
        for action in (0,1):
            embedding=self.action.weight[action][None,None,:]
            conditioned=query[:,None,:].expand(b,e,-1)+embedding
            probs.append(torch.sigmoid(self.head(torch.cat((slot_features,conditioned),-1)).squeeze(-1)))
        return tuple(probs)


class UnifiedPredictiveStatePrototype(nn.Module):
    """Public known-ID streaming state and cost-aware risk selection.

    The global cross-entity cost constraint and inspect-receipt updates are not
    learned here. CASM admissibility is passed as a public boolean mask.
    """
    def __init__(self,entities=8,dim=32,budget=4,costs=None):
        super().__init__()
        self.entities=entities;self.dim=dim;self.budget=budget
        self.observation=nn.Sequential(nn.Linear(16,dim),nn.Tanh())
        self.memory=EntityLocalState(entities,dim)
        self.relevance=BudgetedRelevance(dim*3,entities)
        self.world_predictor=ActionConditionedOSM(dim,3*dim)
        self.costs=costs or CostConvention()

    def encode(self,events,entities,ticks,admit=None):
        if events.ndim!=3 or events.shape[-1]!=16:
            raise ValueError("public event feature tensor [B,T,16] required")
        b,t,_=events.shape
        if entities.shape!=(b,t) or ticks.shape!=(b,t):
            raise ValueError("aligned public entity/time arrays required")
        if admit is None:admit=torch.ones(b,t,device=events.device,dtype=torch.bool)
        if admit.shape!=(b,t) or admit.dtype!=torch.bool:
            raise ValueError("verified commit mask required")
        if not torch.isfinite(events).all() or not torch.isfinite(ticks).all():
            raise ValueError("invalid public observation")
        if (ticks[:,1:]<ticks[:,:-1]).any():raise ValueError("timestamps must be monotonic")
        h=self.observation(events)
        state=self.memory.initial(b,events.device,h.dtype)
        for i in range(t):
            state=self.memory.step(state,h[:,i],entities[:,i],ticks[:,i],admit[:,i])
        return state

    def forward(self,events,entities,ticks,horizon,admissible=None,
                inspect_followup_cost=None,admit=None):
        state=self.encode(events,entities,ticks,admit)
        read=self.memory.read(state)
        chosen,k=self.relevance(read,self.budget)
        # Predict from all public entity slots; retrieval is separately logged
        # and not claimed to reduce OSM computation in this prototype.
        p_wait,p_service=self.world_predictor(read,horizon)
        decisions=select_actions(p_wait,p_service,inspect_followup_cost,
                                 admissible=admissible,convention=self.costs)
        return WorldPrediction(p_wait,p_service,decisions,chosen,k,state)

    def verified_update(self,state,public_event,public_entity,public_tick,accepted):
        if public_event.shape[-1]!=16:raise ValueError("public receipt feature expected")
        return self.memory.step(state,self.observation(public_event),public_entity,
                                public_tick,admit=accepted)
