"""DEV-047 equation-level local temporal and selective diagonal-state controls.

The diagonal control is not a reproduction of Mamba or an optimized SSM kernel.
All variants use the same three-bank storage shape and public routing boundary.
"""
from __future__ import annotations
import torch
from torch import nn
from .unified_state_prototype import EntityLocalState, WorldState, UnifiedPredictiveStatePrototype


class SelectiveDiagonalState(EntityLocalState):
    def __init__(self, entities=8, dim=8):
        super().__init__(entities,dim)
        # Remove unused GRUs: compute/parameter counts must reflect this arm.
        del self.fast_cell, self.slow_cell, self.event_cell, self.slow_gate
        self.decay = nn.Linear(dim+1,3*dim)
        self.value = nn.Linear(dim+1,3*dim)
        self.write = nn.Linear(dim+1,3*dim)
        with torch.no_grad():
            self.decay.bias.copy_(torch.tensor([0.]*dim+[3.]*dim+[6.]*dim))

    def step(self,state,features,entity,tick,admit=None):
        b = features.shape[0]
        if entity.shape!=(b,) or entity.dtype!=torch.long or ((entity<0)|(entity>=self.entities)).any():
            raise ValueError("public entity addresses required")
        if tick.shape!=(b,) or not torch.isfinite(tick).all():
            raise ValueError("finite public ticks required")
        if admit is None:
            admit = torch.ones(b,dtype=torch.bool,device=features.device)
        if admit.shape!=(b,) or admit.dtype!=torch.bool:
            raise ValueError("verified admission mask required")
        row = torch.arange(b,device=features.device)
        x = torch.cat((features,torch.log1p((tick-state.last_seen[row,entity]).clamp_min(0))[:,None]),-1)
        alpha = self.decay(x).sigmoid()
        values = self.value(x).tanh()
        write = self.write(x).sigmoid()
        old = torch.cat((state.fast[row,entity],state.slow[row,entity],state.event[row,entity]),-1)
        new = alpha*old+(1-alpha)*write*values
        banks = []
        for bank,candidate in zip((state.fast,state.slow,state.event),new.chunk(3,-1)):
            updated = bank.clone()
            updated[row,entity] = torch.where(admit[:,None],candidate,bank[row,entity])
            banks.append(updated)
        seen = state.last_seen.clone()
        seen[row,entity] = torch.where(admit,tick,state.last_seen[row,entity])
        return WorldState(*banks,seen)


class TemporalOrderState(EntityLocalState):
    def __init__(self,entities=8,dim=8):
        super().__init__(entities,dim)
        self.order = nn.Linear(2*dim,dim)

    def step(self,state,features,entity,tick,admit=None):
        row = torch.arange(features.shape[0],device=features.device)
        # Retain a previous event feature explicitly; encode order before writing.
        enriched = self.order(torch.cat((state.event[row,entity],features),-1)).tanh()
        updated = super().step(state,enriched,entity,tick,admit)
        accepted = torch.ones(features.shape[0],dtype=torch.bool,device=features.device) if admit is None else admit
        event = state.event.clone()
        event[row,entity] = torch.where(accepted[:,None],features,state.event[row,entity])
        return WorldState(updated.fast,updated.slow,event,updated.last_seen)


def build_variant(variant,seed,entities=8):
    if variant not in ("multiscale_gru","selective_diagonal","temporal_order","fast_gru"):
        raise ValueError("unregistered local memory")
    torch.manual_seed(seed)
    model = UnifiedPredictiveStatePrototype(entities=entities,dim=8,budget=entities)
    if variant=="selective_diagonal":
        model.memory = SelectiveDiagonalState(entities,8)
    elif variant=="temporal_order":
        model.memory = TemporalOrderState(entities,8)
    elif variant=="fast_gru":
        class FastOnly(EntityLocalState):
            def read(self,state):
                return torch.cat((state.fast,torch.zeros_like(state.slow),torch.zeros_like(state.event)),-1)
        model.memory = FastOnly(entities,8)
    return model
