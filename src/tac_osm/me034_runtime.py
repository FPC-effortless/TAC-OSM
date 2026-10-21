"""Portable raw-input pipeline; safe reuse of identical frozen embedding tables."""
import copy
import torch
from torch import nn


class StablePipeline(nn.Module):
    def __init__(self,context,student,share_embedding:bool=False):
        super().__init__()
        self.context=copy.deepcopy(context);self.student=copy.deepcopy(student)
        if share_embedding:
            memory=self.student.memory
            native=hasattr(memory,'core')
            table=memory.core.emb if native else memory.emb
            if table.weight.requires_grad or self.context.emb.weight.requires_grad:
                raise ValueError('sharing requires immutable tables')
            if not torch.equal(table.weight,self.context.emb.weight):
                raise ValueError('sharing requires exactly equal tables')
            if native:memory.core.emb=self.context.emb
            else:memory.emb=self.context.emb

    def forward(self,items:torch.Tensor,meta:torch.Tensor,action:torch.Tensor,
                duration:torch.Tensor)->torch.Tensor:
        offered=torch.cat((items[:,[32,64,96]],action.unsqueeze(-1)),-1)
        durations=torch.cat((meta[:,[32,64,96],0],duration),-1).unsqueeze(-1)
        context=torch.cat((self.context(items,meta,action,duration),durations),-1)
        return self.student(items,meta,offered,context)
