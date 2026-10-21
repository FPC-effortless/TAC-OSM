"""Exact immutable embedding-table sharing; no new learned model or fit."""
import copy
import torch
from torch import nn


class SharedFrozenForecaster(nn.Module):
    def __init__(self,context,student):
        super().__init__()
        if student.memory.emb.weight.requires_grad or context.emb.weight.requires_grad:
            raise ValueError('embedding sharing requires both tables frozen')
        if not torch.equal(context.emb.weight,student.memory.emb.weight):
            raise ValueError('embedding tables differ; sharing would alter predictions')
        self.context=copy.deepcopy(context)
        self.student=copy.deepcopy(student)
        self.student.memory.emb=self.context.emb

    def forward(self,items:torch.Tensor,meta:torch.Tensor,action:torch.Tensor,
                duration:torch.Tensor)->torch.Tensor:
        offered=torch.cat((items[:,[32,64,96]],action.unsqueeze(-1)),-1)
        return self.student(items,meta,offered,self.context(items,meta,action,duration))
