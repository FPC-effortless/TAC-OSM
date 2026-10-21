"""Raw physical-input/output adapter for a frozen selected forecasting model."""
import copy
import torch
from torch import nn


class PhysicalForecast(nn.Module):
    def __init__(self,model,mean:torch.Tensor,scale:torch.Tensor):
        super().__init__();self.model=copy.deepcopy(model)
        self.register_buffer('mean',mean.detach().clone())
        self.register_buffer('scale',scale.detach().clone())
        if bool((self.scale<=0).any()):raise ValueError('positive training scale required')

    def forward(self,history:torch.Tensor)->torch.Tensor:
        return self.model((history-self.mean)/self.scale)*self.scale+self.mean
