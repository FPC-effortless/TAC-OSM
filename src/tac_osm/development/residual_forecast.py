"""Persistence-start point forecasts. Torch-only; no maintenance-action rules."""
import torch
from torch import nn


class ResidualForecast(nn.Module):
    def __init__(self,kind='linear',revin=False,channels=8,context=24,width=32,seed=0):
        super().__init__();torch.manual_seed(seed);self.kind=kind;self.revin=revin;self.channels=channels;self.context=context
        self.register_buffer('location',torch.zeros(channels));self.register_buffer('scale',torch.ones(channels))
        if kind=='linear':self.network=nn.Linear(context*channels,channels)
        elif kind=='mlp':self.network=nn.Sequential(nn.Linear(context*channels,width),nn.GELU(),nn.Linear(width,channels))
        else:raise ValueError(kind)
        head=self.network if kind=='linear' else self.network[-1]
        nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(self.context,self.channels):raise ValueError('Expected batch x24x8 raw analog values')
        if self.revin:
            center=x.mean(1,keepdim=True);scale=x.std(1,keepdim=True,unbiased=False).clamp_min(1e-5)
        else:center=self.location[None,None];scale=self.scale[None,None]
        delta=self.network(((x-center)/scale).flatten(1));return x[:,-1]+delta*scale[:,0]


def specifications():
    return {f'{kind}_{norm}':{'kind':kind,'revin':norm=='revin','channels':8,'context':24,'width':32} for kind in ('linear','mlp') for norm in ('static','revin')}
