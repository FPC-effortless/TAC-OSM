"""Learned real-sensor forecasts, not a scripted repair controller."""
import torch
from torch import nn


def mlp(a,b,c):return nn.Sequential(nn.Linear(a,b),nn.GELU(),nn.Linear(b,c))


class MaintenanceForecast(nn.Module):
    def __init__(self,arm='tac',width=24,aux=24,seed=0):
        super().__init__();torch.manual_seed(seed);self.arm=arm;self.width=width
        self.register_buffer('mean',torch.zeros(15));self.register_buffer('scale',torch.ones(15))
        if arm in ('tac','no_casm'):
            self.sequence=nn.GRU(15,width,batch_first=True);self.query=nn.Linear(width,width);self.key=nn.Linear(width,width);self.value=nn.Linear(width,width)
            if arm=='tac':self.ops=nn.ModuleList([mlp(width,width,width) for _ in range(3)]);self.selector=nn.Linear(width,3)
            else:self.refine=mlp(width,aux,width)
        elif arm=='gru':self.sequence=nn.GRU(15,aux,batch_first=True);self.project=nn.Linear(aux,width)
        elif arm=='mlp':self.sequence=mlp(24*15,aux,width)
        else:raise ValueError(arm)
        self.future=mlp(width,width,30)

    def forward(self,raw):
        x=(raw-self.mean)/self.scale
        if self.arm in ('tac','no_casm'):
            states,h=self.sequence(x);query=self.query(h[0]);weights=(self.key(states)*query[:,None]).sum(-1).div(self.width**.5).softmax(-1);z=h[0]+(weights[:,:,None]*self.value(states)).sum(1)
            if self.arm=='tac':
                select=self.selector(z).softmax(-1);z=z+sum(select[:,i,None]*op(z) for i,op in enumerate(self.ops))
            else:z=z+self.refine(z)
        elif self.arm=='gru':_,h=self.sequence(x);z=self.project(h[0])
        else:z=self.sequence(x.flatten(1))
        p=self.future(z);return {'normalized_mean':p[:,:15],'log_variance':p[:,15:].clamp(-6,6),'mean':p[:,:15]*self.scale+self.mean}


def configurations():
    target=sum(p.numel() for p in MaintenanceForecast().parameters());result={}
    for arm in ('tac','no_casm','gru','mlp'):
        aux=24 if arm=='tac' else min(range(2,256),key=lambda a:abs(sum(p.numel() for p in MaintenanceForecast(arm,24,a).parameters())-target))
        m=MaintenanceForecast(arm,24,aux);count=sum(p.numel() for p in m.parameters());assert abs(count/target-1)<=.05;result[arm]={'width':24,'aux':aux,'parameters':count}
    return result
