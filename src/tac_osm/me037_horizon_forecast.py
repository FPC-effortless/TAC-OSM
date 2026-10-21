"""Native memory and full-window context with bounded horizon-aware corrections."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me036_patch_forecast import NativePatchMemory

ARMS = ('tac_horizon', 'context_horizon', 'gru_horizon', 'ar_adapt')


class HorizonForecaster(nn.Module):
    def __init__(self, arm, horizon):
        super().__init__()
        if arm not in ARMS or horizon not in (96,192,336,720):
            raise ValueError('unregistered architecture/horizon')
        self.arm, self.horizon = arm, horizon
        self.ar = nn.Linear(96,horizon)
        nn.init.constant_(self.ar.weight,1/96);nn.init.zeros_(self.ar.bias)
        self.register_buffer('anchor_weight',self.ar.weight.detach().clone())
        self.register_buffer('anchor_bias',self.ar.bias.detach().clone())
        if arm != 'ar_adapt':
            if arm == 'context_horizon':
                self.encoder = nn.Sequential(nn.Linear(96,60),nn.SiLU(),nn.Linear(60,128))
            else:
                self.context_encoder = nn.Sequential(nn.Linear(96,42),nn.SiLU(),nn.Linear(42,64))
                if arm == 'tac_horizon':self.memory=NativePatchMemory()
                else:
                    self.gru=nn.GRU(8,34,batch_first=True)
                    self.projection=nn.Linear(34,64)
            self.mix=nn.Linear(144,128)
            self.basis=nn.Linear(64,32)
            self.out=nn.Linear(32,horizon)
            nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
            self.gate=nn.Linear(16,2)
            nn.init.zeros_(self.gate.weight)
            with torch.no_grad():self.gate.bias.copy_(torch.tensor([math.log(160/128),0.]))
            self.register_buffer('log_steps',torch.arange(1,horizon+1,dtype=torch.float32).log())

    def initialize_ar(self,state):
        with torch.no_grad():
            self.ar.weight.copy_(state['ar.weight']);self.ar.bias.copy_(state['ar.bias'])
            self.anchor_weight.copy_(self.ar.weight);self.anchor_bias.copy_(self.ar.bias)

    def anchor_loss(self):
        return (self.ar.weight-self.anchor_weight).square().mean()+(self.ar.bias-self.anchor_bias).square().mean()

    def horizon_gate(self,context):
        parameters=self.gate(context)
        tau=96.+288.*torch.sigmoid(parameters[:,:1])
        exponent=2.+4.*torch.sigmoid(parameters[:,1:])
        return torch.sigmoid(exponent*(tau.log()-self.log_steps[None,:]))

    def parts(self,x):
        if x.ndim!=3 or x.size(1)!=96:raise ValueError('expected B x 96 x C history')
        b,_,c=x.shape
        mean=x.mean(1,keepdim=True)
        scale=torch.sqrt((x-mean).square().mean(1,keepdim=True)+1e-5)
        flat=((x-mean)/scale).transpose(1,2).reshape(b*c,96)
        base=self.ar(flat)
        if self.arm=='ar_adapt':
            correction=torch.zeros_like(base)
            alpha=torch.zeros_like(base)
        else:
            patches=flat.reshape(-1,12,8)
            context=torch.cat((patches.mean(-1),flat[:,-1:],
                flat[:,-1:]-flat[:,-8:].mean(-1,keepdim=True),
                mean.transpose(1,2).reshape(-1,1),scale.log().transpose(1,2).reshape(-1,1)),-1)
            if self.arm=='context_horizon':latent=self.encoder(flat)
            else:
                local=self.context_encoder(flat)
                if self.arm=='tac_horizon':state=self.memory(patches)
                else:
                    _,state=self.gru(patches);state=self.projection(state[0])
                latent=torch.cat((local,state),-1)
            a,gate=self.mix(torch.cat((context,latent),-1)).chunk(2,-1)
            raw=self.out(F.silu(self.basis(F.silu(a)*gate)))
            alpha=self.horizon_gate(context)
            correction=alpha*torch.tanh(raw)
        delta=correction.reshape(b,c,self.horizon).transpose(1,2)*scale
        pred=(base+correction).reshape(b,c,self.horizon).transpose(1,2)*scale+mean
        return pred,delta,alpha

    def forward(self,x):return self.parts(x)[0]


def build_model(arm,horizon,seed):
    torch.manual_seed(seed)
    return HorizonForecaster(arm,horizon)
