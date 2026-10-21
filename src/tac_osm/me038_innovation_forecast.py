"""Optional Torch: native TAC dynamics coupled to an innovation memory.

Inspired by associative/delta-rule memories; no global novelty claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me037_horizon_forecast import HorizonForecaster

ARMS=('innovation_tac','context_cross','gru_cross','ar_adapt')

class InnovationMemory(nn.Module):
    """Native fast32/slow32 plus an8x8 addressable, error-corrected state."""
    def __init__(self):
        super().__init__()
        self.enc=nn.Linear(16,32)
        self.transition=nn.Linear(32,32,bias=False)
        self.read_fast=nn.Linear(8,32,bias=False)
        self.gate=nn.Linear(56,32)
        self.slow=nn.Linear(72,32)
        self.slow_gate=nn.Linear(64,32)
        self.query=nn.Linear(64,8)
        self.key=nn.Linear(48,8)
        self.value=nn.Linear(64,8)
        self.write=nn.Linear(64,1)
        self.retention=nn.Parameter(torch.tensor(math.log(.04/.0099)))
        nn.init.orthogonal_(self.transition.weight,gain=.9)
        nn.init.normal_(self.slow.weight,std=.01);nn.init.zeros_(self.slow.bias)
        with torch.no_grad():
            self.gate.bias.copy_(-torch.log(torch.logspace(math.log10(2),math.log10(16),32)-1))
            self.slow_gate.bias.copy_(-torch.log(torch.logspace(math.log10(8),math.log10(64),32)-1))

    @staticmethod
    def address(x):return x/torch.sqrt(x.square().sum(-1,keepdim=True)+1e-5)

    @staticmethod
    def innovation_write(matrix,key,value,beta,rho):
        retained=rho*matrix
        error=value-torch.bmm(retained,key.unsqueeze(-1)).squeeze(-1)
        return retained+beta.unsqueeze(-1)*error.unsqueeze(-1)*key.unsqueeze(-2)

    def step(self,u,f,s,matrix):
        query=self.address(self.query(torch.cat((f,s),-1)))
        recalled=torch.bmm(matrix,query.unsqueeze(-1)).squeeze(-1)
        gate=torch.sigmoid(self.gate(torch.cat((u,f,recalled),-1)))
        f=f+gate*(torch.tanh(self.enc(u)+self.transition(f)+self.read_fast(recalled))-f)
        state=torch.cat((f,s),-1)
        key=self.address(self.key(torch.cat((u,f),-1)))
        value=torch.tanh(self.value(state));beta=torch.sigmoid(self.write(state))
        rho=.95+.0499*torch.sigmoid(self.retention)
        matrix=self.innovation_write(matrix,key,value,beta,rho)
        recalled=torch.bmm(matrix,query.unsqueeze(-1)).squeeze(-1)
        slow_gate=torch.sigmoid(self.slow_gate(state))
        innovation=torch.tanh(self.slow(torch.cat((f-s,s,recalled),-1)))
        s=s+slow_gate*(torch.tanh(s+innovation)-s)
        return f,s,matrix

    def forward(self,patches):
        f=patches.new_zeros((patches.size(0),32));s=torch.zeros_like(f)
        matrix=patches.new_zeros((patches.size(0),8,8))
        for t in range(12):f,s,matrix=self.step(patches[:,t],f,s,matrix)
        h=torch.cat((f,s),-1)
        return h/torch.sqrt(h.square().mean(-1,keepdim=True)+.01)

class InnovationForecaster(HorizonForecaster):
    def __init__(self,arm,horizon):
        if arm not in ARMS:raise ValueError('unregistered arm')
        mapping={'innovation_tac':'tac_horizon','context_cross':'context_horizon',
                 'gru_cross':'gru_horizon','ar_adapt':'ar_adapt'}
        super().__init__(mapping[arm],horizon);self.arm=arm
        if arm=='innovation_tac':
            self.memory=InnovationMemory()
            self.context_encoder=nn.Sequential(nn.Linear(192,26),nn.SiLU(),nn.Linear(26,64))
        elif arm=='context_cross':
            self.encoder=nn.Sequential(nn.Linear(192,50),nn.SiLU(),nn.Linear(50,128))
        elif arm=='gru_cross':
            self.gru=nn.GRU(16,44,batch_first=True);self.projection=nn.Linear(44,64)
            self.context_encoder=nn.Sequential(nn.Linear(192,20),nn.SiLU(),nn.Linear(20,64))

    def parts(self,x):
        if x.ndim!=3 or x.size(1)!=96:raise ValueError('expected B x 96 x C history')
        b,_,c=x.shape;mean=x.mean(1,keepdim=True)
        scale=torch.sqrt((x-mean).square().mean(1,keepdim=True)+1e-5)
        normalized=(x-mean)/scale;flat=normalized.transpose(1,2).reshape(b*c,96)
        base=self.ar(flat)
        if self.arm=='ar_adapt':alpha=torch.zeros_like(base);correction=torch.zeros_like(base)
        else:
            shared=normalized.mean(2,keepdim=True).expand(b,96,c).transpose(1,2).reshape(b*c,96)
            patches=flat.reshape(-1,12,8);tokens=torch.cat((patches,shared.reshape(-1,12,8)),-1)
            context=torch.cat((patches.mean(-1),flat[:,-1:],flat[:,-1:]-flat[:,-8:].mean(-1,keepdim=True),
                mean.transpose(1,2).reshape(-1,1),scale.log().transpose(1,2).reshape(-1,1)),-1)
            window=torch.cat((flat,shared),-1)
            if self.arm=='context_cross':latent=self.encoder(window)
            else:
                local=self.context_encoder(window)
                if self.arm=='innovation_tac':state=self.memory(tokens)
                else:
                    _,state=self.gru(tokens);state=self.projection(state[0])
                latent=torch.cat((local,state),-1)
            a,gate=self.mix(torch.cat((context,latent),-1)).chunk(2,-1)
            raw=self.out(F.silu(self.basis(F.silu(a)*gate)))
            alpha=self.horizon_gate(context);correction=alpha*torch.tanh(raw)
        delta=correction.reshape(b,c,self.horizon).transpose(1,2)*scale
        pred=(base+correction).reshape(b,c,self.horizon).transpose(1,2)*scale+mean
        return pred,delta,alpha

    def forward(self,x):return self.parts(x)[0]

def build_model(arm,horizon,seed):
    torch.manual_seed(seed)
    return InnovationForecaster(arm,horizon)
