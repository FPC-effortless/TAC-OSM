"""Centered state/query corrections with optional frozen common prediction path."""
import math
import torch
from torch import nn


class FrozenContext(nn.Module):
    """R-only common features; no recurrent body or unused source readout weights."""
    def __init__(self, embedding: torch.Tensor, mean: float, scale: float,
                 popularity: torch.Tensor, feature_mean: torch.Tensor,
                 feature_scale: torch.Tensor):
        super().__init__()
        self.emb=nn.Embedding.from_pretrained(embedding.detach().clone(),freeze=True)
        self.register_buffer('duration_mean',torch.tensor(mean))
        self.register_buffer('duration_scale',torch.tensor(scale))
        self.register_buffer('item_popularity',popularity.detach().clone())
        self.register_buffer('halflives',torch.tensor([2.,8.,16.,32.,64.]))
        self.register_buffer('feature_mean',feature_mean.detach().clone())
        self.register_buffer('feature_scale',feature_scale.detach().clone())

    def forward(self,items:torch.Tensor,meta:torch.Tensor,action:torch.Tensor,
                duration:torch.Tensor)->torch.Tensor:
        out=torch.jit.annotate(list[torch.Tensor],[])
        for k in [32,64,96,128]:
            ages=torch.arange(k-1,-1,-1,device=meta.device,dtype=meta.dtype)
            weights=torch.exp2(-ages[:,None]/self.halflives[None,:])
            kernels=meta[:,:k,1]@(weights/weights.sum(0,keepdim=True))
            pooled=torch.cat((self.emb(items[:,:k]).mean(1),meta[:,:k].mean(1)),-1)
            if k<128:
                offered=items[:,k];d=meta[:,k,0:1]
            else:
                offered=action;d=duration
            x=torch.cat((self.emb(offered),(d-self.duration_mean)/self.duration_scale,
                kernels,self.item_popularity[offered].unsqueeze(-1),pooled),-1)
            prior=(meta[:,k-32:k,1].sum(-1,keepdim=True)+1.)/34.
            out.append(torch.cat(((x-self.feature_mean)/self.feature_scale,torch.logit(prior)),-1))
        return torch.stack(out,1)


def split_rms(state:torch.Tensor)->torch.Tensor:
    z,s=state[...,:32],state[...,32:]
    return torch.cat((z/torch.sqrt(z.square().mean(-1,keepdim=True)+.01),
                      s/torch.sqrt(s.square().mean(-1,keepdim=True)+.01)),-1)


class NativeMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(16000,8)
        self.enc=nn.Linear(11,32);self.transition=nn.Linear(32,32,bias=False)
        self.gate=nn.Linear(43,32);self.slow=nn.Linear(64,32);self.slow_gate=nn.Linear(64,32)
        nn.init.constant_(self.gate.bias,math.log(1/7));nn.init.constant_(self.slow_gate.bias,math.log(1/23))

    @torch.jit.export
    def advance(self,u:torch.Tensor,state:torch.Tensor)->torch.Tensor:
        z,s=state[:,:32],state[:,32:]
        q=torch.tanh(self.enc(u)+self.transition(z))
        g=torch.sigmoid(self.gate(torch.cat((u,z),-1)))
        z=(1.-g)*z+g*q;ctx=torch.cat((z,s),-1)
        v=torch.sigmoid(self.slow_gate(ctx))
        return torch.cat((z,(1.-v)*s+v*torch.tanh(self.slow(ctx))),-1)

    def forward(self,items:torch.Tensor,meta:torch.Tensor)->torch.Tensor:
        x=torch.cat((self.emb(items),meta),-1)
        state=x.new_zeros((items.size(0),64));out=torch.jit.annotate(list[torch.Tensor],[])
        for t in range(128):
            state=self.advance(x[:,t],state)
            if t==31 or t==63 or t==95 or t==127:out.append(split_rms(state))
        return torch.stack(out,1)


class PooledMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(16000,8)
        self.pool=nn.Sequential(nn.Linear(11,56),nn.Tanh(),nn.Linear(56,56),
                                nn.Tanh(),nn.Linear(56,64))

    def forward(self,items:torch.Tensor,meta:torch.Tensor)->torch.Tensor:
        values=torch.cat((self.emb(items),meta),-1)
        out=torch.jit.annotate(list[torch.Tensor],[])
        for k in [32,64,96,128]:out.append(split_rms(self.pool(values[:,:k].mean(1))))
        return torch.stack(out,1)


class CommonHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(26,43);self.output=nn.Linear(43,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.output(torch.tanh(self.hidden(x))).squeeze(-1)


class CenteredReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(90,13)
        self.output=nn.Linear(13,1,bias=False)  # A centered output bias would cancel identically.
        self.direct=nn.Linear(64,1,bias=False)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.direct.weight)

    def forward(self,h:torch.Tensor,x:torch.Tensor)->torch.Tensor:
        actual=self.hidden(torch.cat((h,x),-1))
        anchor=self.hidden(torch.cat((torch.zeros_like(h),x),-1))
        return (self.direct(h)+self.output(torch.tanh(actual)-torch.tanh(anchor))).squeeze(-1)


class ConditionalMemory(nn.Module):
    def __init__(self,pooled:bool,joint:bool):
        super().__init__()
        self.memory=PooledMemory() if pooled else NativeMemory()
        self.common=CommonHead();self.correction=CenteredReadout()
        for p in self.common.parameters():p.requires_grad_(joint)

    def forward(self,items:torch.Tensor,meta:torch.Tensor,context:torch.Tensor)->torch.Tensor:
        x=context[...,:26]
        return context[...,26]+self.common(x)+self.correction(self.memory(items,meta),x)


COUNTS={'conditional_native':136236,'joint_native':137441,'conditional_pool':136772}


def make_student(arm,common_state):
    if arm not in COUNTS:raise ValueError(arm)
    net=ConditionalMemory(arm=='conditional_pool',arm=='joint_native')
    net.common.load_state_dict(common_state,strict=True)
    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==COUNTS[arm]
    return net
