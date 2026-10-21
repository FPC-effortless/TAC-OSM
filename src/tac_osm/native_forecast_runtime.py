"""Equivalent compiled native forecasts and fixed-kernel affine segment algebra.

No new learned weights, fits or memory equations. Torch remains optional to the
base package. Native state-dependent recurrence does not admit this kernel scan.
"""
from __future__ import annotations
import torch
from torch import nn


class NativeForecastRuntime(nn.Module):
    __constants__=['has_kernels','state_width']

    def __init__(self, source, has_kernels: bool):
        super().__init__()
        if source.coupling != 0. or not source.rms_readout:
            raise ValueError('only frozen RMS native dynamics are supported')
        self.has_kernels=has_kernels
        self.state_width=74 if has_kernels else 64
        self.emb=source.emb
        self.enc=source.enc
        self.transition=source.transition
        self.gate=source.gate
        self.slow=source.slow
        self.slow_gate=source.slow_gate
        self.head=source.head
        if self.head.in_features != (79 if has_kernels else 73):
            raise ValueError('source head does not match declared feature contract')
        self.register_buffer('duration_mean',source.duration_mean.detach().clone())
        self.register_buffer('duration_scale',source.duration_scale.detach().clone())
        self.register_buffer('item_popularity',source.item_popularity.detach().clone()
            if has_kernels else source.duration_mean.new_empty((0,)))
        half=source.duration_mean.new_tensor([2.,8.,16.,32.,64.])
        self.register_buffer('decay',torch.exp2(-1./half) if has_kernels else half.new_empty((0,)))
        weights=torch.exp2(-torch.arange(127,-1,-1,dtype=half.dtype,device=half.device)[:,None]/half[None,:])
        self.register_buffer('window_weights',weights/weights.sum(0,keepdim=True) if has_kernels else weights[:0])

    @torch.jit.export
    def advance_native(self,u:torch.Tensor,state:torch.Tensor)->torch.Tensor:
        z,s=state[:,:32],state[:,32:64]
        q=torch.tanh(self.enc(u)+self.transition(z))
        g=torch.sigmoid(self.gate(torch.cat((u,z),-1)))
        z=(1.-g)*z+g*q
        context=torch.cat((z,s),-1)
        v=torch.sigmoid(self.slow_gate(context))
        s=(1.-v)*s+v*torch.tanh(self.slow(context))
        return torch.cat((z,s),-1)

    @torch.jit.export
    def stream_step(self,item:torch.Tensor,metadata:torch.Tensor,state:torch.Tensor)->torch.Tensor:
        if state.size(1)!=self.state_width:
            raise ValueError('wrong incremental state width')
        native=self.advance_native(torch.cat((self.emb(item),metadata),-1),state[:,:64])
        if self.has_kernels:
            sums=state[:,64:69]*self.decay+metadata[:,1:2]
            counts=state[:,69:74]*self.decay+1.
            return torch.cat((native,sums,counts),-1)
        return native

    @torch.jit.export
    def readout(self,state:torch.Tensor,action:torch.Tensor,duration:torch.Tensor,
                prior:torch.Tensor)->torch.Tensor:
        if state.size(1)!=self.state_width:
            raise ValueError('wrong incremental state width')
        z,s=state[:,:32],state[:,32:64]
        features=torch.cat((z/torch.sqrt(z.square().mean(-1,keepdim=True)+.01),
                            s/torch.sqrt(s.square().mean(-1,keepdim=True)+.01)),-1)
        d=(duration-self.duration_mean)/self.duration_scale
        parts=torch.cat((features,self.emb(action),d),-1)
        if self.has_kernels:
            if bool(torch.any(state[:,69:74]<=0)):
                raise ValueError('at least one observed feedback is required')
            kernels=state[:,64:69]/state[:,69:74]
            pop=self.item_popularity[action].unsqueeze(-1)
            parts=torch.cat((parts,kernels,pop),-1)
        return self.head(parts)+torch.logit(prior)

    def forward(self,items:torch.Tensor,metadata:torch.Tensor,action:torch.Tensor,
                duration:torch.Tensor)->torch.Tensor:
        if items.size(1)!=128 or metadata.size(1)!=128 or metadata.size(2)!=3:
            raise ValueError('exactly 128 genuine observations required')
        x=torch.cat((self.emb(items),metadata),-1)
        state=x.new_zeros((items.size(0),64))
        for t in range(128): state=self.advance_native(x[:,t],state)
        if self.has_kernels:
            kernels=metadata[:,:,1]@self.window_weights
            state=torch.cat((state,kernels,torch.ones_like(kernels)),-1)
        prior=(metadata[:,-32:,1].sum(-1,keepdim=True)+1.)/34.
        return self.readout(state,action,duration,prior)


def kernel_segment(feedback:torch.Tensor):
    """Affine segment statistics (decay, weighted responses, weight total)."""
    half=feedback.new_tensor([2.,8.,16.,32.,64.])
    n=feedback.size(1)
    decay=torch.exp2(-n/half)
    ages=torch.arange(n-1,-1,-1,device=feedback.device,dtype=feedback.dtype)
    weights=torch.exp2(-ages[:,None]/half[None,:])
    sums=feedback@weights
    counts=weights.sum(0,keepdim=True).expand(feedback.size(0),-1)
    return decay,sums,counts


def compose_kernel_segments(first,second):
    """Apply first, then second. Associative over real-valued affine maps."""
    a,s,w=first; b,t,v=second
    return b*a,b*s+t,b*w+v
