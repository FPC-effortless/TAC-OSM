"""Source-pinned Mamba-2 geometry with explicit reference/official backends.

Reference is a transparent CPU mathematical implementation, NOT an installed
Mamba package, optimized kernel, or an inference-efficiency baseline. Official
backend fails closed unless actual CUDA + pinned upstream source are available.
No fallback is silently substituted and no optimizer is included.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .decision_sys1_ssd_reference_003 import ssd_chunk_reference

UPSTREAM_COMMIT = '6b72c122713bb769cc82c6b8e6d019c53d27d6a1'
UPSTREAM_MODULE_BLOB = '36b16d4715ed2d753697167d65da5e6abba22017'


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x, z=None):
        # Mamba2 norm_before_gate=False: apply SiLU gate BEFORE normalization.
        if z is not None:
            x = x*F.silu(z)
        # FP64 is a mathematical reference, not upstream Triton FP64 support.
        work = x if x.dtype == torch.float64 else x.float()
        return (work*torch.rsqrt(work.square().mean(-1,keepdim=True)+self.eps)
                *self.weight).to(x.dtype)


@dataclass(frozen=True)
class MambaState:
    conv: torch.Tensor
    ssm: torch.Tensor


class Mamba2ReferenceBlock(nn.Module):
    """Restricted genuine SSD geometry: 1 group, full SSM, scalar D/head.

    Includes x/B/C causal convolution, input-dependent dt/B/C, per-head A,
    D skip, gated RMS normalization and output projection. Not a simple EMA.
    State updates are out-of-place so the reference supports autograd.
    """
    def __init__(self, d_model=64, d_state=64, headdim=32, expand=2, d_conv=4):
        super().__init__()
        self.d_model,self.d_state,self.headdim = d_model,d_state,headdim
        self.d_inner,self.d_conv = expand*d_model,d_conv
        if min(d_model,d_state,headdim,expand,d_conv) < 1 or self.d_inner%headdim:
            raise ValueError('invalid Mamba-2 geometry')
        self.nheads = self.d_inner//headdim
        self.conv_dim = self.d_inner+2*d_state
        self.in_proj = nn.Linear(d_model,2*self.d_inner+2*d_state+self.nheads,bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim,self.conv_dim,d_conv,
                              groups=self.conv_dim,padding=d_conv-1,bias=True)
        dt = torch.exp(torch.rand(self.nheads)*(math.log(.1)-math.log(.001))+math.log(.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt+torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.nheads).uniform_(1,16).log())
        self.D = nn.Parameter(torch.ones(self.nheads))
        for p in (self.dt_bias,self.A_log,self.D):
            p._no_weight_decay = True
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner,d_model,bias=False)

    def initial(self,batch):
        p = self.in_proj.weight
        return MambaState(p.new_zeros(batch,self.conv_dim,self.d_conv),
                          p.new_zeros(batch,self.nheads,self.headdim,self.d_state))

    def _check(self,x,state):
        if x.ndim!=2 or x.shape[1]!=self.d_model or not torch.isfinite(x).all():
            raise ValueError('invalid block input')
        p=self.in_proj.weight
        if x.dtype!=p.dtype or x.device!=p.device:
            raise ValueError('input dtype/device differs from model')
        expected=((x.shape[0],self.conv_dim,self.d_conv),
                  (x.shape[0],self.nheads,self.headdim,self.d_state))
        for value,shape in zip((state.conv,state.ssm),expected):
            if value.shape!=shape or value.dtype!=x.dtype or value.device!=x.device or not torch.isfinite(value).all():
                raise ValueError('invalid/mismatched state')

    def step(self,x,state):
        self._check(x,state)
        z,xbc,dt = torch.split(self.in_proj(x),[self.d_inner,self.conv_dim,self.nheads],-1)
        conv = torch.cat((state.conv[:,:,1:],xbc[:,:,None]),-1)
        xbc = F.silu((conv*self.conv1d.weight[:,0]).sum(-1)+self.conv1d.bias)
        v,b,c = torch.split(xbc,[self.d_inner,self.d_state,self.d_state],-1)
        v = v.reshape(x.shape[0],self.nheads,self.headdim)
        dt = F.softplus(dt+self.dt_bias)
        a = -self.A_log.exp()
        ssm = state.ssm*(dt*a).exp()[:,:,None,None]+torch.einsum('bh,bn,bhp->bhpn',dt,b,v)
        y = torch.einsum('bhpn,bn->bhp',ssm,c)+self.D[None,:,None]*v
        return self.out_proj(self.norm(y.flatten(1),z)),MambaState(conv,ssm)

    def forward(self,x,state=None,block_len=16):
        if x.ndim!=3 or x.shape[1]<1:
            raise ValueError('nonempty batch/time/width input required')
        state = self.initial(x.shape[0]) if state is None else state
        self._check(x[:,0],state)
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite sequence')
        z,xbc,dt = torch.split(self.in_proj(x),[self.d_inner,self.conv_dim,self.nheads],-1)
        history = torch.cat((state.conv[:,:,1:],xbc.transpose(1,2)),-1)
        conv = F.conv1d(history,self.conv1d.weight,self.conv1d.bias,groups=self.conv_dim).transpose(1,2)
        v,b,c = torch.split(F.silu(conv),[self.d_inner,self.d_state,self.d_state],-1)
        v = v.reshape(*v.shape[:2],self.nheads,self.headdim)
        b = b[:,:,None,:].expand(-1,-1,self.nheads,-1)
        c = c[:,:,None,:].expand(-1,-1,self.nheads,-1)
        dt = F.softplus(dt+self.dt_bias)
        a = dt*(-self.A_log.exp())
        vx = v*dt[:,:,:,None]
        # Padding with zero dt*A and zero writes makes padding a true no-op.
        length=x.shape[1]
        pad=(-length)%block_len
        def pad_time(t):
            return torch.cat((t,t.new_zeros(t.shape[0],pad,*t.shape[2:])),1) if pad else t
        y,ssm = ssd_chunk_reference(pad_time(vx),pad_time(a),pad_time(b),pad_time(c),block_len,state.ssm)
        y = y[:,:length]+v*self.D[None,None,:,None]
        out = self.out_proj(self.norm(y.flatten(-2),z))
        conv_state = torch.cat((state.conv,xbc.transpose(1,2)),-1)[:,:,-self.d_conv:]
        return out,MambaState(conv_state,ssm)


def official_mamba_class():
    if not torch.cuda.is_available():
        raise RuntimeError('official CUDA kernel qualification unavailable; no CPU fallback')
    if importlib.util.find_spec('mamba_ssm') is None:
        raise RuntimeError('official mamba_ssm is not installed')
    from mamba_ssm.modules.mamba2 import Mamba2
    raw=Path(inspect.getfile(Mamba2)).read_bytes()
    blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    if blob!=UPSTREAM_MODULE_BLOB:
        raise RuntimeError(f'upstream module source pin mismatch: {blob}')
    return Mamba2


class Mamba2RiskModel(nn.Module):
    """~100K two-layer Mamba-2 risk reference; backend is mandatory/explicit."""
    def __init__(self, backend, context_size=212, width=64, layers=2, head_width=61):
        super().__init__()
        if backend not in ('reference','official'):
            raise ValueError('explicit reference or official backend required')
        self.backend,self.context_size=backend,context_size
        self.input=nn.Linear(context_size,width)
        self.norms=nn.ModuleList(RMSNorm(width) for _ in range(layers))
        factory = Mamba2ReferenceBlock if backend=='reference' else official_mamba_class()
        self.blocks=nn.ModuleList(factory(d_model=width,d_state=64,headdim=32,
                                          expand=2,d_conv=4) for _ in range(layers))
        self.norm=RMSNorm(width)
        self.head=nn.Sequential(nn.Linear(context_size+width,head_width),nn.SiLU(),nn.Linear(head_width,4))

    def initial(self,batch):
        if self.backend=='reference':
            return tuple(block.initial(batch) for block in self.blocks)
        return tuple(MambaState(*block.allocate_inference_cache(batch,0)) for block in self.blocks)

    def forward(self,x,state=None):
        if x.ndim!=3 or x.shape[-1]!=self.context_size or x.shape[1]<1:
            raise ValueError('invalid risk sequence shape')
        z=self.input(x)
        if self.backend=='official':
            if state is not None:
                raise ValueError('official full-sequence nonzero carry requires separate qualification')
            for norm,block in zip(self.norms,self.blocks):
                z=z+block(norm(z))
            return self.head(torch.cat((x,self.norm(z)),-1)),None
        state=self.initial(x.shape[0]) if state is None else state
        if len(state)!=len(self.blocks):
            raise ValueError('state layer count mismatch')
        fresh=[]
        for norm,block,s in zip(self.norms,self.blocks,state):
            value,s=block(norm(z),s)
            z=z+value
            fresh.append(s)
        return self.head(torch.cat((x,self.norm(z)),-1)),tuple(fresh)

    def step(self,x,state):
        if x.ndim!=2 or x.shape[-1]!=self.context_size or len(state)!=len(self.blocks):
            raise ValueError('invalid risk input/state')
        z=self.input(x)
        fresh=[]
        for norm,block,s in zip(self.norms,self.blocks,state):
            if self.backend=='reference':
                value,s=block.step(norm(z),s)
            else:
                # Official inference mutates buffers; isolate caller-owned state.
                value,conv,ssm=block.step(norm(z)[:,None],s.conv.clone(),s.ssm.clone())
                value,s=value[:,0],MambaState(conv,ssm)
            z=z+value
            fresh.append(s)
        return self.head(torch.cat((x,self.norm(z)),-1)),tuple(fresh)


def count_and_state_receipt():
    model=Mamba2RiskModel('reference')
    state=model.initial(1)
    return {'parameters':sum(p.numel() for p in model.parameters()),
            'state_scalars_per_entity':sum(s.conv.numel()+s.ssm.numel() for s in state),
            'backend':'CPU_MATHEMATICAL_REFERENCE',
            'upstream_commit':UPSTREAM_COMMIT,'official_kernel_qualified':False,
            'cuda_available':torch.cuda.is_available(),'optimizer_steps':0}
