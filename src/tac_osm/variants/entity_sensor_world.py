"""TAC-Temporal candidate: source-aligned traces and shared associative scan.

The thirteen encoded sensor channels include sin/cos of the published wind tag.
No physical action or invented geography is supplied. States belong to one
source station/model instance; queries never transfer state between owners.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .retention_controlled_world import ControlledGateLogits
from .parallel_affine_memory import associative_affine_schedule,work_efficient_affine_prefix


def diagonal_affine_prefix(a,b,initial):
    """Inclusive chronological elementwise affine scan; O(T) work/logT span."""
    if a.ndim!=3 or b.shape!=a.shape or initial.shape!=(a.shape[0],a.shape[2]) or a.shape[1]<1:
        raise ValueError('matching nonempty [B,T,C] affine coefficients required')
    B,T,C=a.shape;n=1<<(T-1).bit_length()
    if n>T:
        a=torch.cat((a,a.new_ones(B,n-T,C)),1);b=torch.cat((b,b.new_zeros(B,n-T,C)),1)
    levels=[(a,b)]
    while levels[-1][0].shape[1]>1:
        p,q=levels[-1];levels.append((p[:,1::2]*p[:,::2],q[:,1::2]+p[:,1::2]*q[:,::2]))
    p=a.new_ones(B,1,C);q=b.new_zeros(B,1,C)
    for depth in range(len(levels)-2,-1,-1):
        cp,cq=levels[depth];rp=cp[:,::2]*p;rq=cq[:,::2]+cp[:,::2]*q
        p=torch.stack((p,rp),2).reshape(B,-1,C);q=torch.stack((q,rq),2).reshape(B,-1,C)
    return ((p*a)*initial[:,None]+q*a+b)[:,:T]


@dataclass(frozen=True)
class EntityEvidenceState:
    source_sum: torch.Tensor  # B,13
    source_mass: torch.Tensor # B,13
    evidence_sum: torch.Tensor # B,62
    count: torch.Tensor       # B,1 int64
    memory: torch.Tensor      # B,8,8
    owner: torch.Tensor       # B int64, actual source station code
    last_hour: torch.Tensor   # B int64, source-naive monotonically increasing hour
    model_token: int
    parameter_versions: tuple

@dataclass(frozen=True)
class SensorForecast:
    log_location: torch.Tensor # B,6, standardized log1p pollutant
    log_scale: torch.Tensor    # B,6, normal scale in the same units


class EntitySensorWorld(nn.Module):
    def __init__(self):
        super().__init__();self.width=88
        self.station=nn.Embedding(12,4)
        self.encoder=nn.Linear(49,48)
        self.address=nn.Linear(48,8);self.content=nn.Linear(48,8)
        self.source_admission=nn.Linear(48,13)
        self.fast_map=nn.Linear(48+88+8,176);self.slow_map=nn.Linear(176+8,176)
        self.memory_gates=ControlledGateLogits(nn.Linear(48+176,2))
        self.query=nn.Linear(48+176,8)
        self.readout=nn.Sequential(nn.Linear(176+8+48,132),nn.SiLU(),nn.Linear(132,12))
        self.retrieved_delta=nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,12))
        with torch.no_grad():
            self.readout[2].weight[:6].zero_();self.readout[2].bias[:6].zero_()
            self.retrieved_delta[2].weight[:6].zero_();self.retrieved_delta[2].bias[:6].zero_()
        self.register_buffer('target_mean',torch.zeros(6));self.register_buffer('target_scale',torch.ones(6))

    def versions(self):return tuple(p._version for p in self.parameters())

    def initial_state(self,owners):
        p=next(self.parameters());B=owners.numel()
        return EntityEvidenceState(p.new_zeros(B,13),p.new_zeros(B,13),p.new_zeros(B,62),
          torch.zeros(B,1,dtype=torch.int64,device=p.device),p.new_zeros(B,8,8),owners.clone(),
          torch.full((B,),-2**60,dtype=torch.int64,device=p.device),id(self),self.versions())

    def forward(self,values,observed,ages,clock,owners,hours,*,state=None,return_trace=False):
        if values.ndim!=3 or values.shape[-1]!=13 or values.shape[1]<1:
            raise ValueError('values require nonempty [B,T,13]')
        B,T,_=values.shape
        if observed.shape!=values.shape or observed.dtype!=torch.bool or ages.shape!=values.shape or clock.shape!=(B,T,6):raise ValueError('mask/age/clock shape mismatch')
        if owners.shape!=(B,) or owners.dtype!=torch.int64 or hours.shape!=(B,T) or hours.dtype!=torch.int64:raise ValueError('owner/hour shape mismatch')
        if torch.any(owners<0) or torch.any(owners>=12):raise ValueError('unknown source entity')
        if any(x.device!=values.device for x in (observed,ages,clock,owners,hours)):raise ValueError('device mismatch')
        if any(x.dtype!=values.dtype or not torch.isfinite(x).all() for x in (ages,clock)) or not torch.isfinite(values).all():raise ValueError('nonfinite or mismatched floating inputs')
        if torch.any(hours[:,1:]<=hours[:,:-1]):raise ValueError('nonchronological source')
        if torch.any(values.abs()>8.00001) or torch.any(ages<0) or torch.any(ages>1):raise ValueError('input normalization contract violated')
        state=self.initial_state(owners) if state is None else state
        if state.model_token!=id(self) or state.parameter_versions!=self.versions():raise ValueError('stale/foreign model state')
        if not torch.equal(state.owner,owners) or torch.any(hours[:,0]<=state.last_hour):raise ValueError('cross-entity or stale/out-of-order state')
        expected=[(state.source_sum,(B,13)),(state.source_mass,(B,13)),(state.evidence_sum,(B,62)),(state.memory,(B,8,8))]
        if any(x.shape!=shape or x.dtype!=values.dtype or x.device!=values.device or not torch.isfinite(x).all() for x,shape in expected):raise ValueError('invalid state payload')
        if state.count.shape!=(B,1) or state.count.dtype!=torch.int64 or torch.any(state.count<0):raise ValueError('invalid evidence count')
        # Unobserved numeric values are unconditionally zeroed. No forward fill.
        safe=torch.where(observed,values,torch.zeros_like(values))
        sid=self.station(owners)[:,None].expand(B,T,4)
        e=self.encoder(torch.cat((safe,observed.to(values.dtype),ages,clock,sid),-1)).tanh()
        beta=observed.to(values.dtype)*self.source_admission(e).sigmoid()
        aa=torch.cat((1-beta,1-beta),-1);bb=torch.cat((beta*safe/8,beta),-1)
        trace=diagonal_affine_prefix(aa,bb,torch.cat((state.source_sum,state.source_mass),-1))
        old=torch.cat((torch.cat((state.source_sum,state.source_mass),-1)[:,None],trace[:,:-1]),1)
        usable=observed.any(-1)
        zero=e.new_zeros(B,T,8)
        basis=self.fast_map(torch.cat((e,e.new_zeros(B,T,88),zero),-1)).chunk(2,-1)[0].tanh()[...,26:]
        basis=torch.where(usable[...,None],basis,torch.zeros_like(basis))
        sums=state.evidence_sum[:,None]+basis.cumsum(1)
        counts=state.count[:,None]+usable.to(torch.int64).cumsum(1)[...,None]
        ctx=torch.cat((old[...,:13]/old[...,13:].clamp_min(1e-6),old[...,13:],
            (sums-basis)/(counts-usable.to(torch.int64)[...,None]).clamp_min(1).to(values.dtype)),-1)
        p,g=self.fast_map(torch.cat((e,ctx,zero),-1)).chunk(2,-1);pf=torch.lerp(ctx,p.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat((pf,ctx,zero),-1)).chunk(2,-1);ps=torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
        alpha,b=self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1);alpha=alpha*(1-b/4)
        k=F.normalize(self.address(e),dim=-1,eps=1e-6);v=F.normalize(self.content(e),dim=-1,eps=1e-6)
        P,Q=associative_affine_schedule(k,v,alpha,b,usable[...,None]);M=work_efficient_affine_prefix(P,Q,state.memory)
        prev=state.memory if T==1 else M[:,-2]
        q=F.normalize(k[:,-1]+self.query(torch.cat((e[:,-1],pf[:,-1],ps[:,-1]),-1)),dim=-1,eps=1e-6)
        r=(prev@q[...,None]).squeeze(-1)
        f,g=self.fast_map(torch.cat((e[:,-1],pf[:,-1],r),-1)).chunk(2,-1);f=torch.lerp(pf[:,-1],f.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat((f,ps[:,-1],r),-1)).chunk(2,-1);s=torch.lerp(ps[:,-1],(ps[:,-1]+d.tanh()).tanh(),.25*g.sigmoid())
        raw=self.readout(torch.cat((f,s,r,e[:,-1]),-1))+self.retrieved_delta(r)
        delta,scale=raw.chunk(2,-1)
        retained=8*trace[:,-1,:6]/trace[:,-1,13:19].clamp_min(1e-6)
        anchor=torch.where(observed[:,-1,:6],safe[:,-1,:6],retained)
        forecast=SensorForecast(anchor+2*delta.tanh(),.05+F.softplus(scale))
        final=EntityEvidenceState(trace[:,-1,:13],trace[:,-1,13:],sums[:,-1],counts[:,-1],M[:,-1],owners.clone(),hours[:,-1],id(self),self.versions())
        if return_trace:return forecast,final,dict(admission=beta,source_trace=trace,context=ctx,memory=M)
        return forecast,final

    def raw_quantiles(self,forecast):
        q=forecast.log_location[...,None]+forecast.log_scale[...,None]*forecast.log_scale.new_tensor([-1.6448536269514722,0.,1.6448536269514722])
        return torch.expm1(q*self.target_scale[None,:,None]+self.target_mean[None,:,None]).clamp_min(0)

    def nll(self,forecast,raw_targets,target_mask):
        y=torch.log1p(raw_targets.clamp_min(0));z=(y-self.target_mean)/self.target_scale
        mu,sigma=forecast.log_location,forecast.log_scale
        ordinary=.5*((z-mu)/sigma).square()+sigma.log()+.5*torch.log(sigma.new_tensor(2*torch.pi))+self.target_scale.log()+y
        threshold=-self.target_mean/self.target_scale
        atom=-torch.special.log_ndtr((threshold-mu)/sigma)
        loss=torch.where(raw_targets==0,atom,ordinary)
        return torch.where(target_mask,loss,torch.zeros_like(loss)).sum()/target_mask.sum().clamp_min(1)
