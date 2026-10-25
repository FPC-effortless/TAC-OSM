"""Shared TAC v1.9: one conditioned state, coherent multi-horizon Student-t.

The learned core remains TAC's input-scheduled fast/slow/associative graph.
A separate closed-form, stable-velocity forecast law defines one future path.
No forecast is written back into the observed-world state. No future action is
assumed observed. Exact linear forecast algebra is established prior art.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .evidence_retrieval_world import EvidenceRetrievalWorld,EvidenceState
from .parallel_affine_memory import associative_affine_schedule,work_efficient_affine_prefix

@dataclass(frozen=True)
class TrajectoryParameters:
    observed: torch.Tensor
    velocity: torch.Tensor
    terminal_velocity: torch.Tensor
    retention: torch.Tensor
    position_noise: torch.Tensor
    velocity_noise: torch.Tensor
    valid: torch.Tensor

@dataclass(frozen=True)
class TrajectoryMoments:
    mean: torch.Tensor  # B,T,H,D
    scale_matrix: torch.Tensor  # B,T,D,H,H, Student-t SCALE, covariance=2*scale


def trajectory_moments(p,horizons):
    """Discrete factual frame horizons, parallel closed-form response Gram.

    x[k+1]=x[k]+v[k]+sx*eps[k],
    v[k+1]=rho*v[k]+(1-rho)*vinf+sv*eta[k].
    Gaussian innovations share a single inverse-gamma scale per channel/forecast,
    giving a coherent df=4 Student-t path. No independent per-horizon noise heads.
    """
    hs=tuple(horizons)
    if (not hs or any(type(h) is not int or not 0<=h<=256 for h in hs)
        or len(set(hs))!=len(hs)):
        raise ValueError('distinct integer horizons in0..256 required')
    h=torch.tensor(hs,device=p.observed.device,dtype=p.observed.dtype)
    rho=p.retention
    def gain(n):
        # Stable (1-rho**n)/(1-rho), including exactly zero n.
        logr=rho.log()
        return -torch.expm1(logr[...,None]*n)/(-torch.expm1(logr))[...,None]
    g=gain(h)
    displacement=g*p.velocity[...,None]+(h-g)*p.terminal_velocity[...,None]
    mu=p.observed[...,None]+displacement
    # Velocity innovation at j first affects position at j+2.
    j=torch.arange(max(max(hs),1),device=h.device,dtype=h.dtype)
    W=(j[None,:]<h[:,None]).to(h.dtype)
    n=(h[:,None]-1-j[None,:]).clamp_min(0)
    # Accumulate the response Gram in FP64. Large correlated future responses
    # should not destroy positive definiteness through FP32 cancellation.
    logr=rho.double().log();n=n.double();W=W.double()
    K=-torch.expm1(logr[...,None,None]*n)/(-torch.expm1(logr))[...,None,None]
    S=p.position_noise.double()[...,None,None].square()*(W@W.T)
    S=S+p.velocity_noise.double()[...,None,None].square()*(K@K.transpose(-1,-2))
    S=torch.where(p.valid[...,None,None,None],S,torch.zeros_like(S))
    # Invalid events neither predict a change nor manufacture a future density.
    mu=torch.where(p.valid[...,None,None],mu,p.observed[...,None].expand_as(mu))
    return TrajectoryMoments(mu.transpose(-1,-2),S)


def joint_student_nll(moments,truth):
    """Proper df4 joint horizon log density, independent observed coordinates.

    Zero-horizon/invalid rows are not scored. Distinct positive horizons and
    positive position noise make the scale matrix positive definite. No jitter
    or altered covariance is silently introduced when Cholesky fails.
    """
    residual=(truth-moments.mean).transpose(-1,-2).double()
    S=moments.scale_matrix.double()
    chol=torch.linalg.cholesky(S)
    q=torch.linalg.solve_triangular(chol,residual[...,None],upper=False).square().sum((-1,-2))
    H=residual.shape[-1];nu=4.
    c=torch.lgamma(S.new_tensor(nu/2))-torch.lgamma(S.new_tensor((nu+H)/2))
    c=c+.5*H*torch.log(S.new_tensor(nu*torch.pi))
    return c+chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)+.5*(nu+H)*torch.log1p(q/nu)


class CoherentTrajectoryWorld(EvidenceRetrievalWorld):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        hidden=self.readout[0].out_features
        self.trajectory_head=nn.Linear(hidden,4*self.state_dim)
        nn.init.normal_(self.trajectory_head.weight,std=.001)
        nn.init.zeros_(self.trajectory_head.bias)

    def load_one_step_parent(self,parent):
        missing,unexpected=self.load_state_dict(parent.state_dict(),strict=False)
        if set(missing)!={'trajectory_head.weight','trajectory_head.bias'} or unexpected:
            raise ValueError('parent does not match unchanged one-step core')

    def condition(self, obs, action, *, valid=None, state=None):
        if (obs.ndim != 3 or action.shape != (*obs.shape[:2], self.action_dim)
                or obs.shape[-1] != self.state_dim or obs.shape[1] == 0):
            raise ValueError('nonempty matching [B,T,D] observation/action required')
        B,T,_ = obs.shape
        if obs.dtype != action.dtype or obs.device != action.device:
            raise ValueError('observation/action dtype/device mismatch')
        if valid is None:
            valid = torch.ones(B,T, device=obs.device, dtype=torch.bool)
        if valid.shape != (B,T) or valid.dtype != torch.bool or valid.device != obs.device:
            raise ValueError('valid must be matching bool [B,T]')
        if not (torch.isfinite(obs).all() and torch.isfinite(action).all()):
            raise ValueError('inputs must be finite, including padding')
        state = self.initial_state(B) if state is None else state
        if (state.basis_sum.shape != (B,self.width) or state.count.shape != (B,1)
            or state.memory.shape != (B,8,8) or state.count.dtype != torch.int64
            or torch.any(state.count < 0)):
            raise ValueError('invalid carried state shape/count')
        if any(x.device != obs.device for x in (state.basis_sum,state.count,state.memory)):
            raise ValueError('state device mismatch')
        if any(x.dtype != obs.dtype or not torch.isfinite(x).all()
               for x in (state.basis_sum,state.memory)):
            raise ValueError('state floating dtype/nonfinite mismatch')
        e = self.encoder(torch.cat((obs,action),-1)).tanh()
        k = F.normalize(self.address(e),dim=-1,eps=1e-6)
        v = F.normalize(self.content(e),dim=-1,eps=1e-6)
        zr = e.new_zeros(B,T,8)
        basis = self.fast_map(torch.cat((e,e.new_zeros(B,T,self.width),zr),-1)).chunk(2,-1)[0].tanh()
        basis = torch.where(valid[...,None],basis,torch.zeros_like(basis))
        sums = state.basis_sum[:,None] + basis.cumsum(1)
        counts = state.count[:,None] + valid.to(torch.int64).cumsum(1)[...,None]
        ctx = (sums-basis)/(counts-valid.to(torch.int64)[...,None]).clamp_min(1).to(obs.dtype)
        p,g = self.fast_map(torch.cat((e,ctx,zr),-1)).chunk(2,-1)
        pf = torch.lerp(ctx,p.tanh(),g.sigmoid())
        d,g = self.slow_map(torch.cat((pf,ctx,zr),-1)).chunk(2,-1)
        ps = torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
        alpha,beta = self.memory_gates(torch.cat((e,pf,ps),-1)).sigmoid().chunk(2,-1)
        alpha = alpha*(1-beta/4)
        P,Q = associative_affine_schedule(k,v,alpha,beta,valid[...,None])
        memory = work_efficient_affine_prefix(P,Q,state.memory)
        previous = torch.cat((state.memory[:,None],memory[:,:-1]),1)
        # Read BEFORE current write, with a separately learned context query.
        q = F.normalize(k + self.query(torch.cat((e,pf,ps),-1)),dim=-1,eps=1e-6)
        r = (previous @ q[...,None]).squeeze(-1)
        f,g = self.fast_map(torch.cat((e,pf,r),-1)).chunk(2,-1)
        f = torch.lerp(pf,f.tanh(),g.sigmoid())
        d,g = self.slow_map(torch.cat((f,ps,r),-1)).chunk(2,-1)
        s = torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
        hidden = self.readout[1](self.readout[0](torch.cat((f,s,r,e),-1)))
        delta = self.readout[2](hidden) + self.retrieved_delta(r)
        v0 = self.delta_scale*delta.tanh()
        extra = self.trajectory_head(hidden)
        vinf,raw_rho,raw_sx,raw_sv = extra.chunk(4,-1)
        amplitude=self.delta_scale*(.02+F.softplus(raw_sx))
        share=.05+.90*raw_sv.sigmoid()
        params = TrajectoryParameters(obs,v0,self.delta_scale*vinf.tanh(),
             .01+.98*raw_rho.sigmoid(),
             amplitude*share.sqrt(),amplitude*(1-share).sqrt(),valid)
        final = EvidenceState(sums[:,-1],counts[:,-1],memory[:,-1])
        return params,final

    def forward(self,obs,action,*,horizons=(1,4,16),valid=None,state=None):
        params,state = self.condition(obs,action,valid=valid,state=state)
        return trajectory_moments(params,horizons),state
