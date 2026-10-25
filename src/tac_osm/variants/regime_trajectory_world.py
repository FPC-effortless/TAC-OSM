"""ME099: one conditional uncertainty regime shared by the complete future path."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .coherent_trajectory_world import CoherentTrajectoryWorld,trajectory_moments,TrajectoryParameters

@dataclass(frozen=True)
class RegimeParameters:
    trajectory: TrajectoryParameters
    logit: torch.Tensor
    broadening: torch.Tensor

@dataclass(frozen=True)
class RegimeTrajectory:
    mean: torch.Tensor
    narrow_scale_matrix: torch.Tensor
    regime_logit: torch.Tensor
    broadening: torch.Tensor


def student4_cdf(z):
    u=z/torch.sqrt(4+z.square())
    return .5+.75*u-.25*u.pow(3)


def student4_pdf(z):
    return 12/(4+z.square()).pow(2.5)


def mixture_quantile(probability,broadening):
    """Positive .95 quantile. First-order implicit derivative; no higher-order claim.

    Root bracket [t4(.95),k*t4(.95)] follows from k>=1 and mixture monotonicity.
    48 FP64 bisections; no tuning against observed DEV errors or coverage.
    """
    p=probability.double();k=broadening.double()
    with torch.no_grad():
        lo=torch.full_like(p,2.13184678632665);hi=lo*k
        for _ in range(48):
            mid=(lo+hi)/2
            c=(1-p)*student4_cdf(mid)+p*student4_cdf(mid/k)
            lo=torch.where(c<.95,mid,lo);hi=torch.where(c>=.95,mid,hi)
        root=(lo+hi)/2
    c=(1-p)*student4_cdf(root)+p*student4_cdf(root/k)
    density=((1-p)*student4_pdf(root)+p*student4_pdf(root/k)/k).detach()
    return root+(.95-c)/density


def marginal_intervals(forecast):
    q=mixture_quantile(forecast.regime_logit.sigmoid(),forecast.broadening)
    std=forecast.narrow_scale_matrix.diagonal(dim1=-2,dim2=-1).transpose(-1,-2).sqrt()
    half=q[...,None]*std
    return forecast.mean-half,forecast.mean+half


def regime_joint_nll(forecast,truth):
    """Proper global mixture of coordinate-factorized joint-horizon Student-t.

    Shared regime across ALL horizons and coordinates, with one Cholesky.
    Component Student-t scales may remain independent across coordinates;
    the global regime couples tail events. This is not epistemic uncertainty.
    """
    e=(truth-forecast.mean).transpose(-1,-2).double()
    S=forecast.narrow_scale_matrix.double();L=torch.linalg.cholesky(S)
    q=torch.linalg.solve_triangular(L,e[...,None],upper=False).square().sum((-1,-2))
    H=e.shape[-1];nu=4.;k=forecast.broadening.double()
    c=torch.lgamma(S.new_tensor((nu+H)/2))-torch.lgamma(S.new_tensor(nu/2))-.5*H*torch.log(S.new_tensor(nu*torch.pi))
    base=c-L.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    narrow=(base-.5*(nu+H)*torch.log1p(q/nu)).sum(-1)
    broad=(base-H*k.log()-.5*(nu+H)*torch.log1p(q/(nu*k.square()))).sum(-1)
    logit=forecast.regime_logit.squeeze(-1).double()
    return -torch.logsumexp(torch.stack((F.logsigmoid(-logit)+narrow,F.logsigmoid(logit)+broad),-1),-1)


class RegimeTrajectoryWorld(CoherentTrajectoryWorld):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.regime_gate=nn.Linear(4*self.state_dim,1)
        nn.init.zeros_(self.regime_gate.weight)
        nn.init.constant_(self.regime_gate.bias,math.log(.15/.85))
        self.raw_broadening=nn.Parameter(torch.tensor(math.log(math.expm1(2.))))

    def load_coherent_parent(self,parent):
        missing,unexpected=self.load_state_dict(parent.state_dict(),strict=False)
        if set(missing)!={'regime_gate.weight','regime_gate.bias','raw_broadening'} or unexpected:
            raise ValueError('parent differs from registered coherent core')

    def condition(self,obs,action,*,valid=None,state=None):
        p,state=super().condition(obs,action,valid=valid,state=state)
        scale=self.delta_scale
        amplitude=(p.position_noise.square()+p.velocity_noise.square()).sqrt()/scale
        features=torch.cat((p.velocity/scale,p.terminal_velocity/scale,p.retention,amplitude.log()),-1)
        return RegimeParameters(p,self.regime_gate(features),1+F.softplus(self.raw_broadening)),state

    def forward(self,obs,action,*,horizons=(1,4,16),valid=None,state=None):
        p,state=self.condition(obs,action,valid=valid,state=state)
        f=trajectory_moments(p.trajectory,horizons)
        return RegimeTrajectory(f.mean,f.scale_matrix,p.logit,p.broadening),state
