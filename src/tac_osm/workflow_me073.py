"""Native autoregressive marked trajectories; imagined state is never observed state.

Timing is a hurdle/lognormal mixture conditioned on the sampled activity.
Bounded per-step parameters do not truncate real delays or promise calibration.
"""
from typing import NamedTuple
import math
import torch
from torch import nn
from torch.nn import functional as F
from .workflow_me069 import EventWorldModel, EventBelief

STEPS=8
HORIZONS=(1,2,4,8)
PARTICLES=256


class StepLaw(NamedTuple):
    decoded: torch.Tensor
    mark_log_probability: torch.Tensor
    log_survival: torch.Tensor


class ClockLaw(NamedTuple):
    log_zero: torch.Tensor
    log_weight: torch.Tensor
    location: torch.Tensor
    log_scale: torch.Tensor


class TrajectoryModel(EventWorldModel):
    def __init__(self,pre):
        super().__init__(pre,"native_tac",64,128)
        del self.decoder,self.output
        self.initial_context=nn.Linear(48,128)
        self.project=nn.Linear(184,96)
        self.step_embedding=nn.Embedding(STEPS,8)
        self.readout=nn.Linear(104,256)
        n=len(pre["activities"])+1
        self.mark_head=nn.Linear(128,n)
        self.existence_head=nn.Linear(128,1)
        self.clock_head=nn.Linear(160,10) # zero, three weights/locations/scales
        self.imagined_encoder=nn.Linear(34,48)
        self.prior_strength=nn.Parameter(torch.zeros(STEPS))
        self.register_buffer("existence_prior",torch.tensor(pre["step_survival_logit"],dtype=torch.float32))
        self.register_buffer("clock_locations",torch.tensor(pre["pair_clock_locations"],dtype=torch.float32))
        self.register_buffer("clock_scales",torch.tensor(pre["pair_clock_scale_logit"],dtype=torch.float32))
        self.register_buffer("clock_zeros",torch.tensor(pre["pair_clock_zero_logit"],dtype=torch.float32))
        for head in (self.mark_head,self.existence_head,self.clock_head):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def initial_from_context(self,encoded):
        # Separate slicing keeps a single context projection and an explicit
        # bounded fast/slow initialization, with no ID or untimed case fields.
        context=.5*torch.tanh(self.initial_context(encoded))
        fast,slow=context.chunk(2,-1)
        return EventBelief(fast,slow,encoded.new_zeros(len(encoded),8,8))

    def step(self,belief,encoded,current,history_prior,index):
        q=F.normalize(self.query(encoded),dim=-1,eps=1e-6)
        recall=torch.einsum('bij,bj->bi',belief.memory,q)
        z=torch.tanh(self.project(torch.cat((belief.fast,belief.slow,recall,encoded),-1)))
        emb=self.step_embedding.weight[index].expand(len(encoded),-1)
        left,right=self.readout(torch.cat((z,emb),-1)).chunk(2,-1)
        decoded=F.silu(left)*right
        prior=history_prior if index==0 else self.transition_prior[current]
        marks=(self.mark_head(decoded)+F.softplus(self.prior_strength[index])*prior).log_softmax(-1)
        survival=F.logsigmoid(self.existence_prior[current]+self.existence_head(decoded).squeeze(-1))
        return StepLaw(decoded,marks,survival)

    def clock(self,law,current,following):
        emb=self.embed_known(self.activity_embedding,following)
        out=self.clock_head(torch.cat((law.decoded,emb),-1))
        # Locations anchored to FIT pair quantiles; bounded learned offsets.
        mu=self.clock_locations[current,following]+2.*torch.tanh(out[:,4:7]/2.)
        sigma=.15+1.1*torch.sigmoid(self.clock_scales[current,following]+out[:,7:10])
        zero=F.logsigmoid(self.clock_zeros[current,following]+out[:,0])
        return ClockLaw(zero,out[:,1:4].log_softmax(-1),mu,sigma.log())

    def advance(self,belief,mark,seconds):
        """Functional imagination update; no observed-state mutation/resource oracle."""
        inp=torch.cat((self.embed_known(self.activity_embedding,mark),
                       torch.log1p(seconds)[:,None]/12.,(seconds==0).to(seconds.dtype)[:,None]),-1)
        encoded=torch.tanh(self.imagined_encoder(inp))
        next_belief=self.observe(belief,encoded,seconds[:,None]/3600.,torch.ones_like(mark,dtype=torch.bool))
        return next_belief,encoded


def log_complement(logp):
    return torch.log(-torch.expm1(logp.clamp_max(-torch.finfo(logp.dtype).eps)))


def clock_nll(law,seconds):
    positive=seconds>0
    occurrence=-torch.where(positive,log_complement(law.log_zero),law.log_zero)
    logt=seconds.clamp_min(1e-4).log()
    z=(logt[:,None]-law.location)/law.log_scale.exp()
    density=(law.log_weight-law.log_scale-.5*z.square()-.5*math.log(2*math.pi)).logsumexp(-1)-logt
    return occurrence, -density


def teacher_path(model,belief,encoded,current,history,target,seconds,present,valid,early_weight=None):
    """Exact autoregressive factual suffix likelihood, never free-run accuracy.

    Future labels enter only likelihood/training; generated trajectories never
    call this routine. Mask after the first absent source record.
    """
    alive=valid.clone();losses=[];joint=[];conditional_marks=[];time_scores=[];masks=[]
    for j in range(STEPS):
        law=model.step(belief,encoded,current,history,j)
        yes=present[:,j];label=target[:,j]
        clock=model.clock(law,current,label)
        occurrence,positive=clock_nll(clock,seconds[:,j])
        existence=-torch.where(yes,law.log_survival,log_complement(law.log_survival))
        mark=-law.mark_log_probability.gather(-1,label[:,None]).squeeze(-1)
        time=occurrence+(seconds[:,j]>0)*positive
        # Omit the target-only log-seconds Jacobian during optimization only.
        time_train=occurrence+(seconds[:,j]>0)*(positive-seconds[:,j].clamp_min(1e-4).log())
        full=existence+yes*(mark+time)
        probability=law.mark_log_probability.exp()
        brier=probability.square().sum(-1)-2*probability.gather(-1,label[:,None]).squeeze(-1)+1
        losses.append((existence+yes*(mark+.2*time_train+.05*brier))*alive)
        joint.append(full*alive);conditional_marks.append(mark);time_scores.append(time);masks.append(alive)
        alive=alive & yes
        belief,encoded=model.advance(belief,label,seconds[:,j]);current=label
    mask=torch.stack(masks,1)
    weights=torch.ones_like(valid,dtype=encoded.dtype) if early_weight is None else early_weight
    loss=(torch.stack(losses,1)*weights[:,None]).sum()/(mask*weights[:,None]).sum().clamp_min(1)
    return loss,torch.stack(joint,1),mask,torch.stack(conditional_marks,1),torch.stack(time_scores,1)


def weighted_quantiles(values,log_weights,levels=(.05,.5,.95)):
    sorted_values,idx=values.sort(-1)
    weights=log_weights.softmax(-1).gather(-1,idx)
    cdf=weights.cumsum(-1);targets=values.new_tensor(levels).expand(len(values),-1).contiguous()
    indices=torch.searchsorted(cdf.contiguous(),targets).clamp_max(values.shape[-1]-1)
    return sorted_values.gather(-1,indices)


@torch.no_grad()
def generate(model,belief,encoded,current,history,particles=PARTICLES,keep_paths=8):
    """Deterministic quasi-Monte-Carlo marginalization of genuine sampled paths.

    Sample conditional marks/time and integrate record survival by weights.
    Conditional-on-survival quantiles need not be ordered across horizons:
    survivor selection changes the population. Each path's accumulated clock
    is nondecreasing. An autoregressive path is not necessarily process-legal.
    """
    b=len(encoded);n=model.transition_prior.shape[1]
    original=belief
    belief=EventBelief(*(v.repeat_interleave(particles,0) for v in belief))
    encoded=encoded.repeat_interleave(particles,0);current=current.repeat_interleave(particles)
    history=history.repeat_interleave(particles,0)
    uniforms=torch.quasirandom.SobolEngine(STEPS*4,scramble=True,seed=73000).draw(particles,dtype=encoded.dtype)
    elapsed=encoded.new_zeros(b*particles);logw=elapsed.clone()
    outputs=[];all_paths=[];first_mark=None
    for j in range(STEPS):
        law=model.step(belief,encoded,current,history,j)
        if j==0:first_mark=law.mark_log_probability.view(b,particles,n)[:,0]
        u=uniforms[:,j*4:j*4+4][None,:,:].expand(b,-1,-1).reshape(-1,4)
        mark=torch.searchsorted(law.mark_log_probability.exp().cumsum(-1).contiguous(),u[:,:1].contiguous()).squeeze(-1).clamp_max(n-1)
        clock=model.clock(law,current,mark)
        comp=torch.searchsorted(clock.log_weight.exp().cumsum(-1).contiguous(),u[:,2:3].contiguous()).squeeze(-1).clamp_max(2)
        mu=clock.location.gather(-1,comp[:,None]).squeeze(-1)
        sigma=clock.log_scale.exp().gather(-1,comp[:,None]).squeeze(-1)
        normal=torch.special.ndtri(u[:,3].clamp(1e-7,1-1e-7))
        seconds=torch.where(u[:,1]<=clock.log_zero.exp(),torch.zeros_like(mu),(mu+sigma*normal).exp())
        elapsed=elapsed+seconds;logw=logw+law.log_survival
        marks=mark.view(b,particles);times=elapsed.view(b,particles);weights=logw.view(b,particles)
        all_paths.append(torch.stack((marks.to(times.dtype),times,weights),-1)[:,:keep_paths])
        if j+1 in HORIZONS:
            conditional=encoded.new_zeros(b,n).scatter_add_(1,marks,weights.softmax(-1))
            if j==0:conditional=first_mark.exp() # exact one-step marginal
            survival=(weights.logsumexp(-1)-math.log(particles)).exp()
            joint=torch.cat((survival[:,None]*conditional,1-survival[:,None]),-1)
            # Registered normalized smoothing of the finite-particle mass.
            joint=(joint+1e-6)/(1+(n+1)*1e-6)
            quantiles=weighted_quantiles(times,weights)
            late=(weights.softmax(-1)*(times>86400.)).sum(-1)
            zero=(weights.softmax(-1)*(times==0)).sum(-1)
            outputs.append(torch.cat((joint,quantiles,late[:,None],zero[:,None]),-1))
        belief,encoded=model.advance(belief,mark,seconds);current=mark
    if not all(torch.isfinite(x).all() for x in outputs):raise ValueError('nonfinite generated future')
    # A functional API leaves the original factual state unchanged.
    assert belief is not original
    return torch.stack(outputs,1),torch.stack(all_paths,2)
