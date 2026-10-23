"""ME080: pure native-state evolution and coherent multi-horizon Laplace mixtures.

Reuses gated delta memory (Gated Delta Networks), mixture density networks
(Bishop, 1994), and distributional quantiles. This is not an SSM or a novelty
claim for those established mechanisms. State is explicit and caller-owned.
"""
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from .kinematic_world import NativeCore, HORIZONS


class WorldState(NamedTuple):
    fast: torch.Tensor
    slow: torch.Tensor
    association: torch.Tensor
    previous: torch.Tensor
    velocity: torch.Tensor
    seen: torch.Tensor

    def detached(self):
        return WorldState(*(v.detach() for v in self))


class StreamingMixtureWorld(nn.Module):
    architecture = 'native_streaming_mixture'

    def __init__(self, identified=None, width=123):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(57,64),nn.SiLU(),nn.Linear(64,64),nn.Tanh())
        self.core=NativeCore()
        self.gated_readout=nn.Linear(170,width*2)
        self.heads=nn.Linear(width,255)
        nn.init.normal_(self.heads.weight,std=.001)
        nn.init.zeros_(self.heads.bias)
        with torch.no_grad():
            self.heads.bias[:42].fill_(-2.)  # mixture, not factual motion probability
            self.heads.bias[168:210].fill_(-2.)
            self.heads.bias[210:252].fill_(-4.)
            self.heads.bias[252:].fill_(-2.)
        defaults={'delta_scale':torch.ones(14)*.01,'servo_gain':torch.zeros(14),
                  'servo_step_limit':torch.ones(14)*.1,'action_mean':torch.zeros(14),
                  'action_scale':torch.ones(14),'state_mean':torch.zeros(14),
                  'state_scale':torch.ones(14)}
        for key,value in defaults.items():
            self.register_buffer(key,torch.as_tensor(value if identified is None else identified[key]).float().clone())
        self.register_buffer('horizons',torch.tensor(HORIZONS).float())

    def initial_state(self, x):
        return WorldState(x.new_zeros(len(x),32),x.new_zeros(len(x),32),
                          x.new_zeros(len(x),8,8),torch.zeros_like(x),
                          torch.zeros_like(x),x.new_zeros(len(x),1))

    def features(self, x, action, state):
        velocity=(x-state.previous)*state.seen
        acceleration=(velocity-state.velocity)*state.seen
        command=(action*self.action_scale+self.action_mean-self.state_mean)/self.state_scale-x
        feature=torch.cat((x,command,torch.asinh(velocity/self.delta_scale),
                           torch.asinh(acceleration/self.delta_scale),torch.ones_like(x[:,:1])/15),-1)
        return feature,velocity,command

    def advance(self, observed, state, address=None, input_projection=None):
        # Pure tensor DAG. Observation-only projections can be computed once for
        # a whole training chunk; recurrent terms keep their nonlinear semantics.
        address=self.core.address(observed) if address is None else address
        key,value,retention,write=torch.split(address,(8,8,1,1),-1)
        key=key/(key.square().sum(-1,keepdim=True)+1e-6).sqrt()
        read=(state.association*key[:,None,:]).sum(-1)
        projected=F.linear(observed,self.core.fast_map.weight[:,:64],self.core.fast_map.bias) if input_projection is None else input_projection
        c,g=(projected+F.linear(torch.cat((state.fast,state.slow),-1),self.core.fast_map.weight[:,64:])).chunk(2,-1)
        fast=torch.lerp(state.fast,c.tanh(),g.sigmoid())
        innovation,rate=self.core.slow_map(torch.cat((fast-state.slow,state.slow),-1)).chunk(2,-1)
        slow=torch.lerp(state.slow,(state.slow+innovation.tanh()).tanh(),rate.sigmoid())
        decayed=retention.sigmoid()[:,:,None]*state.association
        error=value.tanh()-(decayed*key[:,None,:]).sum(-1)
        association=decayed+write.sigmoid()[:,:,None]*error[:,:,None]*key[:,None,:]
        association=association/torch.maximum(association.square().sum((1,2),keepdim=True).clamp_min(1e-12).sqrt()/4,torch.ones_like(association[:,:1,:1]))
        representation=self.core.project(torch.cat((fast,slow,read),-1))
        return representation,fast,slow,association

    def decode(self,x,observed,representation,feature,velocity,command):
        context=torch.cat((observed,representation,feature[:,14:56]),-1)
        value,gate=self.gated_readout(context).chunk(2,-1)
        out=self.heads(value*F.silu(gate))
        mix,raw,b0,b1,vg,sg=out[:,:252].reshape(len(x),6,3,14).unbind(1)
        horizon=self.horizons[None,:,None]
        bound=torch.maximum(4*self.servo_step_limit,torch.ones_like(self.delta_scale)*.05)*horizon
        servo=torch.clamp(self.servo_gain*command,-self.servo_step_limit,self.servo_step_limit)
        delta=bound*torch.tanh((horizon*(vg.sigmoid()*velocity[:,None]+sg.sigmoid()*servo[:,None])+bound*raw.tanh())/bound)
        unit=self.delta_scale*horizon
        return {'current':x[:,None].expand_as(delta),'moving_delta':delta,
                'quiet_scale':(unit*.1*(.01+F.softplus(b0))).clamp_min(1e-6),
                'moving_scale':(unit*(.01+F.softplus(b1))).clamp_min(1e-6),
                'mixture_logit':mix,'motion_logit':out[:,252:]}

    def step(self,x,action,state=None):
        if x.ndim!=2 or x.shape[-1]!=14 or action.shape!=x.shape:
            raise ValueError('same-shape observed Bx14 state and already-known command required')
        if not torch.isfinite(x).all() or not torch.isfinite(action).all():
            raise ValueError('finite observed inputs required')
        state=self.initial_state(x) if state is None else state
        feature,velocity,command=self.features(x,action,state)
        observed=self.encoder(feature)
        representation,f,s,m=self.advance(observed,state)
        prediction=self.decode(x,observed,representation,feature,velocity,command)
        return prediction,WorldState(f,s,m,x,velocity,torch.ones_like(state.seen))

    def forward(self,x,actions,state=None,return_state=False):
        if x.ndim!=3 or x.shape[-1]!=14 or actions.shape!=x.shape or x.shape[1]<1:
            raise ValueError('nonempty BxTx14 observed histories required')
        if not torch.isfinite(x).all() or not torch.isfinite(actions).all():
            raise ValueError('finite observed inputs required')
        state=self.initial_state(x[:,0]) if state is None else state
        # Chunk features depend only on observed data and incoming physical cache.
        previous=torch.cat((state.previous[:,None],x[:,:-1]),1)
        seen=torch.cat((state.seen[:,None],torch.ones_like(x[:,1:,:1])),1)
        velocity=(x-previous)*seen
        previous_velocity=torch.cat((state.velocity[:,None],velocity[:,:-1]),1)
        acceleration=(velocity-previous_velocity)*seen
        command=(actions*self.action_scale+self.action_mean-self.state_mean)/self.state_scale-x
        feature=torch.cat((x,command,torch.asinh(velocity/self.delta_scale),torch.asinh(acceleration/self.delta_scale),torch.ones_like(x[:,:,:1])/15),-1)
        observed=self.encoder(feature)
        addresses=self.core.address(observed)
        projections=F.linear(observed,self.core.fast_map.weight[:,:64],self.core.fast_map.bias)
        predictions=[]
        for t in range(x.shape[1]):
            representation,f,s,m=self.advance(observed[:,t],state,addresses[:,t],projections[:,t])
            predictions.append(self.decode(x[:,t],observed[:,t],representation,feature[:,t],velocity[:,t],command[:,t]))
            state=WorldState(f,s,m,x[:,t],velocity[:,t],torch.ones_like(state.seen))
        result={key:torch.stack([p[key] for p in predictions],1) for key in predictions[0]}
        return (result,state) if return_state else result


def log_density(prediction,target):
    delta=target-prediction['current']
    l0=F.logsigmoid(-prediction['mixture_logit'])-(2*prediction['quiet_scale']).log()-delta.abs()/prediction['quiet_scale']
    l1=F.logsigmoid(prediction['mixture_logit'])-(2*prediction['moving_scale']).log()-(delta-prediction['moving_delta']).abs()/prediction['moving_scale']
    return torch.logaddexp(l0,l1)


def distribution_loss(prediction,targets,current,valid,importance,physical_scale,normalizer):
    # Proper marginal likelihood with source-derived coordinate scale; the
    # Jacobian is constant in parameters. No future label is an input or router.
    nll=(-log_density(prediction,targets)+physical_scale.log()).mean(-1)
    nll=(nll*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
    event=(((targets-current[:,:,None]).abs()*physical_scale).mean(-1)>=.025).float()
    bce=F.binary_cross_entropy_with_logits(prediction['motion_logit'],event,reduction='none')
    risk=(bce*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
    return (importance*(nll+.2*risk)).mean()


def quantile(prediction,probability):
    if not 0<probability<1:
        raise ValueError('quantile probability must be in (0,1)')
    # Float64 distribution inversion/scoring, frozen prospectively. Neural
    # weights remain float32. These are marginal, not joint trajectory bands.
    p={k:v.double() for k,v in prediction.items()}
    radius=40*torch.maximum(p['quiet_scale'],p['moving_scale'])
    low=torch.minimum(p['moving_delta'],torch.zeros_like(radius))-radius
    high=torch.maximum(p['moving_delta'],torch.zeros_like(radius))+radius
    weight=p['mixture_logit'].sigmoid()
    for _ in range(40):
        mid=(low+high)*.5
        def laplace(z):return .5+.5*z.sign()*(-torch.expm1(-z.abs()))
        cdf=(1-weight)*laplace(mid/p['quiet_scale'])+weight*laplace((mid-p['moving_delta'])/p['moving_scale'])
        right=cdf<probability
        low=torch.where(right,mid,low);high=torch.where(right,high,mid)
    return p['current']+(low+high)*.5
