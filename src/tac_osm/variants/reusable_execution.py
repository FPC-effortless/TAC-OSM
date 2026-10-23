"""Execution-only reuse for the frozen ME080 architecture; no new parameters.

Decode independent time positions in one GEMM, or only the requested last
position. The nonlinear recurrent transition is deliberately unchanged.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .streaming_mixture_world import StreamingMixtureWorld, WorldState


class BatchedMixtureWorld(StreamingMixtureWorld):
    def _execute(self,x,actions,state=None,last_only=False):
        if x.ndim!=3 or x.shape[-1]!=14 or actions.shape!=x.shape or x.shape[1]<1:
            raise ValueError('nonempty BxTx14 observed histories required')
        if not torch.isfinite(x).all() or not torch.isfinite(actions).all():
            raise ValueError('finite observed inputs required')
        state=self.initial_state(x[:,0]) if state is None else state
        previous=torch.cat((state.previous[:,None],x[:,:-1]),1)
        seen=torch.cat((state.seen[:,None],torch.ones_like(x[:,1:,:1])),1)
        velocity=(x-previous)*seen
        acceleration=(velocity-torch.cat((state.velocity[:,None],velocity[:,:-1]),1))*seen
        command=(actions*self.action_scale+self.action_mean-self.state_mean)/self.state_scale-x
        features=torch.cat((x,command,torch.asinh(velocity/self.delta_scale),torch.asinh(acceleration/self.delta_scale),torch.ones_like(x[:,:,:1])/15),-1)
        observed=self.encoder(features)
        addresses=self.core.address(observed)
        projected=F.linear(observed,self.core.fast_map.weight[:,:64],self.core.fast_map.bias)
        representations=[]
        for t in range(x.shape[1]):
            representation,f,s,m=self.advance(observed[:,t],state,addresses[:,t],projected[:,t])
            if not last_only or t==x.shape[1]-1:representations.append(representation)
            state=WorldState(f,s,m,x[:,t],velocity[:,t],torch.ones_like(state.seen))
        if last_only:
            result=self.decode(x[:,-1],observed[:,-1],representations[-1],features[:,-1],velocity[:,-1],command[:,-1])
        else:
            b,t=x.shape[:2]
            flatten=lambda v:v.reshape(b*t,v.shape[-1])
            result=self.decode(flatten(x),flatten(observed),flatten(torch.stack(representations,1)),flatten(features),flatten(velocity),flatten(command))
            result={k:v.reshape(b,t,*v.shape[1:]) for k,v in result.items()}
        return result,state

    def forward(self,x,actions,state=None,return_state=False):
        output,state=self._execute(x,actions,state)
        return (output,state) if return_state else output

    def forecast_last(self,x,actions,state=None,return_state=False):
        output,state=self._execute(x,actions,state,last_only=True)
        return (output,state) if return_state else output


@dataclass(frozen=True)
class _Entry:
    sequence: int
    observed: torch.Tensor
    command: torch.Tensor
    state: WorldState
    forecast: dict


def _clone(values):return {k:v.detach().clone() for k,v in values.items()}


class EntityInferenceCache:
    """Caller-scoped, ordered inference effects; no hidden model-global state.

New sequences start at zero. Exact retry of the latest observation reuses the
forecast without a neural call. Conflicting retries and out-of-order events
are rejected. Explicit reset is required after gaps or historical revisions.
No tensor/data/weight mutations via `.data` are supported by this runtime.
    """
    def __init__(self,model,*,model_revision,preprocessing_revision,max_entities=1024):
        if not model_revision or not preprocessing_revision or type(max_entities)!=int or max_entities<1:
            raise ValueError('explicit model/preprocessing revisions and positive capacity required')
        if model.training:raise ValueError('inference cache requires model.eval()')
        self.model=model
        self.namespace=(str(model_revision),str(preprocessing_revision))
        self.max_entities=max_entities
        self._entries={}
        self._identity=self._model_identity()

    def _model_identity(self):
        return tuple((name,id(v),v._version,str(v.dtype),str(v.device),tuple(v.shape))
                     for name,v in list(self.model.named_parameters())+list(self.model.named_buffers()))

    def reset(self,entity=None):
        if entity is None:self._entries.clear()
        else:self._entries.pop(entity,None)
        # This does not accept new weights. Construct a new versioned handler.

    @torch.no_grad()
    def step(self,entity,sequence,observed,known_command):
        if self.model.training or self._identity!=self._model_identity():
            self._entries.clear()
            raise RuntimeError('model changed; construct a new versioned inference cache')
        if not isinstance(entity,str) or not entity or type(sequence)!=int or sequence<0:
            raise ValueError('explicit entity and nonnegative integer sequence required')
        reference=next(self.model.parameters())
        for value in (observed,known_command):
            if value.shape!=(1,14) or value.dtype!=reference.dtype or value.device!=reference.device or not torch.isfinite(value).all():
                raise ValueError('finite 1x14 observations/commands matching model device and dtype required')
        previous=self._entries.get(entity)
        if previous and sequence==previous.sequence:
            if not torch.equal(observed,previous.observed) or not torch.equal(known_command,previous.command):
                raise ValueError('conflicting retry; reset and replay corrected history')
            return _clone(previous.forecast)
        if sequence!=(previous.sequence+1 if previous else 0):
            raise ValueError('gap or historical revision; reset and replay verified prefix')
        if previous is None and len(self._entries)>=self.max_entities:
            raise RuntimeError('entity capacity reached; explicitly retire a stream')
        forecast,state=self.model.step(observed,known_command,None if previous is None else previous.state)
        stored_state=WorldState(*(v.detach().clone() for v in state))
        self._entries[entity]=_Entry(sequence,observed.detach().clone(),known_command.detach().clone(),stored_state,_clone(forecast))
        return _clone(forecast)


def mixture_quantiles(prediction,probabilities=(.05,.5,.95)):
    """Invert several quantiles together, sharing conversions and CDF constants."""
    probabilities=tuple(probabilities)
    if not probabilities or not all(0<p<1 for p in probabilities):
        raise ValueError('nonempty probabilities strictly inside (0,1) required')
    p={k:v.double() for k,v in prediction.items()}
    radius=40*torch.maximum(p['quiet_scale'],p['moving_scale'])
    shape=(len(probabilities),)+tuple(radius.shape)
    low=(torch.minimum(p['moving_delta'],torch.zeros_like(radius))-radius).expand(shape)
    high=(torch.maximum(p['moving_delta'],torch.zeros_like(radius))+radius).expand(shape)
    levels=radius.new_tensor(probabilities).reshape((len(probabilities),)+(1,)*radius.ndim)
    weight=p['mixture_logit'].sigmoid()
    for _ in range(40):
        mid=(low+high)*.5
        z0=mid/p['quiet_scale'];z1=(mid-p['moving_delta'])/p['moving_scale']
        c0=.5+.5*z0.sign()*(-torch.expm1(-z0.abs()))
        c1=.5+.5*z1.sign()*(-torch.expm1(-z1.abs()))
        right=(1-weight)*c0+weight*c1<levels
        low=torch.where(right,mid,low);high=torch.where(right,high,mid)
    return p['current'][None]+(low+high)*.5
