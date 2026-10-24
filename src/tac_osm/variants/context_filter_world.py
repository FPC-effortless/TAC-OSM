"""Native TAC plus input-selective affine kinematics and known case context.

The diagonal filter branch has an associative composition; the coupled native
fast/slow/associative recurrence keeps its actual serial dependency.
"""
from typing import NamedTuple
import math
import torch
from torch import nn
from torch.nn import functional as F
from .median_anchored_world import MedianAnchoredWorld
from .clock_value_world import ClockValueState


class ContextFilterState(NamedTuple):
    fast:torch.Tensor
    slow:torch.Tensor
    association:torch.Tensor
    previous:torch.Tensor
    velocity:torch.Tensor
    seen:torch.Tensor
    command:torch.Tensor
    age:torch.Tensor
    encoded_history:torch.Tensor
    encoded_time:torch.Tensor
    raw_history:torch.Tensor
    raw_time:torch.Tensor
    clock:torch.Tensor
    level:torch.Tensor
    filtered_velocity:torch.Tensor


def affine_prefix(retention,innovation,initial):
    """Inclusive causal parallel-prefix composition, without unstable divisions."""
    a,b=retention,innovation;offset=1
    while offset<a.shape[1]:
        olda,oldb=a,b
        a=torch.cat((olda[:,:offset],olda[:,offset:]*olda[:,:-offset]),1)
        b=torch.cat((oldb[:,:offset],oldb[:,offset:]+olda[:,offset:]*oldb[:,:-offset]),1)
        offset*=2
    return a*initial[:,None]+b


class ContextFilterWorld(MedianAnchoredWorld):
    architecture='native_context_selective_kinematic_world'

    def __init__(self,identified=None):
        super().__init__(identified)
        self.task_embedding=nn.Embedding(5,64)
        self.clock_encoder=nn.Linear(6,64)
        self.filter_rates=nn.Linear(64,28)
        self.filter_context=nn.Linear(28,32)
        nn.init.zeros_(self.task_embedding.weight)
        nn.init.zeros_(self.clock_encoder.weight);nn.init.zeros_(self.clock_encoder.bias)
        nn.init.zeros_(self.filter_rates.weight);nn.init.constant_(self.filter_rates.bias,math.log(49.))
        nn.init.normal_(self.filter_context.weight,std=.001);nn.init.zeros_(self.filter_context.bias)

    def task_context(self,task_ids,x):
        if task_ids is None:return x.new_zeros(len(x),64)
        known=(task_ids>=0)[:,None]
        return torch.where(known,self.task_embedding(task_ids.clamp_min(0)),x.new_zeros(len(x),64))

    def initial_state(self,x,task_ids=None):
        s=super().initial_state(x);context=self.task_context(task_ids,x)
        s=s._replace(fast=context[:,:32].tanh(),slow=context[:,32:].tanh())
        return ContextFilterState(*s,x.clone(),torch.zeros_like(x))

    def visual_tensor_graph(self,x,actions,elapsed,images,state=None,last_only=False,*,task_ids,clock_context):
        state=self.initial_state(x[:,0],task_ids) if state is None else state
        ratio=elapsed*15
        previous=torch.cat((state.previous[:,None],x[:,:-1]),1)
        seen=torch.cat((state.seen[:,None],torch.ones_like(x[:,1:,:1])),1)
        velocity=(x-previous)*seen/ratio
        acceleration=(velocity-torch.cat((state.velocity[:,None],velocity[:,:-1]),1))*seen/ratio
        command=(actions*self.action_scale+self.action_mean-self.state_mean)/self.state_scale-x
        prior_commands=torch.cat((state.command[:,None],actions[:,:-1]),1)
        change=(actions-prior_commands)*self.action_scale/self.state_scale*seen
        clock=state.clock[:,None]+ratio.cumsum(1)
        reset=(x!=previous)|((torch.arange(x.shape[1],device=x.device)[None,:,None]==0)&(state.seen[:,None]==0))
        last_change=torch.cummax(torch.where(reset,clock,state.clock[:,None]-state.age[:,None]),1).values
        age=clock-last_change
        lag_source=torch.cat((state.raw_history,x),1)
        lag_bank=lag_source.unfold(1,8,1).permute(0,1,3,2)
        lag_time=torch.cat((state.raw_time,clock[:,:,0]),1).unfold(1,8,1)
        lag_age=clock-lag_time
        temporal_weights=torch.softmax(-lag_age/8,2)
        local_level=(temporal_weights[:,:,:,None]*lag_bank).sum(2)
        innovation=(x-local_level)/self.quiet_noise
        features=torch.cat((x,command,torch.asinh(velocity/self.delta_scale),torch.asinh(acceleration/self.delta_scale),
             torch.asinh(change/self.delta_scale),torch.log1p(age)/3.4965075614664802,torch.log1p(ratio)/.6931471805599453,torch.asinh(innovation)),-1)
        visual=self.visual_encoder(images.reshape(-1,1,32,32)).reshape(*x.shape[:2],64)
        base_encoded=self.encoder(features)+self.visual_gain.sigmoid()*visual+self.task_context(task_ids,x[:,0])[:,None]+self.clock_encoder(clock_context)
        retention=torch.exp(-F.softplus(self.filter_rates(base_encoded))*ratio)
        values=torch.cat((x,velocity),-1)
        filtered=affine_prefix(retention,(1-retention)*values,torch.cat((state.level,state.filtered_velocity),-1))
        level,filtered_velocity=filtered.chunk(2,-1)
        evidence=torch.cat((torch.asinh((level-x)/self.quiet_noise),torch.asinh(filtered_velocity/self.delta_scale)),-1)
        filter_feature=F.silu(self.filter_context(evidence))
        encoded=base_encoded+.1*torch.cat((filter_feature,torch.zeros_like(filter_feature)),-1)
        taps=torch.cat((state.encoded_history,encoded),1).unfold(1,17,1)
        tap_times=torch.cat((state.encoded_time,clock[:,:,0]),1).unfold(1,17,1)
        tap_age=clock-tap_times
        fast=(taps[:,:,:,12:]*torch.exp(-tap_age[:,:,None,12:]/4)*self.local_fast.weight[:,0][None,None]).sum(-1)+self.local_fast.bias
        slow=(taps[:,:,:,::4]*torch.exp(-tap_age[:,:,None,::4]/16)*self.local_slow.weight[:,0][None,None]).sum(-1)+self.local_slow.bias
        local=encoded+.1*(F.silu(fast)+F.silu(slow))
        addresses=self.core.address(encoded);projected=F.linear(encoded,self.core.fast_map.weight[:,:64],self.core.fast_map.bias)
        representations=[]
        for t in range(x.shape[1]):
            z,f,s,m=self.timed_advance(encoded[:,t],state,ratio[:,t],addresses[:,t],projected[:,t])
            if not last_only or t==x.shape[1]-1:representations.append(z)
            state=ClockValueState(f,s,m,x[:,t],velocity[:,t],torch.ones_like(state.seen),actions[:,t],age[:,t],
                state.encoded_history,state.encoded_time,state.raw_history,state.raw_time,clock[:,t])
        final=state._replace(encoded_time=torch.cat((state.encoded_time,clock[:,:,0]),1)[:,-16:],encoded_history=torch.cat((state.encoded_history,encoded),1)[:,-16:],raw_history=lag_source[:,-7:],
                            raw_time=torch.cat((state.raw_time,clock[:,:,0]),1)[:,-7:])
        if last_only:
            out=self.decode(x[:,-1],local[:,-1],representations[-1],features[:,-1],filtered_velocity[:,-1],command[:,-1],lag_bank[:,-1],lag_age[:,-1])
        else:
            b,t=x.shape[:2];flat=lambda v:v.reshape(b*t,v.shape[-1])
            out=self.decode(flat(x),flat(local),flat(torch.stack(representations,1)),flat(features),flat(filtered_velocity),flat(command),lag_bank.reshape(b*t,8,14),lag_age.reshape(b*t,8))
            out={k:v.reshape(b,t,*v.shape[1:]) for k,v in out.items()}
        return out,ContextFilterState(*final,level[:,-1],filtered_velocity[:,-1])


    def execute(self,x,actions,elapsed,state=None,last_only=False,*,images=None,task_ids=None,clock_context=None):
        if x.ndim!=3 or x.shape[-1]!=14 or actions.shape!=x.shape or x.shape[1]<1:raise ValueError('BxTx14 observed histories required')
        if elapsed is None or elapsed.shape!=(*x.shape[:2],1) or not torch.isfinite(elapsed).all() or not (elapsed>0).all():raise ValueError('positive observed elapsed required')
        if images is None or images.shape!=(*x.shape[:2],1,32,32) or not torch.isfinite(images).all() or (images<0).any() or (images>1).any():raise ValueError('aligned gray32 observed images required')
        if task_ids is None or task_ids.shape!=(len(x),) or task_ids.dtype!=torch.int64 or (task_ids < -1).any() or (task_ids>4).any():raise ValueError('known task ID0..4 or explicit unknown-1 required')
        if clock_context is None or clock_context.shape!=(*x.shape[:2],6) or not torch.isfinite(clock_context).all():raise ValueError('six observed capture-clock features required')
        if not torch.isfinite(x).all() or not torch.isfinite(actions).all():raise ValueError('finite numeric observations required')
        return self.visual_tensor_graph(x,actions,elapsed,images,state,last_only,task_ids=task_ids,clock_context=clock_context)

    def forward(self,x,actions,state=None,return_state=False,*,elapsed=None,images=None,task_ids=None,clock_context=None):
        p,s=self.execute(x,actions,elapsed,state,images=images,task_ids=task_ids,clock_context=clock_context)
        return (p,s) if return_state else p

    def forecast_last(self,x,actions,state=None,return_state=False,*,elapsed=None,images=None,task_ids=None,clock_context=None):
        p,s=self.execute(x,actions,elapsed,state,last_only=True,images=images,task_ids=task_ids,clock_context=clock_context)
        return (p,s) if return_state else p

    def step(self,x,action,state=None,*,elapsed=None,images=None,task_ids=None,clock_context=None):
        return self.execute(x[:,None],action[:,None],None if elapsed is None else elapsed[:,None],state,last_only=True,images=None if images is None else images[:,None],task_ids=task_ids,clock_context=None if clock_context is None else clock_context[:,None])
