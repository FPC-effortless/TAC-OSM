"""Native time-aware state and precision-preserving causal observation retrieval.

All inputs are observed by the query cutoff. Capture elapsed time is distinct
from the publisher's reconstructed uniform retained-frame timestamp grid.
"""
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from .equilibrium_parallel_world import EquilibriumParallelWorld


class ClockValueState(NamedTuple):
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
    def detached(self):return ClockValueState(*(v.detach() for v in self))


class ClockValueWorld(EquilibriumParallelWorld):
    architecture='native_clock_value_world'
    def __init__(self,identified=None):
        super().__init__(identified)
        self.encoder=nn.Sequential(nn.Linear(99,64),nn.SiLU(),nn.Linear(64,64),nn.Tanh())
        self.gated_readout=nn.Linear(202,232)
        self.heads=nn.Linear(116,197)
        nn.init.normal_(self.heads.weight,std=.001);nn.init.zeros_(self.heads.bias)
        with torch.no_grad():
            self.heads.bias[:14].fill_(-2.)
            self.heads.bias[56:70].fill_(-2.)
            self.heads.bias[70:84].fill_(-4.)
            self.heads.bias[182:196].fill_(2.)
            self.heads.bias[196:].fill_(-2.)

    def initial_state(self,x):
        initial=super().initial_state(x)
        return ClockValueState(*initial,x.new_zeros(len(x),16),x[:,None].expand(-1,7,-1).clone(),x.new_zeros(len(x),7),x.new_zeros(len(x),1))

    def timed_advance(self,observed,state,ratio,address,projected):
        key,value,retention,write=torch.split(address,(8,8,1,1),-1)
        key=key/(key.square().sum(-1,keepdim=True)+1e-6).sqrt()
        decayed=torch.exp(F.logsigmoid(retention)*ratio)[:,:,None]*state.association
        read=(decayed*key[:,None,:]).sum(-1)
        c,g=(projected+F.linear(torch.cat((state.fast,state.slow),-1),self.core.fast_map.weight[:,64:])).chunk(2,-1)
        fast_gate=-torch.expm1(F.logsigmoid(-g)*ratio)
        fast=torch.lerp(state.fast,c.tanh(),fast_gate)
        innovation,rate=self.core.slow_map(torch.cat((fast-state.slow,state.slow),-1)).chunk(2,-1)
        slow_gate=-torch.expm1(F.logsigmoid(-rate)*ratio)
        slow=torch.lerp(state.slow,(state.slow+innovation.tanh()).tanh(),slow_gate)
        error=value.tanh()-read
        memory=decayed+write.sigmoid()[:,:,None]*error[:,:,None]*key[:,None,:]
        memory=memory/torch.maximum(memory.square().sum((1,2),keepdim=True).clamp_min(1e-12).sqrt()/4,torch.ones_like(memory[:,:1,:1]))
        representation=self.core.project(torch.cat((fast,slow,read),-1))
        return representation,fast,slow,memory

    def tensor_graph(self,x,actions,elapsed,state=None,last_only=False):
        state=self.initial_state(x[:,0]) if state is None else state
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
        encoded=self.encoder(features)
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
            out=self.decode(x[:,-1],local[:,-1],representations[-1],features[:,-1],velocity[:,-1],command[:,-1],lag_bank[:,-1],lag_age[:,-1])
        else:
            b,t=x.shape[:2];flat=lambda v:v.reshape(b*t,v.shape[-1])
            out=self.decode(flat(x),flat(local),flat(torch.stack(representations,1)),flat(features),flat(velocity),flat(command),lag_bank.reshape(b*t,8,14),lag_age.reshape(b*t,8))
            out={k:v.reshape(b,t,*v.shape[1:]) for k,v in out.items()}
        return out,final

    def execute(self,x,actions,elapsed,state=None,last_only=False):
        if x.ndim!=3 or x.shape[-1]!=14 or actions.shape!=x.shape or x.shape[1]<1:raise ValueError('BxTx14 causal history required')
        elapsed=x.new_full((*x.shape[:2],1),1/15) if elapsed is None else elapsed
        if elapsed.shape!=(*x.shape[:2],1) or not torch.isfinite(elapsed).all() or not (elapsed>0).all():raise ValueError('positive observed BxTx1 capture elapsed seconds required')
        if not torch.isfinite(x).all() or not torch.isfinite(actions).all():raise ValueError('finite inputs required')
        return self.tensor_graph(x,actions,elapsed,state,last_only)

    def forward(self,x,actions,state=None,return_state=False,*,elapsed=None):
        output,state=self.execute(x,actions,elapsed,state)
        return (output,state) if return_state else output

    def forecast_last(self,x,actions,state=None,return_state=False,*,elapsed=None):
        output,state=self.execute(x,actions,elapsed,state,last_only=True)
        return (output,state) if return_state else output

    def step(self,x,action,state=None,*,elapsed=None):
        return self.execute(x[:,None],action[:,None],None if elapsed is None else elapsed[:,None],state,last_only=True)

    def decode(self,x,observed,representation,feature,velocity,command,lag_bank,lag_age):
        b=len(x);n=len(self.horizons)
        context=torch.cat((observed,representation,feature[:,14:84]),-1)
        query=torch.cat((context[:,None].expand(b,n,198),self.horizon_features[None].expand(b,n,4)),-1)
        value,gate=self.gated_readout(query).chunk(2,-1);out=self.heads(value*F.silu(gate))
        mix,raw,b0,b1,vg,sg=out[:,:,:84].reshape(b,n,6,14).unbind(2)
        logits=out[:,:,84:196].reshape(b,n,8,14)-lag_age[:,None,:,None]/8
        lag_weights=logits.softmax(2)
        quiet_delta=(lag_weights*(lag_bank[:,None]-x[:,None,None])).sum(2)
        h=self.horizons[None,:,None]
        servo=torch.clamp(self.servo_gain*command,-self.servo_step_limit,self.servo_step_limit)
        bound=torch.maximum(4*self.servo_step_limit,torch.ones_like(self.delta_scale)*.05)*h
        moving_delta=bound*torch.tanh((h*(vg.sigmoid()*velocity[:,None]+sg.sigmoid()*servo[:,None])+bound*raw.tanh())/bound)
        quiet=(self.quiet_noise*h*(.01+F.softplus(b0))).clamp_min(1e-7)
        moving=2*quiet+self.delta_scale*h*(.01+F.softplus(b1))
        return {'current':x[:,None]+quiet_delta,'moving_delta':moving_delta-quiet_delta,'quiet_scale':quiet,
                'moving_scale':moving,'mixture_logit':mix,'motion_logit':out[:,:,196]}
