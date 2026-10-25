"""Native TAC with causal parallel filters, observed change clocks and equilibrium means.

Pure parameter-differentiable tensor graph. Observation change detection is a
nonlearned discrete feature; the nonlinear native recurrence stays sequential.
"""
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from .horizon_query_world import HorizonQueryWorld


class ParallelState(NamedTuple):
    fast: torch.Tensor
    slow: torch.Tensor
    association: torch.Tensor
    previous: torch.Tensor
    velocity: torch.Tensor
    seen: torch.Tensor
    command: torch.Tensor
    age: torch.Tensor
    encoded_history: torch.Tensor

    def detached(self):return ParallelState(*(v.detach() for v in self))


class EquilibriumParallelWorld(HorizonQueryWorld):
    architecture='native_equilibrium_parallel'
    def __init__(self,identified=None):
        super().__init__(identified)
        self.encoder=nn.Sequential(nn.Linear(85,64),nn.SiLU(),nn.Linear(64,64),nn.Tanh())
        self.local_fast=nn.Conv1d(64,64,5,groups=64)
        self.local_slow=nn.Conv1d(64,64,5,groups=64,dilation=4)
        self.gated_readout=nn.Linear(202,300)
        self.heads=nn.Linear(150,99)
        nn.init.normal_(self.heads.weight,std=.001);nn.init.zeros_(self.heads.bias)
        with torch.no_grad():
            self.heads.bias[:14].fill_(-2.)
            self.heads.bias[56:70].fill_(-2.)
            self.heads.bias[70:84].fill_(-4.)
            self.heads.bias[98:].fill_(-2.)
        noise=torch.ones(14)*1e-4 if identified is None else torch.as_tensor(identified['quiet_noise'])
        self.register_buffer('quiet_noise',noise.float().clone())

    def initial_state(self,x):
        return ParallelState(x.new_zeros(len(x),32),x.new_zeros(len(x),32),x.new_zeros(len(x),8,8),
          torch.zeros_like(x),torch.zeros_like(x),x.new_zeros(len(x),1),torch.zeros_like(x),
          torch.zeros_like(x),x.new_zeros(len(x),16,64))

    def tensor_graph(self,x,actions,state=None,last_only=False):
        """No data-dependent Python guards; callers validate outside this graph."""
        state=self.initial_state(x[:,0]) if state is None else state
        previous=torch.cat((state.previous[:,None],x[:,:-1]),1)
        seen=torch.cat((state.seen[:,None],torch.ones_like(x[:,1:,:1])),1)
        velocity=(x-previous)*seen
        acceleration=(velocity-torch.cat((state.velocity[:,None],velocity[:,:-1]),1))*seen
        command=(actions*self.action_scale+self.action_mean-self.state_mean)/self.state_scale-x
        prior_commands=torch.cat((state.command[:,None],actions[:,:-1]),1)
        change=(actions-prior_commands)*self.action_scale/self.state_scale*seen
        ticks=torch.arange(1,x.shape[1]+1,device=x.device,dtype=x.dtype)[None,:,None]
        reset=(velocity!=0)|((ticks==1)&(state.seen[:,None]==0))
        last_change=torch.cummax(torch.where(reset,ticks,-state.age[:,None]),1).values
        age=ticks-last_change
        features=torch.cat((x,command,torch.asinh(velocity/self.delta_scale),torch.asinh(acceleration/self.delta_scale),
                            torch.asinh(change/self.delta_scale),torch.log1p(age)/3.4965075614664802,torch.ones_like(x[:,:,:1])/15),-1)
        encoded=self.encoder(features)
        fast_context=torch.cat((state.encoded_history[:,-4:],encoded),1).transpose(1,2)
        slow_context=torch.cat((state.encoded_history,encoded),1).transpose(1,2)
        local=encoded+.1*(F.silu(self.local_fast(fast_context))+F.silu(self.local_slow(slow_context))).transpose(1,2)
        addresses=self.core.address(encoded)
        projections=F.linear(encoded,self.core.fast_map.weight[:,:64],self.core.fast_map.bias)
        representations=[]
        for t in range(x.shape[1]):
            representation,f,s,m=self.advance(encoded[:,t],state,addresses[:,t],projections[:,t])
            if not last_only or t==x.shape[1]-1:representations.append(representation)
            state=ParallelState(f,s,m,x[:,t],velocity[:,t],torch.ones_like(state.seen),actions[:,t],age[:,t],state.encoded_history)
        final=state._replace(encoded_history=torch.cat((state.encoded_history,encoded),1)[:,-16:])
        if last_only:
            out=self.decode(x[:,-1],local[:,-1],representations[-1],features[:,-1],velocity[:,-1],command[:,-1])
        else:
            b,t=x.shape[:2];flat=lambda v:v.reshape(b*t,v.shape[-1])
            out=self.decode(flat(x),flat(local),flat(torch.stack(representations,1)),flat(features),flat(velocity),flat(command))
            out={k:v.reshape(b,t,*v.shape[1:]) for k,v in out.items()}
        return out,final

    def _execute(self,x,actions,state=None,last_only=False):
        if x.ndim!=3 or x.shape[-1]!=14 or actions.shape!=x.shape or x.shape[1]<1:
            raise ValueError('actual BxTx14 observed history and already-known commands required')
        if not torch.isfinite(x).all() or not torch.isfinite(actions).all():raise ValueError('finite observations required')
        return self.tensor_graph(x,actions,state,last_only)

    def step(self,x,action,state=None):
        if x.ndim!=2 or x.shape[-1]!=14 or action.shape!=x.shape:raise ValueError('Bx14 actual observations/known commands required')
        return self._execute(x[:,None],action[:,None],state,last_only=True)

    def decode(self,x,observed,representation,feature,velocity,command):
        b=len(x);n=len(self.horizons)
        context=torch.cat((observed,representation,feature[:,14:84]),-1)
        query=torch.cat((context[:,None].expand(b,n,198),self.horizon_features[None].expand(b,n,4)),-1)
        value,gate=self.gated_readout(query).chunk(2,-1)
        out=self.heads(value*F.silu(gate))
        mix,raw,b0,b1,vg,sg,quiet_raw=out[:,:,:98].reshape(b,n,7,14).unbind(2)
        h=self.horizons[None,:,None]
        acceleration=self.delta_scale*torch.sinh(feature[:,42:56])
        change=self.delta_scale*torch.sinh(feature[:,56:70])
        support=2*velocity.abs()+.5*acceleration.abs()+change.abs()
        quiet_delta=h*velocity[:,None]*quiet_raw.tanh()
        moving_delta=h*(velocity[:,None]*vg.sigmoid()+support[:,None]*raw.tanh()+change.abs()[:,None]*command[:,None].tanh()*sg.sigmoid())
        quiet_center=x[:,None]+quiet_delta
        quiet=(self.quiet_noise*h*(.01+F.softplus(b0))).clamp_min(1e-7)
        moving=2*quiet+self.delta_scale*h*(.01+F.softplus(b1))
        return {'current':quiet_center,'moving_delta':moving_delta-quiet_delta,
                'quiet_scale':quiet,'moving_scale':moving,'mixture_logit':mix,'motion_logit':out[:,:,98]}
