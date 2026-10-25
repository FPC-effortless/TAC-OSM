"""Integrated causal camera/native predictive state; no future images or controls.

CNN and feature preparation are parallel over batch/time. Only native coupled
fast/slow/delta-state evolution is serial. One shared horizon readout reuses the
result. Image resizing/normalization is fixed and source-verified externally.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .clock_value_world import ClockValueState
from .bounded_value_world import BoundedValueWorld


class VisualWorld(BoundedValueWorld):
    architecture='native_causal_visual_world'

    def __init__(self,identified=None):
        super().__init__(identified)
        self.gated_readout=nn.Linear(202,216)
        self.heads=nn.Linear(108,211)
        self.visual_encoder=nn.Sequential(
            nn.Conv2d(1,8,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(8,16,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(16,16,3,stride=2,padding=1),nn.SiLU(),
            nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),nn.Linear(64,64),nn.Tanh())
        self.visual_gain=nn.Parameter(torch.full((64,),-2.))
        nn.init.normal_(self.visual_encoder[-2].weight,std=.001)
        nn.init.zeros_(self.visual_encoder[-2].bias)

    def visual_tensor_graph(self,x,actions,elapsed,images,state=None,last_only=False):
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
        visual=self.visual_encoder(images.reshape(-1,1,32,32)).reshape(*x.shape[:2],64)
        encoded=self.encoder(features)+self.visual_gain.sigmoid()*visual
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

    def execute(self,x,actions,elapsed,state=None,last_only=False,*,images=None):
        if x.ndim!=3 or x.shape[-1]!=14 or actions.shape!=x.shape or x.shape[1]<1:raise ValueError('BxTx14 causal history required')
        elapsed=x.new_full((*x.shape[:2],1),1/15) if elapsed is None else elapsed
        if elapsed.shape!=(*x.shape[:2],1) or not torch.isfinite(elapsed).all() or not (elapsed>0).all():raise ValueError('positive observed capture elapsed required')
        if images is None or images.shape!=(*x.shape[:2],1,32,32):raise ValueError('Aligned BxTx1x32x32 observed images required')
        if not torch.isfinite(images).all() or (images<0).any() or (images>1).any():raise ValueError('Images must be finite gray pixels scaled to [0,1]')
        if not torch.isfinite(x).all() or not torch.isfinite(actions).all():raise ValueError('finite numeric inputs required')
        return self.visual_tensor_graph(x,actions,elapsed,images,state,last_only)

    def forward(self,x,actions,state=None,return_state=False,*,elapsed=None,images=None):
        output,state=self.execute(x,actions,elapsed,state,images=images)
        return (output,state) if return_state else output

    def forecast_last(self,x,actions,state=None,return_state=False,*,elapsed=None,images=None):
        output,state=self.execute(x,actions,elapsed,state,last_only=True,images=images)
        return (output,state) if return_state else output

    def step(self,x,action,state=None,*,elapsed=None,images=None):
        return self.execute(x[:,None],action[:,None],None if elapsed is None else elapsed[:,None],state,last_only=True,images=None if images is None else images[:,None])

    def decode(self,x,observed,representation,feature,velocity,command,lag_bank,lag_age):
        b=len(x);n=len(self.horizons)
        context=torch.cat((observed,representation,feature[:,14:84]),-1)
        query=torch.cat((context[:,None].expand(b,n,198),self.horizon_features[None].expand(b,n,4)),-1)
        value,gate=self.gated_readout(query).chunk(2,-1);out=self.heads(value*F.silu(gate))
        mix,raw,b0,b1,vg,sg=out[:,:,:84].reshape(b,n,6,14).unbind(2)
        logits=out[:,:,84:196].reshape(b,n,8,14)-lag_age[:,None,:,None]/8
        lag_weights=logits.softmax(2)
        innovation=(lag_weights*(lag_bank[:,None]-x[:,None,None])).sum(2)
        h=self.horizons[None,:,None]
        quiet_unit=self.quiet_noise*h.sqrt()
        quiet_delta=innovation/(1+(innovation/quiet_unit).abs())
        translation=self.delta_scale*h*out[:,:,197:211].tanh()
        servo=torch.clamp(self.servo_gain*command,-self.servo_step_limit,self.servo_step_limit)
        bound=torch.maximum(4*self.servo_step_limit,torch.ones_like(self.delta_scale)*.05)*h
        moving_delta=bound*torch.tanh((h*(vg.sigmoid()*velocity[:,None]+sg.sigmoid()*servo[:,None])+bound*raw.tanh())/bound)
        predictive_unit=torch.maximum(self.quiet_noise,.01*self.delta_scale)
        quiet=(predictive_unit*h*(.01+F.softplus(b0))).clamp_min(1e-7)
        moving=2*quiet+self.delta_scale*h*(.01+F.softplus(b1))
        return {'current':x[:,None]+quiet_delta+translation,'moving_delta':moving_delta-quiet_delta,'quiet_scale':quiet,
                'moving_scale':moving,'mixture_logit':mix,'motion_logit':out[:,:,196]}
