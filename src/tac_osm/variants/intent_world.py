"""Joint future logged-reference and physical-state prediction from causal history.

References are predictions under observed behavior, not chosen actions or
identified causal interventions. No true future reference enters inference.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .raw_router_world import RawRouterWorld


class IntentWorld(RawRouterWorld):
    architecture='native_joint_reference_world'

    def __init__(self,identified=None):
        super().__init__(identified)
        self.coordinate_router[0]=nn.Linear(35,64)
        self.reference_head=nn.Linear(64,42)
        nn.init.normal_(self.reference_head.weight,std=.001);nn.init.zeros_(self.reference_head.bias)
        with torch.no_grad():
            self.reference_head.bias[14:28].fill_(-2.)
            self.reference_head.bias[28:].fill_(-4.)

    def decode(self,x,observed,representation,feature,velocity,command,lag_bank,lag_age):
        b=len(x);n=len(self.horizons)
        context=torch.cat((observed,representation,feature[:,14:84]),-1)
        query=torch.cat((context[:,None].expand(b,n,198),self.horizon_features[None].expand(b,n,4)),-1)
        value,gate=self.gated_readout(query).chunk(2,-1);hidden=value*F.silu(gate);out=self.heads(hidden)
        mix,raw,b0,b1,vg,sg=out[:,:,:84].reshape(b,n,6,14).unbind(2)
        raw_reference,reference_change,raw_lead=self.reference_head(hidden[:,:,:64]).chunk(3,-1)
        reference_delta=4*reference_change.sigmoid()*raw_reference.tanh()
        lead=raw_lead.sigmoid()
        reference_prediction=x[:,None]+command[:,None]+reference_delta
        h=self.horizons[None,:,None]
        lag_weights=(out[:,:,84:196].reshape(b,n,8,14)-lag_age[:,None,:,None]/8).softmax(2)
        innovation=(lag_weights*(lag_bank[:,None]-x[:,None,None])).sum(2)
        unit=self.quiet_noise*h.sqrt();quiet_delta=innovation/(1+(innovation/unit).abs())
        servo=torch.clamp(self.servo_gain*(command[:,None]+lead*reference_delta),-self.servo_step_limit,self.servo_step_limit)
        bound=torch.maximum(4*self.servo_step_limit,torch.ones_like(self.delta_scale)*.05)*h
        moving_delta=bound*torch.tanh((h*(vg.sigmoid()*velocity[:,None]+sg.sigmoid()*servo)+bound*raw.tanh())/bound)
        predictive_unit=torch.maximum(self.quiet_noise,.01*self.delta_scale)
        quiet=(predictive_unit*h*(.01+F.softplus(b0))).clamp_min(1e-7)
        moving=2*quiet+self.delta_scale*h*(.01+F.softplus(b1))
        relative=torch.asinh((lag_bank-x[:,None])/self.delta_scale).transpose(1,2)
        times=(torch.log1p(lag_age)/3.4965075614664802)[:,None].expand(b,14,8)
        current=torch.stack((x,torch.asinh(command/self.delta_scale),feature[:,28:42],feature[:,42:56],
             feature[:,56:70],feature[:,70:84],feature[:,85:99],feature[:,84:85].expand(b,14)),-1)
        local=torch.cat((relative,times,current),-1)
        predicted_intent=torch.stack((torch.asinh(reference_delta/self.delta_scale),reference_change.sigmoid(),lead),-1)
        local_query=torch.cat((local[:,None].expand(b,n,14,24),self.horizon_features[None,:,None].expand(b,n,14,4),
             self.coordinate_embedding.weight[None,None].expand(b,n,14,4),predicted_intent),-1)
        evidence=self.coordinate_router(local_query).squeeze(-1)
        return {'current':x[:,None]+quiet_delta,'moving_delta':moving_delta-quiet_delta,'quiet_scale':quiet,
                'moving_scale':moving,'mixture_logit':mix+evidence,'motion_logit':out[:,:,196],
                'reference_prediction':reference_prediction,'reference_change_logit':reference_change,'reference_lead':lead}
