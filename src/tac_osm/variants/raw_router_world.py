"""Native global state plus parallel coordinate-local causal evidence routing."""
import torch
from torch import nn
from torch.nn import functional as F
from .precision_world import PrecisionWorld


class RawRouterWorld(PrecisionWorld):
    architecture='native_raw_coordinate_router_world'

    def __init__(self,identified=None):
        super().__init__(identified)
        self.coordinate_embedding=nn.Embedding(14,4)
        self.coordinate_router=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.normal_(self.coordinate_embedding.weight,std=.02)
        nn.init.normal_(self.coordinate_router[-1].weight,std=.001)
        nn.init.zeros_(self.coordinate_router[-1].bias)

    def decode(self,x,observed,representation,feature,velocity,command,lag_bank,lag_age):
        b=len(x);n=len(self.horizons)
        context=torch.cat((observed,representation,feature[:,14:84]),-1)
        query=torch.cat((context[:,None].expand(b,n,198),self.horizon_features[None].expand(b,n,4)),-1)
        value,gate=self.gated_readout(query).chunk(2,-1);out=self.heads(value*F.silu(gate))
        mix,raw,b0,b1,vg,sg=out[:,:,:84].reshape(b,n,6,14).unbind(2)
        h=self.horizons[None,:,None]
        lag_weights=(out[:,:,84:196].reshape(b,n,8,14)-lag_age[:,None,:,None]/8).softmax(2)
        innovation=(lag_weights*(lag_bank[:,None]-x[:,None,None])).sum(2)
        unit=self.quiet_noise*h.sqrt();quiet_delta=innovation/(1+(innovation/unit).abs())
        servo=torch.clamp(self.servo_gain*command,-self.servo_step_limit,self.servo_step_limit)
        bound=torch.maximum(4*self.servo_step_limit,torch.ones_like(self.delta_scale)*.05)*h
        moving_delta=bound*torch.tanh((h*(vg.sigmoid()*velocity[:,None]+sg.sigmoid()*servo[:,None])+bound*raw.tanh())/bound)
        predictive_unit=torch.maximum(self.quiet_noise,.01*self.delta_scale)
        quiet=(predictive_unit*h*(.01+F.softplus(b0))).clamp_min(1e-7)
        moving=2*quiet+self.delta_scale*h*(.01+F.softplus(b1))
        # All inputs are actual prefix values; no future response or control.
        relative=torch.asinh((lag_bank-x[:,None])/self.delta_scale).transpose(1,2)
        times=(torch.log1p(lag_age)/3.4965075614664802)[:,None].expand(b,14,8)
        current=torch.stack((x,torch.asinh(command/self.delta_scale),feature[:,28:42],feature[:,42:56],
             feature[:,56:70],feature[:,70:84],feature[:,85:99],feature[:,84:85].expand(b,14)),-1)
        local=torch.cat((relative,times,current),-1)
        local_query=torch.cat((local[:,None].expand(b,n,14,24),self.horizon_features[None,:,None].expand(b,n,14,4),
             self.coordinate_embedding.weight[None,None].expand(b,n,14,4)),-1)
        evidence=self.coordinate_router(local_query).squeeze(-1)
        return {'current':x[:,None]+quiet_delta,'moving_delta':moving_delta-quiet_delta,'quiet_scale':quiet,
                'moving_scale':moving,'mixture_logit':mix+evidence,'motion_logit':out[:,:,196]}
