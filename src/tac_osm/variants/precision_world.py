"""Native predictive-noise calibration and short-step residual trust regions."""
import torch
from torch.nn import functional as F
from .clock_value_world import ClockValueWorld


class PrecisionWorld(ClockValueWorld):
    architecture='native_precision_clock_value_world'

    def decode(self,x,observed,representation,feature,velocity,command,lag_bank,lag_age):
        b=len(x);n=len(self.horizons)
        context=torch.cat((observed,representation,feature[:,14:84]),-1)
        query=torch.cat((context[:,None].expand(b,n,198),self.horizon_features[None].expand(b,n,4)),-1)
        value,gate=self.gated_readout(query).chunk(2,-1);out=self.heads(value*F.silu(gate))
        mix,raw,b0,b1,vg,sg=out[:,:,:84].reshape(b,n,6,14).unbind(2)
        h=self.horizons[None,:,None]
        lag_weights=(out[:,:,84:196].reshape(b,n,8,14)-lag_age[:,None,:,None]/8).softmax(2)
        innovation=(lag_weights*(lag_bank[:,None]-x[:,None,None])).sum(2)
        unit=self.quiet_noise*h.sqrt()
        quiet_delta=innovation/(1+(innovation/unit).abs())
        servo=torch.clamp(self.servo_gain*command,-self.servo_step_limit,self.servo_step_limit)
        bound=torch.maximum(4*self.servo_step_limit,torch.ones_like(self.delta_scale)*.05)*h
        trust=1-.9*torch.exp(-(h-1))
        moving_delta=bound*torch.tanh((h*(vg.sigmoid()*velocity[:,None]+sg.sigmoid()*servo[:,None])+trust*bound*raw.tanh())/bound)
        # Predictive innovation units are distinct from quantized-observation
        # noise. A smooth learned scale remains active on every coordinate.
        predictive_unit=torch.maximum(self.quiet_noise,.01*self.delta_scale)
        quiet=(predictive_unit*h*(.01+F.softplus(b0))).clamp_min(1e-7)
        moving=2*quiet+self.delta_scale*h*(.01+F.softplus(b1))
        return {'current':x[:,None]+quiet_delta,'moving_delta':moving_delta-quiet_delta,'quiet_scale':quiet,
                'moving_scale':moving,'mixture_logit':mix,'motion_logit':out[:,:,196]}
