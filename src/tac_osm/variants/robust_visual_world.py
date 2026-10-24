"""Native visual state with exact-hold, local-noise and learned-transition regimes.

Modes combine as a probability distribution, not an average of point guesses.
Every input, including the three-value robust observation, is causal.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .visual_world import VisualWorld


class RobustVisualWorld(VisualWorld):
    architecture='native_three_regime_visual_world'

    def __init__(self,identified=None):
        super().__init__(identified)
        # Keep meaningful native prediction fields; replace112lag logits with
        #42new identifiable local-noise fields. No unused output/parameter padding.
        self.heads=nn.Linear(108,99)
        self.jitter_head=nn.Linear(108,42)
        nn.init.zeros_(self.jitter_head.weight);nn.init.zeros_(self.jitter_head.bias)
        with torch.no_grad():
            self.jitter_head.bias[:14].fill_(-2.)
            self.jitter_head.bias[14:28].fill_(.541324854612918)

    def decode(self,x,observed,representation,feature,velocity,command,lag_bank,lag_age):
        b=len(x);n=len(self.horizons)
        context=torch.cat((observed,representation,feature[:,14:84]),-1)
        query=torch.cat((context[:,None].expand(b,n,198),self.horizon_features[None].expand(b,n,4)),-1)
        value,gate=self.gated_readout(query).chunk(2,-1);hidden=value*F.silu(gate)
        # One GEMM, shared hidden tensor; separate parameter groups for new rows.
        out=F.linear(hidden,torch.cat((self.heads.weight,self.jitter_head.weight),0),torch.cat((self.heads.bias,self.jitter_head.bias),0))
        motion,raw,bhold,bdyn,vg,sg=out[:,:,:84].reshape(b,n,6,14).unbind(2)
        translation=out[:,:,84:98];risk=out[:,:,98]
        jitter,bmicro,blend=out[:,:,99:].reshape(b,n,3,14).unbind(2)
        h=self.horizons[None,:,None]
        servo=torch.clamp(self.servo_gain*command,-self.servo_step_limit,self.servo_step_limit)
        bound=torch.maximum(4*self.servo_step_limit,torch.ones_like(self.delta_scale)*.05)*h
        dynamic_delta=bound*torch.tanh((h*(vg.sigmoid()*velocity[:,None]+sg.sigmoid()*servo[:,None])+bound*raw.tanh())/bound)
        dynamic=x[:,None]+dynamic_delta+self.delta_scale*h*translation.tanh()
        robust=lag_bank[:,-3:].median(1).values
        micro=x[:,None]+(1-blend.sigmoid())*(robust-x)[:,None]
        hold=x[:,None].expand(b,n,14)
        noise=(self.quiet_noise*h.sqrt()*(.01+F.softplus(bhold))).clamp_min(1e-7)
        local=2*noise+(self.quiet_noise*self.delta_scale).sqrt()*h.sqrt()*(.01+F.softplus(bmicro))
        wide=2*local+self.delta_scale*h*(.01+F.softplus(bdyn))
        return {'locations':torch.stack((hold,micro,dynamic),-2),'scales':torch.stack((noise,local,wide),-2),
                'component_logits':torch.stack((torch.zeros_like(motion),jitter,motion),-2),'motion_logit':risk}
