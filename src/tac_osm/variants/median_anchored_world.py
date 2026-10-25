"""Native visual model with explicit learned median and recentered multimodal shape.

The predictive median is independent of shape mixing weights/scales. Shape remains
multimodal; its implicit median derivative is retained when translating its CDF.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .robust_visual_world import RobustVisualWorld
from .multimode_distribution import implicit_median


class MedianAnchoredWorld(RobustVisualWorld):
    architecture='native_median_anchored_visual_world'

    def __init__(self,identified=None):
        super().__init__(identified)
        self.decision_head=nn.Linear(108,42)
        nn.init.zeros_(self.decision_head.weight);nn.init.zeros_(self.decision_head.bias)
        with torch.no_grad():
            self.decision_head.bias[:14].fill_(-2.)
            self.decision_head.bias[14:28].fill_(7.75)

    def decode(self,x,observed,representation,feature,velocity,command,lag_bank,lag_age):
        b=len(x);n=len(self.horizons)
        context=torch.cat((observed,representation,feature[:,14:84]),-1)
        query=torch.cat((context[:,None].expand(b,n,198),self.horizon_features[None].expand(b,n,4)),-1)
        value,gate=self.gated_readout(query).chunk(2,-1);hidden=value*F.silu(gate)
        out=F.linear(hidden,torch.cat((self.heads.weight,self.jitter_head.weight,self.decision_head.weight),0),torch.cat((self.heads.bias,self.jitter_head.bias,self.decision_head.bias),0))
        motion,raw,bhold,bdyn,vg,sg=out[:,:,:84].reshape(b,n,6,14).unbind(2)
        translation=out[:,:,84:98];risk=out[:,:,98]
        jitter,bmicro,blend=out[:,:,99:141].reshape(b,n,3,14).unbind(2)
        point_gate,timescale,retrieval_gate=out[:,:,141:].reshape(b,n,3,14).unbind(2)
        h=self.horizons[None,:,None]
        servo=torch.clamp(self.servo_gain*command,-self.servo_step_limit,self.servo_step_limit)
        bound=torch.maximum(4*self.servo_step_limit,torch.ones_like(self.delta_scale)*.05)*h
        dynamic_delta=bound*torch.tanh((h*(vg.sigmoid()*velocity[:,None]+sg.sigmoid()*servo[:,None])+bound*raw.tanh())/bound)
        dynamic=x[:,None]+dynamic_delta+self.delta_scale*h*translation.tanh()
        median3=lag_bank[:,-3:].median(1).values
        band=(self.quiet_noise*self.delta_scale).sqrt().clamp_min(1e-7)
        deviation=(lag_bank[:,None]-median3[:,None,None])/band
        scores=-lag_age[:,None,:,None]/(.25+F.softplus(timescale))[:,:,None]-torch.log1p(deviation.square())
        retrieved=(scores.softmax(2)*lag_bank[:,None]).sum(2)
        robust=median3[:,None]+retrieval_gate.sigmoid()*(retrieved-median3[:,None])
        micro=x[:,None]+(1-blend.sigmoid())*(robust-x[:,None])
        point=micro+point_gate.sigmoid()*(dynamic-micro)
        noise=(self.quiet_noise*h.sqrt()*(.01+F.softplus(bhold))).clamp_min(1e-7)
        local=2*noise+(self.quiet_noise*self.delta_scale).sqrt()*h.sqrt()*(.01+F.softplus(bmicro))
        wide=2*local+self.delta_scale*h*(.01+F.softplus(bdyn))
        base={'locations':torch.stack((x[:,None].expand(b,n,14),micro,dynamic),-2),'scales':torch.stack((noise,local,wide),-2),
              'component_logits':torch.stack((torch.zeros_like(motion),jitter,motion),-2)}
        q=implicit_median(base)
        shape={k:v.double() for k,v in base.items()}
        shape['locations']=shape['locations']+(point.double()-q).unsqueeze(-2)
        return {**shape,'median':point.double(),'motion_logit':risk}
