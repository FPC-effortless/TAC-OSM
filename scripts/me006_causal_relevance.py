"""ME006 future-conditioned retrieval; not causal intervention identification.
Only past/current observations enter forward. Authentic future outcomes train the
readout via a proper prediction loss outside forward.
"""
import torch
from torch import nn

class CausalRelevanceAdapter(nn.Module):
    def __init__(self,backbone,mode="associative",features=12,width=16):
        super().__init__()
        if mode not in ("associative","direct"):raise ValueError("unknown arm")
        self.mode=mode;self.features=features;self.width=width
        self.backbone=backbone
        for p in self.backbone.parameters():p.requires_grad_(False)
        self.encode=nn.Linear(2*features,width)
        self.query=nn.Linear(width,width)
        self.decode=nn.Linear(width,2*features)
        nn.init.zeros_(self.decode.weight);nn.init.zeros_(self.decode.bias)
        self.fast_rate_logit=nn.Parameter(torch.tensor(-3.0))
        self.decay_logit=nn.Parameter(torch.tensor(-4.0))
    def forward(self,x,return_details=False):
        if x.ndim!=3 or x.shape[-1]!=2*self.features:
            raise ValueError("expected [B,T,2F] causal observed inputs")
        b,T,_=x.shape
        state=x.new_zeros(b,self.width,self.width);last=None
        for t in range(T):
            mask=x[:,t,self.features:].clamp(0,1)
            value=torch.where(mask>0,x[:,t,:self.features],torch.zeros_like(mask))
            current=torch.tanh(self.encode(torch.cat((value,mask),dim=-1)))
            if last is not None and self.mode=="associative":
                rate=.05*torch.sigmoid(self.fast_rate_logit)
                decay=.01*torch.sigmoid(self.decay_logit)
                state=(1-decay)*state+rate*torch.einsum("bd,be->bde",current,last)/self.width
                norm=state.flatten(1).norm(dim=1,keepdim=True).clamp_min(1e-9)
                state=state/(norm/2).clamp_min(1.0)[:,None,:]
            last=current
        query=torch.tanh(self.query(last))
        read=torch.bmm(state,query.unsqueeze(-1)).squeeze(-1) if self.mode=="associative" else query
        masked_x=torch.cat((torch.where(x[:,:,self.features:]>0,x[:,:,:self.features],
                          torch.zeros_like(x[:,:,:self.features])),x[:,:,self.features:].clamp(0,1)),dim=-1)
        baseline=self.backbone(masked_x)
        residual=self.decode(read).reshape(b,2,self.features)
        prediction=baseline+residual
        if not return_details:return prediction
        return {"forecast":prediction,"baseline":baseline,"fast_state":state,
                "relevance_read":read,"residual":residual}

def future_gain(prediction,target,baseline):
    """Gaussian fixed-variance future conditional relevance, not original LM-CDL."""
    if prediction.shape!=target.shape or prediction.shape!=baseline.shape:
        raise ValueError("target/output shapes mismatch")
    return .5*((baseline-target).square()-(prediction-target).square()).mean()
