"""ME045 conditional multiscale forecast experts with genuine hard compute skipping.

Frozen original context/AR from 1000 authentic FIT training updates.
Separate fully trained experts share ME044's fixed multiscale horizon basis and
physically bounded correction. Router scores only available historic features.
No policy/action/counterfactual or System2 capability is claimed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me044_multiscale_decoder import MultiscaleForecaster

ARMS=("conditional_native","conditional_gru40","conditional_gru16")
EXPERT_PREFIX=("branch.","branch_projection.","coefficients.","trust.","router.")


class ConditionalMultiscaleForecaster(MultiscaleForecaster):
    def __init__(self,arm,horizon,period):
        if arm not in ARMS:raise ValueError("unregistered conditional expert arm")
        super().__init__("pyramid_native" if arm=="conditional_native" else "pyramid_gru",
                         horizon,period)
        self.conditional_arm=arm
        if arm=="conditional_gru16":
            self.branch=nn.GRU(16,16,batch_first=True)
            self.branch_projection=nn.Linear(16,64)
        # 16 real contextual observations and 128 learned cheap-context features.
        self.router=nn.Sequential(nn.Linear(144,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.threshold=.5

    def lock_frozen_backbone(self,cheap):
        """Initialize exact cheap parameters from an authentic pretrained model."""
        candidate=self.state_dict()
        source=cheap.state_dict()
        for name,value in source.items():
            if name not in candidate:raise KeyError("missing full original context "+name)
            candidate[name]=value.detach().clone()
        self.load_state_dict(candidate,strict=True)
        for name,p in self.named_parameters():
            if not name.startswith(EXPERT_PREFIX):
                p.requires_grad_(False)
        if sum(p.requires_grad for n,p in self.named_parameters()
               if not n.startswith(EXPERT_PREFIX))!=0:
            raise AssertionError("cheap backbone not actually frozen")

    def routing_logits(self,features):
        _,_,ctx,latent,_,_,_=features
        return self.router(torch.cat((ctx.detach(),latent.detach()),-1))

    def _selected_expert(self,features,which):
        """Execute recurrent expert ONLY for actually selected channels."""
        _,tokens,ctx,latent,window,_,_=features
        tokens=tokens.index_select(0,which)
        ctx=ctx.index_select(0,which).detach()
        latent=latent.index_select(0,which).detach()
        if self.conditional_arm=="conditional_native":
            state=self.branch(tokens,latent)
        else:
            paired=(tokens[:,0::2]+3*tokens[:,1::2])*.25
            size=self.branch.hidden_size
            _,hidden=self.branch(paired,.1*torch.tanh(latent[:,:size]).unsqueeze(0))
            state=self.branch_projection(hidden[0])
        inp=torch.cat((ctx,latent,state),-1)
        coeff=self.coefficients(inp)
        trust=torch.sigmoid(self.trust(inp))
        normalized_delta=.5*trust*torch.tanh((coeff@self.forecast_modes)/.5)
        return normalized_delta,coeff,trust

    def routed(self,x,mode="adaptive",threshold=.5):
        """Return full physical forecast, cheap forecast, probabilities, executed.

        mode=cheap avoids expensive expert entirely. mode=always executes all.
        On adaptive mode use nonzero index-select before branch forward.
        """
        if mode not in ("adaptive","always","cheap"):
            raise ValueError("unknown registered serving mode")
        feat=self.features(x)
        cheap,_=self.cheap_from(feat)
        logits=self.routing_logits(feat)
        probabilities=torch.sigmoid(logits)
        if mode=="cheap":
            chosen=torch.zeros_like(probabilities[:,0],dtype=torch.bool)
        elif mode=="always":
            chosen=torch.ones_like(probabilities[:,0],dtype=torch.bool)
        else:
            chosen=probabilities[:,0]>threshold
        which=torch.nonzero(chosen,as_tuple=False).flatten()
        if which.numel()==0:
            return cheap,cheap,probabilities,chosen
        _,_,_,_,_,mean,scale=feat
        d,_,_=self._selected_expert(feat,which)
        # Flattened (batch*channels,horizon) physical residual in exactly
        # the same channel-first layout used by the original ME044 model.
        flat=d.new_zeros((probabilities.size(0),self.horizon))
        full_scale=scale.transpose(1,2).reshape(-1,1)
        phys=d*full_scale.index_select(0,which)
        flat=flat.index_copy(0,which,phys)
        b,_,c=mean.shape
        residual=flat.reshape(b,c,self.horizon).transpose(1,2)
        return cheap+residual,cheap,probabilities,chosen

    def forward(self,x):
        return self.routed(x)[0]

    def expert_router_loss(self,x,y):
        """Ground gate labels in actual FIT futures; never synthetic outcomes."""
        feat=self.features(x)
        cheap,_=self.cheap_from(feat)
        all_idx=torch.arange(feat[1].size(0),device=x.device)
        normal_delta,coeff,_=self._selected_expert(feat,all_idx)
        _,_,_,_,_,_,scale=feat
        b,_,c=scale.shape
        physical=(normal_delta*scale.transpose(1,2).reshape(-1,1))
        full=cheap+physical.reshape(b,c,self.horizon).transpose(1,2)
        # Per actual channel-window forecast error; actual future used ONLY
        # for supervised FIT objective and NEVER supplied to serving router.
        ec=(cheap.detach()-y).square().mean(1).reshape(-1,1)
        ef=(full-y).square().mean(1).reshape(-1,1)
        improved=(ec.detach()-ef.detach()>.01*ec.detach()).to(ec.dtype)
        logits=self.routing_logits(feat)
        p=torch.sigmoid(logits)
        router_loss=.05*F.binary_cross_entropy_with_logits(logits,improved)
        router_loss=router_loss+(p*(ef.detach()-ec.detach()+.001)).mean()
        target=((y-cheap.detach())/scale).transpose(1,2).reshape(-1,self.horizon)
        projected=(target@self.forecast_modes.T/self.horizon).detach()
        expert_loss=(full-y).square().mean()
        expert_loss=expert_loss+.05*(coeff-projected).square().mean()
        expert_loss=expert_loss+.01*physical.square().mean()
        return expert_loss+router_loss,expert_loss,router_loss

    def active_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_model(arm,horizon,seed,period,pretrained_context):
    torch.manual_seed(seed)
    m=ConditionalMultiscaleForecaster(arm,horizon,period)
    m.lock_frozen_backbone(pretrained_context)
    return m
