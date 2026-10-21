"""ME049 real three-arm randomized visit/conversion/spend outcome intelligence.

Unlike invalid ME048 cohort, each split contains genuine randomized outcomes
for ALL actions. Native recurrent dimensions are feature-coordinate processing,
not a learned timeline. Three models match active trainable param within 1%.
"""
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me040_dualrate_innovation import PairedInnovationMemory

ARMS=("action_context","action_gru","action_tac")
ACTION_NAMES=("No E-Mail","Mens E-Mail","Womens E-Mail")
GOALS=("visit","conversion","log1p_spend")
GRU_HIDDEN=57
CONTEXT_HIDDEN=125


class ThreeActionWorldModel(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError("ME049 unknown action architecture")
        self.arm=arm
        if arm=="action_tac":
            self.coordinate=nn.Linear(1,16)
            self.state=PairedInnovationMemory()
            output_width=64
        elif arm=="action_gru":
            self.state=nn.GRU(1,GRU_HIDDEN,batch_first=True)
            output_width=GRU_HIDDEN
        else:
            self.state=nn.Sequential(nn.Linear(12,CONTEXT_HIDDEN),
                    nn.SiLU(),nn.Linear(CONTEXT_HIDDEN,64))
            output_width=64
        self.goal=nn.Embedding(3,16)
        self.head=nn.Sequential(nn.Linear(output_width+16,96),
                    nn.SiLU(),nn.Linear(96,3))
        nn.init.constant_(self.head[-1].bias,-2.)
        self.register_buffer("fitted_mean",torch.zeros(12))
        self.register_buffer("fitted_std",torch.ones(12))

    def set_fit_normalization(self,mean,std):
        if mean.shape!=(12,) or std.shape!=(12,) or not torch.isfinite(std).all() or not (std>0).all():
            raise ValueError("train-only validated normalization mandatory")
        self.fitted_mean.copy_(mean)
        self.fitted_std.copy_(std)

    def forward(self,x,goal):
        if x.ndim!=2 or x.size(1)!=12:
            raise ValueError("12 real pre-treatment features per customer")
        if goal.shape!=(len(x),):
            raise ValueError("one typed goal per actual historical customer")
        state_input=((x-self.fitted_mean)/self.fitted_std).clamp(-12.,12.)
        if self.arm=="action_tac":
            h=self.state(self.coordinate(state_input.unsqueeze(-1)))
        elif self.arm=="action_gru":
            _,end=self.state(state_input.unsqueeze(-1))
            h=end[-1]
        else:
            h=self.state(state_input)
        return self.head(torch.cat((h,self.goal(goal)),dim=-1))

    def active_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def actual_logged_action_loss(net,x,action,visit,conversion,spend):
    if (action<0).any() or (action>2).any():
        raise ValueError("invalid randomized treatment action")
    def pred(goalidx):
        goal=torch.full((len(x),),goalidx,dtype=torch.long,device=x.device)
        return net(x,goal).gather(1,action[:,None])[:,0]
    observed_visit=pred(0)
    observed_conv=pred(1)
    observed_log_spend=pred(2)
    # All labels come from the SINGLE actual randomized intervention.
    loss=F.binary_cross_entropy_with_logits(observed_visit,visit)
    loss=loss+.3*F.binary_cross_entropy_with_logits(observed_conv,conversion)
    loss=loss+.1*F.smooth_l1_loss(observed_log_spend,torch.log1p(spend))
    return loss


def test_math_preflight():
    """Synthetic numerical fixtures test software only: NO performance claim."""
    torch.set_num_threads(1)
    torch.manual_seed(49049)
    counts={}
    for arm in ARMS:
        model=ThreeActionWorldModel(arm)
        x=torch.randn(3,12)
        a=torch.tensor([0,1,2])
        visit=torch.tensor([0.,1.,0.])
        conv=torch.tensor([0.,0.,0.])
        spend=torch.tensor([0.,0.,0.])
        y=model(x,torch.tensor([0,1,2]))
        assert y.shape==(3,3) and torch.isfinite(y).all()
        loss=actual_logged_action_loss(model,x,a,visit,conv,spend)
        model.zero_grad(set_to_none=True);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad),arm
        counts[arm]=model.active_parameters()
    parity=max(counts.values())/min(counts.values())
    if parity>=1.01:
        raise AssertionError("real treatment active parameter matching failed "+str(counts))
    return {"status":"ME049_NUMERICAL_NO_SYNTHETIC_SUCCESS",
            "active_parameter_counts":counts,"max_min_ratio":parity,
            "all_action_labels_genuine_only_when_real_training":True,
            "heldout_actions_loaded":False}
