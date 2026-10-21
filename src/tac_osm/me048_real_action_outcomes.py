"""ME048 action-conditioned real outcome predictions from a randomized experiment.

Criteo v2.1 anonymized features are NOT a time sequence. The TAC/GRU cores
here are recurrent feature computation, NOT evidence of persistent world state.
Only observed randomized treatments and visit/conversion outcomes train heads.
"""
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me040_dualrate_innovation import PairedInnovationMemory

ARMS=("action_context","action_gru","action_tac")
GOALS=("visit","conversion")


class ActionOutcomeModel(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError("unregistered action learner")
        self.arm=arm
        if arm=="action_tac":
            self.coordinate=nn.Linear(1,16)
            self.state=PairedInnovationMemory()
        elif arm=="action_gru":
            self.state=nn.GRU(1,64,batch_first=True)
        else:
            self.state=nn.Sequential(nn.Linear(12,128),nn.SiLU(),
                                     nn.Linear(128,64))
        self.goal=nn.Embedding(2,16)
        self.action_head=nn.Sequential(
             nn.Linear(64+16,96),nn.SiLU(),nn.Linear(96,2))
        nn.init.constant_(self.action_head[-1].bias,-3.)
        self.register_buffer("trained_mean",torch.zeros(12))
        self.register_buffer("trained_std",torch.ones(12))

    def set_training_normalization(self,mean,std):
        if mean.shape!=(12,) or std.shape!=(12,) or not (std>0).all():
            raise ValueError("train-only 12 covariate means and scales needed")
        self.trained_mean.copy_(mean)
        self.trained_std.copy_(std)

    def forward(self,x,goal):
        if x.ndim!=2 or x.size(-1)!=12:raise ValueError("12 real covariates required")
        if goal.shape!=(len(x),):raise ValueError("one validated goal per user")
        v=(x-self.trained_mean)/self.trained_std
        v=v.clamp(-15,15)
        if self.arm=="action_tac":
            state=self.state(self.coordinate(v.unsqueeze(-1)))
        elif self.arm=="action_gru":
            _,last=self.state(v.unsqueeze(-1))
            state=last[-1]
        else:
            state=self.state(v)
        return self.action_head(torch.cat((state,self.goal(goal)),dim=-1))

    def probability(self,x,goal):
        return torch.sigmoid(self.forward(x,goal))

    def num_active(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def true_observed_action_loss(net,x,a,visit,conversion):
    if not torch.isin(a,torch.tensor([0,1],device=a.device)).all():
        raise ValueError("treatment must be observed real 0/1")
    gvisit=torch.zeros(len(x),dtype=torch.long,device=x.device)
    gconv=torch.ones(len(x),dtype=torch.long,device=x.device)
    observedvisit=net.forward(x,gvisit).gather(1,a[:,None])[:,0]
    observedconv=net.forward(x,gconv).gather(1,a[:,None])[:,0]
    loss=F.binary_cross_entropy_with_logits(observedvisit,visit)
    loss=loss+.3*F.binary_cross_entropy_with_logits(observedconv,conversion)
    return loss


def preflight():
    """Software-only numerical invariants, no synthetic causal evidence."""
    torch.set_num_threads(1)
    torch.manual_seed(48048)
    result={}
    for arm in ARMS:
        m=ActionOutcomeModel(arm)
        x=torch.randn(5,12)
        g=torch.tensor([0,1,0,1,0])
        a=torch.tensor([0,1,0,1,1])
        prob=m.probability(x,g)
        assert prob.shape==(5,2) and torch.isfinite(prob).all()
        loss=true_observed_action_loss(m,x,a,torch.ones(5),torch.zeros(5))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in m.parameters() if p.requires_grad)
        result[arm]=m.num_active()
    return {"status":"ME048_MATH_SOFTWARE_ONLY_PASS","active_coefficients":result,
            "treatment_possibilities":2,"goals":GOALS,
            "authentic_action_test_results_computed":False,
            "sequence_of_logged_user_states_available":False}
