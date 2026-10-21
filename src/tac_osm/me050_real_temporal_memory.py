"""ME050: real event history models; training labels must be authentic KuaiRand clicks.

Native recurrent transitions are unchanged ME038 innovation memory. The current
candidate video is a logged exposure (an observed action), not a planned action.
No synthetic sequence task is used as performance evidence.
"""
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me038_innovation_forecast import InnovationMemory

ARMS=("tac_chronological","tac_reversed","gru_chronological","context_history")
HISTORY=12
BUCKETS=4096


class RealHistoryPredictor(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS: raise ValueError("unregistered arm")
        self.arm=arm
        self.video=nn.Embedding(BUCKETS,8)
        self.token=nn.Linear(11,16)
        if arm.startswith("tac_"):
            self.memory=InnovationMemory()
        elif arm=="gru_chronological":
            self.memory=nn.GRU(16,64,batch_first=True)
        else:
            self.memory=nn.Sequential(nn.Linear(16,64),nn.SiLU(),nn.Linear(64,64))
        self.head=nn.Sequential(nn.Linear(73,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.constant_(self.head[-1].bias,-2.)

    def forward(self, history_video, history_numeric, current_video, current_hour):
        n=len(history_video)
        if history_video.shape!=(n,HISTORY) or history_numeric.shape!=(n,HISTORY,3):
            raise ValueError("requires 12 strictly earlier genuine event tokens")
        if current_video.shape!=(n,) or current_hour.shape!=(n,1):
            raise ValueError("requires current observed candidate and pre-click time")
        h=torch.cat((self.video(history_video),history_numeric),dim=-1)
        tokens=torch.tanh(self.token(h))
        if self.arm=="tac_reversed":
            tokens=tokens.flip(1)
        if self.arm.startswith("tac_"):
            state=self.memory(tokens)
        elif self.arm=="gru_chronological":
            _,hidden=self.memory(tokens)
            state=hidden[-1]
        else:
            state=self.memory(tokens.mean(1))
        candidate=torch.cat((state,self.video(current_video),current_hour),dim=-1)
        return self.head(candidate).squeeze(-1)

    def num_active(self):
        return sum(x.numel() for x in self.parameters() if x.requires_grad)


def real_observed_click_loss(model, history_video, history_numeric,
                             current_video, current_hour, clicked):
    if clicked.ndim!=1 or not torch.isin(clicked,torch.tensor([0.,1.],device=clicked.device)).all():
        raise ValueError("requires actual observed binary clicks")
    return F.binary_cross_entropy_with_logits(
        model(history_video,history_numeric,current_video,current_hour),clicked)
