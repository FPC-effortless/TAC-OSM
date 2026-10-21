"""ME051: TAC predictive-retention on REAL long human movie rating histories.

No generated rating data used as intelligence evidence. Each token is an
ordered aggregate of up to eight strictly earlier authentic user actions.
TAC fast/slow+8x8 dynamics are unchanged from ME038 except the number of
updates (32); this is NOT an SSM or a Transformer.
"""
import torch
from torch import nn
from tac_osm.me038_innovation_forecast import InnovationMemory

ARMS=("tac_old","tac_recent","gru_old","gru_recent","context_recent")
HORIZONS=(16,64,256)
TOKENS=32
CHUNK=8
MOVIES=3953
AUX_WEIGHT=.20


class LongInnovationMemory(InnovationMemory):
    def forward(self,tokens):
        if tokens.ndim!=3 or tokens.shape[1:]!=(TOKENS,16):
            raise ValueError("expected 32 authentic aggregate tokens of width 16")
        f=tokens.new_zeros((tokens.size(0),32))
        s=torch.zeros_like(f)
        matrix=tokens.new_zeros((tokens.size(0),8,8))
        for t in range(TOKENS):
            f,s,matrix=self.step(tokens[:,t],f,s,matrix)
        state=torch.cat((f,s),-1)
        return state/torch.sqrt(state.square().mean(-1,keepdim=True)+.01)


class RealRetentionPredictor(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError("unknown ME051 architecture")
        self.arm=arm
        self.movie=nn.Embedding(MOVIES,8,padding_idx=0)
        self.token=nn.Linear(11,16)
        if arm.startswith("tac_"):self.state=LongInnovationMemory()
        elif arm.startswith("gru_"):self.state=nn.GRU(16,64,batch_first=True)
        else:self.state=nn.Sequential(nn.Linear(16,64),nn.SiLU(),nn.Linear(64,64))
        self.pred=nn.Sequential(nn.Linear(72,64),nn.SiLU(),nn.Linear(64,1))
        self.retrieve=nn.Linear(64,1)
        nn.init.zeros_(self.movie.weight.data[0])

    def latent(self,hist_movie,hist_rating,hist_gap,valid):
        b=hist_movie.size(0)
        if (hist_movie.shape!=(b,256) or hist_rating.shape!=(b,256)
                or hist_gap.shape!=(b,256) or valid.shape!=(b,256)):
            raise ValueError("one real 256-slot history/mask per event")
        if not torch.isin(valid,torch.tensor([0.,1.],device=valid.device)).all():
            raise ValueError("history must indicate actual, not fabricated events")
        mask=valid.reshape(b,TOKENS,CHUNK)
        counts=mask.sum(-1,keepdim=True).clamp_min(1.)
        item=(self.movie(hist_movie).reshape(b,TOKENS,CHUNK,8)*mask.unsqueeze(-1)).sum(2)/counts
        rating=(hist_rating.reshape(b,TOKENS,CHUNK)*mask).sum(-1,keepdim=True)/counts
        gap=(hist_gap.reshape(b,TOKENS,CHUNK)*mask).sum(-1,keepdim=True)/counts
        filled=(mask.sum(-1,keepdim=True)>0).to(rating.dtype)
        token=torch.tanh(self.token(torch.cat((item,rating,gap,filled),dim=-1)))*filled
        if self.arm.startswith("tac_"):z=self.state(token)
        elif self.arm.startswith("gru_"):
            _,last=self.state(token);z=last[-1]
        else:z=self.state(token.sum(1)/filled.sum(1).clamp_min(1.))
        return z

    def forward(self,hist_movie,hist_rating,hist_gap,valid,candidate):
        if candidate.shape!=(hist_movie.size(0),):raise ValueError("real candidate ID shape invalid")
        z=self.latent(hist_movie,hist_rating,hist_gap,valid)
        guess=3.+2.*torch.tanh(self.pred(torch.cat((z,self.movie(candidate)),dim=-1)).squeeze(-1))
        recalled=3.+2.*torch.tanh(self.retrieve(z).squeeze(-1))
        return guess,recalled

    def num_active(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def real_quarter_target(history_rating,valid,oldest):
    """Factual past mean over earliest/latest quarter of actual H observed events."""
    if history_rating.shape!=valid.shape:raise ValueError("history target/mask mismatch")
    counts=valid.sum(-1,keepdim=True)
    if (counts<16).any():raise ValueError("at least 16 genuinely prior events needed")
    positions=valid.cumsum(-1)
    quarter=torch.floor(counts/4.)
    if oldest:
        chosen=(positions<=quarter)&(valid>0)
    else:
        chosen=(positions>counts-quarter)&(valid>0)
    values=history_rating*2.+3. # Input normalization of genuine original star ratings
    return (values*chosen).sum(-1)/chosen.sum(-1).clamp_min(1)


def genuine_rating_loss(model,batch,oldest):
    predicted,recalled=model(batch["movie"],batch["rating"],batch["gap"],batch["valid"],batch["candidate"])
    old=real_quarter_target(batch["rating"],batch["valid"],oldest=oldest)
    main=((predicted-batch["target"])/2.).square().mean()
    aux=((recalled-old)/2.).square().mean()
    return main+AUX_WEIGHT*aux,main.detach(),aux.detach()
