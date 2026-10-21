"""ME034 stable-input recurrence and preregistered practical controls.

Torch is optional: this module is never imported by the base package.
"""
import torch
from torch import nn
from tac_osm.me032_conditional_memory import split_rms, CenteredReadout
from tac_osm.me033_semantic_moments import build_model as parent_model, SemanticStudent


class ProjectedGRU(nn.Module):
    """Active input projection + standard GRU; fixed injective output coordinates.

    All 6,984 learned body coefficients participate in event updates. The fixed
    64x36 orthonormal map supplies the existing 64D readout without dead padding.
    It carries no trained information and does not enlarge the 36D recurrent state.
    """
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(16000,8)
        self.input=nn.Linear(11,24)
        self.cell=nn.GRUCell(24,36)
        g=torch.Generator().manual_seed(34034)
        q,_=torch.linalg.qr(torch.randn(64,36,generator=g))
        self.register_buffer('coordinates',q)

    def forward(self,items:torch.Tensor,meta:torch.Tensor,offered:torch.Tensor)->torch.Tensor:
        u=torch.tanh(self.input(torch.cat((self.emb(items),meta),-1)))
        h=u.new_zeros((items.size(0),36))
        out=torch.jit.annotate(list[torch.Tensor],[])
        for t in range(128):
            h=self.cell(u[:,t],h)
            if t==31 or t==63 or t==95 or t==127:
                out.append(split_rms(h@self.coordinates.t()))
        return torch.stack(out,1)


class CurrentOnly(nn.Module):
    """Trained history-free reference, rather than an out-of-distribution deletion."""
    def __init__(self,global_logit:float):
        super().__init__()
        self.hidden=nn.Linear(10,43);self.output=nn.Linear(43,1)
        self.register_buffer('global_logit',torch.tensor(global_logit))
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,items:torch.Tensor,meta:torch.Tensor,offered:torch.Tensor,
                context:torch.Tensor)->torch.Tensor:
        x=torch.cat((context[...,:9],context[...,14:15]),-1)
        return self.global_logit+self.output(torch.tanh(self.hidden(x))).squeeze(-1)


class EWMABank(nn.Module):
    """ME021's 12 causal feature definitions, extended to each actual prefix."""
    def __init__(self,popularity:torch.Tensor,mean:torch.Tensor,scale:torch.Tensor):
        super().__init__()
        self.register_buffer('popularity',popularity.detach().clone())
        self.register_buffer('mean',mean.detach().clone())
        self.register_buffer('scale',scale.detach().clone())
        self.register_buffer('halflives',torch.tensor([2.,4.,8.,16.,32.,64.,128.]))
        self.net=nn.Sequential(nn.Linear(12,16),nn.Tanh(),nn.Linear(16,1))

    @torch.jit.export
    def features(self,items:torch.Tensor,meta:torch.Tensor,offered:torch.Tensor,
                 context:torch.Tensor)->torch.Tensor:
        out=torch.jit.annotate(list[torch.Tensor],[])
        # context is R-normalized; caller supplies offered duration separately via
        # the known current metadata at early prefixes and context's inverse below.
        for j in range(4):
            k=(j+1)*32;y=meta[:,:k,1]
            ages=torch.arange(k-1,-1,-1,device=meta.device,dtype=meta.dtype)
            weights=torch.exp2(-ages[:,None]/self.halflives[None,:])
            kernels=y@(weights/weights.sum(0,keepdim=True))
            # Raw duration is supplied in context's extra (28th) auxiliary channel.
            duration=context[:,j,27:28]
            x=torch.cat(((y[:,-32:].sum(-1,keepdim=True)+1.)/34.,
                (y.sum(-1,keepdim=True)+1.)/float(k+2),kernels,
                self.popularity[offered[:,j]].unsqueeze(-1),duration,
                meta[:,k-32:k,2].mean(-1,keepdim=True)),-1)
            out.append(x)
        return torch.stack(out,1)

    def forward(self,items:torch.Tensor,meta:torch.Tensor,offered:torch.Tensor,
                context:torch.Tensor)->torch.Tensor:
        return self.net((self.features(items,meta,offered,context)-self.mean)/self.scale).squeeze(-1)


COUNTS={'native_frozen':8236,'native_adapt':136236,'history_frozen':8772,
        'common_adapt':1205,'gru_frozen':8244,'ewma_mlp':225,'current_only':517}


def build_model(arm,common_state,embedding,seed,popularity,ewma_mean,ewma_scale,global_logit):
    if arm not in COUNTS:raise ValueError(arm)
    if arm in ('native_frozen','native_adapt','history_frozen','common_adapt'):
        net=parent_model('native_adapt' if arm=='native_frozen' else arm,common_state,embedding,seed)
        if arm=='native_frozen':net.memory.core.emb.weight.requires_grad_(False)
    else:
        torch.manual_seed(seed)
        if arm=='gru_frozen':
            net=SemanticStudent(False,True);net.memory=ProjectedGRU()
            with torch.no_grad():net.memory.emb.weight.copy_(embedding)
            net.memory.emb.weight.requires_grad_(False)
            net.common.load_state_dict(common_state,strict=True)
            torch.manual_seed(seed+1);net.correction=CenteredReadout()
        elif arm=='ewma_mlp':net=EWMABank(popularity,ewma_mean,ewma_scale)
        else:net=CurrentOnly(float(global_logit))
    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==COUNTS[arm]
    return net
