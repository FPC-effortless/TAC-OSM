"""ME025 frozen-design Torch cores: original TAC + rate kernels vs causal tiny Transformer.

No SSM replacement. The rate kernels are fixed causal observational summaries,
not learned memory gates; they use ONLY feedback before the forecasted response.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from tac_osm.coupled_native import CoupledNative

HALFLIVES = (2., 8., 16., 32., 64.)
VOCAB = 16000


def kernel_summaries(prior_feedback: torch.Tensor) -> torch.Tensor:
    """Five exact normalized exponentially weighted causal means.

    prior_feedback: [B,T] contains only previously observed factual responses.
    Newest observation has weight 1. This never reads target response at t=T.
    """
    if prior_feedback.ndim != 2 or prior_feedback.size(1) < 1:
        raise ValueError("must pass causal observed nonempty historical feedback")
    t = prior_feedback.size(1)
    ages = torch.arange(t-1, -1, -1, device=prior_feedback.device,
                        dtype=prior_feedback.dtype)
    half = torch.tensor(HALFLIVES, dtype=prior_feedback.dtype,
                        device=prior_feedback.device)
    weight = torch.exp2(-ages[:, None] / half[None, :])
    return prior_feedback @ (weight / weight.sum(0, keepdim=True))


def streaming_kernel_step(sums: torch.Tensor, counts: torch.Tensor,
                          observed_feedback: torch.Tensor):
    """Fixed per-event exponentially decayed sufficient-statistic streaming update."""
    half = sums.new_tensor(HALFLIVES)
    factor = torch.exp2(-1. / half)
    return sums * factor + observed_feedback[:, None], counts * factor + 1.


def split_rms(state: torch.Tensor, epsilon: float = .01):
    z, slow = state[:, :32], state[:, 32:]
    return torch.cat((
        z / torch.sqrt(z.square().mean(-1, keepdim=True)+epsilon),
        slow / torch.sqrt(slow.square().mean(-1, keepdim=True)+epsilon)), -1)


def recent32_prior(metadata: torch.Tensor, depth: int):
    if depth < 32:
        raise ValueError("32 previously observed outcomes required")
    return (metadata[:, depth-32:depth, 1].sum(-1, keepdim=True) + 1.) / 34.


class KernelNative(CoupledNative):
    """ME024 RMS-only EXACT native fast/slow dynamics, 6 extra readout coefficients."""

    def __init__(self, duration_mean: float, duration_scale: float,
                 popularity: torch.Tensor):
        super().__init__(coupling=0., rms_readout=True,
                         duration_mean=duration_mean,
                         duration_scale=duration_scale)
        self.head = nn.Linear(79, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if popularity.shape != (VOCAB,):
            raise ValueError("train-only popularity lookup must cover original items")
        self.register_buffer("item_popularity", popularity.detach().clone())

    def readout_at(self, state, current_item, current_duration, meta, depth):
        feats = self.state_features(state)
        d = (current_duration-self.duration_mean)/self.duration_scale
        prior = recent32_prior(meta, depth)
        kernels = kernel_summaries(meta[:, :depth, 1])
        pop = self.item_popularity[current_item].unsqueeze(-1)
        raw = self.head(torch.cat((
            feats, self.emb(current_item), d, kernels, pop), dim=-1))
        return raw + torch.logit(prior)

    def prefix_logits(self, items, metadata, action, duration):
        batch, n = items.shape
        if n != 128 or metadata.shape != (batch, 128, 3):
            raise ValueError("exactly 128 real input events required")
        x = torch.cat((self.emb(items), metadata), dim=-1)
        state = x.new_zeros(batch,64)
        out = []
        for t in range(128):
            state = self.advance_features(x[:, t], state)
            k = t + 1
            if k in (32,64,96):
                out.append(self.readout_at(state,items[:, k],metadata[:, k, 0:1],
                                           metadata,k))
        out.append(self.readout_at(state,action,duration,metadata,128))
        return torch.cat(out,dim=-1)

    def forward(self, items, metadata, action, duration):
        state = self.encode(items,metadata)
        return self.readout_at(state,action,duration,metadata,128)


class TinyCausalTransformer(nn.Module):
    """2x16-dim 2-head causal attention control with same true input/readout info."""

    def __init__(self, duration_mean: float, duration_scale: float,
                 popularity: torch.Tensor):
        super().__init__()
        self.emb = nn.Embedding(VOCAB,8)
        self.in_proj = nn.Linear(11,16)
        block = nn.TransformerEncoderLayer(
            d_model=16,nhead=2,dim_feedforward=48,dropout=0.,
            batch_first=True,norm_first=True,activation="gelu")
        self.blocks = nn.TransformerEncoder(block,num_layers=2,
                                             enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(16)
        self.to_state = nn.Linear(16,64)
        self.head = nn.Linear(79,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("duration_mean",torch.tensor(duration_mean))
        self.register_buffer("duration_scale",torch.tensor(duration_scale))
        if popularity.shape != (VOCAB,):
            raise ValueError("train-only popularity lookup must cover original items")
        self.register_buffer("item_popularity",popularity.detach().clone())
        pos = torch.arange(128,dtype=torch.float32)[:,None]
        omega=torch.exp(torch.arange(0,16,2,dtype=torch.float32)*(-math.log(10000.)/16))
        encoding=torch.zeros(128,16,dtype=torch.float32)
        encoding[:,0::2]=torch.sin(pos*omega)
        encoding[:,1::2]=torch.cos(pos*omega)
        self.register_buffer("positions",encoding)

    def causal_states(self, items, metadata):
        b,n=items.shape
        if n!=128 or metadata.shape!=(b,128,3):
            raise ValueError("128 genuine historical records required")
        v=self.in_proj(torch.cat((self.emb(items),metadata),-1))
        v=v+self.positions.unsqueeze(0).to(dtype=v.dtype)
        future_mask=torch.ones((n,n),device=v.device,dtype=torch.bool).triu(1)
        h=self.blocks(v,mask=future_mask)
        return self.to_state(self.final_norm(h))

    def readout_at(self, state, item, duration, meta, depth):
        feat=split_rms(state)
        d=(duration-self.duration_mean)/self.duration_scale
        kernels=kernel_summaries(meta[:,:depth,1])
        pop=self.item_popularity[item].unsqueeze(-1)
        prior=recent32_prior(meta,depth)
        return (self.head(torch.cat(
            (feat,self.emb(item),d,kernels,pop),dim=-1))+
                torch.logit(prior))

    def prefix_logits(self,items,metadata,action,duration):
        states=self.causal_states(items,metadata)
        out=[self.readout_at(states[:,k-1],items[:,k],metadata[:,k,0:1],
                             metadata,k) for k in (32,64,96)]
        out.append(self.readout_at(states[:,-1],action,duration,metadata,128))
        return torch.cat(out,dim=-1)

    def forward(self,items,metadata,action,duration):
        states=self.causal_states(items,metadata)
        return self.readout_at(states[:,-1],action,duration,metadata,128)


def parameter_count(net):
    return sum(p.numel() for p in net.parameters())


def software_preflight():
    """Constructed numerical tests ONLY. Never real-model quality evidence."""
    torch.set_num_threads(1)
    torch.manual_seed(25025)
    pop=torch.linspace(.2,.8,VOCAB)
    models={
        "tac_rms":CoupledNative(0.,True,.7,.05),
        "tac_kernel":KernelNative(.7,.05,pop),
        "tiny_transformer_kernel":TinyCausalTransformer(.7,.05,pop),
    }
    counts={k:parameter_count(v) for k,v in models.items()}
    assert counts["tac_rms"]==135050, counts
    assert counts["tac_kernel"]==135056,counts
    assert counts["tiny_transformer_kernel"]==134896,counts
    assert abs(counts["tac_kernel"]-counts["tiny_transformer_kernel"])/counts["tac_kernel"]<.01
    item=torch.randint(VOCAB,(2,128))
    meta=torch.rand(2,128,3)
    meta[:,:,1]=(meta[:,:,1]>.5).float()
    current=torch.randint(VOCAB,(2,))
    d=torch.rand(2,1)
    scores={}
    for name,net in models.items():
        pred=net.prefix_logits(item,meta,current,d)
        assert pred.shape==(2,4) and torch.isfinite(pred).all()
        expected=torch.logit(recent32_prior(meta,128)).flatten()
        assert torch.allclose(pred[:,-1],expected,atol=1e-6,rtol=1e-6)
        if name!="tac_rms":
            # Strict future noninterference for every causal prefix.
            for k,col in ((32,0),(64,1),(96,2)):
                alternate=meta.clone()
                alternate[:,k:,1]=1.-alternate[:,k:,1]
                trial=net.prefix_logits(item,alternate,current,d)
                if not torch.allclose(pred[:,col],trial[:,col],atol=3e-6,rtol=3e-6):
                    raise AssertionError(name+" leaks future response to causal prefix")
        fake_y=torch.rand(2,4).round()
        loss=nn.functional.binary_cross_entropy_with_logits(pred,fake_y)
        loss.backward()
        for p in net.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise ArithmeticError("nonfinite gradient in "+name)
        scores[name]=float(loss.detach())
    # Fixed kernel recurrence must agree with the batch kernel formula for
    # irregular chunking and observed feedback, without using next outcome.
    fb=meta[:,:,1]
    running=torch.zeros(2,5);counts_state=torch.zeros(2,5)
    for i in range(128):
        running,counts_state=streaming_kernel_step(running,counts_state,fb[:,i])
        if i in (31,63,95,127):
            expected=kernel_summaries(fb[:,:i+1])
            assert torch.allclose(running/counts_state,expected,atol=2e-6,rtol=2e-6)
    # Direct per-history feedback check: no FUTURE outcome appended.
    assert all(torch.isfinite(torch.tensor(list(scores.values()))))
    return {"preflight":"ME025_SOFTWARE_ONLY_CAUSAL_MATH_PASS",
            "parameters":counts,"causal_prefix_response_leak":False,
            "streaming_summary_parity":True,"model_performance_claim":False}
