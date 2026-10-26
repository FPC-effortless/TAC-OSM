"""LATFUSE002 no-extra-state repair candidate: real-data training is separately gated.

Changes from LATFUSE001: source-conditioned query, explicitly long-retention
delta prior, lower write intensity at initialization, and fixed multirate
slow-memory channels. These are HYPOTHESES, not trained source-verified gains.
The carried state dimensions/dtypes remain *exactly* those of LATFUSE001 H80.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from tac_osm.latent_fusion_001 import FusedLatentCandidate, FusionState, gated_delta, source_slot_write


class FusedLatentFixedState(FusedLatentCandidate):
    def __init__(self, obs_dim=13, action_dim=6, hidden=80, key_dim=16,
                 val_dim=16, entity_count=16):
        super().__init__(obs_dim, action_dim, hidden, key_dim, val_dim, entity_count)
        # Keep existing module shapes; harden the ORIGINAL random initialized gates.
        # The source-conditioned gate remains fully trainable.
        with torch.no_grad():
            self.retention_map.weight.zero_()
            self.retention_map.bias.fill_(math.log(.995/.005))
            self.write_map.weight.zero_()
            self.write_map.bias.fill_(math.log(.1/.9))
        # Read keys from the BELIEF as well as future-known covariates.
        self.belief_query = nn.Linear(hidden*2, key_dim)
        nn.init.normal_(self.belief_query.weight, std=.002)
        nn.init.zeros_(self.belief_query.bias)
        # Multi-rate spectrum within EXISTING slow coordinates; no state added.
        # 0.90 -> 0.99999. At least one channel survives 512 direct steps.
        rates=torch.exp(torch.linspace(math.log(.90),math.log(.99999),hidden))
        self.register_buffer("slow_rates", rates, persistent=True)

    def step(self, state: FusionState, x, observed, elapsed, last_seen, source_slot):
        b=x.shape[0]
        if (state.fast.shape != (b,self.hidden) or
            state.slow.shape != (b,self.hidden) or
            state.associative.shape != (b,self.val_dim,self.key_dim) or
            state.entities.shape != (b,self.entity_count,self.hidden) or
            state.active.shape != (b,self.entity_count)):
            raise ValueError("state shape mismatch")
        u=self.encode(x,observed,elapsed,last_seen)
        a=torch.tanh(self.temporal_a(u))
        f=a*state.fast+torch.tanh(self.temporal_b(u))
        rate=self.slow_rates.to(dtype=u.dtype,device=u.device).view(1,-1)
        s=rate*state.slow+(1-rate)*torch.tanh(self.slow_input(u))
        key=self.key_map(u)
        m=gated_delta(state.associative,key,torch.tanh(self.val_map(u)),
                      torch.sigmoid(self.retention_map(u)),torch.sigmoid(self.write_map(u)))
        slots,active=source_slot_write(state.entities,state.active,source_slot,
                      torch.tanh(self.entity_map(u)),torch.sigmoid(self.entity_gate(u)))
        return FusionState(f,s,m,slots,active)

    def forecast(self, state: FusionState, source_slot, candidates):
        """Read-only future covariate conditional forecast; NOT do(action)."""
        b=state.fast.shape[0]
        if (candidates.ndim!=3 or candidates.shape[0]!=b or
                candidates.shape[2]!=self.action_dim or source_slot.shape!=(b,)):
            raise ValueError("forecast input shape mismatch")
        if bool(((source_slot<0)|(source_slot>=self.entity_count)).any()):
            raise ValueError("invalid registered source slot")
        if not torch.isfinite(candidates).all():
            raise ValueError("nonfinite future covariates")
        joint=torch.cat((state.fast,state.slow),-1)
        a=torch.tanh(self.action_map(candidates))
        belief_key=self.belief_query(joint).unsqueeze(1)
        q=F.normalize(a+belief_key,dim=-1,eps=1e-8)
        recall=(state.associative.unsqueeze(1)@q.unsqueeze(-1)).squeeze(-1)
        slot=source_slot.long().view(b,1,1).expand(-1,1,self.hidden)
        entity=state.entities.gather(1,slot).squeeze(1)
        entity=entity*state.active.gather(1,source_slot.long().view(b,1)).to(entity.dtype)
        n=candidates.shape[1]
        x=torch.cat((state.fast[:,None,:].expand(-1,n,-1),
                     state.slow[:,None,:].expand(-1,n,-1),
                     entity[:,None,:].expand(-1,n,-1),recall,q),dim=-1)
        mean,raw=self.forecast_head(x).chunk(2,-1)
        return mean,F.softplus(raw)+1e-4


def state_contract(model: FusedLatentFixedState):
    """FP32 nominal per independent stream; model parameters counted separately."""
    state=model.initial(1)
    assert len(state)==5
    assert all(t.dtype==torch.float32 for t in state[:4])
    assert state.active.dtype==torch.bool
    floats=sum(t.numel() for t in state[:4])
    active=state.active.numel()
    if (model.hidden,model.key_dim,model.val_dim,model.entity_count)!=(80,16,16,16):
        raise ValueError("LATFUSE002 registered size must remain H80 key16 value16 N16")
    if (floats,active)!=(1696,16):
        raise RuntimeError("recurrent-state capacity changed")
    return {"float_values":floats,"occupancy_flags":active,"nominal_fp32_bytes":floats*4+active,
            "dtype":"torch.float32",
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "deployed_learned_parameters":sum(p.numel() for p in model.parameters())}
