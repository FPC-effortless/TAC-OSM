"""LATFUSE003: 4-head, source-selective gated-delta within ORIGINAL 6,800-byte state.

Research candidate only. NOT a Transformer attention cache, not a trained or
independently qualified replacement for LATFUSE002, and NOT a causal action model.
The entire original [B,16,16] FP32 associative matrix remains allocated and used
as four [B,16,4] independent key-subspace heads. Each head stores 16 values;
4*16*4=256 recurrent FP32 cells. No per-history keys/values are carried.

Input-conditioned top-2 writes preserve non-selected head matrices exactly;
temperature affects SOFTMAX BETWEEN FOUR HEADS, not fake softmax over absent
historical tokens. Same fast/slow/entity recurrent shapes and FP32 types.
"""

import torch
from torch import nn
from torch.nn import functional as F

from tac_osm.latent_fusion_001 import FusionState, source_slot_write
from tac_osm.latent_fusion_002 import FusedLatentFixedState


HEADS=4
HEAD_KEY_DIM=4
HEAD_VALUE_DIM=16
TOP_K=2
FLOAT_STATE_VALUES=1696
NOMINAL_FP32_BYTES=6800
PARAMETERS=99747


def _router(logits: torch.Tensor, log_temperatures: torch.Tensor, *, top_k: int):
    if logits.ndim!=2 or logits.shape[-1]!=HEADS or log_temperatures.shape!=(HEADS,):
        raise ValueError("registered four-head router dimensionality mismatch")
    if not bool(torch.isfinite(logits).all() and torch.isfinite(log_temperatures).all()):
        raise ValueError("nonfinite routing logits/temperature")
    # Clamp positive temperatures to avoid collapse from floating-point overflow.
    temp=torch.exp(log_temperatures.clamp(-0.69314718,0.69314718))
    scores=logits/temp[None,:]
    if top_k==HEADS:
        return F.softmax(scores,dim=-1),torch.ones_like(scores,dtype=torch.bool)
    if top_k!=TOP_K:
        raise ValueError("only preregistered top-2 writes or all-head reads permitted")
    selected=torch.zeros_like(scores,dtype=torch.bool).scatter(
        1,torch.topk(scores,k=TOP_K,dim=-1).indices,True)
    # Gradients reach selected router logits; non-selected heads receive no
    # current-step write. Head collapse is possible, not denied.
    weights=F.softmax(scores.masked_fill(~selected,torch.finfo(scores.dtype).min),dim=-1)
    return weights,selected


def four_head_delta(matrix:torch.Tensor,key:torch.Tensor,value:torch.Tensor,
                    retention:torch.Tensor,write:torch.Tensor,router_logits:torch.Tensor,
                    observed_any:torch.Tensor,log_temperatures:torch.Tensor):
    """Numerically stable differentiable right-affine update, 4x[16,4] heads.

    M^h_(t+1) = a_h M^h_t + b_h (v_t-a_h M^h_t k_h) k_h^T.
    a_h=alpha when source-observed and selected, otherwise 1.
    b_h=beta*w_h only for source-observed selected heads.
    Constant memory and per-step O(4*16*4) operations (not whole-graph scan).
    """
    b=matrix.shape[0]
    if (matrix.shape!=(b,16,16) or key.shape!=(b,16) or
        value.shape!=(b,16) or retention.shape!=(b,1) or write.shape!=(b,1) or
        router_logits.shape!=(b,4) or observed_any.shape!=(b,) or
        observed_any.dtype!=torch.bool):
        raise ValueError("multi-head matrix/key/value/gate/observed shape mismatch")
    if (matrix.dtype!=torch.float32 or key.dtype!=torch.float32 or
        value.dtype!=torch.float32 or retention.dtype!=torch.float32 or
        write.dtype!=torch.float32 or router_logits.dtype!=torch.float32):
        raise ValueError("registered LATFUSE003 state/update must stay FP32")
    if not bool(torch.isfinite(matrix).all() and torch.isfinite(key).all()
                and torch.isfinite(value).all() and
                torch.isfinite(retention).all() and torch.isfinite(write).all()):
        raise ValueError("nonfinite incoming head-memory values")
    if bool(((retention<0)|(retention>1)|(write<0)|(write>1)).any()):
        raise ValueError("head gates must lie in [0,1]")
    weights,selected=_router(router_logits,log_temperatures,top_k=TOP_K)
    eligible=selected & observed_any[:,None]
    effective_keep=torch.where(eligible,retention.expand(-1,HEADS),torch.ones_like(weights))
    effective_write=weights*write*eligible.to(weights.dtype)
    heads=matrix.reshape(b,HEAD_VALUE_DIM,HEADS,HEAD_KEY_DIM).permute(0,2,1,3)
    k=F.normalize(key.reshape(b,HEADS,HEAD_KEY_DIM),dim=-1,eps=1e-8)
    retained=effective_keep[:,:,None,None]*heads
    prior=(retained@k.unsqueeze(-1)).squeeze(-1)
    updated=retained+effective_write[:,:,None,None]*(
        value[:,None,:,None]-prior.unsqueeze(-1))*k[:,:,None,:]
    result=updated.permute(0,2,1,3).contiguous().reshape(b,16,16)
    if result.shape!=matrix.shape or not bool(torch.isfinite(result).all()):
        raise RuntimeError("four-head matrix numerical update invalid")
    return result,eligible,weights


class FusedLatentFourHead(FusedLatentFixedState):
    """Four isolated key subspaces in original FP32 recurrent state capacity."""

    def __init__(self,obs_dim=13,action_dim=6,hidden=80,key_dim=16,val_dim=16,
                 entity_count=16):
        if (obs_dim,action_dim,hidden,key_dim,val_dim,entity_count)!=(13,6,80,16,16,16):
            raise ValueError("source-authorized LATFUSE003 requires exact registered input/state shape")
        super().__init__(obs_dim,action_dim,hidden,key_dim,val_dim,entity_count)
        self.write_router=nn.Linear(hidden,HEADS)
        self.read_router=nn.Linear(hidden*2,HEADS)
        # Learnable log temperatures apply to competition between heads only.
        self.log_head_temperature=nn.Parameter(torch.zeros(HEADS))
        if sum(p.numel() for p in self.parameters())!=PARAMETERS:
            raise RuntimeError("100k actual head model parameter count changed")

    def step(self,state:FusionState,x,observed,elapsed,last_seen,source_slot):
        b=x.shape[0]
        if (state.fast.shape!=(b,80) or state.slow.shape!=(b,80) or
            state.associative.shape!=(b,16,16) or
            state.entities.shape!=(b,16,80) or state.active.shape!=(b,16) or
            observed.shape!=(b,13) or observed.dtype!=torch.bool or
            source_slot.shape!=(b,)):
            raise ValueError("source inputs or model recurrent state shape mismatch")
        if any(t.dtype!=torch.float32 for t in state[:4]) or state.active.dtype!=torch.bool:
            raise ValueError("LATFUSE003 recurrent state must be FP32 + bool")
        u=self.encode(x,observed,elapsed,last_seen)
        f=torch.tanh(self.temporal_a(u))*state.fast+torch.tanh(self.temporal_b(u))
        rate=self.slow_rates.to(dtype=u.dtype,device=u.device)[None,:]
        slow=rate*state.slow+(1-rate)*torch.tanh(self.slow_input(u))
        has_source=observed.any(-1)
        matrix,_,_=four_head_delta(
            state.associative,self.key_map(u),torch.tanh(self.val_map(u)),
            torch.sigmoid(self.retention_map(u)),torch.sigmoid(self.write_map(u)),
            self.write_router(u),has_source,self.log_head_temperature)
        slots,active=source_slot_write(
            state.entities,state.active,source_slot,torch.tanh(self.entity_map(u)),
            torch.sigmoid(self.entity_gate(u))*has_source[:,None].to(u.dtype))
        return FusionState(f,slow,matrix,slots,active)

    def forecast(self,state:FusionState,source_slot,candidates):
        b=state.fast.shape[0]
        if (candidates.ndim!=3 or candidates.shape[0]!=b or
            candidates.shape[2]!=6 or source_slot.shape!=(b,)):
            raise ValueError("conditional forecast/source shape mismatch")
        if bool(((source_slot<0)|(source_slot>=16)).any()):
            raise ValueError("unknown physical source/slot")
        if not bool(torch.isfinite(candidates).all()):
            raise ValueError("nonfinite future-known calendar")
        joint=torch.cat((state.fast,state.slow),-1)
        a=torch.tanh(self.action_map(candidates))
        belief_key=self.belief_query(joint)[:,None,:]
        n=candidates.shape[1]
        q=F.normalize((a+belief_key).reshape(b,n,HEADS,HEAD_KEY_DIM),dim=-1,eps=1e-8)
        # Each subspace read is self-contained; no within-history softmax.
        M=state.associative.reshape(b,HEAD_VALUE_DIM,HEADS,HEAD_KEY_DIM).permute(0,2,1,3)
        read=torch.einsum("bhvk,bnhk->bnhv",M,q)
        weights,_=_router(self.read_router(joint),self.log_head_temperature,top_k=HEADS)
        retrieval=(read*weights[:,None,:,None]).sum(dim=2)
        slot=source_slot.long().view(b,1,1).expand(-1,1,80)
        entity=state.entities.gather(1,slot).squeeze(1)
        entity=entity*state.active.gather(1,source_slot.long().view(b,1)).to(entity.dtype)
        feat=torch.cat((state.fast[:,None,:].expand(-1,n,-1),
                        state.slow[:,None,:].expand(-1,n,-1),
                        entity[:,None,:].expand(-1,n,-1),
                        retrieval,q.reshape(b,n,16)),dim=-1)
        mean,scale_logits=self.forecast_head(feat).chunk(2,-1)
        return mean,F.softplus(scale_logits)+1e-4

    def source_key_separation_loss(self,encodings,values,observed):
        """Real observation-grounded auxiliary regularization, training only.

        Pairs compared only within SAME physical source history when >=3
        co-observed coordinates differ by RMS >=1 in FIT-normalized source.
        Penalizes near-identical keys *only in the same selected write head*.
        Cannot make >4 independent vectors mutually orthogonal in R^4;
        cannot prove physical events are distinct.
        """
        if (encodings.ndim!=3 or encodings.shape[-1]!=80 or
            values.shape!=observed.shape or values.shape!=(*encodings.shape[:2],13)
            or observed.dtype!=torch.bool):
            raise ValueError("source-aligned observed encoder trace required")
        b,t,_=encodings.shape
        if t<2:
            return encodings.sum()*0
        key=F.normalize(self.key_map(encodings).reshape(b,t,4,4),dim=-1,eps=1e-8)
        logits=self.write_router(encodings)
        scores=logits/torch.exp(self.log_head_temperature.clamp(-.69314718,.69314718))
        chosen=torch.zeros_like(scores,dtype=torch.bool).scatter_(
            -1,torch.topk(scores,k=2,dim=-1).indices,True)
        joint=observed[:,:,None,:]&observed[:,None,:,:]
        overlap=joint.sum(-1)
        diff=(torch.where(joint,(values[:,:,None,:]-values[:,None,:,:]).square(),0)
              .sum(-1)/overlap.clamp_min(1)).sqrt()
        pair_ok=(overlap>=3)&(diff>=1)
        upper=torch.triu(torch.ones(t,t,dtype=torch.bool,device=values.device),diagonal=1)
        eligible=pair_ok[:,:,:,None] & (chosen[:,:,None,:]&chosen[:,None,:,:])
        eligible=eligible & upper[None,:,:,None]
        cos=torch.einsum("bthd,bshd->btsh",key,key).abs()
        penalty=(F.relu(cos-.85)**2)*eligible.to(cos.dtype)
        return penalty.sum()/eligible.sum().clamp_min(1)

    def head_utilization_loss(self,encodings,observed):
        """Mild selected-soft-routing balance; not arbitrary strict equal quotas."""
        if encodings.ndim!=3 or observed.shape!=(*encodings.shape[:2],13):
            raise ValueError("source-aligned routing/mask required")
        p=F.softmax(self.write_router(encodings)/
                    torch.exp(self.log_head_temperature.clamp(-.69314718,.69314718)),dim=-1)
        real=observed.any(-1).to(p.dtype)
        mean=(p*real[...,None]).sum((0,1))/real.sum().clamp_min(1)
        return ((mean-.25)**2).mean()


def exact_state_contract(model):
    state=model.initial(1)
    counted=sum(x.numel() for x in state[:4])
    flags=state.active.numel()
    actual=sum(p.numel() for p in model.parameters() if p.requires_grad)
    deployed=sum(p.numel() for p in model.parameters())
    if (counted,flags,counted*4+flags,actual,deployed)!=(1696,16,6800,99747,99747):
        raise RuntimeError("LATFUSE003 exact model/state budget violated")
    if any(x.dtype!=torch.float32 for x in state[:4]) or state.active.dtype!=torch.bool:
        raise RuntimeError("state dtype changed")
    return dict(fp32_values=counted,occupancy_flags=flags,nominal_bytes=counted*4+flags,
                trainable_parameters=actual,deployed_parameters=deployed,
                head_count=4,head_value_dim=16,head_key_dim=4,
                matrix_cells=4*16*4,top_k_writes=2,
                historical_recurrent_state_unchanged=True)
