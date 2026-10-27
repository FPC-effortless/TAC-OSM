"""LATFUSE004 factorial 2x2: head width versus soft routing independently.

No new recurrent state: H80 fast/slow, [16,16] associative, [16,80]
source slots, [16] occupancy, 1,696 FP32 floats+16 flags=6,800 bytes.
Arms (heads,route) = (4,hard),(4,soft),(2,hard),(2,soft).
2 heads are [16 values x8 keys] each; 4 heads [16 values x4 keys].
Every arm uses the same input-conditioned encoders/forecaster and real
source-only key/route auxiliary objectives. No KV cache or causal do().

Hard4 exactly uses LATFUSE003 step and forecast for an authentic reference.
Per-head geometry and routing losses below are common across all four arms.
Soft routing is differentiable through writes to ALL heads; no stochastic
sampling or uncontrolled training-time head changes. Learnable temperature
is bounded; fixed schedule would require its own new experiment.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from tac_osm.latent_fusion_001 import FusionState,source_slot_write
from tac_osm.latent_fusion_003 import FusedLatentFourHead

ARMS={ "4h-hard":(4,False),"4h-soft":(4,True),
       "2h-hard":(2,False),"2h-soft":(2,True)}
COUNTS={4:99747,2:99261}


def head_weights(logits,log_temperatures,*,head_count,soft):
    if logits.ndim!=2 or logits.shape[-1]!=head_count or log_temperatures.shape!=(head_count,):
        raise ValueError("head-routable logits shape mismatch")
    if not bool(torch.isfinite(logits).all() and torch.isfinite(log_temperatures).all()):
        raise ValueError("nonfinite head router features")
    tau=torch.exp(log_temperatures.clamp(-.69314718,.69314718))
    scores=logits/tau[None,:]
    if soft:
        return scores.softmax(-1),torch.ones_like(scores,dtype=torch.bool)
    k=head_count//2
    chosen=torch.zeros_like(scores,dtype=torch.bool).scatter_(
        -1,torch.topk(scores,k=k,dim=-1).indices,True)
    return scores.masked_fill(~chosen,torch.finfo(scores.dtype).min).softmax(-1),chosen


def partitioned_update(matrix,key,value,retention,write,logits,observed,temperatures,
                       *,head_count,soft):
    """Head-isolated FP32 gated delta. Equal matrix-cell count for all arms.

    Scalar retention for HARD unselected heads=1 and for observed selected
    heads=retention. SOFT uses retention on all observed heads; all get
    source-differentiable writes with beta*softmax router probabilities.
    This intentionally changes head admission dynamics, not matrix capacity.
    """
    b=matrix.shape[0]
    d=16//head_count
    if (matrix.shape!=(b,16,16) or key.shape!=(b,16) or
       value.shape!=(b,16) or retention.shape!=(b,1) or
       write.shape!=(b,1) or logits.shape!=(b,head_count) or
       observed.shape!=(b,) or observed.dtype!=torch.bool):
        raise ValueError("factorial source/update shapes invalid")
    if any(t.dtype!=torch.float32 for t in (matrix,key,value,retention,write,logits,temperatures)):
        raise ValueError("LATFUSE004 requires FP32 update and learned parameters")
    if not all(bool(torch.isfinite(t).all()) for t in (matrix,key,value,retention,write)):
        raise ValueError("nonfinite source memory values")
    if bool(((retention<0)|(retention>1)|(write<0)|(write>1)).any()):
        raise ValueError("invalid bounded gates")
    weights,selection=head_weights(logits,temperatures,head_count=head_count,soft=soft)
    eligible=selection & observed[:,None]
    old=matrix.reshape(b,16,head_count,d).permute(0,2,1,3)
    k=F.normalize(key.reshape(b,head_count,d),dim=-1,eps=1e-8)
    keep=torch.where(eligible,retention.expand(-1,head_count),torch.ones_like(weights))
    beta=write*weights*eligible.to(weights.dtype)
    retained=keep[:,:,None,None]*old
    correction=value[:,None,:,None]-(retained@k.unsqueeze(-1))
    new=retained+beta[:,:,None,None]*correction*k[:,:,None,:]
    result=new.permute(0,2,1,3).contiguous().reshape(b,16,16)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("nonfinite modeled matrix update")
    return result,weights,eligible


class FactorialFusion(FusedLatentFourHead):
    def __init__(self,arm:str):
        if arm not in ARMS:raise ValueError("unknown preregistered factorial arm")
        heads,soft=ARMS[arm]
        super().__init__()
        self.arm=arm
        self.head_count=heads
        self.soft_routing=soft
        self.key_width=16//heads
        if heads==2:
            self.write_router=nn.Linear(80,2)
            self.read_router=nn.Linear(160,2)
            self.log_head_temperature=nn.Parameter(torch.zeros(2))
        self._verify_contract()

    def _verify_contract(self):
        state=self.initial(1)
        count=sum(x.numel() for x in state[:4])
        learned=sum(p.numel() for p in self.parameters() if p.requires_grad)
        deployed=sum(p.numel() for p in self.parameters())
        if ((count,state.active.numel(),count*4+state.active.numel(),
             learned,deployed)!=(1696,16,6800,COUNTS[self.head_count],COUNTS[self.head_count])):
            raise RuntimeError("model size/learned budget/recurrent capacity mismatch")
        if any(x.dtype!=torch.float32 for x in state[:4]) or state.active.dtype!=torch.bool:
            raise RuntimeError("incorrect recurrent dtype")

    def step(self,state:FusionState,x,observed,elapsed,last_seen,source_slot):
        if self.arm=="4h-hard":
            return super().step(state,x,observed,elapsed,last_seen,source_slot)
        b=x.shape[0]
        if (state.fast.shape!=(b,80) or state.slow.shape!=(b,80) or
            state.associative.shape!=(b,16,16) or
            state.entities.shape!=(b,16,80) or state.active.shape!=(b,16) or
            observed.shape!=(b,13) or observed.dtype!=torch.bool or source_slot.shape!=(b,)):
            raise ValueError("recurrent/source input shape mismatch")
        if any(t.dtype!=torch.float32 for t in state[:4]) or state.active.dtype!=torch.bool:
            raise ValueError("recurrent values must be FP32")
        u=self.encode(x,observed,elapsed,last_seen)
        fast=torch.tanh(self.temporal_a(u))*state.fast+torch.tanh(self.temporal_b(u))
        rate=self.slow_rates.to(dtype=u.dtype,device=u.device)[None,:]
        slow=rate*state.slow+(1-rate)*torch.tanh(self.slow_input(u))
        source=observed.any(-1)
        matrix,_,_=partitioned_update(
            state.associative,self.key_map(u),torch.tanh(self.val_map(u)),
            torch.sigmoid(self.retention_map(u)),torch.sigmoid(self.write_map(u)),
            self.write_router(u),source,self.log_head_temperature,
            head_count=self.head_count,soft=self.soft_routing)
        entity,active=source_slot_write(
            state.entities,state.active,source_slot,torch.tanh(self.entity_map(u)),
            torch.sigmoid(self.entity_gate(u))*source[:,None].to(u.dtype))
        return FusionState(fast,slow,matrix,entity,active)

    def forecast(self,state:FusionState,source_slot,candidates):
        if self.arm=="4h-hard":
            return super().forecast(state,source_slot,candidates)
        b=state.fast.shape[0]
        if (candidates.ndim!=3 or candidates.shape[0]!=b or
            candidates.shape[-1]!=6 or source_slot.shape!=(b,)):
            raise ValueError("source forecast query shape mismatch")
        if bool(((source_slot<0)|(source_slot>=16)).any()) or not bool(torch.isfinite(candidates).all()):
            raise ValueError("source ID or future-known calendar invalid")
        n=candidates.shape[1];h=self.head_count;d=self.key_width
        joint=torch.cat((state.fast,state.slow),-1)
        k=torch.tanh(self.action_map(candidates))+self.belief_query(joint)[:,None]
        q=F.normalize(k.reshape(b,n,h,d),dim=-1,eps=1e-8)
        mat=state.associative.reshape(b,16,h,d).permute(0,2,1,3)
        r=torch.einsum("bhvk,bnhk->bnhv",mat,q)
        weights,_=head_weights(self.read_router(joint),self.log_head_temperature,
                               head_count=h,soft=True)
        read=(r*weights[:,None,:,None]).sum(2)
        idx=source_slot.long().view(b,1,1).expand(-1,1,80)
        entity=state.entities.gather(1,idx).squeeze(1)
        entity=entity*state.active.gather(1,source_slot.long().view(b,1)).to(entity.dtype)
        features=torch.cat((
            state.fast[:,None].expand(-1,n,-1),state.slow[:,None].expand(-1,n,-1),
            entity[:,None].expand(-1,n,-1),read,q.reshape(b,n,16)),dim=-1)
        mean,logsd=self.forecast_head(features).chunk(2,-1)
        return mean,F.softplus(logsd)+1e-4

    def common_key_loss(self,embeddings,values,observed):
        """Identical source-defined training objective for all 4 arms.

        Weighted same-head pair cosine hinge on physical sensor-distinct
        co-observed past values; no future target used in loss.
        Routing probability-based weighting is differentiable even in the
        hard writer arm. This measures a proxy, not verified factual recall.
        """
        if (embeddings.ndim!=3 or embeddings.shape[-1]!=80 or
            values.shape!=(*embeddings.shape[:2],13) or
            observed.shape!=values.shape or observed.dtype!=torch.bool):
            raise ValueError("nonmatching source-visible encoder and masks")
        b,t,_=embeddings.shape;h=self.head_count;d=self.key_width
        if t<2:return embeddings.sum()*0
        key=F.normalize(self.key_map(embeddings).reshape(b,t,h,d),dim=-1,eps=1e-8)
        logits=self.write_router(embeddings)
        p=F.softmax(logits/torch.exp(self.log_head_temperature.clamp(-.69314718,.69314718)),
                    dim=-1)
        joint=observed[:,:,None,:]&observed[:,None,:,:]
        n=joint.sum(-1)
        diff=(torch.where(joint,(values[:,:,None,:]-values[:,None,:,:]).square(),0)
              .sum(-1)/n.clamp_min(1)).sqrt()
        pair=((n>=3)&(diff>=1))
        upper=torch.triu(torch.ones(t,t,dtype=torch.bool,device=values.device),diagonal=1)
        pair=pair & upper[None]
        cos=torch.einsum("bthd,bshd->btsh",key,key).abs()
        weighted=p[:,:,None,:]*p[:,None,:,:]*pair[:,:,:,None].to(p.dtype)
        hinge=F.relu(cos-.85).square()
        return (weighted*hinge).sum()/weighted.sum().clamp_min(1)

    def common_head_balance(self,encodings,observed):
        if encodings.ndim!=3 or observed.shape!=(*encodings.shape[:2],13):
            raise ValueError("source head balance shapes invalid")
        probs=F.softmax(
           self.write_router(encodings)/
           torch.exp(self.log_head_temperature.clamp(-.69314718,.69314718)),dim=-1)
        eligibility=observed.any(-1).to(probs.dtype)
        usage=(probs*eligibility[:,:,None]).sum((0,1))/eligibility.sum().clamp_min(1)
        return ((usage-(1/self.head_count))**2).mean()


def contract(arm):
    m=FactorialFusion(arm)
    return dict(arm=arm,heads=m.head_count,key_dim=m.key_width,
       head_value_dim=16,per_stream_fp32_scalars=1696,per_stream_occupancy_flags=16,
       per_stream_nominal_bytes=6800,
       matrix_cells=m.head_count*16*m.key_width,
       active_parameters=sum(p.numel() for p in m.parameters()),
       source_future_is_known_calendar_not_action=True)
