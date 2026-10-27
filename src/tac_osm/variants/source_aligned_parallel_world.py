"""PAR-MEM-010 opt-in source-aligned two-head memory on parallel TAC.

This is a NEW UNTRAINED candidate, not a replacement of any source-trained
checkpoint and not a claim of historic content recovery.

Reuse from original code:
* DIAGKEY004: exactly SAME source-known key at write and historical query;
  read-only queries and separate corrective-content/retention/write gates.
* ME100: state is bound to a genuine caller-supplied source/entity owner,
  chronological event ordinals, and explicit observed-only admission.
* LATFUSE004: soft routing across two independent 8D-key/16D-value heads.
* LATFUSE018: a real physical historical-content head, whose TARGET must
  be provided separately only in a training loss AFTER causal retrieval.

The original PAR-TIME time-aware backbone is preserved unmodified and
continues to use its own separate 8x8 associative state. This option
ADDS 2x16x8=256 FP32 state scalars. Total is 154+256=410 FP32 scalars
plus a boolean and count and 2 int64 per stream, not 154/152 and not the
full LATFUSE 1696-value entity-slot architecture. No event-KV sidecar is
available during deployment; the indexed historical *key* is recomputed
from public source-known event ordinal without consulting historic content.

Per-write time-parallel kernels are truly input-only and can scan via the
existing original affine (8x8) monoid. All four 8x8 value blocks use one
shared update schedule. Causal retrieval reads the PREVIOUS prefix; no
current event can be returned to its own causal prediction. No claim of
global neural latency speedup or full-network Lean refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .parallel_affine_memory import (
    associative_affine_schedule, work_efficient_affine_prefix,
    preserve_nonwrite_prefix,
)
from .parallel_time_age_world import TimeAwareParallelResidualWorld, TimeAwareStreamState

PERIODS = (7, 13, 29, 59)
HEADS = 2
VALUE_WIDTH = 16
KEY_WIDTH = 8
EXTRA_MEMORY_FLOATS = HEADS * VALUE_WIDTH * KEY_WIDTH


@dataclass(frozen=True)
class AlignedSourceStreamState:
    native: TimeAwareStreamState
    aligned_memory: torch.Tensor  # [B,2,16,8], independent of backbone 8x8
    source_owner: torch.Tensor  # [B] int64 exact externally supplied entity ID
    last_event_ordinal: torch.Tensor  # [B] int64; -1 before initial observation


class SourceAlignedParallelWorld(nn.Module):
    """An opt-in, six-family-compatible research core, not a seventh family.

    The historical query's requested ordinal MUST refer to the same verified
    source stream and past event. The module can check monotonicity and owner
    identity; it CANNOT attest that an arbitrary old ordinal actually carried
    a sensor measurement. That requires separate source provenance.

    The ordinal Fourier function is input-only and shared exactly at write
    and read; CONTENT is separately learned from observed current events.
    """

    def __init__(self, state_dim=14, action_dim=14, width=88,
                 decoder=132, ffn_hidden=12):
        super().__init__()
        self.backbone = TimeAwareParallelResidualWorld(
            state_dim=state_dim, action_dim=action_dim, width=width,
            decoder=decoder, ffn_hidden=ffn_hidden,
            preserve_masked_prefix_bytes=True,
        )
        self.source_key_rotations = nn.Parameter(torch.stack(
            (torch.eye(KEY_WIDTH),torch.roll(torch.eye(KEY_WIDTH),shifts=1,dims=1)),0))
        self.source_content = nn.Linear(48, VALUE_WIDTH)
        self.write_router = nn.Linear(48, HEADS)
        self.read_router = nn.Linear(KEY_WIDTH, HEADS)
        self.retention = nn.Linear(48, HEADS)
        self.write_strength = nn.Linear(48, HEADS)
        self.historical_content = nn.Linear(VALUE_WIDTH, state_dim)
        self.predictive_memory_delta = nn.Linear(VALUE_WIDTH, state_dim, bias=False)
        self.predictive_memory_logit = nn.Parameter(torch.tensor(-4.0))
        # Exact original backbone prediction at initialization, with trainable
        # predictive memory connection. Historical-content head stays active.
        nn.init.zeros_(self.predictive_memory_delta.weight)
        with torch.no_grad():
            nn.init.zeros_(self.retention.weight)
            self.retention.bias.fill_(5.3)
            nn.init.zeros_(self.write_strength.weight)
            self.write_strength.bias.fill_(-2.0)

    @property
    def state_dim(self):
        return self.backbone.state_dim

    @property
    def action_dim(self):
        return self.backbone.action_dim

    @staticmethod
    def _integer_source(x, shape, device, name):
        if (not torch.is_tensor(x) or x.shape != shape or x.device != device
            or x.dtype != torch.int64):
            raise ValueError(name+" must be source-provided int64 with correct shape/device")
        if (x < 0).any().item():
            raise ValueError(name+" must not contain negative indices or owners")

    @staticmethod
    def _event_encoding(ordinal, dtype):
        # int64 modulo preserves exact source ordinal residue at high indexes
        # where directly converting the full index to FP32 loses low bits.
        phases = []
        for period in PERIODS:
            angles=(ordinal.remainder(period)).to(dtype)* (2.0*math.pi/period)
            phases.extend((angles.cos(),angles.sin()))
        return torch.stack(phases,dim=-1)

    def _keys(self, ordinal, dtype):
        phi=self._event_encoding(ordinal,dtype)
        transformed=torch.einsum("...i,hij->...hj",phi,
                                 self.source_key_rotations.to(dtype=dtype))
        return F.normalize(transformed,dim=-1,eps=1e-6), phi

    def _writer(self, embedding):
        value=self.source_content(embedding).tanh()
        route=self.write_router(embedding).softmax(-1)
        alpha=self.retention(embedding).sigmoid()
        beta=self.write_strength(embedding).sigmoid()*route
        return value,route,alpha,beta

    def _read(self, memory, event_ordinal):
        # memory [B,...,2,16,8] and event_index [B,...]; source-key and
        # router depend ONLY on the requested old source-known ordinal.
        key,phi=self._keys(event_ordinal,memory.dtype)
        heads=torch.einsum("...hvk,...hk->...hv",memory,key)
        router=self.read_router(phi).softmax(-1)
        read=(heads*router.unsqueeze(-1)).sum(dim=-2)
        return read,router

    @staticmethod
    def _source_batch_contract(obs, source_ids, event_index, query_index):
        if obs.ndim!=3 or obs.shape[1]<1:
            raise ValueError("source batch must be [B,T,D] with nonempty T")
        B,T=obs.shape[:2]
        SourceAlignedParallelWorld._integer_source(
            source_ids,(B,),obs.device,"source owner")
        SourceAlignedParallelWorld._integer_source(
            event_index,(B,T),obs.device,"event ordinal")
        if (event_index[:,1:]<=event_index[:,:-1]).any().item():
            raise ValueError("source event ordinal must strictly increase within each owner")
        if query_index is not None:
            SourceAlignedParallelWorld._integer_source(
                query_index,(B,T),obs.device,"requested historic event")
            if (query_index>=event_index).any().item():
                raise ValueError("historical read must precede current event")
        return B,T

    @staticmethod
    def _query_ordinal(events, query):
        if query is not None:
            return query,torch.ones_like(events,dtype=torch.bool)
        # Default read is last chronological ordinal. If missing/no-write,
        # it is a source-known address probe, NOT evidence of an old value.
        has_prior=events>0
        return (events-1).clamp_min(0),has_prior

    def init_stream(self, source_ids):
        if not torch.is_tensor(source_ids) or source_ids.ndim!=1:
            raise ValueError("initial stream requires source owner int64 [B]")
        self._integer_source(source_ids,source_ids.shape,source_ids.device,"source owner")
        p=next(self.parameters())
        if p.device!=source_ids.device:
            raise ValueError("source owner device differs from model")
        B=source_ids.shape[0]
        return AlignedSourceStreamState(
            self.backbone.init_stream(B),
            p.new_zeros(B,HEADS,VALUE_WIDTH,KEY_WIDTH),
            source_ids.clone(),
            torch.full_like(source_ids,-1),
        )

    def forward(self, observation, observed_action, *, elapsed_dt, write_mask,
                source_ids, event_index, query_event_index=None,
                return_trace=False):
        B,T=self._source_batch_contract(
            observation,source_ids,event_index,query_event_index)
        if (write_mask.shape!=(B,T) or write_mask.dtype!=torch.bool
            or write_mask.device!=observation.device):
            raise ValueError("source observed admission is required boolean [B,T]")
        base,native=self.backbone(
            observation,observed_action,elapsed_dt=elapsed_dt,
            write_mask=write_mask,return_trace=True)
        embedding=native["embedding"]     # actual input-only source encoder
        value,route,alpha,beta=self._writer(embedding)
        key,_=self._keys(event_index,embedding.dtype)
        # Flatten 2 heads x 2 value halves into 4 independent original
        # 8x8 chronological affine monoid scans (B*4 original matrices).
        shapes=(B,T,HEADS,2)
        keys=key.unsqueeze(-2).expand(*shapes,KEY_WIDTH)
        contents=value.reshape(B,T,1,2,KEY_WIDTH).expand(*shapes,KEY_WIDTH)
        a=alpha.unsqueeze(-1).unsqueeze(-1).expand(*shapes,1)
        b=beta.unsqueeze(-1).unsqueeze(-1).expand(*shapes,1)
        admitted=write_mask[:,:,None,None,None].expand(*shapes,1)
        def flatten(x):
            return x.permute(0,2,3,1,4).reshape(B*HEADS*2,T,x.shape[-1])
        P,Q=associative_affine_schedule(
            flatten(keys),flatten(contents),flatten(a),flatten(b),
            flatten(admitted))
        scanned=work_efficient_affine_prefix(P,Q).reshape(
            B,HEADS,2,T,KEY_WIDTH,KEY_WIDTH)
        matrices=scanned.permute(0,3,1,2,4,5).reshape(
            B,T,HEADS,VALUE_WIDTH,KEY_WIDTH)
        matrices=preserve_nonwrite_prefix(matrices,write_mask)
        before=torch.cat((matrices.new_zeros(B,1,HEADS,VALUE_WIDTH,KEY_WIDTH),
                          matrices[:,:-1]),dim=1)
        old_index,has_prior=self._query_ordinal(event_index,query_event_index)
        read,read_routes=self._read(before,old_index)
        read=torch.where(has_prior.unsqueeze(-1),read,torch.zeros_like(read))
        corrected=self._predictive_correction(base,read)
        if return_trace:
            return corrected,{**native,"aligned_memory":matrices,
                "aligned_before_write":before,"aligned_old_read":read,
                "aligned_write_key":key,"aligned_write_route":route,
                "aligned_read_route":read_routes,"aligned_retention":alpha,
                "aligned_strength":beta,"source_event_ordinal":event_index}
        return corrected

    def _predictive_correction(self, prediction, read):
        scale=self.backbone.core.delta_scale
        return (prediction+scale *
                self.predictive_memory_logit.sigmoid() *
                self.predictive_memory_delta(read).tanh())

    def stream_step(self, state, observation, observed_action, *,
                    elapsed_dt, write_mask, source_ids, event_index,
                    query_event_index=None, return_trace=False):
        if not isinstance(state,AlignedSourceStreamState) or observation.ndim!=2:
            raise ValueError("expected initialized source-owned streaming state and [B,D]")
        B=observation.shape[0]
        self._integer_source(source_ids,(B,),observation.device,"source owner")
        self._integer_source(event_index,(B,),observation.device,"event ordinal")
        if (state.source_owner.shape!=(B,) or
            not torch.equal(state.source_owner,source_ids) or
            state.last_event_ordinal.shape!=(B,) or
            (event_index<=state.last_event_ordinal).any().item()):
            raise ValueError("cross-owner, duplicate or nonchronological stream event")
        if (state.aligned_memory.shape!=(B,HEADS,VALUE_WIDTH,KEY_WIDTH)
            or state.aligned_memory.dtype!=observation.dtype
            or state.aligned_memory.device!=observation.device):
            raise ValueError("source aligned memory state wrong shape/dtype/device")
        if query_event_index is not None:
            self._integer_source(query_event_index,(B,),observation.device,
                                 "requested historic event")
            if (query_event_index>=event_index).any().item():
                raise ValueError("historical query must precede current source event")
        base,updated_native,native=self.backbone.stream_step(
            state.native,observation,observed_action,elapsed_dt=elapsed_dt,
            write_mask=write_mask,return_trace=True)
        e=native["embedding"]
        value,route,alpha,beta=self._writer(e)
        keys,_=self._keys(event_index,e.dtype)
        old_index,has_prior=self._query_ordinal(event_index,query_event_index)
        read,read_route=self._read(state.aligned_memory,old_index)
        read=torch.where(has_prior.unsqueeze(-1),read,torch.zeros_like(read))
        corrected=self._predictive_correction(base,read)
        # Corrective gated-delta WRITE: M_next = alpha*M + beta*(v-alpha*M*k)*kT
        old=state.aligned_memory
        old_projection=torch.einsum("bhvk,bhk->bhv",old,keys)
        innovation=value[:,None,:]-alpha[:,:,None]*old_projection
        candidate=alpha[:,:,None,None]*old+(
            beta[:,:,None,None]*innovation.unsqueeze(-1)*keys.unsqueeze(-2))
        if (write_mask.shape!=(B,) or write_mask.dtype!=torch.bool
            or write_mask.device!=observation.device):
            raise ValueError("source write mask must be boolean [B]")
        memory=torch.where(write_mask[:,None,None,None],candidate,old)
        next_state=AlignedSourceStreamState(
            updated_native,memory,state.source_owner,event_index.clone())
        if return_trace:
            return corrected,next_state,{
                **native,"aligned_memory":memory,
                "aligned_before_write":old,"aligned_old_read":read,
                "aligned_write_key":keys,"aligned_write_route":route,
                "aligned_read_route":read_route,"aligned_retention":alpha,
                "aligned_strength":beta}
        return corrected,next_state

    def historic_query(self, state, *, source_ids, requested_event_index):
        """Read ONLY, returning normalized-content estimates for old events.

        The old target measurement and the original content-conditioned
        write key are NOT passed to this method. Multiple independent
        source-authorized queries reuse the exact same unchanged state.
        """
        if not isinstance(state,AlignedSourceStreamState):
            raise ValueError("query needs source-owned persistent stream")
        B=state.aligned_memory.shape[0]
        self._integer_source(source_ids,(B,),state.aligned_memory.device,"source owner")
        self._integer_source(requested_event_index,(B,),state.aligned_memory.device,
                             "historical requested ordinal")
        if (not torch.equal(source_ids,state.source_owner)
            or (requested_event_index>state.last_event_ordinal).any().item()):
            raise ValueError("query must target the same owner and a known past ordinal")
        read,_=self._read(state.aligned_memory,requested_event_index)
        return self.historical_content(read)

    def contract(self):
        params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        deployed=sum(p.numel() for p in self.parameters())
        B=1
        st=self.init_stream(torch.zeros(B,dtype=torch.int64,device=next(self.parameters()).device))
        original=st.native.fast_sum.numel()+st.native.memory.numel()+(
            st.native.elapsed_clock.numel()+st.native.last_write_clock.numel())
        extra=st.aligned_memory.numel()
        if not (95000<=params<=105000 and params==deployed
                and original==self.backbone.core.width+66 and extra==256):
            raise RuntimeError("TAC 100k original+new source memory contract failed")
        return dict(original_time_aware_fp32=original,added_source_aligned_fp32=extra,
                    total_fp32_per_stream=original+extra,
                    owner_and_last_event_int64_per_stream=2,
                    native_write_seen_boolean_per_stream=1,
                    native_event_count_integer_per_stream=1,
                    learned_trainable_parameters=params,
                    deployed_parameters=deployed,
                    two_soft_heads=True,
                    read_write_same_source_known_address=True,
                    original_backbone_architecture_preserved=True,
                    trained_source_model=False,
                    demonstrated_source_historical_skill=False)
