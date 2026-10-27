"""LATFUSE007 source-timed historic-event query candidate — UNTRAINED.

Built on exact LATFUSE004 2h-soft source encoder, delta writer and original
future forecasting graph. No writer, retention or source normalization changes.
Adds a HISTORICAL TIME-CONDITIONED query to the existing 256-cell [16x16]
associative matrix, trained only with authorized original-source EVENT target
after the forward pass. Binary high-PM2.5 label derived from source-recorded
PM2.5 at historic timestamp, NEVER passed as query. History span changes
training compute, so equal-FLOP comparisons are NOT claimed.

All original persistent stream values remain fast80+slow80+associative256+
16*80 owner source slots =1696 FP32 values +16 bool flags =6800 nominal bytes.
Historical event head adds 129 trained/deployed parameters over base2H99261,
for EXACT 99390. Parameters alone do not prove matched training FLOPs.
No new latent history cache, Transformer attention, or claimed world-control.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from tac_osm.latent_fusion_001 import FusionState
from tac_osm.latent_fusion_004_factorial import FactorialFusion,head_weights

EXPECTED_TRAINABLE=99390

class SourceTimedEventQuery(FactorialFusion):
    """Source-recorded historic event query, with NO source label in input."""

    def __init__(self):
        super().__init__("2h-soft")
        self.historic_clock_query=nn.Linear(6,16)
        self.historic_event_head=nn.Linear(16,1)
        actual=sum(p.numel() for p in self.parameters())
        if actual!=EXPECTED_TRAINABLE:
            raise RuntimeError("LATFUSE007 original full learned weight count mismatched")
        st=self.initial(1)
        size=sum(z.numel() for z in st[:4])*4+st.active.numel()
        if size!=6800:raise RuntimeError("unregistered FP32 live source state changes")

    def historic_event_logits(self,state:FusionState,source_slot:torch.Tensor,
                              historic_known_clock:torch.Tensor):
        """[B,N,6] known calendar OF REQUESTED PAST t-L, no original value.

        The query can learn CONTENT address by conditioning on already
        authorized source-trained current fast/slow belief plus past clock.
        Identical two-head partition, no per-token KV cache. The original
        forecast query remains unchanged.
        """
        b=state.fast.shape[0]
        if (state.fast.shape!=(b,80) or state.slow.shape!=(b,80) or
            state.associative.shape!=(b,16,16) or
            state.entities.shape!=(b,16,80) or state.active.shape!=(b,16) or
            source_slot.shape!=(b,) or
            historic_known_clock.ndim!=3 or
            historic_known_clock.shape[0]!=b or
            historic_known_clock.shape[-1]!=6):
            raise ValueError("source-registered event query/state shape mismatch")
        if (historic_known_clock.dtype!=torch.float32 or
            any(z.dtype!=torch.float32 for z in state[:4]) or
            state.active.dtype!=torch.bool):
            raise ValueError("event query must use original FP32 tensor state")
        if (not bool(torch.isfinite(historic_known_clock).all()) or
            bool(((source_slot<0)|(source_slot>=16)).any())):
            raise ValueError("nonfinite or unregistered source/clock query")
        n=historic_known_clock.shape[1]
        joined=torch.cat((state.fast,state.slow),dim=-1)
        # Query uses historical source known TIME, not measured PM2.5 value.
        q=(torch.tanh(self.historic_clock_query(historic_known_clock))+
           self.belief_query(joined)[:,None,:])
        key=F.normalize(q.reshape(b,n,2,8),dim=-1,eps=1e-8)
        memory=state.associative.reshape(b,16,2,8).permute(0,2,1,3)
        lookup=torch.einsum("bhvk,bnhk->bnhv",memory,key)
        weights,_=head_weights(self.read_router(joined),self.log_head_temperature,
                     head_count=2,soft=True)
        response=(lookup*weights[:,None,:,None]).sum(2)
        event_logits=self.historic_event_head(response).squeeze(-1)
        if event_logits.shape!=(b,n) or not bool(torch.isfinite(event_logits).all()):
            raise RuntimeError("source event prediction numeric invalid")
        return event_logits

    def original_state_and_weight_contract(self):
        st=self.initial(1)
        return dict(original_recurrent_fp32_values=sum(v.numel() for v in st[:4]),
           original_bool_source_flags=st.active.numel(),
           original_nominal_memory_bytes=4*sum(v.numel() for v in st[:4])+st.active.numel(),
           trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
           deployed_parameters=sum(p.numel() for p in self.parameters()),
           historic_query_only_no_new_state=True,
           original_beta_and_alpha_writer_unchanged=True,
           source_event_labels_not_input=True)
