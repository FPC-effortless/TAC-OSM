"""LATFUSE008 matched native historical-query control without associative retrieval.

Use exactly the same 99,390 live/deployed learned parameter tensors and
identical 6,800 byte FP32 recurrent state as LATFUSE007. The unchanged
2h-soft original writer and six-hour forecast still use the original matrix.
Only the historical-event decoder's information pathway changes:
LATFUSE007 historical clock -> stored associative M; here historical clock
-> current fast/slow multiplicative feature matrix, NOT persistent M.

A low-rank cross-feature matrix is an actual nonassociative interaction, not
no-op FLOP padding. This comparison isolates whether the stored associative
matrix is needed for factual historical queries, conditional on fast/slow.
It does NOT isolate all backbone training interactions, since the common
future forecasting objective still reads the same associative memory.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

from tac_osm.latent_fusion_004_factorial import head_weights
from tac_osm.latent_fusion_007_historic_query import SourceTimedEventQuery

class SourceTimedNonassociativeQuery(SourceTimedEventQuery):
    """History query uses only current fast/slow and requested historical clock."""

    def __init__(self):
        super().__init__()
        original=self.original_state_and_weight_contract()
        if (original["trainable_parameters"]!=99390 or
            original["original_nominal_memory_bytes"]!=6800):
            raise RuntimeError("LATFUSE008 live FP32 source budget mismatch")

    def historic_event_logits(self,state,source_slot,historic_known_clock):
        b=state.fast.shape[0]
        if (state.fast.shape!=(b,80) or state.slow.shape!=(b,80) or
            state.associative.shape!=(b,16,16) or
            state.entities.shape!=(b,16,80) or state.active.shape!=(b,16) or
            historic_known_clock.ndim!=3 or
            historic_known_clock.shape[0]!=b or
            historic_known_clock.shape[-1]!=6 or
            source_slot.shape!=(b,)):
            raise ValueError("matched memoryless historical event query shapes")
        if (historic_known_clock.dtype!=torch.float32 or
            any(z.dtype!=torch.float32 for z in state[:4]) or
            state.active.dtype!=torch.bool):
            raise ValueError("FP32 carried state required")
        if (not bool(torch.isfinite(historic_known_clock).all()) or
            bool(((source_slot<0)|(source_slot>=16)).any())):
            raise ValueError("invalid known source/timestamp")
        n=historic_known_clock.shape[1]
        joint=torch.cat((state.fast,state.slow),dim=-1)
        clock_query=(torch.tanh(self.historic_clock_query(historic_known_clock))+
                    self.belief_query(joint)[:,None,:])
        key=F.normalize(clock_query.reshape(b,n,2,8),dim=-1,eps=1e-8)
        # Meaningful interaction of current non-associative fast and slow
        # state; no read of state.associative's *values*. The shape matches
        # the original [value16, head2, key8] contracted read kernel.
        features=state.fast[:,:16].unsqueeze(-1)*state.slow[:,:16].unsqueeze(1)
        pseudo=features.reshape(b,16,2,8).permute(0,2,1,3)
        looked_up=torch.einsum("bhvk,bnhk->bnhv",pseudo,key)
        weights,_=head_weights(self.read_router(joint),self.log_head_temperature,
                               head_count=2,soft=True)
        read=(looked_up*weights[:,None,:,None]).sum(2)
        out=self.historic_event_head(read).squeeze(-1)
        if out.shape!=(b,n) or not bool(torch.isfinite(out).all()):
            raise RuntimeError("nonassociative genuine clock event output not finite")
        return out
