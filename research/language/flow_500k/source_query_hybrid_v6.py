"""GRAPH010: frozen-backbone, GRAPH009-exact-first learned miss-only query.

All backbones (TAC, GRU, SSM, Transformer) must be supplied exactly the
same fixed-source count cache AND same learned query when benchmarking.
No hidden target input, model-specific oracle or user authorization claim.
"""
import torch
from torch import nn
from learned_context_query_v6 import LearnedContextQuery

class SourceQueryHybridV6(nn.Module):
    def __init__(self,graph009_hybrid,query_encoder:LearnedContextQuery,*,fusion=.6):
        super().__init__()
        if not isinstance(query_encoder,LearnedContextQuery) or not 0<=fusion<=1:
            raise ValueError("fixed source query model and valid fusion required")
        if graph009_hybrid.order!=3 or graph009_hybrid.cache.capacity!=query_encoder.capacity:
            raise ValueError("same exact source horizon and context order required")
        if not torch.allclose(graph009_hybrid.fit_prior,query_encoder.fit_prior,atol=1e-6):
            raise ValueError("source FIT prior mismatch")
        self.exact=graph009_hybrid
        self.learned=query_encoder
        self.fusion=fusion

    def forward(self,tokens,*,start=0,source_ids=None,initial_state=None,return_state=False,
                return_diagnostics=False):
        if initial_state is None:
            if start!=0:raise ValueError("nonzero start requires complete carry")
            base_state=learned_state=None
        else:
            if (initial_state.get("count")!=start or
                initial_state.get("source_ids")!=source_ids or
                set(initial_state)!={"count","source_ids","exact","learned"}):
                raise ValueError("owner or count mismatch")
            base_state=initial_state["exact"];learned_state=initial_state["learned"]
        base,newbase=self.exact(tokens,start=start,source_ids=source_ids,
                                 initial_state=base_state,return_state=True)
        query,newquery=self.learned.scan(tokens,start=start,source_ids=source_ids,
                                        initial_state=learned_state)
        # GRAPH009 already handles exact 3gram cache hits; learned matcher
        # is permitted ONLY where no exact context was available.
        if bool((query["exact_hit"]&query["hit"]).any()):
            raise ValueError("learned query used exact-match context")
        gate=query["hit"].to(dtype=base.dtype)*self.fusion
        posterior=query["posterior"].to(dtype=base.dtype,device=base.device)
        probs=(1-gate[...,None])*base+gate[...,None]*posterior
        parts=[probs]
        if return_state:parts.append(dict(count=start+tokens.shape[1],
           source_ids=source_ids,exact=newbase,learned=newquery))
        if return_diagnostics:parts.append({
            "approximate_hit":query["hit"],"exact_hit":query["exact_hit"],
            "approximate_query_weight":gate,
            "added_trainable_parameters":self.learned.learned_parameters(),
            "query_cache_event_budget_per_owner":self.learned.capacity,
            "extra_source_cache_bytes":"must measure actual serialized events, not just model weights"})
        return tuple(parts) if len(parts)>1 else probs
