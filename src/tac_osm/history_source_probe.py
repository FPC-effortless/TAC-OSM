"""Source-reviewed adapter for unchanged fd201207 historical-query models.

No labels/raw/future observations accepted. Model family is explicit; this is
not a default adapter for other architectures. Local AD is unimplemented.
"""
from __future__ import annotations

from dataclasses import dataclass,replace
import torch
from .research_probe import ResearchProbeAdapter


@dataclass(frozen=True)
class HistorySourceWindow:
    values: torch.Tensor
    valid: torch.Tensor
    age: torch.Tensor
    last: torch.Tensor
    owners: torch.Tensor
    hours: torch.Tensor
    past_clocks: torch.Tensor
    future_clock: torch.Tensor


def validate(model,s):
    if type(s) is not HistorySourceWindow:raise ValueError('explicit feature-only history window required')
    b=len(s.owners)
    if any(t.shape!=(b,88,13) for t in (s.values,s.valid,s.age,s.last)) or s.valid.dtype!=torch.bool:
        raise ValueError('original causal history shapes required')
    if s.hours.shape!=(b,88) or not torch.equal(s.hours[:,1:]-s.hours[:,:-1],torch.ones_like(s.hours[:,1:])):
        raise ValueError('original chronological source hours required')
    if s.past_clocks.shape!=(b,8,6) or s.future_clock.shape!=(b,1,6):raise ValueError('legal calendar shapes required')
    if s.owners.dtype!=torch.long or not bool(((s.owners>=0)&(s.owners<12)).all()):raise ValueError('original station IDs required')
    if model.__class__.__module__ not in ('tac_osm.latent_fusion_007_historic_query','tac_osm.latent_fusion_008_nonassoc_control'):
        raise ValueError('adapter is only for the reviewed original graph')
    if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in (s.values,s.age,s.last,s.past_clocks,s.future_clock)):
        raise ValueError('finite FP32 source features required')


def advance(model,state,s,i):
    return model.step(state,s.values[:,i],s.valid[:,i],s.age[:,i],s.last[:,i],s.owners)


def forecast(model,state,s):
    historical=model.historic_event_logits(state,s.owners,s.past_clocks)
    mean,sd=model.forecast(state,s.owners,s.future_clock)
    return dict(historical=historical,future_mean=mean,future_sd=sd)


def batch_item(s,i):
    return replace(s,**{k:getattr(s,k)[i:i+1] for k in s.__dataclass_fields__})


def adapter():
    names=('fast','slow','associative','entities')
    return ResearchProbeAdapter(name='fd201207-source-history-feature-only',validate=validate,steps=lambda s:88,
        start=lambda m,s:m.initial(len(s.owners)),advance=advance,forecast=forecast,
        state_views=lambda state:{k:getattr(state,k) for k in names},
        patch_state=lambda state,name,value:state._replace(**{name:value}),
        batch_size=lambda s:len(s.owners),batch_item=batch_item)
