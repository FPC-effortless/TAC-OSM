"""Source-owned CSL-125-R1 adapter for repository-native generic diagnostics.

Only prefix tokens and a *currently observable* query token enter the model;
no labels or post-action information are present. Candidate permutation is
INCONCLUSIVE_NOT_DECLARED: this language/tool-head graph has no legal candidate
axis. The adapter never executes a tool or treats a logit as calibrated safety.
"""
from __future__ import annotations
from dataclasses import replace
from collections.abc import Mapping
import torch
from .csl125 import CSL125State
from .csl125_r1 import CSL125R1,R1LayerState
from .research_probe import ResearchProbeAdapter


def _validate(model,sample):
    if not isinstance(model,CSL125R1):
        raise ValueError('CSL125R1 adapter requires the R1 model')
    if not isinstance(sample,Mapping) or set(sample)!={'prefix','query_token','stream_ids','depth'}:
        raise ValueError('input boundary requires ONLY prefix/query_token/stream_ids/depth')
    prefix, query = sample['prefix'],sample['query_token']
    if (not isinstance(prefix,torch.Tensor) or not isinstance(query,torch.Tensor) or
        prefix.ndim!=2 or query.ndim!=2 or prefix.shape[0]!=query.shape[0] or
        prefix.shape[0]<1 or prefix.shape[1]<1 or query.shape[1]!=1 or
        prefix.dtype!=torch.long or query.dtype!=torch.long):
        raise ValueError('invalid causal token dimensions')
    ids=sample['stream_ids']
    if (not isinstance(ids,tuple) or len(ids)!=prefix.shape[0] or
        len(set(ids))!=len(ids) or any(not isinstance(v,str) or not v for v in ids)):
        raise ValueError('one unique stream ID required per row')
    if sample['depth'] not in model.config.rungs:
        raise ValueError('depth must be registered before diagnostics')
    if prefix.device != next(model.parameters()).device or query.device!=prefix.device:
        raise ValueError('device mismatch')
    if torch.any(prefix<0) or torch.any(prefix>=model.config.vocab_size) or torch.any(query<0) or torch.any(query>=model.config.vocab_size):
        raise ValueError('out-of-vocabulary token IDs')


def _start(m,s):
    return m.initial_state(s['prefix'].shape[0],s['prefix'].device,m.embedding.weight.dtype,
                           depth=s['depth'],stream_ids=s['stream_ids'])


def _advance(m,state,s,t):
    _,new=m(s['prefix'][:,t:t+1],depth=s['depth'],state=state,
            stream_ids=s['stream_ids'],emit_token_logits=False)
    return new


def _forecast(m,state,s):
    outs,_=m(s['query_token'],depth=s['depth'],state=state,
             stream_ids=s['stream_ids'],emit_token_logits=False)
    x=outs[s['depth']]
    return {'tool_logits':x['tool_logits'],'field_logits':x['field_logits'][:,-1],
            'accept_logit':x['accept_logit']}


def _parts(state):
    if not isinstance(state,CSL125State):raise ValueError('invalid R1 state')
    out={}
    for i,s in enumerate(state.layers):
        if not isinstance(s,R1LayerState):raise ValueError('non-R1 layer state')
        for k in ('fast','slow','fast_mass','slow_mass','conv_tail'):
            out[f'layer{i}.{k}']=getattr(s,k)
    for i,a in enumerate(state.attention):
        if a is not None:
            out[f'kv{i}.k']=a.k;out[f'kv{i}.v']=a.v
    return out


def _patch(state,name,value):
    before=_parts(state)
    if name not in before or value.shape!=before[name].shape:
        raise ValueError('unknown or misshaped patch state component')
    if name.startswith('layer'):
        i,part=name[5:].split('.',1);idx=int(i)
        layers=list(state.layers)
        layers[idx]=replace(layers[idx],**{part:value})
        return replace(state,layers=tuple(layers))
    if name.startswith('kv'):
        i,part=name[2:].split('.',1);idx=int(i)
        caches=list(state.attention)
        caches[idx]=replace(caches[idx],**{part:value})
        return replace(state,attention=tuple(caches))
    raise ValueError('invalid state path')


def _gradient_graph(m,state,s):
    sources={name:t.detach().clone().requires_grad_(True)
             for name,t in _parts(state).items()}
    leaf=state
    for name,t in sources.items():
        leaf=_patch(leaf,name,t)
    return _forecast(m,leaf,s),sources


def _batch_item(s,index):
    return {**s,'prefix':s['prefix'][index:index+1],
            'query_token':s['query_token'][index:index+1],
            'stream_ids':(s['stream_ids'][index],)}


csl125_r1_probe_adapter=ResearchProbeAdapter(
    name='csl125-r1-causal-token-v1',validate=_validate,
    steps=lambda sample:sample['prefix'].shape[1],start=_start,advance=_advance,
    forecast=_forecast,state_views=_parts,patch_state=_patch,
    batch_size=lambda sample:sample['prefix'].shape[0],batch_item=_batch_item,
    gradient_graph=_gradient_graph,permute_candidates=None)
