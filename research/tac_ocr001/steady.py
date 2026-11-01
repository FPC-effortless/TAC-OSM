"""Optional frozen TorchScript *steady-state* TAC inference, same weights.

One token, filled KV cache, fixed batch/reference shape. Python validates
owners/config/weight versions outside the traced tensor kernel. This is not
a new trained architecture, a mixed-precision kernel, or a GPU speed claim.
"""
from __future__ import annotations

from dataclasses import fields
import math
import time
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from research.langcache104.model import LayerState104
from .model import DecoderState


NAMES=tuple(f.name for f in fields(LayerState104))


class _SteadyGraph(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.source=model

    def forward(self,tokens,visual_keys,visual_values,clock,*flat_state):
        model,c=self.source,self.source.config
        b=tokens.shape[0]
        x=model.embedding(tokens)
        outputs=[]
        mask=torch.ones_like(tokens,dtype=torch.bool)
        for i,block in enumerate(model.layers):
            prior=LayerState104(*flat_state[i*7:(i+1)*7])
            q=model.cross_queries[i](model.cross_norms[i](x)).reshape(b,1,c.n_heads,-1).transpose(1,2)
            v=F.scaled_dot_product_attention(q,visual_keys,visual_values,dropout_p=0.)
            x=x+model.cross_outputs[i](v.transpose(1,2).reshape(b,1,c.d_model))/math.sqrt(c.n_layers)
            normalized=block.memory_norm(x)
            fast,fm,fz,fe=block.fast(normalized,prior.fast_memory,prior.fast_mass,mask,"recurrent")
            slow,sm,sz,se=block.slow(normalized,prior.slow_memory,prior.slow_mass,mask,"recurrent")
            query=block.local_query(normalized).reshape(b,1,c.n_heads,c.key_dim).transpose(1,2)
            key=block.local_key(normalized).reshape(b,1,c.n_heads,c.key_dim).transpose(1,2)
            value=block.local_value(normalized).reshape(b,1,c.n_heads,c.value_dim).transpose(1,2)
            phase=clock.double()*block.local_frequencies.double()
            co,si=phase.cos().to(x.dtype)[None,None,None],phase.sin().to(x.dtype)[None,None,None]
            def rotate(z):
                a,b_part=z.reshape(b,c.n_heads,1,c.key_dim//2,2).unbind(-1)
                return torch.stack((co*a-si*b_part,si*a+co*b_part),-1).flatten(-2)
            query,key=rotate(query),rotate(key)
            keys=torch.cat((prior.recent_key,key),2)
            values=torch.cat((prior.recent_value,value),2)
            # Filled W-1 history + one current token: every key is legal.
            recent=F.scaled_dot_product_attention(query,keys,values,dropout_p=0.).transpose(1,2)
            pieces=torch.cat((fast,slow,fast-slow,fast*slow,recent,fe[...,None],se[...,None]),-1).flatten(-2)
            x=x+block.residual_scale*block.perspective_out(pieces)
            normalized=block.ffn_norm(x)
            x=x+block.residual_scale*block.ffn_out(block.ffn_value(normalized)*F.silu(block.ffn_gate(normalized)))
            outputs.extend((fm,fz,sm,sz,prior.rotary,keys[:,:,1:].clone(),values[:,:,1:].clone()))
        return (model.output(model.output_norm(x)),*outputs)


class SteadyTACSession:
    """Compile once per model/batch/reference shape, after regular prefill."""
    def __init__(self,model,references,state):
        if model.config.family!="tac" or model.training or model.config.window<2:
            raise ValueError("eval TAC and a nonempty local cache required")
        if state.clock<model.config.window-1 or state.signature!=model.signature or state.owners!=references.owners:
            raise ValueError("prefill full local window with same-owner state first")
        if references.model_tag!=model.model_tag:
            raise ValueError("same model visual references required")
        self.model,self.owners=model,state.owners
        self.reference_shape=tuple(references.keys.shape)
        self.reference_schema=tuple((tuple(x.shape),x.dtype,x.device) for x in (references.keys,references.values))
        self.state_schema=tuple((tuple(x.shape),x.dtype,x.device) for x in state.tensors().values())
        self.versions=tuple(p._version for p in model.parameters())
        device=model.embedding.weight.device
        args=(torch.zeros((len(self.owners),1),device=device,dtype=torch.long),references.keys,references.values,
              torch.tensor(state.clock,device=device,dtype=torch.float64),*state.tensors().values())
        started=time.perf_counter()
        with torch.inference_mode(),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always",torch.jit.TracerWarning)
            traced=torch.jit.trace(_SteadyGraph(model).eval(),args,check_trace=True,strict=True)
            self.graph=torch.jit.freeze(traced.eval())
        self.compile_seconds=time.perf_counter()-started
        self.trace_warnings=sorted({str(w.message) for w in caught})

    def step(self,tokens,references,*,owners,state):
        c=self.model.config
        if torch.is_grad_enabled():
            raise ValueError("frozen graph is inference only; use no_grad/inference_mode")
        if (tuple(owners)!=self.owners or state.owners!=self.owners or state.signature!=self.model.signature or
                references.owners!=self.owners or references.model_tag!=self.model.model_tag or
                tuple(references.keys.shape)!=self.reference_shape):
            raise ValueError("steady kernel owner/config/reference shape changed")
        if tuple(p._version for p in self.model.parameters())!=self.versions:
            raise ValueError("weights changed; rebuild frozen graph")
        if (tuple((tuple(x.shape),x.dtype,x.device) for x in state.tensors().values())!=self.state_schema or
                tuple((tuple(x.shape),x.dtype,x.device) for x in (references.keys,references.values))!=self.reference_schema):
            raise ValueError("fixed-shape steady graph state/reference schema changed")
        if (tokens.shape!=(len(self.owners),1) or tokens.dtype!=torch.long or
                (tokens<0).any() or (tokens>=259).any() or
                type(state.clock) is not int or not c.window-1<=state.clock<2**53):
            raise ValueError("legal one-token full-cache inference required")
        if any(layer.recent_key.shape[2]!=c.window-1 for layer in state.layers):
            raise ValueError("steady kernel requires full cache")
        values=self.graph(tokens,references.keys,references.values,
            torch.tensor(state.clock,device=tokens.device,dtype=torch.float64),*state.tensors().values())
        layers=tuple(LayerState104(*values[1+i*7:1+(i+1)*7]) for i in range(c.n_layers))
        return values[0],DecoderState(self.owners,self.model.signature,state.clock+1,layers)
