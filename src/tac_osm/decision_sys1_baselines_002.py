"""Untrained, parameter-matched GRU and cached causal Transformer references.

Exact torch2.10 source package recorded by preflight. These are instantiated
engineering candidates, not tuned/qualified scientific competitors. Mamba-2 is
required by the protocol and intentionally NOT replaced by an invented SSM.
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


class GRURiskModel(nn.Module):
    def __init__(self, context_size=212, hidden_size=64, layers=2, head_width=77):
        super().__init__()
        self.gru = nn.GRU(context_size, hidden_size, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(context_size+hidden_size, head_width),
                                  nn.SiLU(), nn.Linear(head_width, 4))

    def initial(self, batch):
        p = next(self.parameters())
        return p.new_zeros(self.gru.num_layers, batch, self.gru.hidden_size)

    def forward(self, x, state=None):
        states, state = self.gru(x, self.initial(x.shape[0]) if state is None else state)
        return self.head(torch.cat((x, states), -1)), state

    def step(self, x, state):
        y, state = self.forward(x[:, None, :], state)
        return y[:, 0], state


@dataclass(frozen=True)
class KVCache:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    length: int


class TransformerRiskModel(nn.Module):
    """Two pre-norm layers, four heads, sinusoidal positions, no truncation.

    Full causal and one-event KV paths use the SAME learned parameter tensors.
    Cache is caller-owned, grows with history and is not counted as constant state.
    Dropout is zero in this candidate; tuning remains a separate equal-budget gate.
    """
    def __init__(self, context_size=212, width=64, heads=4, layers=2,
                 ff_width=128, head_width=68):
        super().__init__()
        if width % heads or width % 2:
            raise ValueError('width must divide heads and be even')
        self.width, self.heads = width, heads
        self.input = nn.Linear(context_size, width)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(width, heads,
                ff_width, dropout=0., activation='gelu', batch_first=True,
                norm_first=True) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(context_size+width, head_width),
                                  nn.SiLU(), nn.Linear(head_width, 4))

    def position(self, length, offset, reference):
        pos = torch.arange(offset, offset+length, device=reference.device,
                           dtype=reference.dtype)[:, None]
        freq = torch.exp(torch.arange(0, self.width, 2, device=reference.device,
                         dtype=reference.dtype)*(-math.log(10000.)/self.width))
        angles = pos*freq
        return torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)

    def forward(self, x):
        z = self.input(x)+self.position(x.shape[1], 0, x)
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            z = layer(z, src_mask=mask)
        return self.head(torch.cat((x, self.norm(z)), -1))

    def step(self, x, cache=None):
        b = x.shape[0]
        offset = 0 if cache is None else cache.length
        if cache is not None and (len(cache.layers) != len(self.layers) or offset < 1):
            raise ValueError('invalid cache')
        z = self.input(x[:, None, :])+self.position(1, offset, x)
        fresh = []
        d = self.width//self.heads
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            qkv = F.linear(layer.norm1(z), a.in_proj_weight, a.in_proj_bias)
            q, k, v = [t.reshape(b, 1, self.heads, d).transpose(1,2) for t in qkv.chunk(3, -1)]
            if cache is not None:
                old_k, old_v = cache.layers[i]
                if old_k.shape != (b,self.heads,offset,d) or old_v.shape != old_k.shape:
                    raise ValueError('cache shape mismatch')
                k, v = torch.cat((old_k,k),2), torch.cat((old_v,v),2)
            # Only available past/current keys exist; is_causal=True here would
            # wrongly mask all but the first key for a one-token query.
            value = F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
            z = z+a.out_proj(value.transpose(1,2).reshape(b,1,self.width))
            n = layer.norm2(z)
            z = z+layer.linear2(layer.activation(layer.linear1(n)))
            fresh.append((k,v))
        return self.head(torch.cat((x,self.norm(z[:,0])), -1)), KVCache(tuple(fresh),offset+1)


def mamba2_runtime_status():
    """No silent CPU shim or surrogate may be described as qualified Mamba-2."""
    import importlib.util
    installed = importlib.util.find_spec('mamba_ssm') is not None
    return {'installed': installed, 'cuda': torch.cuda.is_available(),
            'qualified': False, 'reason': 'source/kernel/geometry/parity qualification pending'}
