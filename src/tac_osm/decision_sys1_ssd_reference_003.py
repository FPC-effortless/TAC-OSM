# Copyright (c) 2024, Albert Gu and Tri Dao.
# SPDX-License-Identifier: Apache-2.0
# Selected functions from state-spaces/mamba, commit
# 6b72c122713bb769cc82c6b8e6d019c53d27d6a1, modules/ssd_minimal.py.
# Modified only to remove CUDA imports/test and use local torch reshapes instead
# of einops. Equations retain the upstream chunk factorization. This is a CPU
# mathematical reference, NOT the official fused runtime or a speed competitor.
import torch
from torch.nn import functional as F


def segsum(x):
    t = x.shape[-1]
    z = x[..., :, None].expand(*x.shape, t)
    strict = torch.ones(t, t, device=x.device, dtype=torch.bool).tril(-1)
    z = z.masked_fill(~strict, 0).cumsum(-2)
    return z.masked_fill(~torch.ones_like(strict).tril(), -torch.inf)


def ssd_chunk_reference(x, a, b, c, block_len, initial_state=None):
    """X=(batch,time,heads,head_dim); A=(batch,time,heads); B,C=...state.

    Caller discretizes x with dt, and a with dt*A. Input length must divide
    block_len. B,C may already be expanded from a single group to all heads.
    """
    if not (x.dtype == a.dtype == b.dtype == c.dtype):
        raise ValueError('SSD dtype mismatch')
    batch, length, heads, head_dim = x.shape
    if block_len < 1 or length < 1 or length % block_len:
        raise ValueError('positive length divisible by block_len required')
    chunks, n = length//block_len, b.shape[-1]
    x = x.reshape(batch, chunks, block_len, heads, head_dim)
    a = a.reshape(batch, chunks, block_len, heads).permute(0,3,1,2)
    b = b.reshape(batch, chunks, block_len, heads, n)
    c = c.reshape(batch, chunks, block_len, heads, n)
    ac = a.cumsum(-1)
    decay = segsum(a).exp()
    diagonal = torch.einsum('bclhn,bcshn,bhcls,bcshp->bclhp',c,b,decay,x)
    states = torch.einsum('bclhn,bhcl,bclhp->bchpn',b,(ac[...,-1:]-ac).exp(),x)
    initial = torch.zeros_like(states[:,:1]) if initial_state is None else initial_state[:,None]
    states = torch.cat((initial,states),1)
    inter = segsum(F.pad(ac[...,-1],(1,0))).exp()
    new_states = torch.einsum('bhzc,bchpn->bzhpn',inter,states)
    boundary, final = new_states[:,:-1],new_states[:,-1]
    off = torch.einsum('bclhn,bchpn,bhcl->bclhp',c,boundary,ac.exp())
    return (diagonal+off).reshape(batch,length,heads,head_dim),final
