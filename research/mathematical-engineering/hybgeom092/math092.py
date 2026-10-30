"""HYBGEOM092: exact algebraic dissection of HYBGEOM091 fixed Euclidean decoder.

No learning, no observations other than current byte and current learned embeddings.
All returned logits are centered over the 256 candidates, as in original091.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


def corrections(embeddings: torch.Tensor, current: torch.Tensor, strength: torch.Tensor,
                *, permutation: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if embeddings.ndim != 2 or embeddings.shape[0] != 256:
        raise ValueError('original 256-row byte vocabulary')
    if current.ndim != 1 or current.dtype != torch.long or current.min() < 0 or current.max() > 255:
        raise ValueError('observed byte indices, not next-byte labels')
    if strength.numel() != 1 or not bool(torch.isfinite(strength).all()) or not 0 <= float(strength.detach()) <= .02:
        raise ValueError('original strength bounds')
    if embeddings.device != current.device or not torch.isfinite(embeddings).all():
        raise ValueError('embedding finite/device')
    batch = embeddings[current]
    distance = (embeddings[None, :, :] - batch[:, None, :]).square().sum(-1)
    original = -strength * (distance - distance.mean(-1,keepdim=True))
    dot = 2*strength * (batch @ embeddings.T)
    dot = dot - dot.mean(-1, keepdim=True)
    norm = -strength * embeddings.square().sum(-1)[None, :]
    norm = norm - norm.mean(-1, keepdim=True)
    combined = dot + norm
    if permutation is None:
        permutation = torch.randperm(256,device=embeddings.device,generator=torch.Generator(device=embeddings.device).manual_seed(92))
    if permutation.shape != (256,) or not torch.equal(permutation.sort().values,torch.arange(256,device=embeddings.device)):
        raise ValueError('fixed blind vocabulary permutation')
    wrong = 2*strength * (batch @ embeddings[permutation].T)
    wrong = wrong - wrong.mean(-1,keepdim=True)
    return {'original_distance':original,'algebraic':combined,'similarity':dot,
            'vocab_norm':norm,'shuffled_similarity':wrong}


def assert_equivalence(terms:dict[str,torch.Tensor], *, tolerance:float=6e-5) -> float:
    a,b=terms['original_distance'],terms['algebraic']
    error=float((a-b).detach().abs().max())
    if error>tolerance or not torch.isfinite(a).all():
        raise ValueError('Euclidean vs dot-plus-norm algebra mismatch')
    return error
