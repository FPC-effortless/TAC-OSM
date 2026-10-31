"""SENSE-TOK-005 independent Python reference for two operators ONLY.

This does not establish full-model Lean proof or source equivalence.
"""
from __future__ import annotations
from math import exp, sqrt, isfinite
from collections.abc import Sequence


def position_pool_reference(vectors: Sequence[Sequence[float]], recency: Sequence[float]) -> tuple[float,...]:
    """Stable per-channel chronological softmax pooling."""
    if not vectors or not recency or any(len(v)!=len(recency) for v in vectors):
        raise ValueError("nonempty equal-width observed vectors required")
    if any(not isfinite(v) for v in recency) or any(not isfinite(x) for v in vectors for x in v):
        raise ValueError("nonfinite reference input")
    out=[]
    for j in range(len(recency)):
        a=[float(i)*float(recency[j]) for i in range(len(vectors))]
        top=max(a)
        w=[exp(z-top) for z in a]
        out.append(sum(w[i]*vectors[i][j] for i in range(len(vectors)))/sum(w))
    return tuple(out)


def code_id_reference(vector: Sequence[float], codes: Sequence[Sequence[float]]) -> int:
    """First-argmax cosine code ID, with F.normalize-equivalent epsilon."""
    if not vector or not codes or any(len(c)!=len(vector) for c in codes):
        raise ValueError("nonempty equal-width vectors/codes required")
    if any(not isfinite(x) for x in vector) or any(not isfinite(x) for c in codes for x in c):
        raise ValueError("nonfinite reference input")
    norm=max(sqrt(sum(x*x for x in vector)),1e-12)
    best,bestval=0,float("-inf")
    for i,c in enumerate(codes):
        cnorm=max(sqrt(sum(x*x for x in c)),1e-12)
        sim=sum(x*y for x,y in zip(vector,c))/(norm*cnorm)
        if sim>bestval: best,bestval=i,sim
    return best
