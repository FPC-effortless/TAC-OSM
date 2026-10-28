"""GRAPH018: primary-anchored Euclidean gradient projection, SOFTWARE ONLY.

Related prior art: PCGrad (Yu et al., NeurIPS 2020); this is an asymmetric
one-primary/two-auxiliary adaptation. It protects the *first-order* main-loss
gradient inner product, not a finite-step AdamW update or predictive quality.
"""
from __future__ import annotations
import math
from collections.abc import Mapping
import torch

HORIZONS=(1,4,16)

def primary_guard(primary:torch.Tensor,auxiliary:torch.Tensor,*,epsilon:float=1e-12):
    """g_aux' = g_aux - min(0,<g_aux,g_main>)/||g_main||^2*g_main.

    This assures <g_main,g_aux'> >= 0 in exact Euclidean arithmetic.
    """
    if not isinstance(primary,torch.Tensor) or not isinstance(auxiliary,torch.Tensor) or primary.shape!=auxiliary.shape:
        raise ValueError("matching tensors required")
    if primary.dtype!=auxiliary.dtype or primary.device!=auxiliary.device or not primary.is_floating_point():
        raise ValueError("matching floating tensor dtype/device required")
    if not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError("positive finite numerical floor required")
    if not torch.isfinite(primary).all() or not torch.isfinite(auxiliary).all():
        raise ValueError("finite gradient tensors required")
    dot=(primary.double()*auxiliary.double()).sum()
    squared=(primary.double()*primary.double()).sum()
    if squared<=epsilon:
        return auxiliary.clone(),False
    if dot>=0:
        return auxiliary.clone(),False
    projected=auxiliary-(dot/squared).to(primary.dtype)*primary
    return projected,True

def protected_backward(model:torch.nn.Module,losses:Mapping[int,torch.Tensor],*,weights=None):
    """Form a protected first-order gradient for all trainable parameters.

    NO optimizer step, source access, checkpoint changes or model training.
    Gradient surgery alters an optimization direction; the claimed inequality
    does not transfer to finite learning rates or adaptive optimizer steps.
    """
    weights={4:.15,16:.15} if weights is None else weights
    if set(losses)!={1,4,16} or set(weights)!={4,16}:
        raise ValueError("exact registered horizon objectives required")
    if any(type(w) not in (float,int) or not math.isfinite(w) or w<0 for w in weights.values()):
        raise ValueError("nonnegative finite auxiliary task weights required")
    for h,loss in losses.items():
        if not isinstance(loss,torch.Tensor) or loss.ndim!=0 or not bool(torch.isfinite(loss).item()):
            raise ValueError(f"nonfinite or non-scalar loss horizon {h}")
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no learned parameters")
    g={h:torch.autograd.grad(losses[h],params,retain_graph=h!=16,allow_unused=True)
       for h in (1,4,16)}
    clips={4:0,16:0};examined={4:0,16:0}
    model.zero_grad(set_to_none=True)
    for i,p in enumerate(params):
        main=torch.zeros_like(p) if g[1][i] is None else g[1][i]
        total=main.clone()
        for h in (4,16):
            aux=torch.zeros_like(p) if g[h][i] is None else g[h][i]
            adjusted,changed=primary_guard(main,aux)
            clips[h]+=int(changed)
            examined[h]+=int(g[h][i] is not None and g[1][i] is not None)
            total=total+float(weights[h])*adjusted
        p.grad=total.detach()
    return {"opposing_component_removed":clips,"shared_tensors_examined":examined,
            "precision":"Euclidean first-order only, NO AdamW or validation-loss guarantee",
            "training":"NO UPDATE APPLIED"}
