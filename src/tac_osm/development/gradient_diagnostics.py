"""Failure-only gradient observations; never change clipping or optimizer policy."""
import math
import torch


def scalar(value):
    if math.isnan(value):return 'nan'
    if math.isinf(value):return 'inf' if value>0 else '-inf'
    return value


def gradient_health(named_parameters):
    rows=[];norms32=[];norms64=[];nonfinite=0
    for name,parameter in named_parameters:
        if parameter.grad is None:continue
        gradient=parameter.grad.detach()
        if gradient.is_sparse:gradient=gradient.coalesce().values()
        gradient=gradient.cpu()
        bad=int((~torch.isfinite(gradient)).sum());nonfinite+=bad
        n32=float(torch.linalg.vector_norm(gradient.float()))
        n64=float(torch.linalg.vector_norm(gradient.double()))
        norms32.append(n32);norms64.append(n64)
        rows.append(dict(name=name,elements=gradient.numel(),nonfinite_elements=bad,
            norm_float32=scalar(n32),norm_float64=scalar(n64)))
    total32=float(torch.linalg.vector_norm(torch.tensor(norms32,dtype=torch.float32))) if rows else 0.
    total64=math.hypot(*norms64)
    if nonfinite:classification='nonfinite_gradient_elements'
    elif not math.isfinite(total32) and math.isfinite(total64):classification='finite_gradients_float32_norm_overflow'
    else:classification='finite_norms_or_different_failure'
    return dict(classification=classification,nonfinite_elements=nonfinite,
        total_norm_float32=scalar(total32),total_norm_float64=scalar(total64),parameters=rows,
        clipping_changed=False,optimizer_step_applied_by_diagnostic=False)
