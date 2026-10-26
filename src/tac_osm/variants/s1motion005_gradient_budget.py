"""S1MOTION005: research-only source-agnostic auxiliary-gradient budget.

Same 95,907 parameters and forward as VelocityAnchoredSystem1, no added
state/labels/actuation. Computes point and auxiliary grads separately for ALL
three TAC arms (matched autograd calls). No hindsight gating or DEV access.
This is an engineering kernel, not qualified intelligence or safety.
"""
from __future__ import annotations
import math
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from .s1motion004_gradient_routes import outputs

RHO = .10
MAX_BETA = .25
EPS = 1e-12
SMALL_POINT_NORM = 1e-14
ARMS=("tac_shared","tac_isolated","tac_projected")


def project_auxiliary(
    point_grads: Sequence[torch.Tensor],
    auxiliary_grads: Sequence[torch.Tensor],
    *, rho:float=RHO,beta_cap:float=MAX_BETA,
):
    """Project and norm-budget globally across the shared backbone only.

    All lists are shape-matched. Run on detached torch tensors; returns
    updated gradients and numeric diagnostics. Point gradients unchanged
    when auxiliary gradient opposes the point direction. Ideal SGD has
    a non-harmful FIRST-ORDER directional derivative; AdamW does not inherit it.
    """
    if not (0<=rho<=1 and 0<=beta_cap<=1 and
            math.isfinite(rho) and math.isfinite(beta_cap)):
        raise ValueError("invalid fixed hyperparameters")
    if len(point_grads)==0 or len(point_grads)!=len(auxiliary_grads):
        raise ValueError("nonempty corresponding gradient lists required")
    if any(p.shape!=a.shape or p.dtype!=a.dtype or
           p.device!=a.device or not torch.isfinite(p).all().item() or
           not torch.isfinite(a).all().item()
           for p,a in zip(point_grads,auxiliary_grads)):
        raise ValueError("nonfinite/mismatched gradient source; fail closed")
    # Double precision accumulation avoids avoidable FP32 dot cancellation.
    gp2=sum(float((x.detach().double().square()).sum())
             for x in point_grads)
    ga2=sum(float((x.detach().double().square()).sum())
             for x in auxiliary_grads)
    dot=sum(float((x.detach().double()*y.detach().double()).sum())
            for x,y in zip(point_grads,auxiliary_grads))
    if not all(math.isfinite(x) for x in (gp2,ga2,dot)):
        raise ValueError("nonfinite gradient statistics; fail closed")
    gp_norm=math.sqrt(max(gp2,0.))
    ga_norm=math.sqrt(max(ga2,0.))
    if gp_norm <= SMALL_POINT_NORM:
        q=[torch.zeros_like(x) for x in point_grads]
        beta=0.
    else:
        projection=(min(dot,0.)/gp2)  # only remove opposing component
        q=[a-projection*p for p,a in zip(point_grads,auxiliary_grads)]
        q2=sum(float((x.detach().double().square()).sum()) for x in q)
        if q2 < 0 or not math.isfinite(q2):
            raise ValueError("nonfinite projected norm")
        qn=math.sqrt(q2)
        beta=min(beta_cap,rho*gp_norm/(qn+EPS))
    result=[p+beta*x for p,x in zip(point_grads,q)]
    if any(not torch.isfinite(x).all().item() for x in result):
        raise ValueError("nonfinite proposed gradient")
    signed=sum(float((p.detach().double()*r.detach().double()).sum())
        for p,r in zip(point_grads,result))
    diagnostics=dict(
        point_norm=gp_norm,auxiliary_norm=ga_norm,
        raw_point_auxiliary_dot=dot,conflict=dot<0,
        beta=float(beta),projected_point_dot=float(signed),
        local_first_order_sgd_nonharm=(signed>=-1e-11*max(1.,gp2)),
        projected_budget_fraction_max=rho,
        actual_AdamW_descent_guarantee=False)
    return result, diagnostics


def decompose_losses(model,x,u,y,event,previous):
    pred,logit,scale=outputs(model,x,u,previous)
    point=(pred-y).square().mean()
    auxiliary=(.1*F.binary_cross_entropy_with_logits(logit,event)+
        .01*(.5*((pred-y)/scale).square()+scale.log()).mean())
    return point,auxiliary


def compute_gradients(model:nn.Module, batch, *, arm:str):
    """Same two autograd.grad passes for shared, isolated, projected TAC."""
    if arm not in ARMS:
        raise ValueError("unknown research arm")
    x,u,y,event,previous=batch
    point,aux=decompose_losses(model,x,u,y,event,previous)
    if not torch.isfinite(point+aux).item():
        raise ValueError("nonfinite original source objective")
    params=dict(model.named_parameters())
    names=list(params)
    base=[n for n in names if not n.startswith("core.reactive_head.")]
    head=[n for n in names if n.startswith("core.reactive_head.")]
    if not head or not base:
        raise ValueError("TAC source parameter roles not found; fail closed")
    names0=[params[n] for n in names]
    gp_raw=torch.autograd.grad(point,names0,retain_graph=True,allow_unused=True)
    ga_raw=torch.autograd.grad(aux,names0,allow_unused=True)
    gp={n:(g.detach() if g is not None else torch.zeros_like(params[n]))
        for n,g in zip(names,gp_raw)}
    ga={n:(g.detach() if g is not None else torch.zeros_like(params[n]))
        for n,g in zip(names,ga_raw)}
    if any(not torch.isfinite(v).all().item() for v in (*gp.values(),*ga.values())):
        raise ValueError("nonfinite original objective gradient")
    for n in head:
        if gp[n].abs().sum().item()!=0:
            raise ValueError("point gradient entered auxiliary output head")
    if arm=="tac_projected":
        combined,meta=project_auxiliary([gp[n] for n in base],
                                         [ga[n] for n in base])
        actual=dict(zip(base,combined))
    elif arm=="tac_isolated":
        actual={n:gp[n] for n in base}
        meta=dict(beta=0.,conflict=None,
                  projected_point_dot=sum(float(gp[n].double().square().sum())
                                          for n in base))
    else:
        actual={n:gp[n]+ga[n] for n in base}
        meta=dict(beta=1.,conflict=None,
                  projected_point_dot=sum(float((gp[n].double()*actual[n].double()).sum())
                                          for n in base))
    for n in head:
        actual[n]=ga[n]
    return point,aux,actual,meta


def step_model(model, optimizer, batch, *, arm):
    """FIT-only empirical AdamW update audit; identical two-pass AD budgets."""
    optimizer.zero_grad(set_to_none=True)
    point,aux,grads,meta=compute_gradients(model,batch,arm=arm)
    params=dict(model.named_parameters())
    # First-order actual optimizer preconditioned update is measured below,
    # not inferred from ideal-SGD projection positivity.
    before={n:params[n].detach().clone() for n in params}
    for name,p in params.items():
        p.grad=grads[name].clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()
    actual_point_dot=sum(float(
        (grads[n].detach().double()*(params[n].detach()-before[n]).double()).sum())
        for n in params if not n.startswith("core.reactive_head."))
    if not math.isfinite(actual_point_dot):
        raise ValueError("nonfinite actual source step")
    with torch.no_grad():
        x,u,y,event,previous=batch
        pred,_,_=outputs(model,x,u,previous)
        post=(pred-y).square().mean().item()
    outcome=dict(meta,
        pre_point_fit=float(point.detach()),post_point_fit=float(post),
        actual_grad_dot_parameter_delta=float(actual_point_dot),
        optimizer_preconditioned_step_verified_finite=True,
        number_of_core_weights=sum(params[n].numel() for n in params
                                   if not n.startswith("core.reactive_head.")))
    return float((point+aux).detach()),outcome
