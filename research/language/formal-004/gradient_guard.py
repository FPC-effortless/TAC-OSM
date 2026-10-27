"""MLAG-003 source-shared gradient conflict surgery, research only.

Projects auxiliary historical gradients onto the half-space that is locally
non-antagonistic with next-character gradients on explicitly shared source
memory encoder tensors. Does not touch native, decoder, or history-only
parameters, data order, model equations or learned parameter count.

First-order *raw gradient* guarantee only. An AdamW step, finite precision,
clipping and nonlinear loss landscape are NOT certified to improve language.
"""
import torch

SOURCE_PREFIXES=(
    'rotation', 'source_content.', 'source_router.', 'source_retention.',
    'source_strength.', 'source_silu_mod.')

def source_shared(net):
    return [(n,p) for n,p in net.named_parameters()
            if n.startswith(SOURCE_PREFIXES)]

def gradient_guard_backward(net,main_loss,history_loss,optimizer,eps=1e-20):
    selected=source_shared(net)
    if not selected:raise ValueError('source encoder parameters missing')
    params=[p for n,p in selected]
    gm=torch.autograd.grad(main_loss,params,retain_graph=True,allow_unused=True)
    gh=torch.autograd.grad(history_loss,params,retain_graph=True,allow_unused=True)
    dot=sum((g.detach().double()*h.detach().double()).sum().item()
            for g,h in zip(gm,gh) if g is not None and h is not None)
    norm=sum(g.detach().double().square().sum().item() for g in gm if g is not None)
    # A single backward computes the exact original gradient; surgery only
    # changes selected shared-source parameters if the two loss signals conflict.
    optimizer.zero_grad(set_to_none=True)
    (main_loss+history_loss).backward()
    projected = (norm>eps and dot<0)
    if projected:
        coeff=dot/norm
        with torch.no_grad():
            for (_,p),g in zip(selected,gm):
                if g is not None and p.grad is not None:
                    p.grad.sub_(g,alpha=coeff)
    return {'projected':bool(projected),'raw_dot':float(dot),'main_squared_norm':float(norm),
            'gradient_rule':'g_main + g_history - min(0, <g_main,g_history>)/||g_main||²*g_main'}
