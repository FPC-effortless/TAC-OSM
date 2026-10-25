"""Deterministic PCGrad on shared-graph losses; no optimizer or model weights here.

Yu et al., Gradient Surgery for Multi-Task Learning (NeurIPS2020).
Projection reduces pairwise raw-gradient conflict, not a guarantee about AdamW
steps, held-out loss, probability calibration or physical decisions.
"""
import torch


def task_gradient_matrix(losses,parameters):
    """One batched VJP, equivalent to independent reverse sensitivities per task.

    PyTorch backend batching/fallback determines actual hardware speed. This
    shares the forward graph; it does not claim fewer mathematical gradient FLOPs.
    """
    losses=tuple(losses);parameters=tuple(parameters)
    output=torch.stack(losses)
    gradients=torch.autograd.grad(output,parameters,grad_outputs=torch.eye(len(losses),dtype=output.dtype,device=output.device),is_grads_batched=True,allow_unused=False)
    matrix=torch.cat([g.detach().reshape(len(losses),-1).double() for g in gradients],1)
    if not torch.isfinite(matrix).all():raise RuntimeError('Nonfinite real task gradient')
    return matrix


def projected_task_gradients(losses,parameters,*,rotation=0):
    parameters=tuple(parameters);vectors=list(task_gradient_matrix(losses,parameters).unbind(0))
    corrected=[];conflicts=0
    for i,vector in enumerate(vectors):
        current=vector.clone();others=[j for j in range(len(vectors)) if j!=i]
        if others:
            r=rotation%len(others);others=others[r:]+others[:r]
        for j in others:
            reference=vectors[j];dot=current.dot(reference);norm=reference.square().sum()
            if dot<0 and norm>0:
                current=current-dot/norm*reference;conflicts+=1
        corrected.append(current)
    total=torch.stack(corrected).sum(0)
    if not torch.isfinite(total).all():raise RuntimeError('Nonfinite projected gradient')
    offset=0
    for p in parameters:
        p.grad=total[offset:offset+p.numel()].reshape_as(p).to(p.dtype).clone();offset+=p.numel()
    return {'tasks':len(losses),'projected_conflicting_pairs':conflicts,'raw_task_gradient_norms':[float(v.norm()) for v in vectors],'projected_total_norm':float(total.norm())}
