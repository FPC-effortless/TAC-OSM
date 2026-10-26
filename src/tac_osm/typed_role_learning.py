"""ME110 coordinate-typed learning on the unchanged native TAC graph.

Point losses alone shape shared representations/location coordinates. Auxiliary
losses alone shape declared uncertainty coordinates. Per-role clipping avoids
letting auxiliary magnitudes rescale point gradients. This custom update field
is not claimed to be the gradient of one scalar objective.
"""
import torch
from .role_optimizer import BlockProtectedAdamW
from .inward_role_guard import inward_candidates

ROWS=('readout.2.','retrieved_delta.2.','current_head.','current_residual.output.')


def uncertainty_masks(model):
    masks=[]
    for name,p in model.named_parameters():
        mask=torch.zeros_like(p,dtype=torch.bool)
        if name=='state_noise' or name.startswith(('width_head.','width_residual.')):
            mask.fill_(True)
        elif name.startswith(ROWS):
            if p.shape[0]!=12: raise ValueError('expected six location + six variance rows')
            mask[6:]=True
        masks.append(mask)
    if len(masks)!=45: raise ValueError('ME110 requires original45 tensors')
    return masks


def clip_role(grads):
    norm=sum(g.double().square().sum() for g in grads).sqrt()
    coefficient=(1./(norm+1e-6)).clamp(max=1.)
    return [g*coefficient.to(g.dtype) for g in grads],float(norm)


def routed_gradients(model,point_loss,auxiliary_loss):
    params=tuple(model.parameters());masks=uncertainty_masks(model)
    pgrads=torch.autograd.grad(point_loss,params,retain_graph=True,allow_unused=True)
    point=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,pgrads)]
    forbidden=sum(int(torch.count_nonzero(g[mask])) for g,mask in zip(point,masks))
    if forbidden: raise RuntimeError('point loss reached uncertainty-only coordinates')
    ids=[i for i,m in enumerate(masks) if bool(m.any())]
    ugrads=torch.autograd.grad(auxiliary_loss,[params[i] for i in ids],allow_unused=True)
    auxiliary=[torch.zeros_like(p) for p in params]
    for i,g in zip(ids,ugrads):
        if g is not None: auxiliary[i]=torch.where(masks[i],g.detach(),torch.zeros_like(g))
    outside=sum(int(torch.count_nonzero(a[~mask])) for a,mask in zip(auxiliary,masks))
    if outside: raise RuntimeError('uncertainty loss reached point-only coordinates')
    pc,pnorm=clip_role(point);uc,unorm=clip_role(auxiliary)
    for p,g,u,mask in zip(params,pc,uc,masks):
        p.grad=torch.where(mask,u,g)
        if not torch.isfinite(p.grad).all(): raise RuntimeError('nonfinite routed gradient')
    return point,dict(point_preclip_norm=pnorm,uncertainty_preclip_norm=unorm,
        point_on_uncertainty_nonzero=forbidden,auxiliary_on_point_nonzero=outside,
        point_coordinates=sum(int((~m).sum()) for m in masks),
        uncertainty_coordinates=sum(int(m.sum()) for m in masks))


class TypedRoleAdamW(BlockProtectedAdamW):
    @torch.no_grad()
    def propose(self,point_gradients):
        raw,moments,_=super().propose([torch.zeros_like(p) for p in self._params])
        candidates,receipt=inward_candidates(self._params,point_gradients,raw,self.blocks)
        return candidates,moments,receipt
