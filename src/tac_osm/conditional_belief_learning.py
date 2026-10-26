"""ME111 explicit point, uncertainty and restricted reconstruction-mean roles."""
import torch
from .typed_role_learning import uncertainty_masks,clip_role,TypedRoleAdamW


def reconstruction_mean_masks(model):
    masks=[]
    for name,p in model.named_parameters():
        mask=torch.zeros_like(p,dtype=torch.bool)
        if name.startswith(('current_head.','current_residual.output.')):mask[:6]=True
        masks.append(mask)
    if sum(int(m.sum()) for m in masks)!=696:raise ValueError('expected696 current-mean coordinates')
    return masks


def routed_gradients(model,point_loss,auxiliary_loss,reconstruction_loss):
    params=tuple(model.parameters());umasks=uncertainty_masks(model);rmasks=reconstruction_mean_masks(model)
    def grad(loss):
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return [torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,gs)]
    point=grad(point_loss);recon=grad(.3*reconstruction_loss)
    if any(bool(g[m].any()) for g,m in zip(point,umasks)):raise RuntimeError('point reachesU')
    ids=[i for i,m in enumerate(umasks) if bool(m.any())]
    gs=torch.autograd.grad(auxiliary_loss,[params[i] for i in ids],allow_unused=True)
    uncertainty=[torch.zeros_like(p) for p in params]
    for i,g in zip(ids,gs):
        if g is not None:uncertainty[i]=torch.where(umasks[i],g.detach(),torch.zeros_like(g))
    combined=[g+torch.where(r,h,torch.zeros_like(h)) for g,h,r in zip(point,recon,rmasks)]
    pc,pnorm=clip_role(combined);uc,unorm=clip_role(uncertainty)
    for p,g,u,m in zip(params,pc,uc,umasks):
        p.grad=torch.where(m,u,g)
        if not torch.isfinite(p.grad).all():raise RuntimeError('nonfinite routed gradient')
    return point,dict(point_and_reconstruction_preclip_norm=pnorm,uncertainty_preclip_norm=unorm,
        point_on_uncertainty_nonzero=0,auxiliary_on_point_nonzero=0,
        reconstruction_mean_gradient_norm=float(sum(h[r].double().square().sum() for h,r in zip(recon,rmasks)).sqrt()),
        reconstruction_mean_coordinates=696,point_coordinates=100253,uncertainty_coordinates=4410)
