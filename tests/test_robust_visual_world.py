import torch
from tac_osm.variants.robust_visual_world import RobustVisualWorld
from tac_osm.variants.visual_world import VisualWorld
def inputs(t=7):
    torch.manual_seed(93093);torch.set_num_threads(1)
    x=torch.randn(2,t,14,dtype=torch.float64)*.01
    return x,torch.randn_like(x)*.01,torch.full((2,t,1),1/15,dtype=x.dtype),torch.rand(2,t,1,32,32,dtype=x.dtype)

from scripts.me093_train_world import load_initial
from tac_osm.variants.multimode_distribution import log_density,mixture_crps,implicit_median


def test_three_regime_budget_dynamic_parent_preservation_and_nonzero_gradients():
    old=VisualWorld().double();m=RobustVisualWorld().double();load_initial(m,old.state_dict())
    assert sum(p.numel() for p in m.parameters())==96595
    x,a,dt,im=inputs();p=m(x,a,elapsed=dt,images=im);previous=old(x,a,elapsed=dt,images=im)
    torch.testing.assert_close(p['locations'][...,2,:],previous['current']+previous['moving_delta'],rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(p['locations'][...,0,:],x[:,:,None].expand(-1,-1,3,-1),rtol=0,atol=0)
    assert (p['scales'][...,1:,:]>=2*p['scales'][...,:-1,:]).all()
    target=x[:,:,None]+.001
    loss=(mixture_crps(p,target)-.2*log_density(p,target)+implicit_median(p).sub(target).abs()).mean()+p['motion_logit'].square().mean()
    loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
    assert m.jitter_head.bias.grad.abs().sum()>0


def test_three_regime_prefix_causality_and_chunk_step_equivalence():
    m=RobustVisualWorld().double();x,a,dt,im=inputs()
    whole,last=m(x,a,elapsed=dt,images=im,return_state=True)
    xx=x.clone();xx[:,4:]+=1;ii=im.clone();ii[:,4:]=1-ii[:,4:]
    alt=m(xx,a,elapsed=dt,images=ii)
    for k in whole:torch.testing.assert_close(whole[k][:,:4],alt[k][:,:4],rtol=0,atol=0)
    _,state=m(x[:,:3],a[:,:3],elapsed=dt[:,:3],images=im[:,:3],return_state=True)
    end,state=m(x[:,3:],a[:,3:],state,elapsed=dt[:,3:],images=im[:,3:],return_state=True)
    stream=None;parts=[]
    for t in range(x.shape[1]):
        out,stream=m.step(x[:,t],a[:,t],stream,elapsed=dt[:,t],images=im[:,t]);parts.append(out)
    for k in whole:
        torch.testing.assert_close(whole[k][:,3:],end[k],rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(whole[k],torch.stack([r[k] for r in parts],1),rtol=1e-10,atol=1e-10)
    for a,b,c in zip(last,state,stream):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10);torch.testing.assert_close(a,c,rtol=1e-10,atol=1e-10)
