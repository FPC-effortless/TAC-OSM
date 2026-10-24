import torch
from tac_osm.variants.median_anchored_world import MedianAnchoredWorld
from tac_osm.variants.multimode_distribution import cdf,implicit_median,log_density,mixture_crps
from tac_osm.variants.median_anchored_distribution import mixture_quantiles
from scripts.me094_train_world import load_initial
from tac_osm.variants.robust_visual_world import RobustVisualWorld


def inputs(t=5):
    torch.set_num_threads(1);torch.manual_seed(9494);x=torch.randn(2,t,14,dtype=torch.float64)*.01
    return x,torch.randn_like(x),torch.full((2,t,1),1/15,dtype=x.dtype),torch.rand(2,t,1,32,32,dtype=x.dtype)


def test_translated_multimodal_cdf_has_independent_median_and_correct_shape_gradients():
    base={'locations':torch.tensor([[-.4],[.1],[1.2]],dtype=torch.float64,requires_grad=True),'scales':torch.tensor([[.1],[.3],[1.]],dtype=torch.float64,requires_grad=True),'component_logits':torch.tensor([[.2],[-.3],[.1]],dtype=torch.float64,requires_grad=True)}
    point=torch.tensor([.25],dtype=torch.float64,requires_grad=True)
    def translated(p):return {**p,'locations':p['locations']+(point-implicit_median(p)).unsqueeze(-2),'median':point}
    p=translated(base);torch.testing.assert_close(cdf(p,point),torch.full_like(point,.5),rtol=0,atol=1e-10)
    q=implicit_median(p);grads=torch.autograd.grad(q.sum(),tuple(base.values())+(point,),retain_graph=True)
    for g in grads[:-1]:torch.testing.assert_close(g,torch.zeros_like(g),rtol=0,atol=2e-8)
    torch.testing.assert_close(grads[-1],torch.ones_like(point),rtol=0,atol=2e-8)
    loss=log_density(p,torch.tensor([.7],dtype=point.dtype)).sum();analytic=torch.autograd.grad(loss,base['component_logits'])[0]
    for i in range(3):
        plus={k:v.detach().clone() for k,v in base.items()};minus={k:v.detach().clone() for k,v in base.items()};plus['component_logits'][i]+=1e-5;minus['component_logits'][i]-=1e-5
        numeric=(log_density(translated(plus),point.new_tensor([.7]))-log_density(translated(minus),point.new_tensor([.7])))/(2e-5)
        torch.testing.assert_close(analytic[i],numeric,rtol=2e-5,atol=2e-6)


def test_budget_active_parameters_quantile_consistency_and_final_only_decode():
    m=MedianAnchoredWorld().double();load_initial(m,RobustVisualWorld().double().state_dict());assert sum(p.numel() for p in m.parameters())==101173
    x,a,dt,im=inputs();p=m(x,a,elapsed=dt,images=im);last=m.forecast_last(x,a,elapsed=dt,images=im)
    for k in p:torch.testing.assert_close(p[k][:,-1],last[k],rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(cdf(p,p['median']),torch.full_like(p['median'],.5),rtol=0,atol=1e-8)
    lo,med,hi=mixture_quantiles(p);torch.testing.assert_close(med,p['median'],rtol=0,atol=0)
    for value,level in [(lo,.05),(hi,.95)]:torch.testing.assert_close(cdf(p,value),torch.full_like(value,level),rtol=0,atol=1e-8)
    target=x[:,:,None]+.001;loss=(mixture_crps(p,target)-.2*log_density(p,target)+20*(p['median']-target).abs()).mean()+p['motion_logit'].square().mean();loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
    assert m.decision_head.weight.grad.abs().sum()>0


def test_prefix_causality_and_chunk_step_state_equivalence():
    m=MedianAnchoredWorld().double();x,a,dt,im=inputs(5);whole,final=m(x,a,elapsed=dt,images=im,return_state=True)
    xx=x.clone();xx[:,3:]+=1;ii=im.clone();ii[:,3:]=1-ii[:,3:];alt=m(xx,a,elapsed=dt,images=ii)
    for k in whole:torch.testing.assert_close(whole[k][:,:3],alt[k][:,:3],rtol=0,atol=0)
    _,s=m(x[:,:2],a[:,:2],elapsed=dt[:,:2],images=im[:,:2],return_state=True)
    end,s=m(x[:,2:],a[:,2:],s,elapsed=dt[:,2:],images=im[:,2:],return_state=True);stream=None;parts=[]
    for t in range(5):
        p,stream=m.step(x[:,t],a[:,t],stream,elapsed=dt[:,t],images=im[:,t]);parts.append(p)
    for k in whole:
        torch.testing.assert_close(whole[k][:,2:],end[k],rtol=1e-8,atol=1e-8);torch.testing.assert_close(whole[k],torch.stack([p[k] for p in parts],1),rtol=1e-8,atol=1e-8)
    for a,b,c in zip(final,s,stream):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10);torch.testing.assert_close(a,c,rtol=1e-10,atol=1e-10)
