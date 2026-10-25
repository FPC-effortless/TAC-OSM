import torch
from tac_osm.variants.regime_trajectory_world import (
 RegimeTrajectoryWorld,RegimeTrajectory,mixture_quantile,student4_cdf,regime_joint_nll,marginal_intervals)
from tac_osm.variants.coherent_trajectory_world import CoherentTrajectoryWorld,joint_student_nll,TrajectoryMoments

torch.set_num_threads(1)

def test_quantile_root_and_implicit_gradient():
    p=torch.tensor([.001,.15,.9],dtype=torch.float64,requires_grad=True)
    k=torch.tensor(3.,dtype=torch.float64,requires_grad=True)
    q=mixture_quantile(p,k)
    torch.testing.assert_close((1-p)*student4_cdf(q)+p*student4_cdf(q/k),torch.full_like(p,.95),rtol=0,atol=1e-12)
    assert torch.autograd.gradcheck(mixture_quantile,(p,k),eps=1e-5,atol=1e-6,rtol=1e-5)

def test_parent_points_global_mixture_and_carry():
    torch.manual_seed(99);parent=CoherentTrajectoryWorld().double();m=RegimeTrajectoryWorld().double();m.load_coherent_parent(parent)
    assert sum(p.numel() for p in m.parameters())==103380
    x=torch.randn(2,9,14,dtype=torch.float64);a=torch.randn_like(x)
    f,s=m(x,a);old,_=parent(x,a)
    torch.testing.assert_close(f.mean,old.mean,rtol=0,atol=0)
    one=RegimeTrajectory(f.mean,f.narrow_scale_matrix,f.regime_logit,torch.tensor(1.,dtype=torch.float64))
    truth=f.mean.detach()+.02
    expected=joint_student_nll(TrajectoryMoments(f.mean,f.narrow_scale_matrix),truth).sum(-1)
    torch.testing.assert_close(regime_joint_nll(one,truth),expected,rtol=1e-10,atol=1e-10)
    lo,hi=marginal_intervals(f)
    loss=regime_joint_nll(f,truth).mean()/42+(hi-lo).mean()
    grads=torch.autograd.grad(loss,tuple(m.parameters()))
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)
    left,st=m(x[:,:4],a[:,:4]);right,st=m(x[:,4:],a[:,4:],state=st)
    torch.testing.assert_close(f.mean,torch.cat((left.mean,right.mean),1),rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(f.regime_logit,torch.cat((left.regime_logit,right.regime_logit),1),rtol=1e-10,atol=1e-11)
