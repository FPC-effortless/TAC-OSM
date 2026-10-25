import torch
import pytest
from tac_osm.variants.coherent_trajectory_world import (
 CoherentTrajectoryWorld,trajectory_moments,TrajectoryParameters,joint_student_nll)
from tac_osm.variants.evidence_retrieval_world import EvidenceRetrievalWorld

torch.set_num_threads(1)

def test_parent_one_step_and_chunk_gradients():
    torch.manual_seed(98);parent=EvidenceRetrievalWorld().double();m=CoherentTrajectoryWorld().double();m.load_one_step_parent(parent)
    assert sum(p.numel() for p in m.parameters())==103322
    x=torch.randn(2,9,14,dtype=torch.float64,requires_grad=True);u=torch.randn_like(x,requires_grad=True)
    full,s=m(x,u)
    torch.testing.assert_close(full.mean[:,:,0],parent(x,u),rtol=1e-12,atol=1e-12)
    a,st=m(x[:,:4],u[:,:4]);b,st=m(x[:,4:],u[:,4:],state=st)
    torch.testing.assert_close(full.mean,torch.cat((a.mean,b.mean),1),rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(full.scale_matrix,torch.cat((a.scale_matrix,b.scale_matrix),1),rtol=1e-10,atol=1e-11)
    loss=full.mean.square().mean()+joint_student_nll(full,full.mean.detach()+.03).mean()
    grads=torch.autograd.grad(loss,tuple(m.parameters())+(x,u),retain_graph=True)
    other=torch.cat((a.mean,b.mean),1).square().mean()+torch.cat((joint_student_nll(a,a.mean.detach()+.03),joint_student_nll(b,b.mean.detach()+.03)),1).mean()
    grad2=torch.autograd.grad(other,tuple(m.parameters())+(x,u))
    for g,g2 in zip(grads,grad2):
        assert torch.isfinite(g).all() and g.norm()>0
        torch.testing.assert_close(g,g2,rtol=1e-7,atol=1e-9)

def test_semigroup_and_joint_covariance():
    torch.manual_seed(9);m=CoherentTrajectoryWorld().double();x=torch.randn(2,3,14,dtype=torch.float64)
    p,_=m.condition(x,torch.randn_like(x));whole=trajectory_moments(p,(0,1,4,16))
    assert torch.equal(whole.mean[:,:,0],x)
    assert torch.equal(whole.scale_matrix[...,0,:],torch.zeros_like(whole.scale_matrix[...,0,:]))
    a=trajectory_moments(p,(4,));v=p.terminal_velocity+p.retention.pow(4)*(p.velocity-p.terminal_velocity)
    q=TrajectoryParameters(a.mean[:,:,0],v,p.terminal_velocity,p.retention,p.position_noise,p.velocity_noise,p.valid)
    second=trajectory_moments(q,(12,))
    torch.testing.assert_close(second.mean[:,:,0],whole.mean[:,:,3],rtol=1e-11,atol=1e-12)
    pos=trajectory_moments(p,(1,4,16))
    assert torch.linalg.eigvalsh(pos.scale_matrix).min()>0
    marginal=trajectory_moments(p,(16,))
    torch.testing.assert_close(pos.scale_matrix[...,-1,-1],marginal.scale_matrix[...,0,0])
    torch.testing.assert_close(pos.mean[:,:,-1],marginal.mean[:,:,0])
    with pytest.raises(ValueError):trajectory_moments(p,(1,1))

def test_padding_and_causality():
    m=CoherentTrajectoryWorld();x=torch.randn(2,12,14);a=torch.randn_like(x)
    mask=torch.zeros(2,12,dtype=torch.bool);f,s=m(x,a,valid=mask)
    assert torch.equal(s.memory,torch.zeros_like(s.memory)) and not s.count.any()
    assert torch.equal(f.mean,x[:,:,None].expand_as(f.mean))
    f,_=m(x,a);x[:,8:]+=20;g,_=m(x,a)
    torch.testing.assert_close(f.mean[:,:8],g.mean[:,:8],rtol=0,atol=1e-6)


def test_shared_conditioning_reuse_and_extreme_rates():
    m=CoherentTrajectoryWorld();calls=[]
    hook=m.encoder.register_forward_hook(lambda *_:calls.append(1))
    x=torch.randn(2,5,14);u=torch.randn_like(x)
    p,_=m.condition(x,u)
    trajectory_moments(p,(1,4));trajectory_moments(p,(8,16))
    assert len(calls)==1
    hook.remove()
    with torch.no_grad():
        m.trajectory_head.weight.zero_()
        for extreme in [-80.,80.]:
            m.trajectory_head.bias.fill_(extreme)
            f,_=m(x,u,horizons=(1,4,256))
            assert torch.isfinite(f.mean).all() and torch.isfinite(f.scale_matrix).all()
            torch.linalg.cholesky(f.scale_matrix)
    from tac_osm.variants.parallel_six import instantiate_untrained_core,PARALLEL_FAMILIES
    for family in PARALLEL_FAMILIES:
        z=instantiate_untrained_core(family,state_dim=14,action_dim=14,candidate='coherent_trajectory')
        assert type(z) is CoherentTrajectoryWorld
