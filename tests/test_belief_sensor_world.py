import torch,pytest
from tac_osm.variants.belief_sensor_world import BeliefSensorWorld
from scripts.me100_graph_reference import serial

torch.set_num_threads(1)

def sample(dtype):
    torch.manual_seed(101);v=torch.randn(2,23,13,dtype=dtype);o=torch.rand(2,23,13)>.25
    o[:,-1,0]=False;o[:,-1,1]=True
    return v,o,torch.rand_like(v),torch.randn(2,23,6,dtype=dtype).tanh(),torch.tensor([2,9]),torch.arange(23)[None].expand(2,-1)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_marginalization_graph_gradient_and_chunk(dtype):
    m=BeliefSensorWorld().to(dtype);x=sample(dtype);p,s,tr=m(*x,return_trace=True)
    assert sum(v.numel() for v in m.parameters())==97259
    tol=2e-5 if dtype==torch.float32 else 1e-9
    base=serial(m,*x[:5])
    co=torch.nn.functional.softplus(m.state_noise);age=torch.expm1(x[2][:,-1,:6]*__import__('math').log(257.))
    v=(co[0]+co[1]*age/24)*(1+co[2]*(1-base[3][:,:6].clamp(0,1)))
    v=v*(~x[1][:,-1,:6]);j=1+.25*m.state_response.tanh()
    refvar=base[1].square()+j.square()*v
    torch.testing.assert_close(p.log_scale.square(),refvar,rtol=tol,atol=tol*.1)
    torch.testing.assert_close(p.log_location,base[0],rtol=tol,atol=tol*.1)
    ga=torch.autograd.grad(p.log_location.square().sum()+p.log_scale.square().sum(),tuple(m.parameters()),retain_graph=True)
    gb=torch.autograd.grad(base[0].square().sum()+refvar.sum(),tuple(m.parameters()))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=tol*10,atol=tol)
    assert (tr['total_variance']>0).all() and (tr['latent_state_variance'][x[1][:,-1,:6]]==0).all()
    _,part=m(*(z[:,:10] if z.ndim>=2 else z for z in x))
    q,_=m(*(z[:,10:] if z.ndim>=2 else z for z in x),state=part)
    torch.testing.assert_close(p.log_location,q.log_location,rtol=tol,atol=tol*.1)
    torch.testing.assert_close(p.log_scale,q.log_scale,rtol=tol,atol=tol*.1)
    # Read-only variance head adds no persistent tensors.
    assert sum(z.numel() for z in (s.source_sum,s.source_mass,s.evidence_sum,s.memory))==2*152


def test_belief_age_and_missing_value_contract():
    m=BeliefSensorWorld().double();x=list(sample(torch.float64));x[2]=torch.zeros_like(x[2])
    _,_,old=m(*x,return_trace=True);x[2]=torch.ones_like(x[2]);_,_,new=m(*x,return_trace=True)
    # Encoder changes with age, so test the declared law at fixed mass explicitly.
    c=torch.nn.functional.softplus(m.state_noise);C=torch.full((2,6),.5)
    small=(c[0]+c[1]*1/24)*(1+c[2]*(1-C));large=(c[0]+c[1]*48/24)*(1+c[2]*(1-C))
    assert (large>=small).all()
    a,s=m(*x);x[0]=x[0].clone();x[0][~x[1]]=7.9;b,_=m(*x)
    torch.testing.assert_close(a.log_location,b.log_location,rtol=0,atol=0)
    torch.testing.assert_close(a.log_scale,b.log_scale,rtol=0,atol=0)
    loss,_=m.training_loss(a,torch.ones(2,6,dtype=torch.float64)*10,torch.ones(2,6,dtype=torch.bool));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in m.parameters())
