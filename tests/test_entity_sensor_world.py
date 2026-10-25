import pytest,torch
from tac_osm.variants.entity_sensor_world import EntitySensorWorld,diagonal_affine_prefix

torch.set_num_threads(1)

def inputs():
    torch.manual_seed(100);v=torch.randn(2,17,13,dtype=torch.float64)
    o=torch.rand(2,17,13)>.2;a=torch.rand_like(v);c=torch.randn(2,17,6,dtype=torch.float64).tanh();owner=torch.tensor([1,9]);h=torch.arange(17)[None].expand(2,-1)
    return v,o,a,c,owner,h

def test_diagonal_scan_value_and_gradient():
    a=torch.rand(2,19,26,dtype=torch.float64,requires_grad=True);b=torch.randn_like(a,requires_grad=True);init=torch.randn(2,26,dtype=torch.float64)
    y=diagonal_affine_prefix(a,b,init);s=init;ys=[]
    for j in range(19):s=a[:,j]*s+b[:,j];ys.append(s)
    ref=torch.stack(ys,1);torch.testing.assert_close(y,ref,rtol=1e-10,atol=1e-11)
    g=torch.autograd.grad(y.square().sum(),(a,b),retain_graph=True);gr=torch.autograd.grad(ref.square().sum(),(a,b))
    for x,z in zip(g,gr):torch.testing.assert_close(x,z,rtol=1e-9,atol=1e-10)

def test_stream_source_masks_gradients_and_bounds():
    m=EntitySensorWorld().double();x,o,a,c,ids,h=inputs()
    assert 95000<=sum(p.numel() for p in m.parameters())<=105000
    full,st,tr=m(x,o,a,c,ids,h,return_trace=True)
    _,state=m(x[:,:8],o[:,:8],a[:,:8],c[:,:8],ids,h[:,:8]);short,s2=m(x[:,8:],o[:,8:],a[:,8:],c[:,8:],ids,h[:,8:],state=state)
    torch.testing.assert_close(full.log_location,short.log_location,rtol=1e-9,atol=1e-10)
    for z,w in [(st.memory,s2.memory),(st.source_sum,s2.source_sum),(st.source_mass,s2.source_mass)]:torch.testing.assert_close(z,w,rtol=1e-9,atol=1e-10)
    assert st.source_mass.min()>=0 and st.source_mass.max()<=1+1e-12
    assert (st.source_sum.abs()<=st.source_mass+1e-12).all()
    x2=x.clone();x2[~o]=7.9
    changed,_=m(x2,o,a,c,ids,h)
    torch.testing.assert_close(full.log_location,changed.log_location,rtol=0,atol=0)
    loss=m.nll(full,torch.rand(2,6,dtype=torch.float64)*10,torch.ones(2,6,dtype=torch.bool));loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in m.parameters())

def test_owner_version_clock_and_missing_identity():
    m=EntitySensorWorld().double();x,o,a,c,ids,h=inputs();_,st=m(x,o,a,c,ids,h)
    with pytest.raises(ValueError):m(x,o,a,c,ids.flip(0),h+20,state=st)
    with pytest.raises(ValueError):m(x,o,a,c,ids,h,state=st)
    v=m.initial_state(ids);empty=torch.zeros_like(o)
    _,final=m(x,empty,a,c,ids,h,state=v)
    assert torch.equal(final.source_sum,v.source_sum) and torch.equal(final.memory,v.memory)
    with torch.no_grad():m.encoder.bias.add_(.01)
    with pytest.raises(ValueError):m(x,o,a,c,ids,h+20,state=st)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_entire_graph_against_independent_serial_reference(dtype):
    from scripts.me100_graph_reference import serial
    m=EntitySensorWorld().to(dtype);x,o,a,c,ids,h=inputs();x=x.to(dtype);a=a.to(dtype);c=c.to(dtype)
    # Exercise nonzero mean branch independently of the special initialization.
    with torch.no_grad():m.readout[2].weight[:6].normal_(0,.01)
    pred,state=m(x,o,a,c,ids,h);ref=serial(m,x,o,a,c,ids)
    tol=2e-5 if dtype==torch.float32 else 1e-9
    for got,want in zip((pred.log_location,pred.log_scale,state.source_sum,state.source_mass,state.evidence_sum,state.memory),ref):
        torch.testing.assert_close(got,want,rtol=tol,atol=tol*.1)
    ga=torch.autograd.grad(pred.log_location.square().sum()+pred.log_scale.square().sum(),tuple(m.parameters()))
    gb=torch.autograd.grad(ref[0].square().sum()+ref[1].square().sum(),tuple(m.parameters()))
    for v,w in zip(ga,gb):torch.testing.assert_close(v,w,rtol=tol*10,atol=tol)
