"""Constructed engineering falsifiers; NEVER scientific or real-world evidence."""
import pytest
import torch
from tac_osm.latent_fusion_001 import (FusedLatentCandidate, affine_compose,
  affine_prefix, gated_delta, source_slot_write)


def case():
    torch.manual_seed(7)
    m = FusedLatentCandidate(4, 3, hidden=8,key_dim=5,val_dim=6,entity_count=3)
    s = m.initial(2)
    x = torch.tensor([[1., float('nan'),3.,4.],[2.,4.,6.,8.]])
    observed = torch.tensor([[True,False,True,True],[True,True,True,True]])
    elapsed = torch.ones(2,4)
    last = torch.ones(2,4)
    ids = torch.tensor([0,2])
    candidates = torch.randn(2,4,3)
    return m,s,x,observed,elapsed,last,ids,candidates


def test_affine_prefix_and_gradients_are_serial_equivalent():
    torch.manual_seed(3)
    a=(torch.randn(15,2,4,dtype=torch.float64)*.2).requires_grad_()
    b=torch.randn(15,2,4,dtype=torch.float64,requires_grad=True)
    z=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    prefix=affine_prefix(a,b,z)
    h=z; steps=[]
    for t in range(15):
        h=a[t]*h+b[t]; steps.append(h)
    serial=torch.stack(steps)
    torch.testing.assert_close(prefix,serial,rtol=1e-12,atol=1e-12)
    ga=torch.autograd.grad(prefix.square().sum(),(a,b,z),retain_graph=True)
    gb=torch.autograd.grad(serial.square().sum(),(a,b,z))
    for x,y in zip(ga,gb): torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-11)


def test_affine_associativity_in_real_float_tolerance():
    a,b,c=[(torch.randn(3,dtype=torch.float64)*.25,torch.randn(3,dtype=torch.float64)) for _ in range(3)]
    left=affine_compose(affine_compose(a,b),c)
    right=affine_compose(a,affine_compose(b,c))
    for x,y in zip(left,right):torch.testing.assert_close(x,y,rtol=1e-14,atol=1e-14)


def test_gated_delta_exact_one_key_update_and_no_write():
    m=torch.zeros(2,4,5)
    k=torch.nn.functional.one_hot(torch.tensor([1,2]),5).float()
    v=torch.randn(2,4)
    out=gated_delta(m,k,v,torch.ones(2,1),torch.ones(2,1))
    torch.testing.assert_close((out@k.unsqueeze(-1)).squeeze(-1),v)
    torch.testing.assert_close(gated_delta(m,k,v,torch.ones(2,1),torch.zeros(2,1)),m)
    with pytest.raises(ValueError,match='gate'):
        gated_delta(m,k,v,torch.full((2,1),1.4),torch.ones(2,1))


def test_source_slot_isolation_refuses_out_of_capacity():
    slots=torch.randn(2,3,8)
    live=torch.zeros(2,3,dtype=torch.bool)
    ids=torch.tensor([0,2]);v=torch.randn(2,8)
    out,active=source_slot_write(slots,live,ids,v,torch.ones(2,1))
    assert torch.equal(out[0,1:],slots[0,1:])
    assert torch.equal(out[1,:2],slots[1,:2])
    torch.testing.assert_close(out[0,0],v[0]);torch.testing.assert_close(out[1,2],v[1])
    assert active.tolist()==[[True,False,False],[False,False,True]]
    same,same_live=source_slot_write(slots,live,ids,v,torch.zeros(2,1))
    assert torch.equal(same,slots) and torch.equal(same_live,live)
    with pytest.raises(ValueError,match='capacity'):
        source_slot_write(slots,live,torch.tensor([0,3]),v,torch.ones(2,1))


def test_fusion_causal_source_mask_and_readonly_candidate_permutation():
    m,s,x,mask,dt,last,ids,candidates=case()
    new=m.step(s,x,mask,dt,last,ids)
    assert all(torch.isfinite(t).all() for t in new[:4])
    assert torch.equal(s.associative,torch.zeros_like(s.associative))
    assert torch.equal(new.entities[0,1:],s.entities[0,1:])
    assert torch.equal(new.entities[1,:2],s.entities[1,:2])
    before=[t.clone() for t in new]
    mean,scale=m.forecast(new,ids,candidates)
    assert mean.shape == (2,4,4) and scale.shape==(2,4,4)
    assert bool((scale>0).all())
    perm=[2,0,3,1]
    other=m.forecast(new,ids,candidates[:,perm])
    torch.testing.assert_close(mean[:,perm],other[0],rtol=0,atol=0)
    torch.testing.assert_close(scale[:,perm],other[1],rtol=0,atol=0)
    assert all(torch.equal(p,q) for p,q in zip(before,new))
    absent=x.clone();absent[0,1]=float('inf')
    # nonobserved channel is never included in arithmetic.
    res=m.step(s,absent,mask,dt,last,ids)
    for p,q in zip(new,res):torch.testing.assert_close(p,q)


def test_fusion_gradient_backpropagates_into_each_operator():
    m,s,x,mask,dt,last,ids,candidates=case()
    z=m.step(s,x,mask,dt,last,ids)
    mean,scale=m.forecast(z,ids,candidates)
    loss=mean.square().sum()+scale.square().sum()
    loss.backward()
    for part in [m.embed,m.temporal_a,m.temporal_b,m.slow_input,m.key_map,m.val_map,
                 m.retention_map,m.write_map,m.entity_map,m.entity_gate,m.action_map,m.forecast_head]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in part.parameters())


def test_invalid_time_observation_and_source_ids_fail_closed():
    m,s,x,mask,dt,last,ids,candidates=case()
    with pytest.raises(ValueError,match='elapsed'):
        m.step(s,x,mask,-dt,last,ids)
    with pytest.raises(ValueError,match='source slot'):
        m.forecast(s,torch.tensor([0,3]),candidates)
    with pytest.raises(ValueError,match='capacity'):
        m.step(s,x,mask,dt,last,torch.tensor([0,3]))
    with pytest.raises(ValueError,match='nonfinite source'):
        m.step(s,x,mask,dt,last*float('nan'),ids)


def test_state_memory_size_is_bounded_by_configuration():
    m,s,*_=case()
    primary=sum(t.numel() for t in s[:4])+s.active.numel()
    assert primary==2*(2*8+6*5+3*8+3)
    assert sum(p.numel() for p in m.parameters()) < 95000  # DO NOT TRAIN this toy configuration.
