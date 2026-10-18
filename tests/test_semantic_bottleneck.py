"""Information-path and temporal invariants, not training performance claims."""
import pytest
import torch
from tac_osm.development.belief_bottleneck import BeliefBottleneck,relative_events


def payload():
    torch.set_num_threads(1);g=torch.Generator().manual_seed(6300901);keys=torch.randn(2,2,16,generator=g,dtype=torch.float64);events=torch.randn(2,8,46,generator=g,dtype=torch.float64);events[:,:,6:22]=keys[:,0,None];events[:,::2,6:22]=keys[:,1,None];events[:,:,42]=torch.arange(8,dtype=torch.float64)*10+100;present=torch.ones(2,8,dtype=torch.bool);present[1,6:]=False;candidates=torch.randn(2,5,19,generator=g,dtype=torch.float64);prices=torch.tensor([[.1,.1],[.1,.2]],dtype=torch.float64);return events,present,keys,candidates,prices


@pytest.mark.parametrize('arm',['tac','no_casm','gru','mlp'])
def test_fixed_semantic_belief_blocks_any_parallel_history_latent(arm):
    x,p,k,c,prices=payload();m=BeliefBottleneck(arm=arm,seed=3).double().eval();q=torch.arange(1,13,dtype=torch.float64).log()[None];hook=m.belief.register_forward_hook(lambda module,args,value:q.expand_as(value));a=m(x,p,k,c,prices);b=m(x*1.5,p,k,c,prices);hook.remove()
    assert not torch.allclose(a['latent'],b['latent'])
    for field in ('logits','future','cost'):torch.testing.assert_close(a[field],b[field],atol=0,rtol=0)
    grad=torch.autograd.grad(a['logits'].sum(),a['latent'],allow_unused=True)[0];assert grad is None or grad.abs().max()==0


@pytest.mark.parametrize('mode',['event_gap','relative_unit'])
def test_clock_origin_and_declared_unit_change_do_not_change_decisions(mode):
    x,p,k,c,prices=payload();m=BeliefBottleneck(seed=7,clock_mode=mode).double().eval();a=m(x,p,k,c,prices);changed=x.clone();changed[:,:,42]=changed[:,:,42]*1000+200000
    if mode=='relative_unit':m.clock_unit.fill_(1000)
    b=m(changed,p,k,c,prices)
    for field in ('posterior','logits','future','cost'):torch.testing.assert_close(a[field],b[field],atol=1e-12,rtol=1e-12)
    assert torch.equal(m.choice(a),m.choice(b))


def test_relative_encoding_uses_past_records_only_and_leaves_input_intact():
    x,p,k,_,_=payload();original=x.clone();a=relative_events(x,p,k);prefix=relative_events(x[:,:4],p[:,:4],k);torch.testing.assert_close(a[:,:4],prefix,atol=0,rtol=0);assert torch.equal(x,original);assert a[1,6:,42].abs().max()==0


def test_joint_belief_contains_relations_and_is_a_usable_learning_path():
    x,p,k,c,prices=payload();m=BeliefBottleneck(seed=9).double();o=m(x,p,k,c,prices);assert o['posterior'].shape==(2,12);torch.testing.assert_close(o['health_posterior'].sum(-1),torch.ones(2,dtype=torch.float64));grad=torch.autograd.grad(o['logits'].square().sum(),o['posterior'],retain_graph=True)[0];assert grad.abs().sum()>0


@pytest.mark.parametrize('unit',[0.,-1.,float('inf'),float('nan')])
def test_invalid_clock_unit_cannot_silently_erase_time(unit):
    x,p,k,_,_=payload()
    with pytest.raises(ValueError):relative_events(x,p,k,'relative_unit',unit)
