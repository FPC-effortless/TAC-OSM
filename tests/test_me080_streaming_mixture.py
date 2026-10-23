import pytest
import torch
from tac_osm.variants.streaming_mixture_world import StreamingMixtureWorld,quantile,log_density,distribution_loss


def test_whole_model_budget_and_active_gradients():
    torch.manual_seed(80)
    m=StreamingMixtureWorld()
    assert sum(p.numel() for p in m.parameters())==99816
    x=torch.randn(2,8,14)*.01;a=torch.randn_like(x)
    pred=m(x,a);target=x[:,:,None]+torch.randn(2,8,3,14)*.02
    loss=distribution_loss(pred,target,x,torch.ones(2,8,3,dtype=torch.bool),torch.ones(2),m.state_scale,1.)
    loss.backward()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name


def test_stream_step_chunk_and_interleaving_equivalence():
    torch.manual_seed(80);m=StreamingMixtureWorld().eval()
    x=torch.randn(2,40,14)*.01;a=torch.randn_like(x)
    with torch.no_grad():
        full,state=m(x,a,return_state=True)
        left,carried=m(x[:,:13],a[:,:13],return_state=True)
        right,final=m(x[:,13:],a[:,13:],carried.detached(),True)
        states=[None,None];outputs=[]
        for t in range(40):
            local=[]
            for entity in (1,0):
                p,states[entity]=m.step(x[entity:entity+1,t],a[entity:entity+1,t],states[entity])
                local.append((entity,p))
            outputs.append(dict(local))
        for key in full:
            torch.testing.assert_close(full[key],torch.cat((left[key],right[key]),1),rtol=1e-5,atol=1e-6)
            streamed=torch.stack([torch.cat([row[e][key] for e in (0,1)]) for row in outputs],1)
            torch.testing.assert_close(full[key],streamed,rtol=1e-5,atol=1e-6)
        for v,w in zip(state,final):torch.testing.assert_close(v,w,rtol=1e-5,atol=1e-6)
        assert state.fast.abs().max()<=1 and state.slow.abs().max()<=1
        assert state.association.norm(dim=(1,2)).max()<=4.00001


def test_causal_inputs_and_original_native_update_reference():
    torch.manual_seed(81);m=StreamingMixtureWorld()
    obs=torch.randn(2,8,64)
    expected=m.core(obs)
    state=m.initial_state(torch.zeros(2,14));actual=[]
    for t in range(8):
        r,f,s,assoc=m.advance(obs[:,t],state)
        actual.append(r);state=state._replace(fast=f,slow=s,association=assoc)
    torch.testing.assert_close(expected,torch.stack(actual,1),rtol=1e-5,atol=1e-6)
    x=torch.randn(2,8,14);a=torch.randn_like(x)
    p=m(x,a);x[:,5:]+=20;a[:,5:]-=20;q=m(x,a)
    for key in p:torch.testing.assert_close(p[key][:,:5],q[key][:,:5],rtol=1e-5,atol=1e-6)


def test_mixture_quantiles_match_distribution_and_laplace_reference():
    m=StreamingMixtureWorld();p=m(torch.zeros(2,2,14),torch.zeros(2,2,14))
    p={k:torch.zeros_like(v) if k in ('current','moving_delta','mixture_logit') else torch.ones_like(v) for k,v in p.items()}
    # Equal zero-centred components with unit scale are a unit Laplace.
    for tau in (.05,.5,.95):
        expected=torch.distributions.Laplace(0.,1.).icdf(torch.tensor(tau,dtype=torch.float64))
        torch.testing.assert_close(quantile(p,tau),torch.full_like(p['current'].double(),expected),rtol=1e-8,atol=1e-8)
    density=log_density(p,torch.zeros_like(p['current']))
    torch.testing.assert_close(density,torch.full_like(density,-torch.log(torch.tensor(2.))))
    with pytest.raises(ValueError):quantile(p,1.)


def test_reuse_does_not_encode_old_history_or_mutate_state():
    m=StreamingMixtureWorld();x=torch.randn(2,14);a=torch.randn_like(x)
    _,state=m.step(x,a);snapshot=[v.clone() for v in state]
    calls=[]
    hook=m.encoder.register_forward_pre_hook(lambda module,args:calls.append(tuple(args[0].shape)))
    m.step(x,a,state);hook.remove()
    assert calls==[(2,57)]
    for old,new in zip(snapshot,state):assert torch.equal(old,new)
