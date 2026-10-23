import pytest
import torch
from tac_osm.variants.streaming_mixture_world import StreamingMixtureWorld,distribution_loss
from tac_osm.variants.reusable_execution import BatchedMixtureWorld,EntityInferenceCache


def pair():
    torch.manual_seed(810)
    reference=StreamingMixtureWorld().double();optimized=BatchedMixtureWorld().double()
    optimized.load_state_dict(reference.state_dict(),strict=True)
    return reference,optimized


def test_values_last_state_and_autodiff_are_equivalent():
    reference,optimized=pair();x=torch.randn(2,8,14,dtype=torch.double)*.01;a=torch.randn_like(x)
    target=x[:,:,None]+torch.randn(2,8,3,14,dtype=torch.double)*.02
    losses=[]
    for m in (reference,optimized):
        prediction,state=m(x,a,return_state=True)
        loss=distribution_loss(prediction,target,x,torch.ones(2,8,3,dtype=torch.bool),torch.ones(2),m.state_scale,1.)
        losses.append((prediction,state,torch.autograd.grad(loss,tuple(m.parameters()))))
    for k in losses[0][0]:torch.testing.assert_close(losses[0][0][k],losses[1][0][k],atol=1e-12,rtol=1e-10)
    for g,h in zip(losses[0][2],losses[1][2]):torch.testing.assert_close(g,h,atol=1e-9,rtol=1e-8)
    final,state=optimized.forecast_last(x,a,return_state=True)
    for k in final:torch.testing.assert_close(final[k],losses[0][0][k][:,-1],atol=1e-12,rtol=1e-10)
    for g,h in zip(state,losses[0][1]):torch.testing.assert_close(g,h,atol=1e-12,rtol=1e-10)
    assert sum(p.numel() for p in optimized.parameters())==99816


def test_entity_order_retries_isolation_and_caller_mutation():
    _,m=pair();m.eval();cache=EntityInferenceCache(m,model_revision='original-checkpoint',preprocessing_revision='fit-v1',max_entities=2)
    x=torch.randn(1,14,dtype=torch.double);a=torch.randn_like(x);states={};expected={}
    calls=[];hook=m.encoder.register_forward_pre_hook(lambda *args:calls.append(1))
    for t in range(3):
        for entity in ('a','b'):
            value=x+t+(entity=='b')
            with torch.no_grad():out,states[entity]=m.step(value,a,states.get(entity))
            expected[entity]=out;actual=cache.step(entity,t,value,a)
            for k in out:torch.testing.assert_close(out[k],actual[k])
    before=len(calls);retry=cache.step('a',2,x+2,a);assert len(calls)==before
    retry['motion_logit'].fill_(999)
    assert not torch.equal(cache.step('a',2,x+2,a)['motion_logit'],retry['motion_logit'])
    with pytest.raises(ValueError,match='conflicting'):cache.step('a',2,x+20,a)
    with pytest.raises(ValueError,match='gap'):cache.step('a',4,x,a)
    with pytest.raises(RuntimeError,match='capacity'):cache.step('c',0,x,a)
    cache.reset('b');cache.step('c',0,x,a)
    hook.remove()


def test_cache_invalidates_weights_buffers_and_mode():
    for change in ('parameter','buffer','mode'):
        _,m=pair();m.eval();cache=EntityInferenceCache(m,model_revision='v1',preprocessing_revision='p1')
        x=torch.zeros(1,14,dtype=torch.double);cache.step('a',0,x,x)
        with torch.no_grad():
            if change=='parameter':next(m.parameters()).add_(.01)
            elif change=='buffer':m.delta_scale.mul_(2)
            else:m.train()
        with pytest.raises(RuntimeError,match='model changed'):cache.step('a',1,x,x)


def test_chunk_boundary_preserves_outputs_and_gradients():
    _,m=pair();x=torch.randn(2,9,14,dtype=torch.double)*.01;a=torch.randn_like(x)
    full,state=m(x,a,return_state=True)
    left,carried=m(x[:,:4],a[:,:4],return_state=True)
    right,end=m(x[:,4:],a[:,4:],carried,True)
    for k in full:torch.testing.assert_close(full[k],torch.cat((left[k],right[k]),1),atol=1e-12,rtol=1e-10)
    for u,v in zip(state,end):torch.testing.assert_close(u,v)


def test_fused_quantiles_preserve_original_inversion():
    from tac_osm.variants.reusable_execution import mixture_quantiles
    from tac_osm.variants.streaming_mixture_world import quantile
    _,m=pair();x=torch.randn(2,8,14,dtype=torch.double)*.01
    p=m.forecast_last(x,torch.randn_like(x))
    actual=mixture_quantiles(p)
    expected=torch.stack([quantile(p,t) for t in (.05,.5,.95)])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert (actual[0]<=actual[1]).all() and (actual[1]<=actual[2]).all()
    with pytest.raises(ValueError):mixture_quantiles(p,())
