import numpy as np
import pytest
import torch
from tac_osm.variants.context_filter_world import ContextFilterWorld
from tac_osm.variants.context_filter_execution import capture_clock_features,PhysicalContextGraph,ContextInitializer
from tac_osm.variants.context_filter_cache import ContextFilterCache
from tac_osm.variants.median_anchored_distribution import mixture_quantiles


def test_physical_context_graph_matches_original_clock_preprocessing_and_state_birth():
    torch.manual_seed(9597);torch.set_num_threads(1);m=ContextFilterWorld().double().eval()
    with torch.no_grad():
        m.state_mean.copy_(torch.arange(14,dtype=torch.float64));m.state_scale.fill_(2.);m.action_mean.fill_(.5);m.action_scale.fill_(3.)
        m.task_embedding.weight.normal_(std=.03);m.clock_encoder.weight.normal_(std=.01);m.filter_rates.weight.normal_(std=.02)
    x=torch.rand(2,3,14);a=torch.rand_like(x);dt=torch.full((2,3,1),1/15);pixels=torch.randint(0,256,(2,3,1,32,32),dtype=torch.uint8);ids=torch.tensor([0,4])
    clock=torch.tensor([1788405544.6797798,1788405544.7464464,1788405544.8131132],dtype=torch.float64)[None,:,None].expand(2,-1,-1).clone()
    values=[]
    for p in [1.,4.,16.]:phase=np.remainder(clock.numpy()[:,:,0],p)*(2*np.pi/p);values.extend([np.sin(phase),np.cos(phase)])
    ctx=torch.from_numpy(np.stack(values,-1).astype(np.float32)).double();torch.testing.assert_close(ctx,capture_clock_features(clock),rtol=0,atol=2*torch.finfo(torch.float32).eps)
    xn=((x-m.state_mean.float())/m.state_scale.float()).double();an=((a-m.action_mean.float())/m.action_scale.float()).double();s=m.initial_state(xn[:,0],ids)
    for u,v in zip(s,ContextInitializer(m)(x[:,0],ids)):torch.testing.assert_close(u,v,rtol=0,atol=0)
    raw,state=m.visual_tensor_graph(xn,an,dt.double(),(pixels.float()/255).double(),s,last_only=True,task_ids=ids,clock_context=ctx)
    expected=tuple(q*m.state_scale+m.state_mean for q in mixture_quantiles(raw))+(raw['motion_logit'].sigmoid(),*tuple(state))
    actual=PhysicalContextGraph(m)(x,a,dt,pixels,ids,clock,tuple(s));assert len(actual)==19
    for u,v in zip(expected,actual):torch.testing.assert_close(u,v,rtol=1e-5,atol=1e-6)


def test_context_cache_retries_guard_all_context_and_preserve_mutation_isolation():
    torch.set_num_threads(1);m=ContextFilterWorld().double().eval();calls=[];hook=m.visual_encoder.register_forward_hook(lambda *_:calls.append(1))
    cache=ContextFilterCache(m,model_revision='original-sha',preprocessing_revision='clock-periods-v1')
    x=torch.zeros(1,14,dtype=torch.float64);dt=torch.ones(1,1,dtype=x.dtype)/15;image=torch.zeros(1,1,32,32,dtype=x.dtype);ctx=torch.zeros(1,6,dtype=x.dtype)
    first=cache.step('asset',0,x,x,elapsed=dt,image=image,task_id=1,clock_context=ctx);expected=first['median'].clone();first['median'].add_(10)
    retry=cache.step('asset',0,x,x,elapsed=dt,image=image,task_id=1,clock_context=ctx);assert len(calls)==1
    torch.testing.assert_close(retry['median'],expected,rtol=0,atol=0)
    with pytest.raises(ValueError,match='conflicting'):cache.step('asset',0,x,x,elapsed=dt,image=image,task_id=1,clock_context=ctx+.1)
    with pytest.raises(ValueError,match='operation changed'):cache.step('asset',1,x,x,elapsed=dt,image=image,task_id=2,clock_context=ctx)
    hook.remove()


def test_context_entities_interleave_without_state_crosstalk_and_weight_changes_invalidate():
    torch.manual_seed(9598);torch.set_num_threads(1);m=ContextFilterWorld().double().eval()
    shared=ContextFilterCache(m,model_revision='one',preprocessing_revision='one',max_entities=2)
    isolated={k:ContextFilterCache(m,model_revision='one',preprocessing_revision='one') for k in ['a','b']}
    for t in range(3):
        for tag,tid in [('a',0),('b',4)]:
            x=torch.rand(1,14,dtype=torch.float64)*.01;im=torch.rand(1,1,32,32,dtype=x.dtype);dt=torch.ones(1,1,dtype=x.dtype)/15;ctx=torch.rand(1,6,dtype=x.dtype)
            out=shared.step(tag,t,x,x,elapsed=dt,image=im,task_id=tid,clock_context=ctx);expected=isolated[tag].step(tag,t,x,x,elapsed=dt,image=im,task_id=tid,clock_context=ctx)
            for k in out:torch.testing.assert_close(out[k],expected[k],rtol=0,atol=0)
    with torch.no_grad():m.clock_encoder.weight.add_(.001)
    with pytest.raises(RuntimeError,match='model changed'):shared.step('a',3,x,x,elapsed=dt,image=im,task_id=0,clock_context=ctx)
    assert not shared._entries
