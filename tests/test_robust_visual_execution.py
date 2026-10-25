import pytest
import torch
from tac_osm.variants.robust_visual_world import RobustVisualWorld
from tac_osm.variants.robust_visual_execution import PhysicalRobustVisualGraph
from tac_osm.variants.visual_execution import VisualValueCache
from tac_osm.variants.multimode_distribution import mixture_quantiles


def test_raw_three_regime_graph_preserves_physical_units_and_all_carried_state():
    torch.set_num_threads(1);torch.manual_seed(9393);m=RobustVisualWorld().double().eval()
    with torch.no_grad():
        m.state_mean.copy_(torch.arange(14,dtype=torch.float64));m.state_scale.fill_(2.)
        m.action_mean.fill_(.5);m.action_scale.fill_(3.)
    x=torch.rand(2,3,14);a=torch.rand_like(x);dt=torch.full((2,3,1),1/15);pixels=torch.randint(0,256,(2,3,1,32,32),dtype=torch.uint8)
    xn=((x-m.state_mean.float())/m.state_scale.float()).double();an=((a-m.action_mean.float())/m.action_scale.float()).double();state=m.initial_state(xn[:,0])
    p,s=m.visual_tensor_graph(xn,an,dt.double(),(pixels.float()/255).double(),state,last_only=True)
    expected=tuple(q*m.state_scale+m.state_mean for q in mixture_quantiles(p))+(p['motion_logit'].sigmoid(),*tuple(s))
    graph=PhysicalRobustVisualGraph(m);actual=graph(x,a,dt,pixels,tuple(state))
    assert sum(v.numel() for v in graph.parameters())==96595
    assert len(actual)==17 and sum(v[0].numel() for v in actual[4:])==1331
    for u,v in zip(expected,actual):torch.testing.assert_close(u,v,rtol=0,atol=0)


def test_three_regime_cache_reuses_exact_event_and_isolates_entities():
    torch.set_num_threads(1);torch.manual_seed(9394);m=RobustVisualWorld().double().eval()
    cache=VisualValueCache(m,model_revision='checkpoint-sha',preprocessing_revision='original-gray32',max_entities=2)
    single={k:VisualValueCache(m,model_revision='checkpoint-sha',preprocessing_revision='original-gray32') for k in ['a','b']}
    calls=[];hook=m.visual_encoder.register_forward_hook(lambda *_:calls.append(1))
    for t in range(3):
        for entity in ['a','b']:
            x=torch.rand(1,14,dtype=torch.float64);dt=torch.ones(1,1,dtype=x.dtype)/15;im=torch.rand(1,1,32,32,dtype=x.dtype)
            out=cache.step(entity,t,x,x,elapsed=dt,image=im);expected=single[entity].step(entity,t,x,x,elapsed=dt,image=im)
            before=len(calls);retry=cache.step(entity,t,x,x,elapsed=dt,image=im);assert len(calls)==before
            for k in out:
                torch.testing.assert_close(out[k],expected[k],rtol=0,atol=0);torch.testing.assert_close(out[k],retry[k],rtol=0,atol=0)
            retry['locations'].add_(10)
            torch.testing.assert_close(cache.step(entity,t,x,x,elapsed=dt,image=im)['locations'],out['locations'],rtol=0,atol=0)
    with pytest.raises(ValueError,match='conflicting'):cache.step('b',2,x,x,elapsed=dt,image=im*.99)
    hook.remove()
