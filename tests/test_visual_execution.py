import pytest
import torch
from tac_osm.variants.visual_world import VisualWorld
from tac_osm.variants.visual_execution import VisualValueCache


def test_visual_retry_cannot_reuse_forecast_with_changed_camera_or_mutable_output():
    torch.set_num_threads(1);m=VisualWorld().double().eval();calls=[]
    hook=m.visual_encoder.register_forward_hook(lambda *_:calls.append(1))
    cache=VisualValueCache(m,model_revision='checkpoint',preprocessing_revision='gray32-v1')
    x=torch.zeros(1,14,dtype=torch.float64);dt=torch.ones(1,1,dtype=x.dtype)/15;im=torch.zeros(1,1,32,32,dtype=x.dtype)
    first=cache.step('asset',0,x,x,elapsed=dt,image=im);original=first['current'].clone();first['current'].add_(100)
    retry=cache.step('asset',0,x,x,elapsed=dt,image=im.clone())
    assert len(calls)==1;torch.testing.assert_close(retry['current'],original,rtol=0,atol=0)
    with pytest.raises(ValueError,match='conflicting'):cache.step('asset',0,x,x,elapsed=dt,image=im+.1)
    with pytest.raises(ValueError,match='gap'):cache.step('asset',2,x,x,elapsed=dt,image=im)
    hook.remove()


def test_visual_interleaved_entities_match_isolated_execution_and_invalidate_on_weights():
    torch.manual_seed(9191);torch.set_num_threads(1);m=VisualWorld().double().eval()
    shared=VisualValueCache(m,model_revision='one',preprocessing_revision='one',max_entities=2)
    isolated={k:VisualValueCache(m,model_revision='one',preprocessing_revision='one') for k in ['a','b']}
    dt=torch.ones(1,1,dtype=torch.float64)/15
    for t in range(3):
        for entity in ['a','b']:
            x=torch.rand(1,14,dtype=dt.dtype)*.01;im=torch.rand(1,1,32,32,dtype=dt.dtype)
            out=shared.step(entity,t,x,x,elapsed=dt,image=im);expected=isolated[entity].step(entity,t,x,x,elapsed=dt,image=im)
            for k in out:torch.testing.assert_close(out[k],expected[k],rtol=0,atol=0)
    with torch.no_grad():m.visual_gain.add_(.01)
    with pytest.raises(RuntimeError,match='model changed'):shared.step('a',3,x,x,elapsed=dt,image=im)
    assert not shared._entries


def test_raw_physical_boundary_preserves_original_preprocessing_and_quantile_units():
    from tac_osm.variants.visual_execution import PhysicalVisualGraph
    from tac_osm.variants.reusable_execution import mixture_quantiles
    torch.set_num_threads(1);m=VisualWorld().double().eval()
    with torch.no_grad():
        m.state_mean.copy_(torch.arange(14,dtype=torch.float64));m.state_scale.fill_(2.)
        m.action_mean.fill_(.5);m.action_scale.fill_(3.)
    x=torch.rand(2,3,14);a=torch.rand_like(x);dt=torch.full((2,3,1),1/15);im=torch.randint(0,256,(2,3,1,32,32),dtype=torch.uint8)
    xn=((x-m.state_mean.float())/m.state_scale.float()).double();an=((a-m.action_mean.float())/m.action_scale.float()).double()
    initial=m.initial_state(xn[:,0]);raw,final=m.visual_tensor_graph(xn,an,dt.double(),(im.float()/255).double(),initial,last_only=True)
    expected=tuple(v*m.state_scale+m.state_mean for v in mixture_quantiles(raw))+(raw['motion_logit'].sigmoid(),*tuple(final))
    actual=PhysicalVisualGraph(m)(x,a,dt,im,tuple(initial))
    assert sum(p.numel() for p in PhysicalVisualGraph(m).parameters())==104225
    for u,v in zip(expected,actual):torch.testing.assert_close(u,v,rtol=0,atol=0)
