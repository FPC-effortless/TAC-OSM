import pytest
import torch
from tac_osm.variants.visual_world import VisualWorld


def inputs(t=7):
    torch.manual_seed(91091);torch.set_num_threads(1)
    x=torch.randn(2,t,14,dtype=torch.float64)*.01
    return x,torch.randn_like(x)*.01,torch.full((2,t,1),1/15,dtype=x.dtype),torch.rand(2,t,1,32,32,dtype=x.dtype)


def test_visual_whole_model_budget_and_active_perception_gradients():
    m=VisualWorld().double();assert sum(p.numel() for p in m.parameters())==104225
    x,a,dt,im=inputs();out=m(x,a,elapsed=dt,images=im)
    sum(v.square().mean() for v in out.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.visual_encoder.parameters())
    assert m.visual_gain.grad is not None and m.visual_gain.grad.abs().sum()>0


def test_camera_and_numeric_future_cannot_change_prefix_forecasts():
    m=VisualWorld().double();x,a,dt,im=inputs()
    base=m(x,a,elapsed=dt,images=im)
    x2=x.clone();x2[:,4:]+=2;im2=im.clone();im2[:,4:]=1-im2[:,4:]
    alt=m(x2,a,elapsed=dt,images=im2)
    for k in base:torch.testing.assert_close(base[k][:,:4],alt[k][:,:4],rtol=0,atol=0)


def test_visual_full_chunk_and_step_share_exact_state_contract():
    m=VisualWorld().double();x,a,dt,im=inputs()
    whole,last=m(x,a,elapsed=dt,images=im,return_state=True)
    _,state=m(x[:,:3],a[:,:3],elapsed=dt[:,:3],images=im[:,:3],return_state=True)
    end,state=m(x[:,3:],a[:,3:],state,elapsed=dt[:,3:],images=im[:,3:],return_state=True)
    stream=None;parts=[]
    for t in range(x.shape[1]):
        value,stream=m.step(x[:,t],a[:,t],stream,elapsed=dt[:,t],images=im[:,t]);parts.append(value)
    for k in whole:
        torch.testing.assert_close(whole[k][:,3:],end[k],rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(whole[k],torch.stack([r[k] for r in parts],1),rtol=1e-10,atol=1e-10)
    for u,v,w in zip(last,state,stream):
        torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-10);torch.testing.assert_close(u,w,rtol=1e-10,atol=1e-10)


def test_visual_inputs_are_required_and_checked():
    m=VisualWorld().double();x,a,dt,im=inputs(2)
    with pytest.raises(ValueError,match='Aligned'):m(x,a,elapsed=dt)
    with pytest.raises(ValueError,match='scaled'):m(x,a,elapsed=dt,images=im+1)


def test_visual_warm_mapping_is_explicit_and_keeps_value_gate_units_paired():
    from scripts.me091_train_world import load_initial
    from tac_osm.variants.bounded_value_world import BoundedValueWorld
    old=BoundedValueWorld();new=VisualWorld();load_initial(new,old.state_dict())
    torch.testing.assert_close(new.gated_readout.weight[:108],old.gated_readout.weight[:108],rtol=0,atol=0)
    torch.testing.assert_close(new.gated_readout.weight[108:],old.gated_readout.weight[116:224],rtol=0,atol=0)
    torch.testing.assert_close(new.heads.weight[:197],old.heads.weight[:,:108],rtol=0,atol=0)
    assert sum(p.numel() for n,p in new.named_parameters() if n.startswith('visual_'))==7792


def test_visual_batch_uses_only_images_at_or_before_each_origin():
    from scripts.me091_train_world import batch
    row={'state':torch.zeros(40,14),'action':torch.zeros(40,14),'human':torch.zeros(40,dtype=torch.bool),'human_prefix':torch.zeros(41,dtype=torch.int64),'elapsed':torch.ones(40,1)/15,'images':torch.arange(40,dtype=torch.uint8)[:,None,None,None].expand(-1,1,32,32).clone()}
    before=batch({('task',1):row},[('task',1,8)],8)
    row['images'][9:]=255
    after=batch({('task',1):row},[('task',1,8)],8)
    assert torch.equal(before[-1],after[-1])
    torch.testing.assert_close(before[-1][0,-1],torch.full((1,32,32),8/255),rtol=0,atol=0)


def test_shared_location_translates_entire_distribution_without_changing_shape():
    from tac_osm.variants.streaming_mixture_world import quantile,log_density
    m=VisualWorld().double();x,a,dt,im=inputs(2)
    with torch.no_grad():
        m.heads.weight[-14:].zero_();m.heads.bias[-14:].zero_()
        baseline=m(x,a,elapsed=dt,images=im)
        m.heads.bias[-14:].fill_(.1)
        shifted=m(x,a,elapsed=dt,images=im)
        offset=m.delta_scale*m.horizons[None,None,:,None]*torch.tanh(torch.tensor(.1,dtype=x.dtype))
        torch.testing.assert_close(shifted['current'],baseline['current']+offset,rtol=1e-12,atol=1e-12)
        for k in ['moving_delta','quiet_scale','moving_scale','mixture_logit','motion_logit']:
            torch.testing.assert_close(shifted[k],baseline[k],rtol=0,atol=0)
        target=baseline['current']+.01
        torch.testing.assert_close(log_density(baseline,target),log_density(shifted,target+offset),rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(quantile(shifted,.5),quantile(baseline,.5)+offset,rtol=1e-9,atol=1e-9)
